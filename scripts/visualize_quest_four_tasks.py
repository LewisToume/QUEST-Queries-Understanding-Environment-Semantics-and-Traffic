"""One combined checkpoint, one forward per target, four prediction views (not GT)."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from quest.agent_training import decode_supported_agent_predictions, load_checkpoint_cpu
from quest.cuda_runtime import initialize_cuda_device
from quest.map_training import denormalize_points
from quest.seg_depth_dataset import collate_seg_depth
from quest.seg_depth_diagnostics import color_depth, preview_positions, rgb_from_normalized
from quest.seg_on_v3 import load_seg_on_v3
from quest.seg_teacher import CITYSCAPES_CLASSES, CITYSCAPES_PALETTE
from quest.stage3_split import file_sha256
from quest.teacher_adapters import NAVFORMER_CLASS_SUPPORT
from quest.vector_map_labels import MAP_CLASS_NAMES
from scripts.train_seg_on_v3 import dataset, inputs
from scripts.train_seg_depth import load_context, resolve


def box_outline(center, size, yaw):
    """Canonical metric [dx,dy,dz], yaw about LiDAR Z; return closed XY outline."""
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * np.asarray(size[:2]) / 2
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return corners @ rotation.T + np.asarray(center[:2])


def decode_outputs(output, base, metadata, agent_threshold, map_threshold):
    recipe = metadata["training_config"]["temporal_agent"]["recipe"]
    single = {key: output[key][0].float() for key in ("agent_cls_logits", "agent_boxes", "agent_velocity")}
    agent = decode_supported_agent_predictions(single, NAVFORMER_CLASS_SUPPORT, agent_threshold,
        base.geometry_lift.x_range, base.geometry_lift.y_range,
        recipe["agent_loss"]["z_range"], recipe["agent_loss"]["size_norm"])
    agent = {key: value.detach().cpu().numpy() for key, value in agent.items()}
    xy_range = (base.geometry_lift.x_range[0], base.geometry_lift.y_range[0],
                base.geometry_lift.x_range[1], base.geometry_lift.y_range[1])
    scores, labels = output["map_cls_logits"][0].float().softmax(-1).max(-1)
    keep = (labels != len(MAP_CLASS_NAMES)) & (scores >= map_threshold)
    points = denormalize_points(output["map_points"][0, keep].float(), xy_range).detach().cpu().numpy()
    return agent, {"points_xy_m": points, "labels": labels[keep].cpu().numpy(),
                   "scores": scores[keep].cpu().numpy()}, xy_range


def render(path, images, output, cameras, agent, vectors, xy_range, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure = plt.figure(figsize=(16, 5 + 1.8 * len(cameras)), layout="constrained")
    grid = figure.add_gridspec(len(cameras) + 2, 6, height_ratios=[1.2, 1.2] + [1] * len(cameras))
    agent_axis, map_axis = figure.add_subplot(grid[:2, :3]), figure.add_subplot(grid[:2, 3:])
    agent_colors = ("tab:red", "tab:purple")
    map_colors = ("tab:blue", "tab:orange", "tab:green")
    for center, size, yaw, label in zip(agent["centers"], agent["sizes"], agent["yaw"], agent["labels"]):
        line = box_outline(center, size, yaw)
        agent_axis.plot(line[:, 0], line[:, 1], color=agent_colors[int(label)], linewidth=.8)
        agent_axis.arrow(center[0], center[1], np.cos(yaw), np.sin(yaw),
                         color=agent_colors[int(label)], head_width=.35, length_includes_head=True)
    agent_axis.legend(handles=[Line2D([0], [0], color=c, label=n)
                               for c, n in zip(agent_colors, ("vehicle", "pedestrian"))], fontsize=7)
    for points, label in zip(vectors["points_xy_m"], vectors["labels"]):
        map_axis.plot(points[:, 0], points[:, 1], color=map_colors[int(label)], linewidth=.9)
    map_axis.legend(handles=[Line2D([0], [0], color=c, label=n)
                             for c, n in zip(map_colors, MAP_CLASS_NAMES)], fontsize=7)
    for axis, name in ((agent_axis, "Agent: current frame, supported classes only"),
                       (map_axis, "Vector Map: current frame")):
        axis.set(title=name, xlim=(xy_range[0], xy_range[2]), ylim=(xy_range[1], xy_range[3]),
                 xlabel="LiDAR X (m)", ylabel="LiDAR Y (m)", aspect="equal")
        axis.plot(0, 0, "k+", markersize=8)
        axis.grid(alpha=.2)
    labels = output["seg_logits"][0].argmax(1).cpu().numpy()
    depth = output["depth"][0, :, 0].float().cpu().numpy()
    palette = np.asarray(CITYSCAPES_PALETTE, dtype=np.uint8)
    for index, camera in enumerate(cameras):
        valid = np.isfinite(depth[index]) & (depth[index] > 0)
        panels = (rgb_from_normalized(images[index]), palette[labels[index]],
                  color_depth(depth[index], valid, .1, 80.))
        names = (camera + " RGB", "QUEST Seg / Cityscapes19", "QUEST camera-Z / log 0.1-80m")
        for col, (panel, name) in enumerate(zip(panels, names)):
            axis = figure.add_subplot(grid[index + 2, col * 2:col * 2 + 2])
            axis.imshow(panel)
            axis.set_title(name, fontsize=8)
            axis.axis("off")
    figure.suptitle(title + "\nQUEST predictions only, NOT GT; depth geometry remains experimental", fontsize=10)
    figure.savefig(path, dpi=120)
    plt.close(figure)


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/quest_seg_on_depth_v3_v1_best.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preview-frames", type=int, default=3)
    parser.add_argument("--sample-index", type=int, action="append", help="Raw validation metadata index; repeatable")
    parser.add_argument("--agent-threshold", type=float, default=.25)
    parser.add_argument("--map-threshold", type=float, default=.25)
    args = parser.parse_args(argv)
    if args.preview_frames <= 0 or not all(0 <= x <= 1 for x in (args.agent_threshold, args.map_threshold)):
        parser.error("Require positive preview count and thresholds in [0,1]")
    destination = resolve(args.output_dir or Path("data/diagnostics") / (
        "quest_four_tasks_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("Choose a new/empty --output-dir: " + str(destination))
    checkpoint_path = resolve(args.checkpoint)
    saved = load_checkpoint_cpu(checkpoint_path)
    config, metadata = saved["training_config"], saved["base_metadata"]
    data_config, manifest = load_context({"paths": {
        "dataset_config": config["dataset_config"], "split_manifest": config["split_manifest"]}})
    if any(metadata[key] != manifest[key] for key in ("split_sha256", "metadata_sha256")):
        raise ValueError("Checkpoint metadata/split differs from visualization data")
    data, audit = dataset(config, metadata, data_config, manifest, "validation")
    if audit != saved["label_audits"]["validation"]:
        raise ValueError("Validation labels/context changed since training")
    if args.sample_index:
        positions = [data.source_indices.index(index) for index in dict.fromkeys(args.sample_index)]
    else:
        # Prefer the next frame in each chosen scene to exercise real history, not just cold starts.
        positions = preview_positions(data.images.infos, args.preview_frames)
        positions = [p + 1 if p + 1 < len(data) and data.history_plans[p + 1]["history_index"] == data.source_indices[p]
                     else p for p in positions]
        positions = list(dict.fromkeys(positions))
    device = initialize_cuda_device()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Combined profile requires CUDA bf16")
    model = load_seg_on_v3(saved).to(device).eval()
    report = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path),
        "epoch": saved["epoch"], "split": "validation", "split_sha256": manifest["split_sha256"],
        "one_checkpoint": True, "forward_calls_per_frame": 1, "online_teacher_models": 0,
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "routes": {"agent": "original_current_only", "map": "current_only", "seg": "current_only",
                   "depth": "current_plus_strict_past_masked_at_cold_start"},
        "reference": "PREDICTIONS_ONLY_NOT_GT_NOT_ACCURACY_EVALUATION",
        "depth_geometry": "experimental_image_plane", "agent_threshold": args.agent_threshold,
        "map_threshold": args.map_threshold, "seg_classes": list(CITYSCAPES_CLASSES),
        "depth_display": {"min_m": .1, "max_m": 80., "scale": "log", "black": "nonfinite_or_nonpositive"},
        "frames": []}
    destination.mkdir(parents=True, exist_ok=True)
    for position in positions:
        sample = data[position]
        batch = collate_seg_depth([sample])
        output = model(**inputs(batch, device))
        required = ("agent_cls_logits", "agent_boxes", "agent_velocity", "map_cls_logits", "map_points", "seg_logits", "depth")
        if not all(key in output and bool(torch.isfinite(output[key]).all()) for key in required):
            raise RuntimeError("Missing/nonfinite four-task prediction; no completed report")
        agent, vectors, xy_range = decode_outputs(output, model.base, metadata, args.agent_threshold, args.map_threshold)
        index, token = sample["source_index"], sample["sample_token"]
        filename = f"validation_{index}_four_tasks.png"
        identity = sample["history_identity"]
        render(destination / filename, sample["images"], output, data.images.camera_names, agent, vectors, xy_range,
               f"index={index} token={token} previous={identity['history_index']} dt={identity['dt_s']:.3f}s")
        row = {"source_index": index, "token": token, "history": identity, "figure": filename,
               "output_shapes": {key: list(output[key].shape) for key in required},
               "agent_counts": {name: int((agent["labels"] == i).sum()) for i, name in enumerate(("vehicle", "pedestrian"))},
               "map_counts": {name: int((vectors["labels"] == i).sum()) for i, name in enumerate(MAP_CLASS_NAMES)}}
        report["frames"].append(row)
        print(f"four_tasks=PASS index={index} history={identity['history_index']} figure={filename}", flush=True)
    report["complete"] = True
    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("output_dir=" + str(destination))


if __name__ == "__main__":
    main()
