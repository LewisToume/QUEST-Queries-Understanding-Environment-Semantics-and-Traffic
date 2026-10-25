from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.agent_training import load_checkpoint_cpu
from quest.map_diagnostics import evenly_spaced_indices, frame_geometry_stats, oracle_check
from quest.map_training import denormalize_points, load_stage3_checkpoint, load_vector_capacity_audit, stage3_forward
from quest.model import QUESTModel
from quest.stage3_dataset import collate_stage3
from quest.stage3_split import load_stage3_split
from quest.utils import load_yaml_config
from quest.vector_map_labels import MAP_CLASS_NAMES
from scripts.run_navformer_openscene_teacher import load_infos
from scripts.train_stage3_map import build_dataset, preflight_vectors, resolve


@contextmanager
def override_backbone(model, features):
    def hook(module, inputs, output):
        if features is None:
            return torch.zeros_like(output)
        if features.shape != output.shape:
            raise ValueError("other-frame backbone features have incompatible shapes")
        return features.to(output)
    handle = model.backbone.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def compare_predictions(real, changed, xy_range, confidence_threshold=0.25):
    if any(not bool(torch.isfinite(changed[key]).all()) for key in ("map_cls_logits", "map_points")):
        raise RuntimeError("non-finite map output during feature-dependency diagnostic")
    logit_delta = (real["map_cls_logits"] - changed["map_cls_logits"]).abs()
    point_delta = (denormalize_points(real["map_points"], xy_range)
                   - denormalize_points(changed["map_points"], xy_range)).abs()
    scores, labels = real["map_cls_logits"].softmax(-1).max(-1)
    foreground = (labels != len(MAP_CLASS_NAMES)) & (scores >= confidence_threshold)
    count = int(foreground.sum())
    return {"logit_mad": float(logit_delta.mean()), "point_mad_m": float(point_delta.mean()),
            "foreground_query_count": count,
            "foreground_logit_mad": float(logit_delta[foreground].mean()) if count else None,
            "foreground_point_mad_m": float(point_delta[foreground].mean()) if count else None,
            "foreground_selection": "same query IDs selected from real predictions"}


def save_figure(path, batch, labels, points, record, camera_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("tab:blue", "tab:orange", "tab:green")
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    flat = axes.flatten()
    for line, label, closed in zip(record["points_xy_m"], record["class_ids"], record["is_closed"]):
        line = line.cpu().numpy()
        if bool(closed):
            line = np.concatenate((line, line[:1]))
        for axis in (flat[0], flat[2]):
            axis.plot(line[:, 0], line[:, 1], color=colors[int(label)], linewidth=1)
    for line, label in zip(points.cpu().numpy(), labels.cpu().tolist()):
        for axis in (flat[1], flat[2]):
            axis.plot(line[:, 0], line[:, 1], color=colors[label], linewidth=1,
                      linestyle="--" if axis is flat[2] else "-")
    xy_range = record["xy_range_m"]
    for axis, title in zip(flat[:3], ("nuPlan GT", "QUEST prediction", "GT solid / prediction dashed")):
        axis.set(xlim=(xy_range[0], xy_range[2]), ylim=(xy_range[1], xy_range[3]),
                 xlabel="LiDAR X (m)", ylabel="LiDAR Y (m)", title=title, aspect="equal")
        axis.plot(0, 0, "k+")
    flat[3].axis("off")
    flat[3].text(0, 0.8, "blue: centerline\norange: ped_crossing\ngreen: road_boundary\n"
                  "Camera panels: RGB only\nMap has no measured elevation;\nno guessed ground-plane projection.", fontsize=11)
    images = batch["images"][0].cpu()
    images = (images * images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
              + images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]).clamp(0, 1)
    for axis, image, name in zip(flat[4:], images, camera_names):
        axis.imshow(image.permute(1, 2, 0).numpy())
        axis.set_title(name)
        axis.axis("off")
    fig.suptitle(f"index={record['sample_index']} token={record['token']}")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Read-only Stage 3 train/validation geometry diagnostics")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-per-split", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/diagnostics/stage3_map")
    args = parser.parse_args()
    output = resolve(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("diagnostic output directory is not empty; use a new --output-dir")
    config = load_yaml_config(PROJECT_ROOT / "configs/stage3_map.yaml")
    stage1 = load_yaml_config(PROJECT_ROOT / "configs/stage1.yaml")
    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    checkpoint = load_checkpoint_cpu(resolve(args.checkpoint))
    if checkpoint.get("map_kd_enabled", checkpoint.get("stage") == "map_hybrid_distillation"):
        raise ValueError("this diagnostic currently supports GT-only checkpoints")
    config["stage"] = "map_gt_agent"
    config["loss_weights"]["lambda_map_kd"] = 0.0
    metadata_path = resolve(stage1["dataset"]["metadata_path"])
    manifest = load_stage3_split(resolve(config["paths"]["split_manifest_path"]), load_infos(metadata_path), metadata_path)
    if checkpoint.get("training_config", {}).get("stage3_split_sha256") != manifest["split_sha256"]:
        raise ValueError("checkpoint differs from current split")
    query_count, provenance = load_vector_capacity_audit(resolve(config["paths"]["vector_capacity_audit_path"]), config, manifest)
    if checkpoint.get("vector_gt_provenance") != provenance:
        raise ValueError("checkpoint differs from certified Vector GT provenance")
    model_config.update(C_map=len(MAP_CLASS_NAMES), N_map=query_count, P=20)
    if not resolve(model_config["local_backbone_dir"]).is_dir():
        raise FileNotFoundError("local DINOv2 weights missing")
    model = QUESTModel(**model_config)
    load_stage3_checkpoint(model, None, checkpoint, expected_map_kd_enabled=False)
    del checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    output.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": str(resolve(args.checkpoint)), "split_sha256": manifest["split_sha256"],
              "metric": "custom_class_aware_1.5m; not official mAP", "frames": []}
    previous_features, previous_token = None, None
    for split in ("train", "validation"):
        indices = evenly_spaced_indices(manifest[split]["frames"], args.num_per_split)
        dataset = build_dataset(model_config, config, stage1, None, indices, provenance)
        if preflight_vectors(dataset, query_count) != provenance:
            raise ValueError("diagnostic Vector GT provenance mismatch")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_stage3)
        with torch.no_grad():
            for position, batch in enumerate(loader):
                record = batch["vector_targets"][0]
                oracle_check(record, dataset.quest_range)
                captured = []
                capture_hook = model.backbone.register_forward_hook(
                    lambda module, inputs, output: captured.append(output.detach().cpu()))
                try:
                    predictions = stage3_forward(model, None, batch, device)
                finally:
                    capture_hook.remove()
                with override_backbone(model, None):
                    zero_prediction = stage3_forward(model, None, batch, device)
                dependency = {"real_vs_zero_feature": compare_predictions(
                    predictions, zero_prediction, dataset.quest_range, float(config["eval"]["confidence_threshold"])),
                              "other_frame_token": previous_token, "real_vs_other_frame_feature": None}
                del zero_prediction
                if previous_features is not None:
                    with override_backbone(model, previous_features):
                        other_prediction = stage3_forward(model, None, batch, device)
                    dependency["real_vs_other_frame_feature"] = compare_predictions(
                        predictions, other_prediction, dataset.quest_range, float(config["eval"]["confidence_threshold"]))
                    del other_prediction
                previous_features, previous_token = captured[0], record["token"]
                logits = predictions["map_cls_logits"][0]
                normalized = predictions["map_points"][0]
                if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(normalized).all()):
                    raise RuntimeError(f"non-finite map prediction: {record['token']}")
                scores, labels = logits.softmax(-1).max(-1)
                all_query_labels = labels.cpu().tolist()
                keep = (labels != len(MAP_CLASS_NAMES)) & (scores >= float(config["eval"]["confidence_threshold"]))
                points = denormalize_points(normalized[keep], dataset.quest_range).cpu()
                labels = labels[keep].cpu()
                stats = frame_geometry_stats(labels, points, record)
                _, anchor_visibility, _ = model.geometry_lift.project_reference_points(
                    batch["intrinsics"].to(device), batch["extrinsics"].to(device), (252, 448))
                visible_cells = anchor_visibility.any(dim=1).any(dim=-1)[0].cpu()
                x0, y0, x1, y1 = dataset.quest_range
                gt_points = record["points_xy_m"]
                columns = ((gt_points[..., 0] - x0) / (x1 - x0) * model.bev_w).long().clamp(0, model.bev_w - 1)
                rows = ((gt_points[..., 1] - y0) / (y1 - y0) * model.bev_h).long().clamp(0, model.bev_h - 1)
                coverage = visible_cells[rows * model.bev_w + columns]
                for class_id, name in enumerate(MAP_CLASS_NAMES):
                    selected_class = record["class_ids"] == class_id
                    stats[name]["fraction_in_geometrically_visible_bev_cells"] = (
                        float(coverage[selected_class].float().mean()) if bool(selected_class.any()) else None)
                info = dataset.images.infos[position]
                row = {"split": split, "index": indices[position], "token": record["token"],
                       "scene_token": str(info["scene_token"]), "map_location": record["map_location"],
                       "oracle": "PASS", "per_class": stats,
                       "visibility_note": "BEV anchor projection only; not an occlusion or map-height validity test",
                       "feature_dependency": dependency,
                       "confidence_threshold": float(config["eval"]["confidence_threshold"]),
                       "queries": {"labels": all_query_labels, "scores": scores.cpu().tolist(),
                                   "points_xy_m": denormalize_points(normalized, dataset.quest_range).cpu().tolist()},
                       "gt": {key: record[key].tolist() for key in
                              ("class_ids", "points_xy_m", "is_closed", "length_m")},
                       "intrinsics": batch["intrinsics"][0].tolist(),
                       "sensor2lidar": batch["extrinsics"][0].tolist(),
                       "camera_names": list(stage1["dataset"]["camera_names"]),
                       "lidar2global": np.asarray(info["lidar2global"]).tolist(),
                       "ego2global": np.asarray(info["ego2global"]).tolist()}
                report["frames"].append(row)
                save_figure(output / f"{split}_{indices[position]}.png", batch, labels, points, record, row["camera_names"])
                print(f"{split} index={indices[position]} oracle=PASS " + " ".join(
                    f"{name}:GT={item['gt']},pred={item['predictions']},"
                    f"ordered_hits={item['ordered_matched_1_5m']},chamfer_hits={item['chamfer_matched_1_5m']}"
                    for name, item in stats.items()), flush=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"report={output / 'report.json'} figures={len(report['frames'])}")


if __name__ == "__main__":
    main()
