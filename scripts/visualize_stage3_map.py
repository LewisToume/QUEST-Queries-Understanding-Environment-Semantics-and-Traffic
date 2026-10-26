from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.agent_training import load_checkpoint_cpu
from quest.map_teacher import MapRasterDistillHead
from quest.map_training import (VECTOR_PROVENANCE_KEYS, denormalize_points,
                                load_stage3_checkpoint, load_vector_capacity_audit,
                                stage3_forward, stage3_kd_enabled, stage3_model_config)
from quest.model import QUESTModel
from quest.stage3_dataset import collate_stage3, load_teacher_audit
from quest.stage3_split import load_stage3_split
from quest.utils import load_yaml_config
from quest.vector_map_labels import MAP_CLASS_NAMES
from scripts.train_stage3_map import build_dataset, resolve


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Stage 3 map orientation and vector outputs")
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/stage3_map_diagnostic.png")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    config = load_yaml_config(PROJECT_ROOT / "configs/stage3_map.yaml")
    checkpoint = load_checkpoint_cpu(resolve(args.checkpoint or config["paths"]["checkpoint_path"]))
    kd_enabled = stage3_kd_enabled(config)
    stage1 = load_yaml_config(PROJECT_ROOT / "configs/stage1.yaml")
    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    model_config = stage3_model_config(model_config, config, checkpoint)
    from scripts.run_navformer_openscene_teacher import load_infos
    metadata_path = resolve(stage1["dataset"]["metadata_path"])
    manifest = load_stage3_split(resolve(config["paths"]["split_manifest_path"]),
                                 load_infos(metadata_path), metadata_path)
    query_count, audited_provenance = load_vector_capacity_audit(
        resolve(config["paths"]["vector_capacity_audit_path"]), config, manifest
    )
    model_config.update(C_map=len(MAP_CLASS_NAMES), N_map=query_count, P=20)
    audit = (load_teacher_audit(resolve(config["paths"]["teacher_audit_path"]),
                                expected_split_sha256=manifest["split_sha256"])
             if kd_enabled else None)
    model = QUESTModel(**model_config)
    raster_head = (MapRasterDistillHead(model.hidden_dim, len(audit["teacher_channel_names_or_ids"]))
                   if kd_enabled else None)
    if checkpoint.get("training_config", {}).get("stage3_split_sha256") != manifest["split_sha256"]:
        raise ValueError("visualization checkpoint was trained for a different Stage 3 split")
    if kd_enabled:
        teacher_metadata = checkpoint.get("teacher_channel_metadata")
        if not isinstance(teacher_metadata, dict):
            raise ValueError("hybrid checkpoint lacks teacher metadata")
        for key in ("teacher_checkpoint", "teacher_alignment_version", "teacher_coordinate_frame",
                    "row_axis", "row_direction", "col_direction", "teacher_pc_range"):
            if teacher_metadata.get(key) != audit[key]:
                raise ValueError(f"teacher {key} differs from visualization audit")
        if checkpoint.get("teacher_channel_support_mask") != audit["teacher_channel_support_mask"]:
            raise ValueError("teacher support mask differs from visualization audit")
    load_stage3_checkpoint(model, raster_head, checkpoint, expected_map_kd_enabled=kd_enabled)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    if raster_head is not None:
        raster_head.to(device).eval()
    sample_index = (args.sample_index if args.sample_index is not None
                    else manifest[args.split]["frames"][0]["index"])
    if sample_index not in {row["index"] for row in manifest[args.split]["frames"]}:
        raise ValueError("visualization sample-index is not in the selected manifest split")
    dataset = build_dataset(model_config, config, stage1, audit, [sample_index], audited_provenance)
    sample = dataset[0]
    if checkpoint["vector_gt_provenance"] != audited_provenance or audited_provenance != {
        key: sample["vector_target"][key] for key in VECTOR_PROVENANCE_KEYS
    }:
        raise ValueError("visualization vector GT provenance differs from Stage 3 checkpoint")
    batch = collate_stage3([sample])
    with torch.no_grad():
        result = stage3_forward(model, raster_head, batch, device)
    xy_range = (model.geometry_lift.x_range[0], model.geometry_lift.y_range[0],
                model.geometry_lift.x_range[1], model.geometry_lift.y_range[1])
    classes = result["map_cls_logits"][0].softmax(-1)
    scores, labels = classes.max(-1)
    keep = (labels != len(MAP_CLASS_NAMES)) & (scores >= float(config["eval"]["confidence_threshold"]))
    predicted = denormalize_points(result["map_points"][0, keep], xy_range).cpu()
    pred_labels = labels[keep].cpu()
    colors = ("tab:blue", "tab:orange", "tab:green")
    figure, axes = plt.subplots(2 if kd_enabled else 1, 2,
                                figsize=(12, 12) if kd_enabled else (12, 6),
                                constrained_layout=True)
    axes = np.atleast_2d(axes)
    for point, class_id, closed in zip(batch["vector_targets"][0]["points_xy_m"],
                                       batch["vector_targets"][0]["class_ids"],
                                       batch["vector_targets"][0]["is_closed"]):
        line = point.numpy()
        if bool(closed):
            line = np.concatenate((line, line[:1]))
        axes[0, 0].plot(line[:, 0], line[:, 1], color=colors[int(class_id)], linewidth=1)
    for point, class_id in zip(predicted, pred_labels):
        line = point.numpy()
        axes[0, 1].plot(line[:, 0], line[:, 1], color=colors[int(class_id)], linewidth=1)
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], color=color, label=name)
              for color, name in zip(colors, MAP_CLASS_NAMES)]
    axes[0, 0].legend(handles=legend, loc="upper right")
    axes[0, 1].legend(handles=legend, loc="upper right")
    titles = ["nuPlan vector GT", "QUEST vector prediction"]
    if kd_enabled:
        channel_index = audit["teacher_channel_support_mask"].index(True)
        channel_name = audit["teacher_channel_names_or_ids"][channel_index]
        teacher = np.where(batch["teacher_map_valid"][0].numpy(),
                           batch["teacher_map_aligned"][0, channel_index].numpy(), np.nan)
        student = result["student_map_raster_logits"][0, channel_index].sigmoid().cpu().numpy()
        extent = [xy_range[0], xy_range[2], xy_range[1], xy_range[3]]
        axes[1, 0].imshow(teacher, origin="lower", extent=extent, vmin=0, vmax=1)
        axes[1, 1].imshow(student, origin="lower", extent=extent, vmin=0, vmax=1)
        titles.extend((f"Navformer teacher {channel_name}", f"QUEST raster KD {channel_name}"))
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlim(xy_range[0], xy_range[2])
        axis.set_ylim(xy_range[1], xy_range[3])
        axis.set_aspect("equal")
        axis.plot([0], [0], "k+")
        axis.annotate("+X", xy=(10, 0), xytext=(0, 0), arrowprops={"arrowstyle": "->"})
        axis.annotate("+Y", xy=(0, 10), xytext=(0, 0), arrowprops={"arrowstyle": "->"})
        axis.set_xlabel("LiDAR X (m)")
        axis.set_ylabel("LiDAR Y (m)")
    figure.suptitle(f"metadata_index={sample_index} token={batch['sample_token'][0]} "
                    f"map_kd={'ON' if kd_enabled else 'OFF'}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(f"visualization={args.output}")


if __name__ == "__main__":
    main()
