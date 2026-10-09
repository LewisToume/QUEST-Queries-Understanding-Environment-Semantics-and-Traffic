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
                                load_stage3_checkpoint, load_vector_capacity_audit, stage3_forward)
from quest.model import QUESTModel
from quest.stage3_dataset import collate_stage3, load_teacher_audit
from quest.utils import load_yaml_config
from quest.vector_map_labels import MAP_CLASS_NAMES
from scripts.train_stage3_map import build_dataset, resolve


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Stage 3 map orientation and vector outputs")
    parser.add_argument("--sample-index", type=int, default=5000)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/stage3_map_diagnostic.png")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    config = load_yaml_config(PROJECT_ROOT / "configs/stage3_map.yaml")
    stage1 = load_yaml_config(PROJECT_ROOT / "configs/stage1.yaml")
    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    query_count, audited_provenance = load_vector_capacity_audit(
        resolve(config["paths"]["vector_capacity_audit_path"]), config
    )
    model_config.update(C_map=3, N_map=query_count, P=20)
    model = QUESTModel(**model_config)
    audit = load_teacher_audit(resolve(config["paths"]["teacher_audit_path"]))
    raster_head = MapRasterDistillHead(model.hidden_dim, len(audit["teacher_channel_names_or_ids"]))
    checkpoint = load_checkpoint_cpu(resolve(args.checkpoint or config["paths"]["checkpoint_path"]))
    load_stage3_checkpoint(model, raster_head, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    raster_head.to(device).eval()
    dataset = build_dataset(model, config, stage1, audit, args.sample_index, 1)
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
    keep = (labels != 3) & (scores >= float(config["eval"]["confidence_threshold"]))
    predicted = denormalize_points(result["map_points"][0, keep], xy_range).cpu()
    pred_labels = labels[keep].cpu()
    colors = ("tab:blue", "tab:orange", "tab:green")
    figure, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
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
    channel_index = audit["teacher_channel_support_mask"].index(True)
    channel_name = audit["teacher_channel_names_or_ids"][channel_index]
    teacher = np.where(batch["teacher_map_valid"][0].numpy(),
                       batch["teacher_map_aligned"][0, channel_index].numpy(), np.nan)
    student = result["student_map_raster_logits"][0, channel_index].sigmoid().cpu().numpy()
    extent = [xy_range[0], xy_range[2], xy_range[1], xy_range[3]]
    axes[1, 0].imshow(teacher, origin="lower", extent=extent, vmin=0, vmax=1)
    axes[1, 1].imshow(student, origin="lower", extent=extent, vmin=0, vmax=1)
    titles = ("nuPlan vector GT", "QUEST vector prediction",
              f"Navformer teacher {channel_name}", f"QUEST raster KD {channel_name}")
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
    figure.suptitle(f"metadata_index={args.sample_index} token={batch['sample_token'][0]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(f"visualization={args.output}")


if __name__ == "__main__":
    main()
