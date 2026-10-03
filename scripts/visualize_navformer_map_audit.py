from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.map_teacher import (
    TEACHER_ALIGNMENT_VERSION, TEACHER_COORDINATE_FRAME,
    align_teacher_map_to_quest_bev, resolve_lidar2ego, validate_teacher_record,
)
from quest.map_training import validate_vector_record
from quest.stage3_dataset import load_record
from quest.utils import load_yaml_config
from quest.vector_map_labels import MAP_CLASS_NAMES
from scripts.audit_navformer_map_teacher import rasterize_vectors
from scripts.run_navformer_openscene_teacher import load_infos, select_infos


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize nuPlan vectors and Navformer raster scores before Stage 3 training")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/openscene/meta_datas/meta_data_mini.pkl")
    parser.add_argument("--vector-dir", type=Path, default=PROJECT_ROOT / "data/vector_map_gt")
    parser.add_argument("--teacher-dir", type=Path, default=PROJECT_ROOT / "data/navformer_map_soft")
    parser.add_argument("--audit", type=Path, default=PROJECT_ROOT / "data/navformer_map_audit.json")
    parser.add_argument("--row-axis", choices=["x", "y"])
    parser.add_argument("--row-direction", type=int, choices=[-1, 1])
    parser.add_argument("--col-direction", type=int, choices=[-1, 1])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/map_audit_preview")
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8")) if args.audit.is_file() else {}
    if audit and (audit.get("teacher_alignment_version") != TEACHER_ALIGNMENT_VERSION
                  or audit.get("teacher_coordinate_frame") != TEACHER_COORDINATE_FRAME):
        raise ValueError("existing audit uses old teacher coordinates; regenerate the audit JSON")
    row_axis = args.row_axis if args.row_axis is not None else audit.get("row_axis")
    row_direction = args.row_direction if args.row_direction is not None else audit.get("row_direction")
    col_direction = args.col_direction if args.col_direction is not None else audit.get("col_direction")
    if row_axis not in ("x", "y") or row_direction not in (-1, 1) or col_direction not in (-1, 1):
        raise ValueError("provide candidate row-axis/row-direction/col-direction or an audit JSON; VERIFIED is not required")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    x0, x1 = (float(v) for v in model_config["x_range"])
    y0, y1 = (float(v) for v in model_config["y_range"])
    height, width = int(model_config["bev_h"]), int(model_config["bev_w"])
    xy_range = (x0, y0, x1, y1)
    infos = select_infos(load_infos(args.metadata), args.sample_index, args.num_frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sample_index, info in enumerate(infos, start=args.sample_index):
        token = str(info["token"])
        vector = load_record(args.vector_dir / f"{token}.pt")
        validate_vector_record(vector, token, sample_index, xy_range)
        teacher = load_record(args.teacher_dir / f"{token}.pt")
        scores = validate_teacher_record(teacher, token, sample_index)
        lidar2ego = resolve_lidar2ego(info)
        if not torch.allclose(teacher["teacher_lidar2ego"], lidar2ego, atol=1e-4, rtol=1e-4):
            raise ValueError(f"teacher lidar2ego differs from metadata for {token}")
        aligned, valid = align_teacher_map_to_quest_bev(
            scores, teacher["teacher_pc_range"], xy_range,
            height, width, row_axis, row_direction, col_direction,
            lidar2ego=lidar2ego,
        )
        masks = rasterize_vectors(vector, xy_range, height, width)
        names = list(teacher["teacher_channel_names_or_ids"])
        if audit and names != audit.get("teacher_channel_names_or_ids"):
            raise ValueError(f"teacher channel order differs from audit for {token}")
        panel_count = 1 + len(MAP_CLASS_NAMES) + aligned.shape[0]
        columns = min(4, panel_count)
        rows = math.ceil(panel_count / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False, constrained_layout=True)
        panels = list(axes.flat)
        colors = ("tab:blue", "tab:orange", "tab:green")
        for points, class_id, closed in zip(vector["points_xy_m"], vector["class_ids"], vector["is_closed"]):
            line = points.numpy()
            if bool(closed):
                line = np.concatenate((line, line[:1]))
            panels[0].plot(line[:, 0], line[:, 1], color=colors[int(class_id)], linewidth=1)
        panels[0].set_title("nuPlan vector GT")
        extent = [x0, x1, y0, y1]
        for class_id, name in enumerate(MAP_CLASS_NAMES):
            axis = panels[1 + class_id]
            axis.imshow(masks[class_id].numpy(), origin="lower", extent=extent, vmin=0, vmax=1)
            axis.set_title(f"GT raster: {name}")
        for channel_id, name in enumerate(names):
            axis = panels[1 + len(MAP_CLASS_NAMES) + channel_id]
            axis.imshow(np.where(valid.numpy(), aligned[channel_id].numpy(), np.nan),
                        origin="lower", extent=extent, vmin=0, vmax=1)
            candidate = "unmapped"
            diagnostics = audit.get("channel_diagnostics", [])
            if channel_id < len(diagnostics):
                candidate = diagnostics[channel_id].get("best_matching_gt_class") or "unmapped"
            axis.set_title(f"Teacher {channel_id}: {name}\ncandidate: {candidate}")
        for axis in panels[:panel_count]:
            axis.set_xlim(x0, x1)
            axis.set_ylim(y0, y1)
            axis.set_aspect("equal")
            axis.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                      color="black", linestyle="--", linewidth=0.7)
            axis.plot(0, 0, "k+")
            axis.annotate("+X", xy=(10, 0), xytext=(0, 0), arrowprops={"arrowstyle": "->"})
            axis.annotate("+Y", xy=(0, 10), xytext=(0, 0), arrowprops={"arrowstyle": "->"})
            axis.set_xlabel("LiDAR X (m)")
            axis.set_ylabel("LiDAR Y (m)")
        for axis in panels[panel_count:]:
            axis.axis("off")
        figure.suptitle(f"metadata_index={sample_index} token={token} axis={row_axis}/{row_direction}/{col_direction}")
        path = args.output_dir / f"{sample_index}_{token}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        print(f"index={sample_index} token={token} preview={path}")


if __name__ == "__main__":
    main()
