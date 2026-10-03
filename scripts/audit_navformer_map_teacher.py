from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.map_teacher import align_teacher_map_to_quest_bev, validate_teacher_record
from quest.map_training import validate_vector_record
from quest.stage3_dataset import load_record
from quest.vector_map_labels import MAP_CLASS_NAMES
from scripts.run_navformer_openscene_teacher import load_infos, select_infos


def rasterize_vectors(record: dict, xy_range: tuple[float, float, float, float], height: int, width: int) -> torch.Tensor:
    import cv2

    x0, y0, x1, y1 = xy_range
    masks = np.zeros((len(MAP_CLASS_NAMES), height, width), dtype=np.uint8)
    for class_id, points, closed in zip(record["class_ids"], record["points_xy_m"], record["is_closed"]):
        points = points.numpy()
        pixel = np.stack(((points[:, 0] - x0) * width / (x1 - x0) - 0.5,
                          (points[:, 1] - y0) * height / (y1 - y0) - 0.5), axis=-1)
        pixel = np.round(pixel).astype(np.int32)
        cv2.polylines(masks[int(class_id)], [pixel], bool(closed), 1, 1)
    return torch.from_numpy(masks.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit opaque Navformer raster channels against nuPlan vectors")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/openscene/meta_datas/meta_data_mini.pkl")
    parser.add_argument("--vector-dir", type=Path, default=PROJECT_ROOT / "data/vector_map_gt")
    parser.add_argument("--teacher-dir", type=Path, default=PROJECT_ROOT / "data/navformer_map_soft")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/navformer_map_audit.json")
    parser.add_argument("--row-axis", choices=["x", "y"], required=True)
    parser.add_argument("--row-direction", type=int, choices=[-1, 1], required=True)
    parser.add_argument("--col-direction", type=int, choices=[-1, 1], required=True)
    args = parser.parse_args()
    infos = select_infos(load_infos(args.metadata), args.sample_index, args.num_frames)
    quest_range = (-50.0, -50.0, 50.0, 50.0)
    soft_scores, ground_truth = [], []
    reference = None
    vector_provenance = None
    for source_index, info in enumerate(infos, start=args.sample_index):
        token = str(info["token"])
        vector = load_record(args.vector_dir / f"{token}.pt")
        validate_vector_record(vector, token, source_index, quest_range)
        current_vector_provenance = tuple(vector[key] for key in (
            "num_points", "min_length_m", "map_version", "vector_semantics_version"
        ))
        if vector_provenance is None:
            vector_provenance = current_vector_provenance
        elif current_vector_provenance != vector_provenance:
            raise ValueError(f"vector GT export provenance changed at {token}")
        teacher = load_record(args.teacher_dir / f"{token}.pt")
        soft = validate_teacher_record(teacher, token, source_index)
        signature = (tuple(teacher["teacher_channel_names_or_ids"]),
                     tuple(teacher["teacher_pc_range"]), str(teacher["teacher_checkpoint"]),
                     teacher["teacher_score_kind"], teacher["teacher_config_sha256"],
                     teacher["teacher_checkpoint_size_bytes"], teacher["teacher_checkpoint_mtime_ns"])
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f"teacher channel/range/checkpoint changed at {token}")
        soft_scores.append(align_teacher_map_to_quest_bev(
            soft, teacher["teacher_pc_range"], quest_range, 32, 32,
            args.row_axis, args.row_direction, args.col_direction,
        ))
        ground_truth.append(rasterize_vectors(vector, quest_range, 32, 32))
    predictions = torch.stack(soft_scores)
    gt = torch.stack(ground_truth)
    channels = []
    for channel_index, name in enumerate(reference[0]):
        channel = predictions[:, channel_index]
        thresholded = channel > 0.5
        comparisons = []
        for class_index, class_name in enumerate(MAP_CLASS_NAMES):
            target = gt[:, class_index] > 0.5
            union = int((thresholded | target).sum())
            intersection = int((thresholded & target).sum())
            expanded_prediction = F.max_pool2d(
                thresholded.float().unsqueeze(1), 3, stride=1, padding=1
            ).squeeze(1).bool()
            expanded_target = F.max_pool2d(
                target.float().unsqueeze(1), 3, stride=1, padding=1
            ).squeeze(1).bool()
            tolerant_union = int((expanded_prediction | expanded_target).sum())
            tolerant_intersection = int((expanded_prediction & expanded_target).sum())
            flat_channel = channel.flatten()
            flat_target = target.float().flatten()
            centered_channel = flat_channel - flat_channel.mean()
            centered_target = flat_target - flat_target.mean()
            denominator = centered_channel.norm() * centered_target.norm()
            correlation = float((centered_channel @ centered_target) / denominator) if float(denominator) > 0 else None
            comparisons.append({"gt_class": class_name, "iou_at_0.5": intersection / union if union else 0.0,
                                "dilated_one_cell_iou_at_0.5": (
                                    tolerant_intersection / tolerant_union if tolerant_union else 0.0
                                ),
                                "soft_correlation": correlation})
        best = max(comparisons, key=lambda item: item["iou_at_0.5"])
        item = {
            "channel": name, "mean_soft_score": float(channel.mean()),
            "max_soft_score": float(channel.max()),
            "fraction_gt_0.1": float((channel > 0.1).float().mean()),
            "fraction_gt_0.3": float((channel > 0.3).float().mean()),
            "fraction_gt_0.5": float(thresholded.float().mean()),
            "effectively_empty": bool(float(channel.max()) <= 0.1),
            "comparisons": comparisons, "best_matching_gt_class": best["gt_class"],
        }
        print(json.dumps(item, sort_keys=True))
        channels.append(item)
    audit = {
        "verified": False,
        "review_note": "Review orientation and semantic classes; explicitly set verified=true and support mask only after human inspection.",
        "sample_count": len(infos), "vector_map_classes": list(MAP_CLASS_NAMES),
        "vector_gt_provenance": dict(zip(
            ("num_points", "min_length_m", "map_version", "vector_semantics_version"),
            vector_provenance,
        )),
        "teacher_channel_names_or_ids": list(reference[0]),
        "teacher_checkpoint": reference[2], "teacher_pc_range": list(reference[1]),
        "teacher_score_kind": reference[3],
        "teacher_config_sha256": reference[4],
        "teacher_checkpoint_size_bytes": reference[5],
        "teacher_checkpoint_mtime_ns": reference[6],
        "teacher_schema_version": teacher["schema_version"],
        "row_axis": args.row_axis, "row_direction": args.row_direction,
        "col_direction": args.col_direction,
        "teacher_channel_mapping": [None] * len(reference[0]),
        "teacher_channel_support_mask": [False] * len(reference[0]),
        "teacher_channel_weights": [1.0] * len(reference[0]),
        "channel_diagnostics": channels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"audit={args.output} verified=false")


if __name__ == "__main__":
    main()
