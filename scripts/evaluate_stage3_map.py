from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.agent_training import load_checkpoint_cpu
from quest.map_teacher import MapRasterDistillHead, soft_map_distillation_loss
from quest.map_training import (
    class_aware_vector_matches, denormalize_points, load_stage3_checkpoint,
    stage3_forward,
)
from quest.model import QUESTModel
from quest.stage3_dataset import collate_stage3, load_teacher_audit
from quest.utils import load_yaml_config
from quest.vector_map_labels import MAP_CLASS_NAMES
from scripts.train_stage3_map import build_dataset, preflight_vectors, resolve


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 vector map against nuPlan GT")
    parser.add_argument("--start", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    config = load_yaml_config(PROJECT_ROOT / "configs/stage3_map.yaml")
    stage1 = load_yaml_config(PROJECT_ROOT / "configs/stage1.yaml")
    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    model_config.update(C_map=len(MAP_CLASS_NAMES), N_map=int(config["map"]["map_query_count"]), P=20)
    model = QUESTModel(**model_config)
    audit = load_teacher_audit(resolve(config["paths"]["teacher_audit_path"]))
    start = args.start if args.start is not None else int(config["eval"]["start_index"])
    count = args.count if args.count is not None else int(config["eval"]["num_samples"])
    dataset = build_dataset(model, config, stage1, audit, start, count)
    vector_provenance = preflight_vectors(dataset, int(config["map"]["map_query_count"]))
    loader = DataLoader(dataset, batch_size=int(config["eval"]["batch_size"]),
                        shuffle=False, num_workers=0, collate_fn=collate_stage3)
    raster_head = MapRasterDistillHead(model.hidden_dim, len(audit["teacher_channel_names_or_ids"]))
    checkpoint_path = resolve(args.checkpoint or config["paths"]["checkpoint_path"])
    checkpoint = load_checkpoint_cpu(checkpoint_path)
    if checkpoint.get("vector_gt_provenance") != vector_provenance:
        raise ValueError("evaluation vector GT export provenance differs from Stage 3 checkpoint")
    if tuple(checkpoint.get("map_class_names", ())) != MAP_CLASS_NAMES:
        raise ValueError("checkpoint map taxonomy differs from direct GT")
    teacher_metadata = checkpoint.get("teacher_channel_metadata")
    if not isinstance(teacher_metadata, dict):
        raise ValueError("Stage 3 checkpoint has no teacher channel metadata")
    if teacher_metadata.get("teacher_checkpoint") != audit["teacher_checkpoint"]:
        raise ValueError("teacher audit/checkpoint identity differs from Stage 3 checkpoint")
    if list(checkpoint.get("teacher_channel_support_mask", [])) != audit["teacher_channel_support_mask"]:
        raise ValueError("teacher support mask differs from Stage 3 checkpoint")
    for key in ("teacher_alignment_version", "teacher_coordinate_frame",
                "row_axis", "row_direction", "col_direction", "teacher_pc_range"):
        if teacher_metadata.get(key) != audit[key]:
            raise ValueError(f"teacher {key} differs from Stage 3 checkpoint")
    epoch = load_stage3_checkpoint(model, raster_head, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    raster_head.to(device).eval()
    thresholds = tuple(float(v) for v in config["eval"]["point_error_thresholds_m"])
    primary = max(thresholds)
    metrics = {name: {"gt": 0, "predictions": 0, "hits": {d: 0 for d in thresholds},
                      "point_error_sum": 0.0, "chamfer_sum": 0.0}
               for name in MAP_CLASS_NAMES}
    support = torch.tensor(audit["teacher_channel_support_mask"], device=device, dtype=torch.bool)
    weights = torch.tensor(audit["teacher_channel_weights"], device=device, dtype=torch.float32)
    kd_sum = 0.0
    raster_intersection = raster_union = 0
    frames = 0
    xy_range = (model.geometry_lift.x_range[0], model.geometry_lift.y_range[0],
                model.geometry_lift.x_range[1], model.geometry_lift.y_range[1])
    with torch.no_grad():
        for batch in loader:
            predictions = stage3_forward(model, raster_head, batch, device)
            logits = predictions["student_map_raster_logits"]
            teacher = batch["teacher_map_aligned"].to(device)
            valid = batch["teacher_map_valid"].to(device)
            kd_sum += float(soft_map_distillation_loss(logits, teacher, valid, support, weights))
            student_binary = (logits.sigmoid()[:, support] >= 0.5) & valid[:, None]
            teacher_binary = (teacher[:, support] >= 0.5) & valid[:, None]
            raster_intersection += int((student_binary & teacher_binary).sum())
            raster_union += int((student_binary | teacher_binary).sum())
            for batch_index, record in enumerate(batch["vector_targets"]):
                class_probabilities = predictions["map_cls_logits"][batch_index].softmax(-1)
                scores, labels = class_probabilities.max(-1)
                keep = (labels != len(MAP_CLASS_NAMES)) & (scores >= float(config["eval"]["confidence_threshold"]))
                pred_labels = labels[keep]
                pred_points = denormalize_points(predictions["map_points"][batch_index, keep], xy_range)
                for class_id, name in enumerate(MAP_CLASS_NAMES):
                    metrics[name]["gt"] += int((record["class_ids"] == class_id).sum())
                    metrics[name]["predictions"] += int((pred_labels == class_id).sum())
                for threshold in thresholds:
                    matches = class_aware_vector_matches(pred_labels, pred_points, record, threshold)
                    for _, gt_index, _, _ in matches:
                        metrics[MAP_CLASS_NAMES[int(record["class_ids"][gt_index])]]["hits"][threshold] += 1
                    if threshold == primary:
                        for _, gt_index, point_error, chamfer in matches:
                            entry = metrics[MAP_CLASS_NAMES[int(record["class_ids"][gt_index])]]
                            entry["point_error_sum"] += point_error
                            entry["chamfer_sum"] += chamfer
                frames += 1
    if not frames:
        raise RuntimeError("Stage 3 map evaluation received no frames")
    print(f"checkpoint={checkpoint_path} epoch={epoch} evaluated_frames={frames} vector_GT=nuPlan")
    for name in MAP_CLASS_NAMES:
        item = metrics[name]
        print(f"{name} gt_count={item['gt']} prediction_count={item['predictions']}")
        for threshold in thresholds:
            hits = item["hits"][threshold]
            print(f"{name} threshold_m={threshold:.1f} matched={hits} "
                  f"precision={ratio(hits, item['predictions']):.6f} recall={ratio(hits, item['gt']):.6f}")
        matched = item["hits"][primary]
        point_error = ratio(item["point_error_sum"], matched) if matched else math.nan
        chamfer = ratio(item["chamfer_sum"], matched) if matched else math.nan
        print(f"{name} matched_mean_point_error_m={point_error:.6f} matched_chamfer_m={chamfer:.6f}")
    print(f"teacher_student_raster_kd_bce={kd_sum / len(loader):.6f} "
          f"teacher_student_raster_iou_at_0.5={ratio(raster_intersection, raster_union):.6f}")


if __name__ == "__main__":
    main()
