from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.agent_training import (
    class_aware_center_matches,
    decode_supported_agent_predictions,
    forward_agent_end_to_end,
    load_agent_stage2_checkpoint,
    load_checkpoint_cpu,
    prepare_navformer_agent_batch,
)
from quest.bev_pretraining import topk_center_hits
from quest.dataset import collate_fn
from quest.model import QUESTModel
from quest.proposal_pretraining import proposal_center_hits, rasterize_proposal_targets
from quest.utils import load_yaml_config
from scripts.train_stage1_bev import build_dataset, resolve_path


CLASS_NAMES = ("vehicle", "pedestrian")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 QUEST Agent")
    parser.add_argument("--start", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    args = parse_args()
    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    stage1_config = load_yaml_config(PROJECT_ROOT / "configs/stage1.yaml")
    config = load_yaml_config(PROJECT_ROOT / "configs/stage2_agent.yaml")
    eval_config = config["eval"]
    start = args.start if args.start is not None else int(eval_config["start_index"])
    count = args.count if args.count is not None else int(eval_config["num_samples"])
    dataset = build_dataset(stage1_config, config, start, count)
    loader = DataLoader(
        dataset,
        batch_size=int(eval_config["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QUESTModel(**model_config).to(device).eval()
    checkpoint_path = resolve_path(args.checkpoint or config["paths"]["checkpoint_path"])
    checkpoint = load_checkpoint_cpu(checkpoint_path)
    epoch = load_agent_stage2_checkpoint(model, checkpoint)
    support_mask = checkpoint["trained_class_support_mask"].bool()
    del checkpoint

    top_ks = tuple(int(value) for value in eval_config["top_k"])
    proposal_distances = tuple(float(value) for value in eval_config["proposal_center_distances_m"])
    center_distances = tuple(float(value) for value in eval_config["center_distances_m"])
    match_distance = float(eval_config["match_distance_m"])
    confidence = float(eval_config["confidence_threshold"])
    bev_hits = {top_k: 0 for top_k in top_ks}
    bev_totals = {top_k: 0 for top_k in top_ks}
    exact_hits = {top_k: 0 for top_k in top_ks}
    exact_totals = {top_k: 0 for top_k in top_ks}
    proposal_hits = {(top_k, d): 0 for top_k in top_ks for d in proposal_distances}
    proposal_totals = {(top_k, d): 0 for top_k in top_ks for d in proposal_distances}
    class_metrics = {
        name: {
            "predictions": 0, "gt": 0, "matched": 0,
            "center_error_sum": 0.0, "size_error_sum": 0.0,
            "yaw_error_sum": 0.0, "velocity_error_sum": 0.0,
            "center_hits": {distance: 0 for distance in center_distances},
        }
        for name in CLASS_NAMES
    }
    evaluated = 0

    with torch.no_grad():
        for batch in loader:
            target = prepare_navformer_agent_batch(batch, device)
            predictions = forward_agent_end_to_end(model, batch, device)
            proposal_target = rasterize_proposal_targets(
                target,
                model.geometry_lift.x_range,
                model.geometry_lift.y_range,
                model.bev_h,
                model.bev_w,
            )
            objectness = predictions["proposal_objectness_logits"]
            offsets = predictions["proposal_xy_offsets"]
            logits_grid = objectness.reshape(-1, model.bev_h, model.bev_w)
            positive_grid = proposal_target["positive_mask"].reshape_as(logits_grid)
            for top_k in top_ks:
                hits, total = topk_center_hits(
                    logits_grid,
                    positive_grid,
                    top_k,
                    int(eval_config["tolerance_cells"]),
                )
                bev_hits[top_k] += hits
                bev_totals[top_k] += total
                hits, total = topk_center_hits(logits_grid, positive_grid, top_k, 0)
                exact_hits[top_k] += hits
                exact_totals[top_k] += total
                for distance in proposal_distances:
                    hits, total = proposal_center_hits(
                        objectness,
                        offsets,
                        proposal_target["target_centers_metric"],
                        top_k,
                        distance,
                        model.geometry_lift.x_range,
                        model.geometry_lift.y_range,
                        model.bev_h,
                        model.bev_w,
                    )
                    proposal_hits[top_k, distance] += hits
                    proposal_totals[top_k, distance] += total

            for batch_index in range(objectness.shape[0]):
                single_prediction = {
                    key: predictions[key][batch_index]
                    for key in ("agent_cls_logits", "agent_boxes", "agent_velocity")
                }
                decoded = decode_supported_agent_predictions(
                    single_prediction,
                    support_mask,
                    confidence,
                    model.geometry_lift.x_range,
                    model.geometry_lift.y_range,
                    config["agent_loss"]["z_range"],
                    config["agent_loss"]["size_norm"],
                )
                valid = target["valid_mask"][batch_index]
                gt_labels = target["labels"][batch_index, valid]
                gt_boxes = target["boxes_metric"][batch_index, valid]
                gt_velocity = target["velocity_mps"][batch_index, valid]
                for class_id, name in enumerate(CLASS_NAMES):
                    class_metrics[name]["predictions"] += int((decoded["labels"] == class_id).sum())
                    class_metrics[name]["gt"] += int((gt_labels == class_id).sum())

                for distance in center_distances:
                    matches = class_aware_center_matches(
                        decoded["centers"], decoded["labels"], gt_boxes[:, :3],
                        gt_labels, distance,
                    )
                    for _, gt_index, _ in matches:
                        name = CLASS_NAMES[int(gt_labels[gt_index])]
                        class_metrics[name]["center_hits"][distance] += 1

                matches = class_aware_center_matches(
                    decoded["centers"], decoded["labels"], gt_boxes[:, :3],
                    gt_labels, match_distance,
                )
                for pred_index, gt_index, center_error in matches:
                    name = CLASS_NAMES[int(gt_labels[gt_index])]
                    metrics = class_metrics[name]
                    metrics["matched"] += 1
                    metrics["center_error_sum"] += center_error
                    metrics["size_error_sum"] += float(
                        (decoded["sizes"][pred_index] - gt_boxes[gt_index, 3:6]).abs().mean()
                    )
                    yaw_difference = decoded["yaw"][pred_index] - gt_boxes[gt_index, 6]
                    metrics["yaw_error_sum"] += float(
                        torch.atan2(yaw_difference.sin(), yaw_difference.cos()).abs()
                    )
                    metrics["velocity_error_sum"] += float(torch.linalg.vector_norm(
                        decoded["velocity"][pred_index] - gt_velocity[gt_index]
                    ))
            evaluated += int(objectness.shape[0])
    if not evaluated:
        raise RuntimeError("Stage 2 Agent evaluation received no frames")

    print(f"checkpoint={checkpoint_path} epoch={epoch} evaluated_frames={evaluated}")
    print(f"evaluated_class_support_mask={support_mask.tolist()}")
    print(f"confidence_threshold={confidence:.2f} match_distance_m={match_distance:.1f}")
    print("matched_error_units=center_m,size_m,yaw_rad,velocity_mps")
    for top_k in top_ks:
        print(f"top_{top_k}_bev_center_recall={ratio(bev_hits[top_k], bev_totals[top_k]):.6f}")
        print(
            f"top_{top_k}_exact_bev_center_recall="
            f"{ratio(exact_hits[top_k], exact_totals[top_k]):.6f}"
        )
        for distance in proposal_distances:
            print(
                f"top_{top_k}_center_recall_within_{distance:.1f}m="
                f"{ratio(proposal_hits[top_k, distance], proposal_totals[top_k, distance]):.6f}"
            )
    for name in CLASS_NAMES:
        metrics = class_metrics[name]
        matched = metrics["matched"]
        print(
            f"{name} prediction_count={metrics['predictions']} gt_count={metrics['gt']} "
            f"precision={ratio(matched, metrics['predictions']):.6f} "
            f"recall={ratio(matched, metrics['gt']):.6f}"
        )
        for distance in center_distances:
            print(
                f"{name}_center_recall_within_{distance:.1f}m="
                f"{ratio(metrics['center_hits'][distance], metrics['gt']):.6f}"
            )
        for error in ("center", "size", "yaw", "velocity"):
            mean = metrics[f"{error}_error_sum"] / matched if matched else math.nan
            print(f"{name}_matched_mean_{error}_error={mean:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
