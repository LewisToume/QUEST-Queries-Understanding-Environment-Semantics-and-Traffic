from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from convert_streampetr_to_quest import (
    MAX_AGENT_INSTANCES,
    STREAM_PETR_TO_QUEST,
    XY_RANGE,
    Z_RANGE,
)
from diagnose_streampetr_coordinates import (
    load_metric_agent_gt,
    load_raw_predictions,
    resolve_lidar2ego,
    transform_points,
)
from quest.openscene_dataset import OpenSceneMetadataDataset
from quest.utils import load_yaml_config


CLASS_NAMES = ("vehicle", "pedestrian", "traffic_cone", "generic_object")
CONFIDENCE_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
MODES = ("raw", "lidar_to_ego")
STAGES = ("before_top64", "after_top64")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep StreamPETR confidence thresholds against OpenScene GT"
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--distance-threshold", type=float, default=2.0)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data/pseudo_labels/streampetr",
    )
    return parser.parse_args()


def filter_raw_predictions(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    raw_labels: torch.Tensor,
    confidence_threshold: float,
    apply_top64: bool,
) -> dict[str, torch.Tensor]:
    if boxes.ndim != 2 or boxes.shape[1] != 9:
        raise ValueError(f"boxes must be [N,9], got {tuple(boxes.shape)}")
    if scores.shape != (boxes.shape[0],) or raw_labels.shape != (boxes.shape[0],):
        raise ValueError("StreamPETR boxes, scores, and labels must have equal length")
    xy_min, xy_max = XY_RANGE
    z_min, z_max = Z_RANGE
    keep = (
        (scores >= float(confidence_threshold))
        & (boxes[:, 0] >= xy_min)
        & (boxes[:, 0] <= xy_max)
        & (boxes[:, 1] >= xy_min)
        & (boxes[:, 1] <= xy_max)
        & (boxes[:, 2] >= z_min)
        & (boxes[:, 2] <= z_max)
    )
    indices = torch.nonzero(keep, as_tuple=False).flatten()
    if apply_top64:
        order = torch.argsort(scores[indices], descending=True)
        indices = indices[order[:MAX_AGENT_INSTANCES]]
    return {
        "indices": indices,
        "centers": boxes[indices, :3],
        "scores": scores[indices],
        "labels": STREAM_PETR_TO_QUEST[raw_labels[indices]],
    }


def strict_class_aware_hungarian(
    prediction_centers: torch.Tensor,
    prediction_labels: torch.Tensor,
    gt_centers: torch.Tensor,
    gt_labels: torch.Tensor,
    distance_threshold: float,
) -> list[tuple[int, int, float]]:
    matches = []
    invalid_cost = 1e6
    for class_id in range(len(CLASS_NAMES)):
        prediction_indices = torch.nonzero(
            prediction_labels == class_id, as_tuple=False
        ).flatten()
        gt_indices = torch.nonzero(gt_labels == class_id, as_tuple=False).flatten()
        if not prediction_indices.numel() or not gt_indices.numel():
            continue
        distances = torch.cdist(
            prediction_centers[prediction_indices, :2].float(),
            gt_centers[gt_indices, :2].float(),
            p=2,
        )
        cost = distances.clone()
        cost[distances > float(distance_threshold)] = invalid_cost
        rows, columns = linear_sum_assignment(cost.cpu().numpy())
        for row, column in zip(rows, columns):
            distance = float(distances[row, column])
            prediction_index = int(prediction_indices[row])
            gt_index = int(gt_indices[column])
            if (
                distance <= float(distance_threshold)
                and prediction_labels[prediction_index] == gt_labels[gt_index]
            ):
                matches.append((prediction_index, gt_index, distance))
    matches.sort(key=lambda item: item[0])
    return matches


@dataclass
class ThresholdMetrics:
    samples: int = 0
    total_gt: int = 0
    total_predictions: int = 0
    matched: int = 0
    center_error_sum: float = 0.0
    class_gt: list[int] = field(default_factory=lambda: [0] * len(CLASS_NAMES))
    class_predictions: list[int] = field(
        default_factory=lambda: [0] * len(CLASS_NAMES)
    )
    class_matched: list[int] = field(
        default_factory=lambda: [0] * len(CLASS_NAMES)
    )

    def update(
        self,
        prediction_labels: torch.Tensor,
        gt_labels: torch.Tensor,
        matches: list[tuple[int, int, float]],
    ) -> None:
        self.samples += 1
        self.total_gt += int(gt_labels.numel())
        self.total_predictions += int(prediction_labels.numel())
        for class_id in range(len(CLASS_NAMES)):
            self.class_gt[class_id] += int((gt_labels == class_id).sum())
            self.class_predictions[class_id] += int(
                (prediction_labels == class_id).sum()
            )
        for prediction_index, gt_index, distance in matches:
            prediction_class = int(prediction_labels[prediction_index])
            gt_class = int(gt_labels[gt_index])
            if prediction_class != gt_class:
                continue
            self.matched += 1
            self.center_error_sum += float(distance)
            self.class_matched[gt_class] += 1

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    def summary(self) -> dict[str, Any]:
        per_class = {}
        for class_id, name in enumerate(CLASS_NAMES):
            per_class[name] = {
                "gt": self.class_gt[class_id],
                "predictions": self.class_predictions[class_id],
                "matched": self.class_matched[class_id],
                "precision": self._ratio(
                    self.class_matched[class_id], self.class_predictions[class_id]
                ),
                "recall": self._ratio(
                    self.class_matched[class_id], self.class_gt[class_id]
                ),
            }
        return {
            "samples": self.samples,
            "total_gt": self.total_gt,
            "total_predictions": self.total_predictions,
            "matched": self.matched,
            "precision": self._ratio(self.matched, self.total_predictions),
            "recall": self._ratio(self.matched, self.total_gt),
            "mean_bev_center_error": (
                self.center_error_sum / self.matched if self.matched else math.nan
            ),
            "per_class": per_class,
        }


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _print_threshold_result(
    threshold: float, mode: str, stage: str, metrics: ThresholdMetrics
) -> None:
    if metrics.samples == 0:
        print(f"threshold={threshold:.2f} mode={mode} stage={stage}: unavailable")
        return
    summary = metrics.summary()
    print(f"threshold={threshold:.2f} mode={mode} stage={stage}")
    for key in (
        "total_gt",
        "total_predictions",
        "matched",
        "precision",
        "recall",
        "mean_bev_center_error",
    ):
        value = summary[key]
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    print("  per_class:")
    for name in CLASS_NAMES:
        values = summary["per_class"][name]
        print(
            f"    {name}: gt={values['gt']} predictions={values['predictions']} "
            f"matched={values['matched']} precision={values['precision']:.6f} "
            f"recall={values['recall']:.6f}"
        )


def _print_top64_impact(
    threshold: float,
    mode: str,
    before: ThresholdMetrics,
    after: ThresholdMetrics,
) -> None:
    if before.samples == 0 or after.samples == 0:
        print(f"top64_impact threshold={threshold:.2f} mode={mode}: unavailable")
        return
    before_summary = before.summary()["per_class"]
    after_summary = after.summary()["per_class"]
    print(f"top64_impact threshold={threshold:.2f} mode={mode}")
    for name in ("vehicle", "pedestrian"):
        before_class = before_summary[name]
        after_class = after_summary[name]
        print(f"  {name}:")
        print(
            f"    predictions_before_top64={before_class['predictions']} "
            f"predictions_after_top64={after_class['predictions']} "
            f"dropped_by_top64="
            f"{before_class['predictions'] - after_class['predictions']}"
        )
        print(
            f"    matched_before_top64={before_class['matched']} "
            f"precision_before_top64={before_class['precision']:.6f} "
            f"recall_before_top64={before_class['recall']:.6f}"
        )
        print(
            f"    matched_after_top64={after_class['matched']} "
            f"precision_after_top64={after_class['precision']:.6f} "
            f"recall_after_top64={after_class['recall']:.6f}"
        )


def main() -> int:
    args = parse_args()
    if args.start < 0:
        raise ValueError(f"start must be non-negative, got {args.start}")
    if args.count <= 0:
        raise ValueError(f"count must be positive, got {args.count}")
    if args.distance_threshold <= 0:
        raise ValueError(
            f"distance-threshold must be positive, got {args.distance_threshold}"
        )

    stage1 = load_yaml_config(PROJECT_ROOT / "configs/stage1.yaml")
    dataset_config = dict(stage1["dataset"])
    for key in ("metadata_path", "camera_root"):
        dataset_config[key] = str(_resolve(dataset_config[key]))
    dataset = OpenSceneMetadataDataset(
        max_samples=args.start + args.count,
        **dataset_config,
    )
    raw_dir = _resolve(args.raw_dir)
    metrics = {
        threshold: {
            mode: {stage: ThresholdMetrics() for stage in STAGES} for mode in MODES
        }
        for threshold in CONFIDENCE_THRESHOLDS
    }
    unavailable_transforms = 0

    for offset, complete_index in enumerate(
        range(args.start, args.start + args.count), start=1
    ):
        info = dataset.infos[complete_index]
        token = str(info["token"])
        gt = load_metric_agent_gt(info)
        gt_centers = gt["boxes"][:, :3]
        gt_labels = gt["labels"]
        raw = load_raw_predictions(raw_dir / f"{token}.pt", token)

        lidar_to_ego_centers = None
        transform = resolve_lidar2ego(info)
        if transform is not None:
            lidar_to_ego_centers = transform_points(raw["boxes"][:, :3], transform)
        if lidar_to_ego_centers is None:
            unavailable_transforms += 1

        for threshold in CONFIDENCE_THRESHOLDS:
            stage_predictions = {
                "before_top64": filter_raw_predictions(
                    raw["boxes"],
                    raw["scores"],
                    raw["labels"],
                    threshold,
                    apply_top64=False,
                ),
                "after_top64": filter_raw_predictions(
                    raw["boxes"],
                    raw["scores"],
                    raw["labels"],
                    threshold,
                    apply_top64=True,
                ),
            }
            for stage, predictions in stage_predictions.items():
                raw_matches = strict_class_aware_hungarian(
                    predictions["centers"],
                    predictions["labels"],
                    gt_centers,
                    gt_labels,
                    args.distance_threshold,
                )
                metrics[threshold]["raw"][stage].update(
                    predictions["labels"], gt_labels, raw_matches
                )
                if lidar_to_ego_centers is not None:
                    transformed_centers = lidar_to_ego_centers[
                        predictions["indices"]
                    ]
                    transformed_matches = strict_class_aware_hungarian(
                        transformed_centers,
                        predictions["labels"],
                        gt_centers,
                        gt_labels,
                        args.distance_threshold,
                    )
                    metrics[threshold]["lidar_to_ego"][stage].update(
                        predictions["labels"], gt_labels, transformed_matches
                    )
        print(f"\rprogress {offset}/{args.count} token={token}", end="", flush=True)
    print()
    if unavailable_transforms:
        print(
            f"lidar_to_ego unavailable for {unavailable_transforms}/{args.count} samples"
        )

    for threshold in CONFIDENCE_THRESHOLDS:
        for mode in MODES:
            for stage in STAGES:
                _print_threshold_result(
                    threshold, mode, stage, metrics[threshold][mode][stage]
                )
            _print_top64_impact(
                threshold,
                mode,
                metrics[threshold][mode]["before_top64"],
                metrics[threshold][mode]["after_top64"],
            )

    print("summary_table")
    print(
        "threshold | mode | stage | vehicle_P | vehicle_R | "
        "pedestrian_predictions | pedestrian_P | pedestrian_R"
    )
    for threshold in CONFIDENCE_THRESHOLDS:
        for mode in MODES:
            for stage in STAGES:
                stage_metrics = metrics[threshold][mode][stage]
                if stage_metrics.samples == 0:
                    print(f"{threshold:.2f} | {mode} | {stage} | unavailable")
                    continue
                per_class = stage_metrics.summary()["per_class"]
                vehicle = per_class["vehicle"]
                pedestrian = per_class["pedestrian"]
                print(
                    f"{threshold:.2f} | {mode} | {stage} | "
                    f"{vehicle['precision']:.6f} | {vehicle['recall']:.6f} | "
                    f"{pedestrian['predictions']} | "
                    f"{pedestrian['precision']:.6f} | {pedestrian['recall']:.6f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
