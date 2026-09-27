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

from convert_streampetr_to_quest import STREAM_PETR_TO_QUEST
from quest.openscene_dataset import OpenSceneMetadataDataset
from quest.utils import load_yaml_config


CLASS_NAMES = ("vehicle", "pedestrian", "traffic_cone", "generic_object")
DISTANCE_THRESHOLDS = (2.0, 5.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare converted StreamPETR Agent pseudo-labels with OpenScene GT"
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--pseudo-dir", type=Path, default=PROJECT_ROOT / "data/soft_labels"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data/pseudo_labels/streampetr",
    )
    return parser.parse_args()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalized_center_to_metric(center: torch.Tensor) -> torch.Tensor:
    if center.shape[-1] != 3:
        raise ValueError(f"center must end in xyz, got {tuple(center.shape)}")
    metric = center.float().clone()
    metric[..., 0:2] = metric[..., 0:2] * 100.0 - 50.0
    metric[..., 2] = metric[..., 2] * 10.0 - 5.0
    return metric


def load_agent_soft_label(path: Path, expected_token: str) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Agent pseudo-label missing: {path}")
    payload = _torch_load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"pseudo-label file must contain a dict: {path}")
    token = str(payload.get("token", ""))
    if token != expected_token:
        raise ValueError(f"pseudo-label token mismatch: {token} != {expected_token}")
    agent = payload.get("agent")
    if not isinstance(agent, dict):
        raise ValueError(f"pseudo-label has no agent mapping: {path}")
    required = {"labels", "boxes", "velocity"}
    if not required.issubset(agent):
        raise ValueError(f"Agent pseudo-label requires {sorted(required)}: {path}")
    labels = torch.as_tensor(agent["labels"], dtype=torch.long)
    boxes = torch.as_tensor(agent["boxes"], dtype=torch.float32)
    if labels.ndim != 1 or boxes.shape != (labels.shape[0], 8):
        raise ValueError(
            f"invalid Agent pseudo-label shapes: labels={tuple(labels.shape)} "
            f"boxes={tuple(boxes.shape)}"
        )
    valid = labels >= 0
    return {
        "labels": labels[valid],
        "centers_m": normalized_center_to_metric(boxes[valid, :3]),
    }


def class_aware_matches(
    pseudo_centers: torch.Tensor,
    pseudo_labels: torch.Tensor,
    gt_centers: torch.Tensor,
    gt_labels: torch.Tensor,
    distance_threshold: float,
) -> list[tuple[int, int, float]]:
    matches = []
    invalid_cost = 1e6
    for class_id in range(len(CLASS_NAMES)):
        pseudo_indices = torch.nonzero(
            pseudo_labels == class_id, as_tuple=False
        ).flatten()
        gt_indices = torch.nonzero(gt_labels == class_id, as_tuple=False).flatten()
        if not pseudo_indices.numel() or not gt_indices.numel():
            continue
        distances = torch.cdist(
            pseudo_centers[pseudo_indices].float(),
            gt_centers[gt_indices].float(),
            p=2,
        )
        cost = distances.clone()
        cost[distances > float(distance_threshold)] = invalid_cost
        rows, columns = linear_sum_assignment(cost.cpu().numpy())
        for row, column in zip(rows, columns):
            distance = float(distances[row, column])
            if distance <= float(distance_threshold):
                matches.append(
                    (int(pseudo_indices[row]), int(gt_indices[column]), distance)
                )
    matches.sort(key=lambda item: item[0])
    return matches


def recover_raw_scores(
    raw_path: Path,
    pseudo_centers: torch.Tensor,
    pseudo_labels: torch.Tensor,
    tolerance: float = 1e-3,
) -> torch.Tensor:
    scores = torch.full((pseudo_labels.shape[0],), math.nan, dtype=torch.float32)
    if not raw_path.is_file() or pseudo_labels.numel() == 0:
        return scores
    payload = _torch_load(raw_path)
    raw_boxes = torch.as_tensor(payload["boxes_3d"], dtype=torch.float32)
    raw_scores = torch.as_tensor(payload["scores_3d"], dtype=torch.float32)
    raw_labels = torch.as_tensor(payload["labels_3d"], dtype=torch.long)
    if raw_boxes.ndim != 2 or raw_boxes.shape[1] != 9:
        raise ValueError(f"raw boxes_3d must be [N,9]: {raw_path}")
    if raw_scores.shape != (raw_boxes.shape[0],) or raw_labels.shape != (
        raw_boxes.shape[0],
    ):
        raise ValueError(f"raw StreamPETR prediction lengths differ: {raw_path}")
    if raw_labels.numel() and (
        int(raw_labels.min()) < 0 or int(raw_labels.max()) >= len(STREAM_PETR_TO_QUEST)
    ):
        raise ValueError(f"raw StreamPETR label outside 10-class taxonomy: {raw_path}")
    quest_labels = STREAM_PETR_TO_QUEST[raw_labels]
    for class_id in range(len(CLASS_NAMES)):
        pseudo_indices = torch.nonzero(
            pseudo_labels == class_id, as_tuple=False
        ).flatten()
        raw_indices = torch.nonzero(quest_labels == class_id, as_tuple=False).flatten()
        if not pseudo_indices.numel() or not raw_indices.numel():
            continue
        distances = torch.cdist(
            pseudo_centers[pseudo_indices].float(), raw_boxes[raw_indices, :3], p=2
        )
        rows, columns = linear_sum_assignment(distances.cpu().numpy())
        for row, column in zip(rows, columns):
            if float(distances[row, column]) <= tolerance:
                scores[pseudo_indices[row]] = raw_scores[raw_indices[column]]
    return scores


@dataclass
class CoordinateAccumulator:
    chunks: list[torch.Tensor] = field(default_factory=list)

    def update(self, centers: torch.Tensor) -> None:
        if centers.numel():
            self.chunks.append(centers.detach().cpu().float())

    def summary(self) -> dict[str, list[float] | int]:
        if not self.chunks:
            nan_xyz = [math.nan, math.nan, math.nan]
            return {
                "count": 0,
                "mean": nan_xyz,
                "std": nan_xyz,
                "min": nan_xyz,
                "max": nan_xyz,
            }
        centers = torch.cat(self.chunks, dim=0)
        return {
            "count": centers.shape[0],
            "mean": centers.mean(dim=0).tolist(),
            "std": centers.std(dim=0, unbiased=False).tolist(),
            "min": centers.min(dim=0).values.tolist(),
            "max": centers.max(dim=0).values.tolist(),
        }


@dataclass
class ThresholdMetrics:
    threshold: float
    matched: int = 0
    center_error_sum: float = 0.0
    class_matched: list[int] = field(
        default_factory=lambda: [0] * len(CLASS_NAMES)
    )

    def update(
        self,
        matches: list[tuple[int, int, float]],
        pseudo_labels: torch.Tensor,
    ) -> None:
        self.matched += len(matches)
        self.center_error_sum += sum(match[2] for match in matches)
        for pseudo_index, _, _ in matches:
            self.class_matched[int(pseudo_labels[pseudo_index])] += 1

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    def summary(
        self,
        pseudo_count: int,
        gt_count: int,
        pseudo_class_count: list[int],
        gt_class_count: list[int],
    ) -> dict[str, Any]:
        per_class = {}
        for class_id, name in enumerate(CLASS_NAMES):
            per_class[name] = {
                "pseudo": pseudo_class_count[class_id],
                "gt": gt_class_count[class_id],
                "matched": self.class_matched[class_id],
                "precision": self._ratio(
                    self.class_matched[class_id], pseudo_class_count[class_id]
                ),
                "recall": self._ratio(
                    self.class_matched[class_id], gt_class_count[class_id]
                ),
            }
        return {
            "matched": self.matched,
            "precision": self._ratio(self.matched, pseudo_count),
            "recall": self._ratio(self.matched, gt_count),
            "mean_center_error": (
                self.center_error_sum / self.matched if self.matched else math.nan
            ),
            "per_class": per_class,
        }


def _resolved(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _print_coordinate_stats(name: str, values: dict[str, Any]) -> None:
    print(f"{name}_center_stats:")
    for statistic in ("mean", "std", "min", "max"):
        xyz = values[statistic]
        print(
            f"  {statistic}_x={xyz[0]:.6f} {statistic}_y={xyz[1]:.6f} "
            f"{statistic}_z={xyz[2]:.6f}"
        )


def main() -> int:
    args = parse_args()
    if args.start < 0:
        raise ValueError(f"start must be non-negative, got {args.start}")
    if args.count <= 0:
        raise ValueError(f"count must be positive, got {args.count}")

    stage1 = load_yaml_config(PROJECT_ROOT / "configs/stage1.yaml")
    dataset_config = dict(stage1["dataset"])
    dataset_config["metadata_path"] = str(_resolved(dataset_config["metadata_path"]))
    dataset_config["camera_root"] = str(_resolved(dataset_config["camera_root"]))
    dataset_config["max_agent_instances"] = 64
    dataset = OpenSceneMetadataDataset(
        max_samples=args.start + args.count,
        **dataset_config,
    )

    pseudo_dir = _resolved(args.pseudo_dir)
    raw_dir = _resolved(args.raw_dir)
    pseudo_count = 0
    gt_count = 0
    pseudo_class_count = [0] * len(CLASS_NAMES)
    gt_class_count = [0] * len(CLASS_NAMES)
    pseudo_coordinates = CoordinateAccumulator()
    gt_coordinates = CoordinateAccumulator()
    threshold_metrics = {
        threshold: ThresholdMetrics(threshold) for threshold in DISTANCE_THRESHOLDS
    }
    previews = []

    for offset, complete_index in enumerate(
        range(args.start, args.start + args.count), start=1
    ):
        info = dataset.infos[complete_index]
        token = str(info["token"])
        hard_gt, _ = dataset._load_agent_gt(info)
        valid_gt = hard_gt["labels"] >= 0
        gt_labels = hard_gt["labels"][valid_gt]
        gt_centers = normalized_center_to_metric(hard_gt["boxes"][valid_gt, :3])
        pseudo = load_agent_soft_label(pseudo_dir / f"{token}.pt", token)
        pseudo_labels = pseudo["labels"]
        pseudo_centers = pseudo["centers_m"]

        pseudo_count += int(pseudo_labels.numel())
        gt_count += int(gt_labels.numel())
        for class_id in range(len(CLASS_NAMES)):
            pseudo_class_count[class_id] += int((pseudo_labels == class_id).sum())
            gt_class_count[class_id] += int((gt_labels == class_id).sum())
        pseudo_coordinates.update(pseudo_centers)
        gt_coordinates.update(gt_centers)

        for threshold, metrics in threshold_metrics.items():
            matches = class_aware_matches(
                pseudo_centers,
                pseudo_labels,
                gt_centers,
                gt_labels,
                threshold,
            )
            metrics.update(matches, pseudo_labels)

        if len(previews) < 3:
            scores = recover_raw_scores(
                raw_dir / f"{token}.pt", pseudo_centers, pseudo_labels
            )
            previews.append(
                (token, gt_labels, gt_centers, pseudo_labels, pseudo_centers, scores)
            )
        print(f"\rprogress {offset}/{args.count} token={token}", end="", flush=True)
    print()

    print(f"samples: {args.count}")
    print(f"pseudo_count: {pseudo_count}")
    print(f"gt_count: {gt_count}")
    _print_coordinate_stats("pseudo", pseudo_coordinates.summary())
    _print_coordinate_stats("gt", gt_coordinates.summary())
    print("class_counts:")
    for class_id, name in enumerate(CLASS_NAMES):
        print(
            f"  {name}: pseudo={pseudo_class_count[class_id]} "
            f"gt={gt_class_count[class_id]}"
        )

    for threshold in DISTANCE_THRESHOLDS:
        summary = threshold_metrics[threshold].summary(
            pseudo_count, gt_count, pseudo_class_count, gt_class_count
        )
        print(f"threshold={threshold:g}m")
        print(f"  matched: {summary['matched']}")
        print(f"  precision: {summary['precision']:.6f}")
        print(f"  recall: {summary['recall']:.6f}")
        print(f"  mean_center_error: {summary['mean_center_error']:.6f}")
        print("  per_class:")
        for name in CLASS_NAMES:
            values = summary["per_class"][name]
            print(
                f"    {name}: pseudo={values['pseudo']} gt={values['gt']} "
                f"matched={values['matched']} precision={values['precision']:.6f} "
                f"recall={values['recall']:.6f}"
            )

    print("sample_previews:")
    for token, gt_labels, gt_centers, pseudo_labels, pseudo_centers, scores in previews:
        print(f"token: {token}")
        print("  gt:")
        for label, center in zip(gt_labels[:5], gt_centers[:5]):
            x, y, z = center.tolist()
            print(f"    {CLASS_NAMES[int(label)]} xyz=({x:.3f},{y:.3f},{z:.3f})")
        print("  pseudo:")
        for label, center, score in zip(
            pseudo_labels[:5], pseudo_centers[:5], scores[:5]
        ):
            x, y, z = center.tolist()
            print(
                f"    {CLASS_NAMES[int(label)]} xyz=({x:.3f},{y:.3f},{z:.3f}) "
                f"score={float(score):.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
