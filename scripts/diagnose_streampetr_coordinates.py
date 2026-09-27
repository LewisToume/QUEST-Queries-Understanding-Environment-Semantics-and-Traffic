from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from convert_streampetr_to_quest import (
    STREAM_PETR_CLASSES,
    STREAM_PETR_TO_QUEST,
)
from quest.openscene_dataset import OPENSCENE_AGENT_CLASS_TO_ID, OpenSceneMetadataDataset
from quest.utils import load_yaml_config


REQUIRED_TRANSFORMS = ("lidar2ego", "lidar2global", "ego2global")
QUEST_CLASS_NAMES = ("vehicle", "pedestrian", "traffic_cone", "generic_object")
Z_ERROR_NAMES = (
    "mean_abs_z_error_raw",
    "mean_abs_z_error_raw_plus_half_height",
    "mean_abs_z_error_raw_minus_half_height",
    "mean_abs_z_error_transformed",
    "mean_abs_z_error_transformed_plus_half_height",
    "mean_abs_z_error_transformed_minus_half_height",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose OpenScene GT and StreamPETR raw box coordinates"
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
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


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be [N,3], got {tuple(points.shape)}")
    transform = torch.as_tensor(transform, dtype=points.dtype, device=points.device)
    if transform.shape != (4, 4):
        raise ValueError(f"transform must be [4,4], got {tuple(transform.shape)}")
    homogeneous = torch.cat(
        (points, torch.ones(points.shape[0], 1, dtype=points.dtype, device=points.device)),
        dim=1,
    )
    return (transform @ homogeneous.T).T[:, :3]


def class_aware_nearest_distances(
    source_centers: torch.Tensor,
    source_labels: torch.Tensor,
    gt_centers: torch.Tensor,
    gt_labels: torch.Tensor,
    dimensions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dimensions not in (2, 3):
        raise ValueError(f"dimensions must be 2 or 3, got {dimensions}")
    distances = []
    gt_indices = []
    for source_index in range(source_centers.shape[0]):
        candidates = torch.nonzero(
            gt_labels == source_labels[source_index], as_tuple=False
        ).flatten()
        if not candidates.numel():
            continue
        candidate_distances = torch.linalg.vector_norm(
            gt_centers[candidates, :dimensions]
            - source_centers[source_index, :dimensions],
            dim=1,
        )
        nearest = int(candidate_distances.argmin())
        distances.append(candidate_distances[nearest])
        gt_indices.append(candidates[nearest])
    if not distances:
        return (
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
        )
    return torch.stack(distances).float(), torch.stack(gt_indices).long()


def z_convention_values(z: torch.Tensor, height: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "base": z,
        "plus_half_height": z + height / 2.0,
        "minus_half_height": z - height / 2.0,
    }


def z_convention_errors(
    source_centers: torch.Tensor,
    heights: torch.Tensor,
    source_labels: torch.Tensor,
    gt_centers: torch.Tensor,
    gt_labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    source_indices = []
    nearest_gt_indices = []
    for source_index in range(source_centers.shape[0]):
        candidates = torch.nonzero(
            gt_labels == source_labels[source_index], as_tuple=False
        ).flatten()
        if not candidates.numel():
            continue
        bev_distances = torch.linalg.vector_norm(
            gt_centers[candidates, :2] - source_centers[source_index, :2], dim=1
        )
        source_indices.append(source_index)
        nearest_gt_indices.append(int(candidates[int(bev_distances.argmin())]))
    if not source_indices:
        empty = torch.empty(0, dtype=torch.float32)
        return {name: empty for name in ("base", "plus_half_height", "minus_half_height")}
    source_indices_tensor = torch.tensor(source_indices, dtype=torch.long)
    gt_indices_tensor = torch.tensor(nearest_gt_indices, dtype=torch.long)
    candidates = z_convention_values(
        source_centers[source_indices_tensor, 2], heights[source_indices_tensor]
    )
    gt_z = gt_centers[gt_indices_tensor, 2]
    return {name: (values - gt_z).abs() for name, values in candidates.items()}


def load_raw_predictions(path: Path, expected_token: str) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"StreamPETR raw pseudo-label missing: {path}")
    payload = _torch_load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"raw pseudo-label must contain a dict: {path}")
    token = str(payload.get("token", ""))
    if token != expected_token:
        raise ValueError(f"raw pseudo-label token mismatch: {token} != {expected_token}")
    boxes = torch.as_tensor(payload["boxes_3d"], dtype=torch.float32)
    scores = torch.as_tensor(payload["scores_3d"], dtype=torch.float32)
    labels = torch.as_tensor(payload["labels_3d"], dtype=torch.long)
    if boxes.ndim != 2 or boxes.shape[1] != 9:
        raise ValueError(f"boxes_3d must be [N,9], got {tuple(boxes.shape)}")
    if scores.shape != (boxes.shape[0],) or labels.shape != (boxes.shape[0],):
        raise ValueError(f"raw prediction lengths differ: {path}")
    if labels.numel() and (
        int(labels.min()) < 0 or int(labels.max()) >= len(STREAM_PETR_CLASSES)
    ):
        raise ValueError(f"raw labels outside StreamPETR taxonomy: {path}")
    return {"boxes": boxes, "scores": scores, "labels": labels}


def load_metric_agent_gt(info: dict[str, Any]) -> dict[str, Any]:
    raw_boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float32)
    raw_names = np.asarray(info.get("gt_names", []))
    boxes = []
    labels = []
    names = []
    for box, raw_name in zip(raw_boxes, raw_names):
        name = str(raw_name)
        if name not in OPENSCENE_AGENT_CLASS_TO_ID or len(box) < 7:
            continue
        values = np.asarray(box[:7], dtype=np.float32)
        if not np.isfinite(values).all():
            continue
        boxes.append(torch.from_numpy(values.copy()))
        labels.append(OPENSCENE_AGENT_CLASS_TO_ID[name])
        names.append(name)
    return {
        "boxes": torch.stack(boxes) if boxes else torch.empty(0, 7),
        "labels": torch.tensor(labels, dtype=torch.long),
        "names": names,
    }


def transform_fields(info: dict[str, Any]) -> list[str]:
    fields = []
    for key, value in info.items():
        lowered = key.lower()
        if not any(name in lowered for name in ("lidar", "ego", "global")):
            continue
        array = np.asarray(value)
        if array.ndim >= 1 and array.size <= 64:
            fields.append(key)
    return sorted(fields)


def _mean_or_nan(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _print_transform(info: dict[str, Any], key: str) -> None:
    if key not in info or info[key] is None:
        print(f"{key}: MISSING")
        return
    array = np.asarray(info[key])
    print(f"{key}: shape={array.shape}")
    print(array)


def _print_distance(name: str, values: torch.Tensor) -> None:
    value = float(values.mean()) if values.numel() else math.nan
    print(f"{name}: {value:.6f}")


def main() -> int:
    args = parse_args()
    if args.start < 0:
        raise ValueError(f"start must be non-negative, got {args.start}")
    if args.count <= 0:
        raise ValueError(f"count must be positive, got {args.count}")

    stage1 = load_yaml_config(PROJECT_ROOT / "configs/stage1.yaml")
    dataset_config = dict(stage1["dataset"])
    for key in ("metadata_path", "camera_root"):
        path = Path(dataset_config[key])
        if not path.is_absolute():
            dataset_config[key] = str(PROJECT_ROOT / path)
    dataset = OpenSceneMetadataDataset(
        max_samples=args.start + args.count,
        **dataset_config,
    )
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else PROJECT_ROOT / args.raw_dir

    aggregate = {
        "raw_bev": [],
        "transformed_bev": [],
        "raw_xyz": [],
        "transformed_xyz": [],
        **{name: [] for name in Z_ERROR_NAMES},
    }

    for complete_index in range(args.start, args.start + args.count):
        info = dataset.infos[complete_index]
        token = str(info["token"])
        print("=" * 80)
        print(f"token: {token}")
        print(f"scene_token: {info.get('scene_token')}")
        print(f"timestamp: {info.get('timestamp')}")
        for key in REQUIRED_TRANSFORMS:
            _print_transform(info, key)
        additional_fields = [
            key for key in transform_fields(info) if key not in REQUIRED_TRANSFORMS
        ]
        print("additional_transform_fields:")
        if additional_fields:
            for key in additional_fields:
                print(f"  {key}: shape={np.asarray(info[key]).shape}")
        else:
            print("  NONE")

        gt = load_metric_agent_gt(info)
        print("gt_boxes_first_5:")
        for name, box in zip(gt["names"][:5], gt["boxes"][:5]):
            x, y, z, dx, dy, dz, yaw = box.tolist()
            print(
                f"  class={name} xyz=({x:.6f},{y:.6f},{z:.6f}) "
                f"size=({dx:.6f},{dy:.6f},{dz:.6f}) yaw={yaw:.6f}"
            )

        raw = load_raw_predictions(raw_dir / f"{token}.pt", token)
        order = torch.argsort(raw["scores"], descending=True)
        print("streampetr_raw_first_5_by_score:")
        for index in order[:5]:
            box = raw["boxes"][index]
            label = int(raw["labels"][index])
            x, y, z, dx, dy, dz, yaw, vx, vy = box.tolist()
            print(
                f"  class={STREAM_PETR_CLASSES[label]}->"
                f"{QUEST_CLASS_NAMES[int(STREAM_PETR_TO_QUEST[label])]} "
                f"score={float(raw['scores'][index]):.6f} "
                f"xyz=({x:.6f},{y:.6f},{z:.6f}) "
                f"size=({dx:.6f},{dy:.6f},{dz:.6f}) yaw={yaw:.6f} "
                f"velocity=({vx:.6f},{vy:.6f})"
            )

        raw_centers = raw["boxes"][:, :3]
        raw_heights = raw["boxes"][:, 5]
        quest_labels = STREAM_PETR_TO_QUEST[raw["labels"]]
        gt_centers = gt["boxes"][:, :3]
        raw_bev, _ = class_aware_nearest_distances(
            raw_centers, quest_labels, gt_centers, gt["labels"], dimensions=2
        )
        raw_xyz, _ = class_aware_nearest_distances(
            raw_centers, quest_labels, gt_centers, gt["labels"], dimensions=3
        )
        aggregate["raw_bev"].extend(raw_bev.tolist())
        aggregate["raw_xyz"].extend(raw_xyz.tolist())

        transformed_centers = None
        if "lidar2ego" in info and info["lidar2ego"] is not None:
            lidar2ego = torch.as_tensor(info["lidar2ego"], dtype=torch.float32)
            if lidar2ego.shape == (4, 4):
                transformed_centers = transform_points(raw_centers, lidar2ego)
                print("streampetr_lidar_to_ego_first_5_xyz:")
                for center in transformed_centers[order[:5]]:
                    x, y, z = center.tolist()
                    print(f"  xyz=({x:.6f},{y:.6f},{z:.6f})")
            else:
                print(
                    "streampetr_lidar_to_ego_first_5_xyz: UNAVAILABLE "
                    f"(lidar2ego shape={tuple(lidar2ego.shape)})"
                )
        else:
            print("streampetr_lidar_to_ego_first_5_xyz: UNAVAILABLE")

        if transformed_centers is not None:
            transformed_bev, _ = class_aware_nearest_distances(
                transformed_centers,
                quest_labels,
                gt_centers,
                gt["labels"],
                dimensions=2,
            )
            transformed_xyz, _ = class_aware_nearest_distances(
                transformed_centers,
                quest_labels,
                gt_centers,
                gt["labels"],
                dimensions=3,
            )
        else:
            transformed_bev = torch.empty(0)
            transformed_xyz = torch.empty(0)
        aggregate["transformed_bev"].extend(transformed_bev.tolist())
        aggregate["transformed_xyz"].extend(transformed_xyz.tolist())

        _print_distance("raw_bev_mean_nearest_distance", raw_bev)
        _print_distance("transformed_bev_mean_nearest_distance", transformed_bev)
        _print_distance("raw_xyz_mean_nearest_distance", raw_xyz)
        _print_distance("transformed_xyz_mean_nearest_distance", transformed_xyz)

        raw_z_errors = z_convention_errors(
            raw_centers, raw_heights, quest_labels, gt_centers, gt["labels"]
        )
        transformed_z_errors = (
            z_convention_errors(
                transformed_centers,
                raw_heights,
                quest_labels,
                gt_centers,
                gt["labels"],
            )
            if transformed_centers is not None
            else {name: torch.empty(0) for name in raw_z_errors}
        )
        sample_z = {
            "mean_abs_z_error_raw": raw_z_errors["base"],
            "mean_abs_z_error_raw_plus_half_height": raw_z_errors[
                "plus_half_height"
            ],
            "mean_abs_z_error_raw_minus_half_height": raw_z_errors[
                "minus_half_height"
            ],
            "mean_abs_z_error_transformed": transformed_z_errors["base"],
            "mean_abs_z_error_transformed_plus_half_height": transformed_z_errors[
                "plus_half_height"
            ],
            "mean_abs_z_error_transformed_minus_half_height": transformed_z_errors[
                "minus_half_height"
            ],
        }
        for name, values in sample_z.items():
            aggregate[name].extend(values.tolist())
            _print_distance(name, values)

    print("=" * 80)
    print("summary")
    print(f"samples: {args.count}")
    print(f"raw BEV mean distance: {_mean_or_nan(aggregate['raw_bev']):.6f}")
    print(
        "transformed BEV mean distance: "
        f"{_mean_or_nan(aggregate['transformed_bev']):.6f}"
    )
    print(f"raw xyz mean distance: {_mean_or_nan(aggregate['raw_xyz']):.6f}")
    print(
        "transformed xyz mean distance: "
        f"{_mean_or_nan(aggregate['transformed_xyz']):.6f}"
    )
    for name in Z_ERROR_NAMES:
        print(f"{name}: {_mean_or_nan(aggregate[name]):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
