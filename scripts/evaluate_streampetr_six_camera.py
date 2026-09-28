from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from convert_streampetr_to_quest import STREAM_PETR_TO_QUEST  # noqa: E402
from diagnose_streampetr_coordinates import (  # noqa: E402
    load_metric_agent_gt,
    load_raw_predictions,
)
from evaluate_streampetr_thresholds import (  # noqa: E402
    strict_class_aware_hungarian,
)
from quest.openscene_dataset import (  # noqa: E402
    OpenSceneMetadataDataset,
    resolve_openscene_camera_path,
)
from quest.utils import load_yaml_config  # noqa: E402


SIX_CAMERA_NAMES = (
    "CAM_F0",
    "CAM_B0",
    "CAM_L0",
    "CAM_L2",
    "CAM_R0",
    "CAM_R2",
)
CLASS_NAMES = ("vehicle", "pedestrian")
SCORE_THRESHOLD = 0.25
MAX_DISTANCE_METERS = 50.0
PEDESTRIAN_DISTANCE_BINS = (
    (0.0, 10.0),
    (10.0, 20.0),
    (20.0, 30.0),
    (30.0, 40.0),
    (40.0, 50.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate six-camera StreamPETR predictions on visible OpenScene GT"
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


def camera_visibility_mask(
    lidar_centers: torch.Tensor,
    camera: dict[str, Any],
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Return centers inside one camera's pinhole image bounds."""

    if lidar_centers.ndim != 2 or lidar_centers.shape[1] != 3:
        raise ValueError(
            f"lidar_centers must be [N,3], got {tuple(lidar_centers.shape)}"
        )
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {image_size}")

    dtype = lidar_centers.dtype
    cam_to_lidar = torch.eye(4, dtype=dtype)
    rotation = torch.as_tensor(camera["sensor2lidar_rotation"], dtype=dtype)
    translation = torch.as_tensor(camera["sensor2lidar_translation"], dtype=dtype)
    intrinsic = torch.as_tensor(camera["cam_intrinsic"], dtype=dtype)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("OpenScene camera extrinsic must contain 3x3 rotation and 3D translation")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"camera intrinsic must be [3,3], got {tuple(intrinsic.shape)}")
    cam_to_lidar[:3, :3] = rotation
    cam_to_lidar[:3, 3] = translation
    lidar_to_cam = torch.linalg.inv(cam_to_lidar)

    homogeneous = torch.cat(
        [lidar_centers, torch.ones((lidar_centers.shape[0], 1), dtype=dtype)],
        dim=1,
    )
    camera_centers = homogeneous @ lidar_to_cam.T
    projected = camera_centers[:, :3] @ intrinsic.T
    depth = camera_centers[:, 2]
    safe_depth = depth.clamp_min(torch.finfo(dtype).eps)
    pixel_x = projected[:, 0] / safe_depth
    pixel_y = projected[:, 1] / safe_depth
    return (
        torch.isfinite(camera_centers).all(dim=1)
        & torch.isfinite(pixel_x)
        & torch.isfinite(pixel_y)
        & (depth > 0.0)
        & (pixel_x >= 0.0)
        & (pixel_x < float(width))
        & (pixel_y >= 0.0)
        & (pixel_y < float(height))
    )


def six_camera_visibility_mask(
    info: dict[str, Any],
    lidar_centers: torch.Tensor,
    camera_root: Path,
) -> torch.Tensor:
    visible = torch.zeros(lidar_centers.shape[0], dtype=torch.bool)
    for camera_name in SIX_CAMERA_NAMES:
        camera = info["cams"][camera_name]
        image_path = resolve_openscene_camera_path(camera["data_path"], camera_root)
        with Image.open(image_path) as image:
            image_size = image.size
        visible |= camera_visibility_mask(lidar_centers, camera, image_size)
    return visible


def filter_predictions(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    raw_labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if boxes.ndim != 2 or boxes.shape[1] != 9:
        raise ValueError(f"boxes must be [N,9], got {tuple(boxes.shape)}")
    if scores.shape != (boxes.shape[0],) or raw_labels.shape != (boxes.shape[0],):
        raise ValueError("StreamPETR boxes, scores, and labels must have equal length")
    if raw_labels.numel() and (
        int(raw_labels.min()) < 0 or int(raw_labels.max()) >= len(STREAM_PETR_TO_QUEST)
    ):
        raise ValueError("StreamPETR labels are outside the 10-class taxonomy")

    mapped_labels = STREAM_PETR_TO_QUEST[raw_labels]
    distances = torch.linalg.vector_norm(boxes[:, :2], dim=1)
    keep = (
        (scores >= SCORE_THRESHOLD)
        & (distances <= MAX_DISTANCE_METERS)
        & ((mapped_labels == 0) | (mapped_labels == 1))
    )
    return {
        "centers": boxes[keep, :3],
        "scores": scores[keep],
        "labels": mapped_labels[keep],
    }


def pedestrian_distance_bin(distance: float) -> int | None:
    for index, (lower, upper) in enumerate(PEDESTRIAN_DISTANCE_BINS):
        if lower <= distance < upper or (
            index == len(PEDESTRIAN_DISTANCE_BINS) - 1
            and math.isclose(distance, upper)
        ):
            return index
    return None


@dataclass
class SixCameraMetrics:
    samples: int = 0
    class_gt: list[int] = field(default_factory=lambda: [0, 0])
    class_predictions: list[int] = field(default_factory=lambda: [0, 0])
    class_matched: list[int] = field(default_factory=lambda: [0, 0])
    pedestrian_bin_gt: list[int] = field(
        default_factory=lambda: [0] * len(PEDESTRIAN_DISTANCE_BINS)
    )
    pedestrian_bin_matched: list[int] = field(
        default_factory=lambda: [0] * len(PEDESTRIAN_DISTANCE_BINS)
    )

    def update(
        self,
        prediction_labels: torch.Tensor,
        gt_centers: torch.Tensor,
        gt_labels: torch.Tensor,
        matches: list[tuple[int, int, float]],
    ) -> None:
        self.samples += 1
        for class_id in range(len(CLASS_NAMES)):
            self.class_gt[class_id] += int((gt_labels == class_id).sum())
            self.class_predictions[class_id] += int(
                (prediction_labels == class_id).sum()
            )

        pedestrian_distances = torch.linalg.vector_norm(gt_centers[:, :2], dim=1)
        for gt_index in torch.nonzero(gt_labels == 1, as_tuple=False).flatten():
            bin_index = pedestrian_distance_bin(float(pedestrian_distances[gt_index]))
            if bin_index is not None:
                self.pedestrian_bin_gt[bin_index] += 1

        for prediction_index, gt_index, _ in matches:
            class_id = int(gt_labels[gt_index])
            if int(prediction_labels[prediction_index]) != class_id:
                continue
            self.class_matched[class_id] += 1
            if class_id == 1:
                bin_index = pedestrian_distance_bin(
                    float(pedestrian_distances[gt_index])
                )
                if bin_index is not None:
                    self.pedestrian_bin_matched[bin_index] += 1

    @staticmethod
    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    def summary(self) -> dict[str, Any]:
        per_class = {}
        for class_id, class_name in enumerate(CLASS_NAMES):
            per_class[class_name] = {
                "gt": self.class_gt[class_id],
                "predictions": self.class_predictions[class_id],
                "matched": self.class_matched[class_id],
                "precision": self.ratio(
                    self.class_matched[class_id], self.class_predictions[class_id]
                ),
                "recall": self.ratio(
                    self.class_matched[class_id], self.class_gt[class_id]
                ),
            }
        pedestrian_bins = []
        for index, (lower, upper) in enumerate(PEDESTRIAN_DISTANCE_BINS):
            pedestrian_bins.append(
                {
                    "range": f"{int(lower)}-{int(upper)}m",
                    "gt": self.pedestrian_bin_gt[index],
                    "matched": self.pedestrian_bin_matched[index],
                    "recall": self.ratio(
                        self.pedestrian_bin_matched[index],
                        self.pedestrian_bin_gt[index],
                    ),
                }
            )
        return {
            "samples": self.samples,
            "per_class": per_class,
            "pedestrian_distance_recall": pedestrian_bins,
        }


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


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
    camera_root = Path(dataset_config["camera_root"])
    raw_dir = _resolve(args.raw_dir)
    metrics = SixCameraMetrics()

    for offset, complete_index in enumerate(
        range(args.start, args.start + args.count), start=1
    ):
        info = dataset.infos[complete_index]
        token = str(info["token"])
        gt = load_metric_agent_gt(info)
        gt_distances = torch.linalg.vector_norm(gt["boxes"][:, :2], dim=1)
        visible = six_camera_visibility_mask(
            info, gt["boxes"][:, :3], camera_root
        )
        gt_keep = (
            visible
            & (gt_distances <= MAX_DISTANCE_METERS)
            & ((gt["labels"] == 0) | (gt["labels"] == 1))
        )
        gt_centers = gt["boxes"][gt_keep, :3]
        gt_labels = gt["labels"][gt_keep]

        raw = load_raw_predictions(raw_dir / f"{token}.pt", token)
        predictions = filter_predictions(
            raw["boxes"], raw["scores"], raw["labels"]
        )
        matches = strict_class_aware_hungarian(
            predictions["centers"],
            predictions["labels"],
            gt_centers,
            gt_labels,
            args.distance_threshold,
        )
        metrics.update(
            predictions["labels"], gt_centers, gt_labels, matches
        )
        print(f"\rprogress {offset}/{args.count} token={token}", end="", flush=True)
    print()

    summary = metrics.summary()
    print(f"samples: {summary['samples']}")
    print(f"cameras: {','.join(SIX_CAMERA_NAMES)}")
    print(f"score_threshold: {SCORE_THRESHOLD:.2f}")
    print(f"distance_threshold_m: {args.distance_threshold:.2f}")
    print(f"max_distance_m: {MAX_DISTANCE_METERS:.1f}")
    print("visibility: GT center projects inside at least one six-camera image")
    for class_name in CLASS_NAMES:
        values = summary["per_class"][class_name]
        print(
            f"{class_name}: gt={values['gt']} predictions={values['predictions']} "
            f"matched={values['matched']} precision={values['precision']:.6f} "
            f"recall={values['recall']:.6f}"
        )
    print("pedestrian_recall_by_distance:")
    for values in summary["pedestrian_distance_recall"]:
        print(
            f"  {values['range']}: gt={values['gt']} matched={values['matched']} "
            f"recall={values['recall']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
