from __future__ import annotations

import argparse
import importlib
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from quest.openscene_dataset import OPENSCENE_AGENT_CLASS_TO_ID  # noqa: E402
from run_navformer_openscene_teacher import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_METADATA,
    NAVFORMER_ROOT,
    build_camera_geometry,
    build_model_and_load_checkpoint,
    build_preprocess_transforms,
    load_infos,
    prepare_model_inputs,
    prepare_temporal_metadata,
    preprocess_images,
    require_directory,
    require_file,
    reset_tracking_state,
    run_track_only,
    select_infos,
    track_output_tensors,
)


NAVFORMER_TO_EVALUATION = {0: 0, 2: 1}
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


def load_metric_agent_gt(info: dict[str, Any]) -> dict[str, torch.Tensor]:
    raw_boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float32)
    raw_names = np.asarray(info.get("gt_names", []))
    boxes = []
    labels = []
    for box, raw_name in zip(raw_boxes, raw_names):
        name = str(raw_name)
        if name not in OPENSCENE_AGENT_CLASS_TO_ID or len(box) < 7:
            continue
        values = np.asarray(box[:7], dtype=np.float32)
        if not np.isfinite(values).all():
            continue
        boxes.append(torch.from_numpy(values.copy()))
        labels.append(OPENSCENE_AGENT_CLASS_TO_ID[name])
    return {
        "boxes": torch.stack(boxes) if boxes else torch.empty(0, 7),
        "labels": torch.tensor(labels, dtype=torch.long),
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
            if distance <= float(distance_threshold):
                matches.append((prediction_index, gt_index, distance))
    matches.sort(key=lambda item: item[0])
    return matches


def pedestrian_distance_bin(distance: float) -> int | None:
    for index, (lower, upper) in enumerate(PEDESTRIAN_DISTANCE_BINS):
        if lower <= distance < upper or (
            index == len(PEDESTRIAN_DISTANCE_BINS) - 1
            and math.isclose(distance, upper)
        ):
            return index
    return None


@dataclass
class AgentMetrics:
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
                bin_index = pedestrian_distance_bin(float(pedestrian_distances[gt_index]))
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
                    "range": "{}-{}m".format(int(lower), int(upper)),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Custom 2m center evaluation for Navformer on OpenScene"
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--navformer-root", type=Path, default=NAVFORMER_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--distance-threshold", type=float, default=2.0)
    return parser.parse_args()


def filter_navformer_predictions(
    boxes: Any,
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    box_tensor = boxes.tensor if hasattr(boxes, "tensor") else boxes
    box_tensor = torch.as_tensor(box_tensor).detach().cpu()
    scores = torch.as_tensor(scores).detach().cpu()
    labels = torch.as_tensor(labels).detach().cpu().long()
    if box_tensor.ndim != 2 or box_tensor.shape[1] < 3:
        raise ValueError(
            "Navformer boxes must be [N,>=3], got {}".format(tuple(box_tensor.shape))
        )
    if scores.shape != (box_tensor.shape[0],) or labels.shape != (box_tensor.shape[0],):
        raise ValueError("Navformer boxes, scores, and labels must have equal length")

    distances = torch.linalg.vector_norm(box_tensor[:, :2], dim=1)
    keep = (scores >= SCORE_THRESHOLD) & (distances <= MAX_DISTANCE_METERS)
    keep &= (labels == 0) | (labels == 2)
    kept_labels = labels[keep]
    mapped_labels = torch.empty_like(kept_labels)
    for navformer_label, evaluation_label in NAVFORMER_TO_EVALUATION.items():
        mapped_labels[kept_labels == navformer_label] = evaluation_label
    return {
        "centers": box_tensor[keep, :3].float(),
        "scores": scores[keep].float(),
        "labels": mapped_labels,
    }


def filter_agent_gt(info: dict[str, Any]) -> dict[str, torch.Tensor]:
    gt = load_metric_agent_gt(info)
    distances = torch.linalg.vector_norm(gt["boxes"][:, :2], dim=1)
    keep = (
        (distances <= MAX_DISTANCE_METERS)
        & ((gt["labels"] == 0) | (gt["labels"] == 1))
    )
    return {
        "centers": gt["boxes"][keep, :3].float(),
        "labels": gt["labels"][keep].long(),
    }


def print_summary(metrics: AgentMetrics, distance_threshold: float) -> None:
    summary = metrics.summary()
    print("\nCUSTOM_METRIC = class-aware Hungarian center matching (not official mAP)")
    print("samples: {}".format(summary["samples"]))
    print("score_threshold: {:.2f}".format(SCORE_THRESHOLD))
    print("distance_threshold_m: {:.2f}".format(distance_threshold))
    print("max_distance_m: {:.1f}".format(MAX_DISTANCE_METERS))
    for class_name in CLASS_NAMES:
        values = summary["per_class"][class_name]
        print(
            "{}: GT={} predictions={} matched={} precision={:.6f} recall={:.6f}".format(
                class_name,
                values["gt"],
                values["predictions"],
                values["matched"],
                values["precision"],
                values["recall"],
            )
        )
    print("pedestrian_recall_by_distance:")
    for values in summary["pedestrian_distance_recall"]:
        print(
            "  {}: GT={} matched={} recall={:.6f}".format(
                values["range"], values["gt"], values["matched"], values["recall"]
            )
        )


def main() -> int:
    args = parse_args()
    if args.distance_threshold <= 0:
        raise ValueError("distance-threshold must be positive")
    navformer_root = require_directory(args.navformer_root, "Navformer root")
    config_path = require_file(args.config, "Navformer config")
    checkpoint_path = require_file(args.checkpoint, "Navformer checkpoint")
    metadata_path = require_file(args.metadata, "OpenScene metadata")
    image_root = require_directory(args.image_root, "OpenScene image root")
    if str(navformer_root) not in sys.path:
        sys.path.insert(0, str(navformer_root))

    import cv2
    import mmcv
    from mmcv import Config
    from mmcv.utils import build_from_cfg
    from mmdet.datasets.builder import PIPELINES
    from mmdet3d.core.bbox import get_box_type
    from mmdet3d.models import build_model

    cfg = Config.fromfile(str(config_path))
    custom_imports = cfg.get("custom_imports", {})
    for module_name in custom_imports.get("imports", ["mmdet3d_plugin"]):
        importlib.import_module(module_name)
    transforms, _ = build_preprocess_transforms(cfg, build_from_cfg, PIPELINES)
    infos = select_infos(load_infos(metadata_path), args.sample_index, args.num_frames)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Navformer teacher evaluation")
    device = torch.device("cuda:0")
    model = build_model_and_load_checkpoint(cfg, checkpoint_path, build_model)
    model.to(device).eval()
    reset_tracking_state(model)

    metrics = AgentMetrics()
    previous_scene = None
    for offset, info in enumerate(infos, start=1):
        scene_token = str(info.get("scene_token"))
        if previous_scene is not None and scene_token != previous_scene:
            reset_tracking_state(model)
        geometry = build_camera_geometry(info, image_root, cv2)
        processed, image_tensor = preprocess_images(geometry, transforms, mmcv)
        model_inputs = prepare_model_inputs(
            info, geometry, processed, image_tensor, get_box_type, device
        )
        prepare_temporal_metadata(model, model_inputs)
        with torch.no_grad():
            track = run_track_only(model, model_inputs)
        boxes, scores, labels, track_ids = track_output_tensors(track)
        predictions = filter_navformer_predictions(boxes, scores, labels)
        gt = filter_agent_gt(info)
        matches = strict_class_aware_hungarian(
            predictions["centers"],
            predictions["labels"],
            gt["centers"],
            gt["labels"],
            args.distance_threshold,
        )
        metrics.update(predictions["labels"], gt["centers"], gt["labels"], matches)
        print(
            "progress {}/{} frame_idx={} token={} tracks={} track_ids={}".format(
                offset,
                len(infos),
                info["frame_idx"],
                info["token"],
                int(scores.numel()),
                track_ids.tolist(),
            )
        )
        previous_scene = scene_token

    print_summary(metrics, args.distance_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
