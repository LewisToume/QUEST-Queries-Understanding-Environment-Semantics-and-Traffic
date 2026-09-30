from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_streampetr_coordinates import load_metric_agent_gt  # noqa: E402
from evaluate_streampetr_six_camera import (  # noqa: E402
    MAX_DISTANCE_METERS,
    SCORE_THRESHOLD,
    SixCameraMetrics,
)
from evaluate_streampetr_thresholds import strict_class_aware_hungarian  # noqa: E402
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


def print_summary(metrics: SixCameraMetrics, distance_threshold: float) -> None:
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

    metrics = SixCameraMetrics()
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
