from __future__ import print_function

import argparse
import importlib
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from quest.teachers import navformer_output_to_quest  # noqa: E402
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


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/soft_labels_navformer"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export token-aligned Navformer Agent labels for QUEST Stage2"
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--max-distance", type=float, default=50.0)
    parser.add_argument("--navformer-root", type=Path, default=NAVFORMER_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    return parser.parse_args()


def build_payload(
    token, boxes, scores, labels, score_threshold=0.25, max_distance=50.0
):
    return {
        "token": str(token),
        "agent": navformer_output_to_quest(
            {
                "boxes_3d": boxes,
                "scores_3d": scores,
                "labels_3d": labels,
            },
            score_threshold=score_threshold,
            max_distance_m=max_distance,
        ),
    }


def save_payload(payload, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "{}.pt".format(payload["token"])
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def main():
    args = parse_args()
    navformer_root = require_directory(args.navformer_root, "Navformer root")
    config_path = require_file(args.config, "Navformer config")
    checkpoint_path = require_file(args.checkpoint, "Navformer checkpoint")
    metadata_path = require_file(args.metadata, "OpenScene metadata")
    image_root = require_directory(args.image_root, "OpenScene image root")
    output_dir = Path(args.output_dir).expanduser().resolve()
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
        raise RuntimeError("CUDA is required for Navformer pseudo-label export")
    device = torch.device("cuda:0")
    model = build_model_and_load_checkpoint(cfg, checkpoint_path, build_model)
    model.to(device).eval()

    if args.score_threshold < 0:
        raise ValueError("score-threshold must be non-negative")
    if args.max_distance <= 0:
        raise ValueError("max-distance must be positive")

    previous_scene = None
    saved = 0
    total_labels = 0
    total_vehicle = 0
    total_pedestrian = 0
    for offset, info in enumerate(infos, start=1):
        token = str(info["token"])
        scene_token = str(info.get("scene_token"))
        new_scene = previous_scene is None or scene_token != previous_scene
        if new_scene:
            reset_tracking_state(model)

        geometry = build_camera_geometry(info, image_root, cv2)
        processed, image_tensor = preprocess_images(geometry, transforms, mmcv)
        model_inputs = prepare_model_inputs(
            info, geometry, processed, image_tensor, get_box_type, device
        )
        continuous = prepare_temporal_metadata(model, model_inputs)
        with torch.no_grad():
            track = run_track_only(model, model_inputs)
        boxes, scores, labels, _track_ids = track_output_tensors(track)
        payload = build_payload(
            token,
            boxes,
            scores,
            labels,
            score_threshold=args.score_threshold,
            max_distance=args.max_distance,
        )
        output_path = save_payload(payload, output_dir)
        output_labels = payload["agent"]["labels"]
        valid_count = int((output_labels >= 0).sum())
        vehicle_count = int((output_labels == 0).sum())
        pedestrian_count = int((output_labels == 1).sum())
        total_labels += valid_count
        total_vehicle += vehicle_count
        total_pedestrian += pedestrian_count
        saved += 1
        print(
            "progress={}/{} frame_idx={} token={} scene={} new_scene={} "
            "continuous={} tracks={} valid_total={} vehicle_count={} "
            "pedestrian_count={} output={}".format(
                offset,
                len(infos),
                info["frame_idx"],
                token,
                scene_token,
                new_scene,
                continuous,
                int(scores.numel()),
                valid_count,
                vehicle_count,
                pedestrian_count,
                output_path,
            )
        )
        previous_scene = scene_token

    print("attempted: {}".format(len(infos)))
    print("saved: {}".format(saved))
    print("total labels: {}".format(total_labels))
    print("vehicle labels: {}".format(total_vehicle))
    print("pedestrian labels: {}".format(total_pedestrian))
    print(
        "mean labels/frame: {:.6f}".format(
            float(total_labels) / saved if saved else 0.0
        )
    )
    print("output_dir: {}".format(output_dir))
    print("NAVFORMER_PSEUDO_EXPORT = PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("NAVFORMER_PSEUDO_EXPORT = FAIL", file=sys.stderr)
        print("{}: {}".format(type(error).__name__, error), file=sys.stderr)
        raise
