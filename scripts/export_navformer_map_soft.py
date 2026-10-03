from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from quest.map_teacher import TEACHER_MAP_SCHEMA_VERSION
from run_navformer_openscene_teacher import (
    DEFAULT_IMAGE_ROOT,
    NAVFORMER_ROOT, as_cpu, build_camera_geometry, build_model_and_load_checkpoint,
    build_preprocess_transforms, load_infos, prepare_model_inputs,
    prepare_temporal_metadata, preprocess_images, require_directory, require_file,
    reset_tracking_state, run_track_and_map_without_gt, select_infos,
)


def _channels(value: object, name: str) -> torch.Tensor:
    tensor = as_cpu(value)
    if not torch.is_tensor(tensor):
        raise ValueError(f"Navformer {name} must be a tensor; got {type(value).__name__}")
    while tensor.ndim > 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Navformer {name} must resolve to [K,H,W], got {tuple(tensor.shape)}")
    return tensor.float()


def extract_teacher_probabilities(mapping: dict, input_kind: str) -> torch.Tensor:
    if "lane_score" not in mapping or "score_list" not in mapping:
        raise KeyError("Navformer map output lacks lane_score or drivable score_list")
    if not isinstance(mapping["score_list"], (list, tuple)) or not mapping["score_list"]:
        raise ValueError("Navformer score_list must be a nonempty sequence")
    lanes = _channels(mapping["lane_score"], "lane_score")
    drivable = _channels(mapping["score_list"][-1], "score_list[-1]")
    if lanes.shape[-2:] != drivable.shape[-2:]:
        raise ValueError("Navformer lane/drivable map resolutions differ")
    values = torch.cat((lanes, drivable), dim=0)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Navformer map contains NaN/Inf")
    if input_kind == "logits":
        values = values.sigmoid()
    elif input_kind != "probabilities":
        raise ValueError("--input-kind must explicitly be probabilities or logits")
    if not bool(((values >= 0) & (values <= 1)).all()):
        raise ValueError("Navformer map values are not probabilities; check --input-kind")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Export offline Navformer map probabilities")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/navformer_map_soft")
    parser.add_argument("--navformer-root", type=Path, default=NAVFORMER_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/openscene/meta_datas/meta_data_mini.pkl")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--teacher-pc-range", nargs=4, type=float, required=True,
                        metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--teacher-coordinate-frame", choices=["openscene_lidar_xy"], required=True)
    parser.add_argument("--input-kind", choices=["probabilities", "logits"], required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    navformer_root = require_directory(args.navformer_root, "Navformer root")
    config_path = require_file(
        args.config if args.config.is_absolute() else navformer_root / args.config,
        "map teacher config",
    )
    checkpoint_path = require_file(
        args.checkpoint if args.checkpoint.is_absolute() else navformer_root / args.checkpoint,
        "map teacher checkpoint",
    )
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
    if not cfg.model.get("seg_head"):
        raise RuntimeError("Navformer config has no Pansegformer seg_head")
    try:
        raw_checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        raw_checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = raw_checkpoint.get("state_dict", raw_checkpoint)
    if not isinstance(state, dict) or not any("seg_head." in str(key) for key in state):
        raise RuntimeError("checkpoint has no seg_head weights; Agent-only checkpoint is invalid")
    del raw_checkpoint, state
    for module_name in cfg.get("custom_imports", {}).get("imports", ["mmdet3d_plugin"]):
        importlib.import_module(module_name)
    transforms, _ = build_preprocess_transforms(cfg, build_from_cfg, PIPELINES)
    infos = select_infos(load_infos(metadata_path), args.sample_index, args.num_frames)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Navformer teacher export")
    device = torch.device("cuda:0")
    model = build_model_and_load_checkpoint(cfg, checkpoint_path, build_model).to(device).eval()
    if not getattr(model, "with_seg_head", False):
        raise RuntimeError("built Navformer model has no map seg_head")
    reset_tracking_state(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_scene = None
    for sample_index, info in enumerate(infos, start=args.sample_index):
        scene = str(info["scene_token"])
        if scene != previous_scene:
            reset_tracking_state(model)
        geometry = build_camera_geometry(info, image_root, cv2)
        processed, image_tensor = preprocess_images(geometry, transforms, mmcv)
        model_inputs = prepare_model_inputs(info, geometry, processed, image_tensor, get_box_type, device)
        prepare_temporal_metadata(model, model_inputs)
        with torch.no_grad():
            _, mapping = run_track_and_map_without_gt(model, model_inputs)
        soft = extract_teacher_probabilities(mapping, args.input_kind)
        token = str(info["token"])
        path = args.output_dir / f"{token}.pt"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite map teacher label: {path}")
        lane_count = _channels(mapping["lane_score"], "lane_score").shape[0]
        names = [f"lane_score_{i}" for i in range(lane_count)]
        names += [f"drivable_score_{i}" for i in range(soft.shape[0] - len(names))]
        record = {
            "sample_index": sample_index, "token": token,
            "teacher_map_soft": soft.cpu(), "teacher_map_shape": tuple(soft.shape),
            "teacher_pc_range": tuple(args.teacher_pc_range),
            "teacher_channel_names_or_ids": names,
            "teacher_checkpoint": str(checkpoint_path),
            "teacher_coordinate_frame": args.teacher_coordinate_frame,
            "schema_version": TEACHER_MAP_SCHEMA_VERSION,
        }
        temporary = path.with_suffix(".pt.tmp")
        torch.save(record, temporary)
        temporary.replace(path)
        print(f"index={sample_index} token={token} teacher_map_shape={tuple(soft.shape)}")
        previous_scene = scene


if __name__ == "__main__":
    main()
