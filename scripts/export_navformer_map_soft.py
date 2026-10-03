from __future__ import annotations

import argparse
import hashlib
import importlib
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from quest.map_teacher import (
    TEACHER_ALIGNMENT_VERSION, TEACHER_COORDINATE_FRAME, TEACHER_MAP_SCHEMA_VERSION,
    TEACHER_RAW_SCORE_SEMANTICS, TEACHER_SCORE_KIND, TEACHER_SCORE_TRANSFORM,
    resolve_lidar2ego, validate_teacher_record,
)
from quest.stage3_dataset import load_record
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
    if tensor.ndim != 3:
        raise ValueError(f"Navformer {name} must be [K,H,W], got {tuple(tensor.shape)}")
    return tensor.float()


def extract_teacher_soft_scores(mapping: dict) -> tuple[torch.Tensor, int]:
    if "lane_score" not in mapping or "score_list" not in mapping:
        raise KeyError("Navformer map output lacks lane_score or drivable score_list")
    raw_lanes = _channels(mapping["lane_score"], "lane_score")
    raw_scores = _channels(mapping["score_list"], "score_list")
    if raw_scores.shape[0] < 1:
        raise ValueError("Navformer score_list has no drivable mask channel")
    raw_drivable = raw_scores[-1:]
    lanes = raw_lanes
    drivable = raw_drivable
    if lanes.shape[-2:] != drivable.shape[-2:]:
        raise ValueError("Navformer lane/drivable map resolutions differ")
    raw = torch.cat((lanes, drivable), dim=0)
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("Navformer map contains NaN/Inf")
    return raw.clamp(0.0, 1.0), lanes.shape[0]


def temporal_export_plan(all_infos: list[dict], sample_index: int, num_frames: int) -> list[dict]:
    targets = select_infos(all_infos, sample_index, num_frames)
    target_indices = set(range(sample_index, sample_index + len(targets)))
    scenes = {str(info["scene_token"]) for info in targets}
    by_scene: dict[str, list[tuple[int, dict]]] = {scene: [] for scene in scenes}
    for index, info in enumerate(all_infos):
        scene = str(info.get("scene_token"))
        if scene in by_scene:
            by_scene[scene].append((index, info))
    plan = []
    for scene in sorted(scenes):
        frames = sorted(by_scene[scene], key=lambda pair: float(pair[1]["timestamp"]))
        if len({str(info["token"]) for _, info in frames}) != len(frames):
            raise ValueError(f"duplicate tokens in scene {scene}")
        timestamps = [float(info["timestamp"]) for _, info in frames]
        if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
            raise ValueError(f"non-increasing timestamps in scene {scene}")
        latest_target = max(float(info["timestamp"]) for index, info in frames if index in target_indices)
        first_token = str(frames[0][1]["token"])
        history = hashlib.sha256()
        for index, info in frames:
            if float(info["timestamp"]) > latest_target:
                break
            history.update(f"{info['token']}:{info['timestamp']}\n".encode("utf-8"))
            plan.append({
                "sample_index": index, "info": info, "target": index in target_indices,
                "scene_start_token": first_token, "temporal_history_sha256": history.hexdigest(),
            })
    if {item["sample_index"] for item in plan if item["target"]} != target_indices:
        raise RuntimeError("temporal plan target indices differ from the original metadata slice")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Export offline Navformer Pansegformer soft mask scores")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/navformer_map_soft")
    parser.add_argument("--navformer-root", type=Path, default=NAVFORMER_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/openscene/meta_datas/meta_data_mini.pkl")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--teacher-pc-range", nargs=4, type=float, required=True,
                        metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
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
    plan = temporal_export_plan(load_infos(metadata_path), args.sample_index, args.num_frames)
    for item in plan:
        if item["target"]:
            item["lidar2ego"] = resolve_lidar2ego(item["info"])
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    checkpoint_stat = checkpoint_path.stat()
    expected = {
        "teacher_pc_range": tuple(args.teacher_pc_range),
        "teacher_coordinate_frame": TEACHER_COORDINATE_FRAME,
        "teacher_alignment_version": TEACHER_ALIGNMENT_VERSION,
        "teacher_checkpoint": str(checkpoint_path),
        "teacher_config": str(config_path),
        "teacher_config_sha256": config_hash,
        "teacher_checkpoint_size_bytes": checkpoint_stat.st_size,
        "teacher_checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    skipped_existing = 0
    for item in plan:
        if not item["target"]:
            continue
        token = str(item["info"]["token"])
        path = args.output_dir / f"{token}.pt"
        if not path.exists() or args.overwrite:
            continue
        try:
            record = load_record(path)
            validate_teacher_record(record, token, item["sample_index"])
            for key, value in expected.items():
                if record.get(key) != value:
                    raise ValueError(f"{key} differs from this export configuration")
            if not torch.allclose(record["teacher_lidar2ego"], item["lidar2ego"], atol=1e-4, rtol=1e-4):
                raise ValueError("lidar2ego differs from current OpenScene metadata")
            if (record.get("temporal_history_sha256") != item["temporal_history_sha256"]
                    or record.get("temporal_scene_start_token") != item["scene_start_token"]):
                raise ValueError("temporal history differs from this scene-start inference")
        except Exception as error:
            raise ValueError(f"existing teacher label is invalid: {path}: {error}; use --overwrite to regenerate") from error
        skipped_existing += 1
    target_count = sum(item["target"] for item in plan)
    if skipped_existing == target_count:
        print(f"skipped_existing={skipped_existing} target_count={target_count}; no inference needed")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Navformer teacher export")
    device = torch.device("cuda:0")
    model = build_model_and_load_checkpoint(cfg, checkpoint_path, build_model).to(device).eval()
    if not getattr(model, "with_seg_head", False):
        raise RuntimeError("built Navformer model has no map seg_head")
    reset_tracking_state(model)
    previous_scene = None
    previous_timestamp = None
    saved = warmup = replayed_existing = 0
    for item in plan:
        sample_index, info = item["sample_index"], item["info"]
        scene = str(info["scene_token"])
        timestamp = float(info["timestamp"])
        if scene != previous_scene or (previous_timestamp is not None and timestamp - previous_timestamp > 1.1e6):
            reset_tracking_state(model)
        geometry = build_camera_geometry(info, image_root, cv2)
        processed, image_tensor = preprocess_images(geometry, transforms, mmcv)
        model_inputs = prepare_model_inputs(info, geometry, processed, image_tensor, get_box_type, device)
        prepare_temporal_metadata(model, model_inputs)
        with torch.no_grad():
            _, mapping = run_track_and_map_without_gt(model, model_inputs)
        soft, lane_count = extract_teacher_soft_scores(mapping)
        token = str(info["token"])
        if item["target"]:
            path = args.output_dir / f"{token}.pt"
            if path.exists() and not args.overwrite:
                replayed_existing += 1
                print(f"index={sample_index} token={token} replayed_existing=true saved=false")
            else:
                names = [f"lane_score_{i}" for i in range(lane_count)] + ["drivable_score_0"]
                record = {
                    **expected,
                    "sample_index": sample_index, "token": token,
                    "teacher_map_soft": soft.cpu(), "teacher_map_shape": tuple(soft.shape),
                    "teacher_channel_names_or_ids": names,
                    "teacher_lidar2ego": item["lidar2ego"],
                    "teacher_score_kind": TEACHER_SCORE_KIND,
                    "teacher_raw_score_semantics": TEACHER_RAW_SCORE_SEMANTICS,
                    "teacher_score_transform": TEACHER_SCORE_TRANSFORM,
                    "temporal_mode": "scene_start_to_target",
                    "temporal_scene_start_token": item["scene_start_token"],
                    "temporal_history_sha256": item["temporal_history_sha256"],
                    "schema_version": TEACHER_MAP_SCHEMA_VERSION,
                }
                temporary = path.with_suffix(".pt.tmp")
                torch.save(record, temporary)
                temporary.replace(path)
                saved += 1
                print(f"index={sample_index} token={token} target=true saved=true shape={tuple(soft.shape)}")
        else:
            warmup += 1
            print(f"index={sample_index} token={token} target=false saved=false")
        previous_scene = scene
        previous_timestamp = timestamp
    print(f"targets={target_count} saved={saved} skipped_existing={skipped_existing} "
          f"replayed_existing={replayed_existing} warmup_frames={warmup} inference_frames={len(plan)}")


if __name__ == "__main__":
    main()
