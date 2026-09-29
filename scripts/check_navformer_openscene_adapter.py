from __future__ import annotations

import argparse
import ast
import pickle
import pprint
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


NAVFORMER_ROOT = Path("/home/user/DataDisk/QUEST_WORK/Navformer")
DEFAULT_CONFIG = NAVFORMER_ROOT / "configs/navformer/track_map_nuplan_r50_navtrain.py"
DEFAULT_CHECKPOINT = (
    NAVFORMER_ROOT
    / "data/alg_engine/ckpts/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth"
)
DEFAULT_OPENSCENE_METADATA = Path(
    "/home/user/DataDisk/QUEST_WORK/data/openscene/openscene-v1.0/"
    "meta_datas/meta_data_mini.pkl"
)

DATASET_SOURCE_FILES = (
    NAVFORMER_ROOT / "mmdet3d_plugin/datasets/navsim_openscene_nuplan.py",
    NAVFORMER_ROOT / "mmdet3d_plugin/datasets/navsim_openscene_nuplan_det.py",
)

EXPECTED_DATASET_TYPE = "NavSimOpenSceneE2EDet"
EXPECTED_CAMERA_COUNT = 8
CAMERA_REQUIRED_FIELDS = (
    "data_path",
    "sensor2lidar_rotation",
    "sensor2lidar_translation",
    "cam_intrinsic",
    "distortion",
)

# These are read by NavSimOpenSceneE2EDet and its NavSimOpenSceneE2E parent for
# the track_map config. They describe the merged Navformer annotation format,
# not the stock OpenScene release format.
REQUIRED_FIELD_GROUPS = {
    "identity_and_sequence": (
        "token",
        "frame_idx",
        "timestamp",
        "log_name",
        "log_token",
        "lidar_path",
        "sample_prev",
        "sample_next",
    ),
    "pose_and_motion": (
        "lidar2global",
        "lidar2ego",
        "lidar2ego_rotation",
        "lidar2ego_translation",
        "ego2global",
        "ego2global_rotation",
        "ego2global_translation",
        "can_bus",
    ),
    "camera": ("cams",),
    "detection_and_tracking": (
        "gt_boxes",
        "gt_names",
        "gt_velocity",
        "valid_flag",
        "gt_inds",
        "gt_fut_bbox_lidar",
        "gt_fut_bbox_mask",
        "gt_pre_bbox_lidar",
        "gt_pre_bbox_mask",
    ),
    "map": (
        "map_location",
        "ego2global_translation",
        "ego2global_rotation",
    ),
    "ego_prediction_and_planning": (
        "gt_pre_bbox_sdc_lidar",
        "gt_fut_bbox_sdc_lidar",
        "gt_pre_bbox_sdc_global",
        "gt_fut_bbox_sdc_global",
        "gt_pre_bbox_sdc_mask",
        "gt_fut_bbox_sdc_mask",
        "gt_pre_command_sdc",
        "driving_command",
    ),
}

FIELD_CANDIDATES = {
    "log_name": ("scene_name",),
    "log_token": ("scene_token",),
    "gt_inds": ("instance_tokens", "track_tokens"),
    "gt_velocity": ("gt_velocity_3d",),
    "lidar_path": ("sweeps",),
    "gt_fut_bbox_lidar": ("gt_boxes_st",),
    "gt_pre_bbox_lidar": ("gt_boxes_st",),
}

KEY_GROUP_KEYWORDS = {
    "camera_related": (
        "cam",
        "camera",
        "image",
        "img",
        "sensor",
    ),
    "pose_related": (
        "pose",
        "ego",
        "global",
        "lidar2",
        "can_bus",
    ),
    "calibration_related": (
        "intrinsic",
        "extrinsic",
        "rotation",
        "translation",
        "distortion",
        "lidar2cam",
        "lidar2img",
    ),
    "annotation_related": (
        "gt_",
        "annotation",
        "label",
        "box",
        "velocity",
        "valid",
        "instance",
        "track",
        "map_",
        "roadblock",
        "command",
        "future",
        "fut_",
        "pre_",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect stock OpenScene metadata against Navformer's track_map adapter"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_OPENSCENE_METADATA)
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} MISSING: {path}")
    return path


def load_config(path: Path) -> Any:
    try:
        from mmcv import Config
    except ImportError as error:
        raise RuntimeError(
            "mmcv is required to load the Navformer config; use the existing "
            "Navformer environment"
        ) from error
    return Config.fromfile(str(path))


def config_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def find_config_values(value: Any, target_key: str) -> list[Any]:
    values = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == target_key:
                values.append(child)
            values.extend(find_config_values(child, target_key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            values.extend(find_config_values(child, target_key))
    return values


def load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping) and isinstance(
        checkpoint.get("state_dict"), Mapping
    ):
        return checkpoint["state_dict"]
    if isinstance(checkpoint, Mapping) and checkpoint and all(
        isinstance(key, str) for key in checkpoint
    ) and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    return {}


def load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)


def extract_samples(metadata: Any) -> tuple[list[Any], str]:
    if isinstance(metadata, Mapping):
        for key in ("infos", "samples", "data_list"):
            value = metadata.get(key)
            if isinstance(value, list):
                return value, key
        list_fields = [
            (str(key), value)
            for key, value in metadata.items()
            if isinstance(value, list)
        ]
        if len(list_fields) == 1:
            return list_fields[0][1], list_fields[0][0]
        raise ValueError(
            "metadata dict has no unambiguous sample list; expected infos/samples/data_list"
        )
    if isinstance(metadata, list):
        return metadata, "<top-level-list>"
    raise TypeError(f"unsupported metadata top-level type: {type(metadata).__name__}")


def summarize(value: Any) -> str:
    if value is None:
        return "None"
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
        return summarize(array)
    if isinstance(value, np.ndarray):
        finite = bool(np.isfinite(value).all()) if value.dtype.kind in "fciu" else "n/a"
        base = f"ndarray shape={value.shape} dtype={value.dtype} finite={finite}"
        if value.size <= 16:
            base += f" value={value.tolist()}"
        return base
    if isinstance(value, Mapping):
        return f"{type(value).__name__} keys={sorted(map(str, value.keys()))}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__} len={len(value)}"
    if isinstance(value, (str, bytes)):
        display = repr(value)
        return display if len(display) <= 240 else display[:237] + "..."
    return f"{type(value).__name__}: {value!r}"


def matching_keys(sample: Mapping[str, Any], keywords: Sequence[str]) -> list[str]:
    return sorted(
        str(key)
        for key in sample
        if any(keyword in str(key).lower() for keyword in keywords)
    )


def ast_info_fields(source_paths: Sequence[Path]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for path in source_paths:
        if not path.is_file():
            result[str(path)] = []
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        fields = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            owner = node.value
            if not isinstance(owner, ast.Name) or owner.id not in {"info", "frame_info"}:
                continue
            key_node = node.slice
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                fields.add(key_node.value)
        result[str(path)] = sorted(fields)
    return result


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def print_config_report(config: Any, config_path: Path) -> None:
    print_header("Navformer config")
    model = config_get(config, "model", {})
    data = config_get(config, "data", {})
    model_type = config_get(model, "type", "MISSING")
    dataset_types = {
        split: config_get(config_get(data, split, {}), "type", "MISSING")
        for split in ("train", "val", "test")
    }
    print(f"config: {config_path}")
    print(f"model type: {model_type}")
    print(f"dataset types: {dataset_types}")
    print(f"model num_cams values: {find_config_values(model, 'num_cams')}")
    for split in ("train", "val", "test"):
        split_config = config_get(data, split, {})
        print(f"{split}.ann_file: {config_get(split_config, 'ann_file', 'MISSING')}")
        print(
            f"{split}.nav_filter_path: "
            f"{config_get(split_config, 'nav_filter_path', 'MISSING')}"
        )


def print_checkpoint_report(checkpoint: Any, checkpoint_path: Path) -> None:
    print_header("Checkpoint")
    top_keys = sorted(map(str, checkpoint.keys())) if isinstance(checkpoint, Mapping) else []
    state_dict = checkpoint_state_dict(checkpoint)
    tensor_count = sum(torch.is_tensor(value) for value in state_dict.values())
    parameter_elements = sum(
        int(value.numel()) for value in state_dict.values() if torch.is_tensor(value)
    )
    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint type: {type(checkpoint).__name__}")
    print(f"checkpoint top-level keys: {top_keys if top_keys else 'MISSING'}")
    print(f"state_dict parameter entries: {len(state_dict)}")
    print(f"state_dict tensor entries: {tensor_count}")
    print(f"state_dict total parameter elements: {parameter_elements}")
    print("checkpoint meta:")
    if isinstance(checkpoint, Mapping) and "meta" in checkpoint:
        pprint.pprint(checkpoint["meta"], width=120, sort_dicts=False)
    else:
        print("MISSING")


def print_metadata_report(metadata: Any, metadata_path: Path) -> tuple[list[Any], Mapping[str, Any]]:
    print_header("OpenScene metadata")
    print(f"metadata: {metadata_path}")
    print(f"top-level data type: {type(metadata).__name__}")
    if isinstance(metadata, Mapping):
        print(f"top-level keys: {sorted(map(str, metadata.keys()))}")
    else:
        print("top-level keys: MISSING (top level is not a mapping)")

    samples, sample_container = extract_samples(metadata)
    print(f"sample container: {sample_container}")
    print(f"sample count: {len(samples)}")
    if not samples:
        raise ValueError("metadata sample list is empty")
    first = samples[0]
    if not isinstance(first, Mapping):
        raise TypeError(f"first sample must be a mapping, got {type(first).__name__}")
    print(f"first frame data type: {type(first).__name__}")
    print(f"first frame all keys: {sorted(map(str, first.keys()))}")

    print_header("Camera fields")
    camera_key = "cams" if isinstance(first.get("cams"), Mapping) else None
    if camera_key is None:
        print("cams: MISSING")
        camera_names: list[str] = []
    else:
        cameras = first[camera_key]
        camera_names = list(map(str, cameras.keys()))
        print(f"camera field: {camera_key}")
        print(f"camera count: {len(camera_names)}")
        print(f"8 camera names: {camera_names}")
        for camera_name, camera in cameras.items():
            print(f"camera {camera_name}:")
            if not isinstance(camera, Mapping):
                print(f"  invalid camera data: {summarize(camera)}")
                continue
            print(f"  keys: {sorted(map(str, camera.keys()))}")
            for field in CAMERA_REQUIRED_FIELDS:
                if field in camera:
                    print(f"  {field}: {summarize(camera[field])}")
                else:
                    print(f"  {field}: MISSING")

    for group_name, keywords in KEY_GROUP_KEYWORDS.items():
        print_header(group_name.replace("_", " ").title())
        keys = matching_keys(first, keywords)
        print(f"keys: {keys if keys else 'MISSING'}")
        for key in keys:
            print(f"{key}: {summarize(first[key])}")
    return samples, first


def print_compatibility_report(
    config: Any,
    first: Mapping[str, Any],
    source_field_report: Mapping[str, list[str]],
) -> bool:
    print_header("NavSimOpenSceneE2EDet field comparison")
    missing_fields = []
    for group_name, fields in REQUIRED_FIELD_GROUPS.items():
        print(f"[{group_name}]")
        for field in fields:
            if field in first:
                print(f"  {field}: PRESENT ({summarize(first[field])})")
                continue
            missing_fields.append(field)
            candidates = [name for name in FIELD_CANDIDATES.get(field, ()) if name in first]
            suffix = f"; candidate={candidates}" if candidates else ""
            print(f"  {field}: MISSING{suffix}")

    cameras = first.get("cams") if isinstance(first.get("cams"), Mapping) else {}
    camera_count_ok = len(cameras) == EXPECTED_CAMERA_COUNT
    camera_missing: dict[str, list[str]] = {}
    for camera_name, camera in cameras.items():
        if not isinstance(camera, Mapping):
            camera_missing[str(camera_name)] = list(CAMERA_REQUIRED_FIELDS)
            continue
        missing = [field for field in CAMERA_REQUIRED_FIELDS if field not in camera]
        if missing:
            camera_missing[str(camera_name)] = missing
    print(f"camera count expected={EXPECTED_CAMERA_COUNT} actual={len(cameras)}")
    print(f"camera calibration missing: {camera_missing if camera_missing else 'NONE'}")

    data = config_get(config, "data", {})
    dataset_types = {
        str(config_get(config_get(data, split, {}), "type", "MISSING"))
        for split in ("train", "val", "test")
    }
    dataset_type_ok = dataset_types == {EXPECTED_DATASET_TYPE}
    print(f"expected dataset type: {EXPECTED_DATASET_TYPE}")
    print(f"config dataset types: {sorted(dataset_types)}")

    print("local dataset source field reads:")
    for source, fields in source_field_report.items():
        status = fields if fields else "MISSING SOURCE OR NO FIELDS"
        print(f"  {source}: {status}")

    extra_fields = sorted(
        set(map(str, first.keys()))
        - {field for fields in REQUIRED_FIELD_GROUPS.values() for field in fields}
    )
    print(f"OpenScene fields not in the required-field checklist: {extra_fields}")
    print("Important non-equivalent candidates:")
    print("  scene_name/scene_token require explicit conversion to log_name/log_token")
    print("  instance_tokens/track_tokens require explicit conversion to integer gt_inds")
    print("  gt_boxes_st is not verified as a replacement for Navformer past/future arrays")
    print("  stock metadata does not replace nav_filter YAML, PDM cache, or nuPlan map assets")

    compatible = (
        not missing_fields
        and camera_count_ok
        and not camera_missing
        and dataset_type_ok
    )
    print(f"DIRECT_ADAPTER_COMPATIBLE = {'YES' if compatible else 'NO'}")
    if compatible:
        print("result: existing OpenScene metadata contains the checked direct-input fields")
    else:
        print(
            "result: stock OpenScene metadata cannot be passed directly to the configured "
            "NavSimOpenSceneE2EDet without an explicit adapter/merged-info conversion"
        )
        print(f"missing fields: {sorted(set(missing_fields)) if missing_fields else 'NONE'}")
    return compatible


def main() -> int:
    args = parse_args()
    config_path = require_file(args.config, "config")
    checkpoint_path = require_file(args.checkpoint, "checkpoint")
    metadata_path = require_file(args.metadata, "OpenScene metadata")

    config = load_config(config_path)
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = load_pickle(metadata_path)
    source_field_report = ast_info_fields(DATASET_SOURCE_FILES)

    print_config_report(config, config_path)
    print_checkpoint_report(checkpoint, checkpoint_path)
    _, first = print_metadata_report(metadata, metadata_path)
    print_compatibility_report(config, first, source_field_report)
    print("\nINSPECTION_COMPLETED = YES")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"INSPECTION_COMPLETED = NO", file=sys.stderr)
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise
