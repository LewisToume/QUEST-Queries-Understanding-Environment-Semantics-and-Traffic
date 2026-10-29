"""DVGT-1 native inference adapter; no QUEST model or teacher imported at module load."""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .depth_teacher import validate_intrinsic
from .map_teacher import _validated_transform, resolve_lidar2ego


DVGT_SCHEMA_VERSION = 2
DVGT_ALIGNMENT_VERSION = "dvgt1_first_ego_rdf_to_camera_z_original_pixels_v1"
DVGT_REVISION = "51cf3f6d11fdff8bc7e2bbe1a88f71665ccb2236"
DINOV3_REVISION = "6876159a11b4df116f30f667f8c9888617df0751"
DVGT_CHECKPOINT_SHA256 = "8e1ab60f901cffc3316fa0a09f24ddcf6b90d0221cf7bfea64664c6f83cc1e22"
DVGT_WEIGHT_REVISION = "4918b6ed3547a00fd217d602fabb84ada180dba4"
DVGT_SCALE = 0.1
DVGT_TEACHER_CAMERAS = (
    "CAM_F0", "CAM_B0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2",
)
# Official tools/data_process/dataloaders/base_dataset.py: RDF coordinates -> FLU.
FLU_FROM_RDF = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.],
                         [0., -1., 0., 0.], [0., 0., 0., 1.]])


def build_sequence_plans(infos, target_indices, context_frames=2, *, allow_future=False,
                         timestamp_scale=1e-6, max_gap_s=1.1, allowed_scenes=None):
    """One metadata scan; preserve target indices/tokens and use only real scene context."""
    if (not isinstance(context_frames, int) or not 2 <= context_frames <= 48
            or not math.isfinite(timestamp_scale) or timestamp_scale <= 0
            or not math.isfinite(max_gap_s) or max_gap_s <= 0):
        raise ValueError("Invalid DVGT context length/timestamp units/gap")
    if (not target_indices or len(set(target_indices)) != len(target_indices)
            or any(index < 0 or index >= len(infos) for index in target_indices)):
        raise ValueError("Require unique original metadata target indices")
    scenes = {str(infos[index]["scene_token"]) for index in target_indices}
    if allowed_scenes is not None and not scenes <= set(allowed_scenes):
        raise ValueError("Teacher context would cross the selected manifest split")
    relevant = defaultdict(list)
    for index, info in enumerate(infos):
        scene = str(info["scene_token"])
        if scene in scenes:
            stamp = float(info["timestamp"])
            if not math.isfinite(stamp):
                raise ValueError(f"Nonfinite timestamp in scene {scene}")
            relevant[scene].append(index)
    windows, locations = {}, {}
    for scene, indices in relevant.items():
        indices.sort(key=lambda index: (float(infos[index]["timestamp"]), index))
        stamps = [float(infos[index]["timestamp"]) for index in indices]
        tokens = [str(infos[index]["token"]) for index in indices]
        if len(set(tokens)) != len(tokens) or any(b <= a for a, b in zip(stamps, stamps[1:])):
            raise ValueError(f"Duplicate token or non-increasing scene timestamps: {scene}")
        locations.update({index: position for position, index in enumerate(indices)})
    for index in target_indices:
        scene = str(infos[index]["scene_token"])
        indices, position = relevant[scene], locations[index]
        start = position - context_frames + 1
        if start < 0:
            if not allow_future:
                raise ValueError(f"BLOCKED_TEMPORAL_SEQUENCE index={index}: insufficient past; "
                                 "--allow-future-context explicitly permits offline future context")
            start = 0
        window = tuple(indices[start:start + context_frames])
        if len(window) != context_frames or index not in window:
            raise ValueError(f"BLOCKED_TEMPORAL_SEQUENCE scene={scene}: not enough real frames")
        gaps = [(float(infos[b]["timestamp"]) - float(infos[a]["timestamp"])) * timestamp_scale
                for a, b in zip(window, window[1:])]
        if any(gap <= 0 or gap > max_gap_s for gap in gaps):
            raise ValueError(f"BLOCKED_TEMPORAL_SEQUENCE scene={scene}: gaps_s={gaps}")
        windows.setdefault(window, []).append(index)
    return [{"indices": list(window), "targets": targets,
             "scene_token": str(infos[window[0]]["scene_token"])}
            for window, targets in windows.items()]


def repository_revision(repo, expected):
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True)
    revision = result.stdout.strip()
    if revision != expected:
        raise ValueError(f"Native source revision mismatch: {repo}; expected={expected} actual={revision}")
    dirty = subprocess.run(["git", "-C", str(repo), "diff", "HEAD", "--exit-code"],
                           capture_output=True, text=True, check=False)
    if dirty.returncode:
        raise ValueError(f"Native source has tracked changes: {repo}; refuse unaudited preprocessing/model")
    return revision


@contextmanager
def native_working_directory(repo):
    original = Path.cwd()
    try:
        os.chdir(repo)
        yield
    finally:
        os.chdir(original)


def load_native_file(repo, relative, name):
    path = Path(repo) / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load native helper {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_preprocessors(repo):
    return (load_native_file(repo, "tools/data_process/utils/image.py", "quest_dvgt_image"),
            load_native_file(repo, "dvgt/datasets/transforms/transforms.py", "quest_dvgt_transforms"))


def prepare_image(rgb, intrinsic, distortion, geometry, helpers):
    """Replay official OpenScene annotation + deterministic native eval transforms."""
    image_utils, transforms = helpers
    k = validate_intrinsic(intrinsic).copy()
    original_aspect = rgb.shape[1] / rgb.shape[0]
    image, k, _ = image_utils.process_image_sequentially(
        rgb, k, dist_coeffs=distortion if geometry == "raw_opencv" else None, num_tokens=2500)
    image, k = image_utils.resize_to_aspect_ratio(image, k, original_aspect)
    shape = np.array([int(512 / 1.77) // 16 * 16, 512])
    image, _, k, _ = transforms.crop_image_depth_and_intrinsic_by_pp(image, None, k, image.shape[:2])
    image, _, k, _ = transforms.resize_image_depth_and_intrinsic(
        image, None, k, shape, enable_random_resize=False)
    image, _, k, _ = transforms.crop_image_depth_and_intrinsic_by_pp(
        image, None, k, shape, strict=True)
    if image.shape != (288, 512, 3):
        raise ValueError(f"Unexpected native preprocessing shape: {image.shape}")
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255., k


def load_strict_state(model, payload):
    if not isinstance(payload, Mapping):
        raise ValueError("DVGT checkpoint must contain a state mapping")
    state = payload.get("model", payload)
    if not isinstance(state, Mapping):
        raise ValueError("Invalid DVGT checkpoint['model']")
    expected = model.state_dict()
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    mismatch = [key for key in expected.keys() & state.keys()
                if not isinstance(state[key], torch.Tensor) or expected[key].shape != state[key].shape]
    print(f"checkpoint_missing_keys={missing} unexpected_keys={unexpected} mismatched_shapes={mismatch}")
    if missing or unexpected or mismatch:
        raise ValueError("Incomplete/mismatched DVGT task weights; no random-head label export")
    model.load_state_dict(state, strict=True)
    return {"strict": True, "missing_keys": missing, "unexpected_keys": unexpected,
            "mismatched_shapes": mismatch}


def build_teacher(repo, checkpoint, device, decoder_chunk_frames=1):
    repo = Path(repo).resolve()
    for package in ("dvgt", "dinov3"):
        existing = sys.modules.get(package)
        if existing is not None:
            origin = Path(getattr(existing, "__file__", "") or ".").resolve()
            if repo not in origin.parents:
                raise RuntimeError(f"Conflicting {package} package; use a separate teacher process")
    sys.path.insert(0, str(repo))
    native_hub = torch.hub.load

    def local_backbone(repo_or_dir, model_name, *args, **kwargs):
        if (Path(repo_or_dir).resolve() != repo / "third_party/dinov3"
                or model_name != "dinov3_vitl16" or kwargs.get("source") != "local"):
            raise RuntimeError("Unexpected native hub request; network/model substitution forbidden")
        # Upstream DVGT's None path still defaults to DINO pretrained=True.
        # Construct code only; the full DVGT checkpoint supplies EVERY backbone weight.
        kwargs["pretrained"] = False
        return native_hub(repo_or_dir, model_name, *args, **kwargs)

    with native_working_directory(repo), patch("torch.hub.load", side_effect=local_backbone), patch(
            "torch.hub.load_state_dict_from_url", side_effect=RuntimeError("Automatic downloads forbidden")):
        from dvgt.models.architectures.dvgt1 import DVGT1
        model = DVGT1(dino_v3_weight_path=None, frames_chunk_size=decoder_chunk_frames)
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        audit = load_strict_state(model, payload)
        del payload
    model.requires_grad_(False)
    return model.to(device).eval(), audit


def camera_from_first_ego(first_info, current_info, sensor2lidar):
    first_global = _validated_transform(first_info["ego2global"], "first ego2global").numpy()
    current_global = _validated_transform(current_info["ego2global"], "current ego2global").numpy()
    lidar2ego = resolve_lidar2ego(current_info).numpy()
    camera2lidar = _validated_transform(sensor2lidar, "sensor2lidar").numpy()
    return np.linalg.inv(camera2lidar) @ np.linalg.inv(lidar2ego) @ np.linalg.inv(
        current_global) @ first_global @ FLU_FROM_RDF


def points_to_camera_z(points, transform, gt_scale_factor=DVGT_SCALE):
    if not math.isclose(gt_scale_factor, DVGT_SCALE, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Pinned DVGT-1 checkpoint requires gt_scale_factor=0.1")
    matrix = _validated_transform(transform, "camera_from_first_ego").numpy()
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("Native points must be [H,W,3], not radial depth")
    metric = points / gt_scale_factor
    camera = metric @ matrix[:3, :3].T + matrix[:3, 3]
    return camera[..., 2], camera


def restore_depth(points, confidence, transform, intrinsic, distortion, native_k,
                  source_hw, output_hw, geometry, *, min_depth=.1, max_depth=80.,
                  confidence_min=0., max_reprojection_px=8.):
    """Map raw JPEG pixel rays into native rectified prediction pixels; never fill padding."""
    import cv2

    z, xyz = points_to_camera_z(points, transform)
    confidence = np.asarray(confidence)
    if confidence.shape != z.shape or geometry not in ("raw_opencv", "rectified_pinhole"):
        raise ValueError("Invalid native confidence/image geometry")
    if not (0 < min_depth < max_depth and max_reprojection_px > 0 and confidence_min >= 0):
        raise ValueError("Invalid DVGT mask thresholds")
    native_k = validate_intrinsic(native_k)
    yy, xx = np.indices(z.shape)
    projected = xyz @ native_k.T
    safe_z = np.where(np.isfinite(z) & (z > 0), z, 1.)
    reprojection = np.hypot(projected[..., 0] / safe_z - xx, projected[..., 1] / safe_z - yy)
    finite_positive = np.isfinite(xyz).all(-1) & np.isfinite(confidence) & (z > 0)
    valid = (finite_positive & (z >= min_depth) & (z <= max_depth)
             & (confidence > confidence_min) & (reprojection <= max_reprojection_px))
    out_y, out_x = np.indices(output_hw, dtype=np.float32)
    raw_uv = np.stack(((out_x + .5) * source_hw[1] / output_hw[1] - .5,
                       (out_y + .5) * source_hw[0] / output_hw[0] - .5), -1).reshape(-1, 1, 2)
    k = validate_intrinsic(intrinsic)
    if geometry == "raw_opencv":
        native_uv = cv2.undistortPoints(raw_uv, k, distortion, P=native_k)
    else:
        rays = np.concatenate((raw_uv[:, 0], np.ones((len(raw_uv), 1))), -1) @ np.linalg.inv(k).T
        native_uv = (rays @ native_k.T)[:, :2]
    maps = np.asarray(native_uv).reshape(*output_hw, 2).astype(np.float32)
    inside = (np.isfinite(maps).all(-1) & (maps[..., 0] >= 0) & (maps[..., 0] <= z.shape[1] - 1)
              & (maps[..., 1] >= 0) & (maps[..., 1] <= z.shape[0] - 1))
    maps = np.where(inside[..., None], maps, -1).astype(np.float32)
    restored = cv2.remap(np.where(valid, z, 0).astype(np.float32), maps[..., 0], maps[..., 1], cv2.INTER_LINEAR)
    coverage = cv2.remap(valid.astype(np.float32), maps[..., 0], maps[..., 1], cv2.INTER_LINEAR)
    mask = inside & (coverage >= 1 - 1e-6) & np.isfinite(restored) & (restored >= min_depth) & (restored <= max_depth)
    errors = reprojection[finite_positive]
    stats = {"native_positive_fraction": float(finite_positive.mean()),
             "native_reprojection_median_px": float(np.median(errors)) if len(errors) else None,
             "native_reprojection_p95_px": float(np.percentile(errors, 95)) if len(errors) else None,
             "valid_fraction": float(mask.mean())}
    return (torch.from_numpy(np.where(mask, restored, 0).astype(np.float32))[None],
            torch.from_numpy(mask)[None], stats)


def validate_dvgt_provenance(record):
    source = record.get("provenance", {})
    if (source.get("teacher") != "DVGT-1" or source.get("repo_revision") != DVGT_REVISION
            or source.get("dinov3_revision") != DINOV3_REVISION
            or source.get("checkpoint_sha256") != DVGT_CHECKPOINT_SHA256
            or source.get("gt_scale_factor") != DVGT_SCALE
            or source.get("native_coordinate_frame") != "first_ego_rdf"
            or source.get("pose_source") != "openscene_metadata"
            or source.get("native_saturation_policy") != "unbounded_points_finite_positive_range_reprojection"
            or source.get("teacher_domain") != "trained_and_evaluated_on_openscene"):
        raise ValueError("DVGT checkpoint/domain/coordinate/metric provenance mismatch")
    if (record.get("geometry_verified") is not False
            or record.get("training_label_scope") not in ("image_plane_candidate", "diagnostic_only")):
        raise ValueError("DVGT adaptation is not geometry-approved; keep candidate/diagnostic scope")
    audit = source.get("checkpoint_load", {})
    if (audit.get("strict") is not True or any(audit.get(key) != []
            for key in ("missing_keys", "unexpected_keys", "mismatched_shapes"))):
        raise ValueError("DVGT labels require a strict full-checkpoint load audit")
    sequence = record.get("sequence", {})
    tokens, indices = sequence.get("tokens", []), sequence.get("source_indices", [])
    stamps = sequence.get("timestamps", [])
    position = sequence.get("target_position")
    if (len(tokens) < 2 or len(tokens) != len(indices) or len(set(tokens)) != len(tokens)
            or len(set(indices)) != len(indices) or len(stamps) != len(tokens)
            or not np.isfinite(stamps).all() or any(b <= a for a, b in zip(stamps, stamps[1:]))
            or not isinstance(position, int) or not 0 <= position < len(tokens)
            or tokens[position] != record["token"] or indices[position] != record["source_index"]
            or sequence.get("scene_token") != record["scene_token"]):
        raise ValueError("DVGT sequence/target token/index provenance mismatch")
    uses_future = position < len(tokens) - 1
    if sequence.get("uses_future") is not uses_future or (uses_future and source.get("allow_future_context") is not True):
        raise ValueError("DVGT future context must be explicitly declared and authorized")
