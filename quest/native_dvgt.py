"""Full pretrained DVGT geometry inside QUEST; not a distilled/lightweight replica."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .dvgt_depth import (DVGT_SCALE, DVGT_TEACHER_CAMERAS, DVGT_REVISION, DINOV3_REVISION,
                        DVGT_CHECKPOINT_SHA256, native_working_directory, repository_revision)
from .openscene_dataset import OPENSCENE_CAMERA_NAMES


NATIVE_QUEST_VERSION = "quest_dvgt_native_v4"
NATIVE_TO_QUEST = tuple(DVGT_TEACHER_CAMERAS.index(name) for name in OPENSCENE_CAMERA_NAMES)


def construct_native_core(repo):
    """Code-only reconstruction for saved QUEST weights, without the teacher checkpoint.

    Keep this outside the existing teacher adapter: changing that adapter's hash
    would invalidate provenance of previously exported/resumable teacher labels.
    """
    repo = Path(repo).resolve()
    repository_revision(repo, DVGT_REVISION)
    repository_revision(repo / "third_party/dinov3", DINOV3_REVISION)
    for package in ("dvgt", "dinov3"):
        existing = sys.modules.get(package)
        if existing is not None and repo not in Path(getattr(existing, "__file__", "") or ".").resolve().parents:
            raise RuntimeError("Conflicting native package: " + package)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "third_party/dinov3"))

    def code_only(repo_or_dir, model_name, *args, **kwargs):
        if (Path(repo_or_dir).resolve() != repo / "third_party/dinov3"
                or model_name != "dinov3_vitl16" or kwargs.pop("source", None) != "local"):
            raise RuntimeError("Unexpected native backbone request")
        from dinov3.hub.backbones import dinov3_vitl16

        kwargs["pretrained"] = False
        return dinov3_vitl16(*args, **kwargs)

    with native_working_directory(repo), patch("torch.hub.load", side_effect=code_only), patch(
            "torch.hub.load_state_dict_from_url", side_effect=RuntimeError("Automatic downloads forbidden")):
        from dvgt.models.architectures.dvgt1 import DVGT1

        return DVGT1(dino_v3_weight_path=None, frames_chunk_size=1)


def audit_native_architecture(core):
    """Reject a smaller substitute before labeling this a native DVGT transplant."""
    aggregator, head = core.aggregator, core.point_head
    families = ("intra_view_blocks", "cross_view_blocks", "cross_frame_blocks")
    counts = {name: len(getattr(aggregator, name)) for name in families}
    if (aggregator.embed_dim != 1024 or aggregator.patch_size != 16 or aggregator.depth != 24
            or any(value != 24 for value in counts.values())
            or tuple(aggregator.aa_order) != ("intra_view", "cross_view", "cross_frame")
            or tuple(head.intermediate_layer_idx) != (4, 11, 17, 23)
            or core.ego_pose_head is None):
        raise ValueError("Require the full pinned DVGT-1 architecture, not a reduced replacement")
    return {"backbone": "native_DINOv3_ViT_L16", "width": 1024,
            "geometry_rounds": 24, "attention_blocks": counts,
            "dpt_geometry_levels_zero_based": [4, 11, 17, 23],
            "outputs": ["XYZ_in_first_ego_RDF", "uncalibrated_confidence", "ego_pose"],
            "parameters": sum(p.numel() for p in core.parameters())}


def raw_to_native_grid(intrinsic, distortion, native_intrinsic, source_hw, output_hw=(252, 448)):
    """CPU calibration lookup only; native point values are sampled in torch."""
    import cv2

    native_intrinsic = np.asarray(native_intrinsic, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if (intrinsic.shape != (3, 3) or native_intrinsic.shape != (3, 3)
            or not np.isfinite(intrinsic).all() or not np.isfinite(native_intrinsic).all()
            or min(*source_hw, *output_hw) <= 0):
        raise ValueError("Invalid camera lookup calibration or image dimensions")
    yy, xx = np.indices(output_hw, dtype=np.float32)
    uv = np.stack(((xx + .5) * source_hw[1] / output_hw[1] - .5,
                   (yy + .5) * source_hw[0] / output_hw[0] - .5), -1)
    uv = cv2.undistortPoints(uv.reshape(-1, 1, 2), intrinsic,
                           np.asarray(distortion, dtype=np.float64), P=native_intrinsic)
    return torch.from_numpy(uv.reshape(*output_hw, 2).astype(np.float32))


def native_camera_depth(points, confidence, camera_from_first_ego, native_intrinsics, raw_native_uv,
                        *, min_depth=.1, max_depth=80., max_reprojection_px=8.):
    """Differentiable metric-Z restoration, with explicit invalid pixels (not filled GT).

    Inputs are current-frame points/confidence in native teacher camera order;
    calibration and pixel lookup are in canonical QUEST order. No radial norm,
    per-frame scale fitting or output clipping is substituted for camera Z.
    """
    if (points.ndim != 5 or points.shape[1] != 8 or points.shape[-1] != 3
            or confidence.shape != points.shape[:-1]):
        raise ValueError("Expected native current points [B,8,H,W,3] and matching confidence")
    b, n, h, w = confidence.shape
    if (camera_from_first_ego.shape != (b, n, 4, 4) or native_intrinsics.shape != (b, n, 3, 3)
            or raw_native_uv.ndim != 5 or raw_native_uv.shape[:2] != (b, n)
            or raw_native_uv.shape[-1] != 2 or not 0 < min_depth < max_depth
            or max_reprojection_px <= 0):
        raise ValueError("Native restoration geometry/range mismatch")
    if not all(bool(torch.isfinite(x).all()) for x in (points, confidence, camera_from_first_ego,
                                                      native_intrinsics, raw_native_uv)):
        raise ValueError("Nonfinite native predictions or calibration")
    xyz = points[:, NATIVE_TO_QUEST].float() / DVGT_SCALE
    transform = camera_from_first_ego.float()
    xyz = torch.einsum("bnij,bnhwj->bnhwi", transform[..., :3, :3], xyz) + transform[..., None, None, :3, 3]
    z = xyz[..., 2]
    uvz = torch.einsum("bnij,bnhwj->bnhwi", native_intrinsics.float(), xyz)
    yy, xx = torch.meshgrid(torch.arange(h, device=z.device), torch.arange(w, device=z.device), indexing="ij")
    expected_uv = torch.stack((xx, yy), -1)
    reprojection = (uvz[..., :2] / z.clamp_min(1e-6)[..., None] - expected_uv).norm(dim=-1)
    native_valid = ((z >= min_depth) & (z <= max_depth)
                    & (confidence[:, NATIVE_TO_QUEST] > 0) & (reprojection <= max_reprojection_px))
    uv = raw_native_uv.float()
    inside = ((uv[..., 0] >= 0) & (uv[..., 0] <= w - 1)
              & (uv[..., 1] >= 0) & (uv[..., 1] <= h - 1))
    grid = 2 * (uv + .5) / uv.new_tensor([w, h]) - 1
    oh, ow = grid.shape[-3:-1]
    grid = grid.reshape(b * n, oh, ow, 2)
    restored = F.grid_sample(torch.where(native_valid, z, 0).reshape(b * n, 1, h, w),
                             grid, align_corners=False).reshape(b, n, 1, oh, ow)
    coverage = F.grid_sample(native_valid.float().reshape(b * n, 1, h, w),
                            grid, align_corners=False).reshape(b, n, 1, oh, ow)
    valid = (inside.unsqueeze(2) & (coverage >= 1 - 1e-6)
             & (restored >= min_depth) & (restored <= max_depth))
    return {"depth": torch.where(valid, restored, 0), "depth_valid_mask": valid}


class QUESTNativeDVGT(nn.Module):
    """One composite QUEST module, with an explicit transitional dual-encoder design.

    The old learned Depth branch is removed. Agent/Map keep their original
    current-frame DINOv2 route; full DVGT is the new geometry/depth route.
    The default is pretrained inference. Fine-tuning needs a separate audited
    objective and memory check, not the old 25M-parameter training command.
    """
    def __init__(self, spatial, geometry_core):
        super().__init__()
        if (getattr(spatial, "dense_frontend_config", None) or {}).get("version") != "geometry_dpt_depth_v2":
            raise ValueError("Use the preserved single-frame V2 source, not the regressed temporal Agent")
        if tuple(spatial.camera_names) != OPENSCENE_CAMERA_NAMES:
            raise ValueError("QUEST spatial camera order mismatch")
        self.native_architecture = audit_native_architecture(geometry_core)
        self.spatial_feature_layers = tuple(spatial.dense_frontend_config["layer_indices"])
        spatial.dense_frontend = None
        spatial.dense_frontend_config = None
        spatial.seg_head = spatial.depth_head = None
        self.spatial, self.geometry_core = spatial, geometry_core
        self.requires_grad_(False)
        self.eval()

    def parameter_summary(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "native_geometry": sum(p.numel() for p in self.geometry_core.parameters()),
                "preserved_agent_map": sum(p.numel() for p in self.spatial.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}

    def forward_spatial(self, images, intrinsics, extrinsics, ego_state):
        # Explicit FP32: the existing Agent/Map reference used this route.
        with torch.autocast(device_type=images.device.type, enabled=False):
            features = self.spatial.extract_camera_features(images.float(), layer_indices=self.spatial_feature_layers)
            encoded = self.spatial.encode_camera_features(features, intrinsics.float(), extrinsics.float(),
                ego_state.float(), tuple(images.shape[-2:]), use_ego_state=False)
            output = self.spatial.forward_agent(encoded["bev_tokens"], encoded["bev_features"])
            output.update(self.spatial.forward_map(encoded, intrinsics.float(), extrinsics.float(),
                ego_state.float(), tuple(images.shape[-2:]), include_dense=False))
        return output

    def forward_geometry(self, native_images, restoration):
        if (native_images.ndim != 6 or native_images.shape[0] != 1
                or native_images.shape[1] not in (1, 2) or native_images.shape[2:] != (8, 3, 288, 512)
                or not bool(torch.isfinite(native_images).all())
                or bool(((native_images < 0) | (native_images > 1)).any())):
            raise ValueError("Require raw RGB [1,T,8,3,288,512], T=1 cold start or T=2 real past/current")
        with torch.autocast(device_type=native_images.device.type, dtype=torch.bfloat16,
                            enabled=native_images.device.type == "cuda"):
            native = self.geometry_core(native_images)
        for key in ("points", "points_conf", "absolute_ego_pose_enc"):
            if key not in native or not bool(torch.isfinite(native[key]).all()):
                raise ValueError("Missing/nonfinite native output: " + key)
        with torch.autocast(device_type=native_images.device.type, enabled=False):
            depth = native_camera_depth(native["points"][:, -1], native["points_conf"][:, -1], **restoration)
        return {**depth, "native_points": native["points"], "native_points_conf": native["points_conf"],
                "native_ego_pose_enc": native["absolute_ego_pose_enc"]}

    def forward(self, *, images, intrinsics, extrinsics, ego_state, native_images, restoration):
        return {**self.forward_spatial(images, intrinsics, extrinsics, ego_state),
                **self.forward_geometry(native_images, restoration)}


def publish_native_checkpoint(path, model, provenance):
    """One self-contained weight file, published without overwriting any experiment."""
    if provenance.get("native_strict_load") is not True or provenance.get("runtime_checks_passed") is not True:
        raise ValueError("Native initialization must pass strict load and runtime checks before saving")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError("Refusing to overwrite native QUEST checkpoint or temporary file")
    created = False
    try:
        with temporary.open("xb") as stream:
            created = True
            torch.save({"profile": NATIVE_QUEST_VERSION, "initialization_only": True,
                        "training_started": False, "shared_backbone": False,
                        "available_tasks": ["agent", "map", "depth", "native_geometry"],
                        "geometry_approval": "UNVERIFIED", "native_architecture": model.native_architecture,
                        "provenance": provenance, "model_state_dict": model.state_dict()}, stream)
        os.link(temporary, path)
    finally:
        if created:
            temporary.unlink()


def load_native_checkpoint_state(model, checkpoint):
    """Reconstruct the two component classes first, then strictly restore all weights."""
    if (checkpoint.get("profile") != NATIVE_QUEST_VERSION
            or checkpoint.get("native_architecture") != model.native_architecture
            or checkpoint.get("shared_backbone") is not False
            or checkpoint.get("geometry_approval") != "UNVERIFIED"):
        raise ValueError("Native QUEST checkpoint/profile mismatch")
    return model.load_state_dict(checkpoint["model_state_dict"], strict=True)


def load_native_quest(path, repo, device="cpu"):
    """Load the one-file model; no original teacher/spatial checkpoint is needed."""
    from .model import QUESTModel

    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    provenance = checkpoint.get("provenance", {})
    if (checkpoint.get("profile") != NATIVE_QUEST_VERSION
            or provenance.get("repo_revision") != DVGT_REVISION
            or provenance.get("dinov3_revision") != DINOV3_REVISION
            or provenance.get("native_checkpoint_sha256") != DVGT_CHECKPOINT_SHA256):
        raise ValueError("Native QUEST source/weight provenance mismatch")
    # Local DINOv2 code/config/assets remain required to construct the spatial route.
    # Its final weights, including every task head, are restored from this one file.
    from .utils import project_root

    spatial_config = provenance["spatial_model_config"]
    local_backbone = Path(spatial_config["local_backbone_dir"])
    if not local_backbone.is_absolute():
        local_backbone = project_root() / local_backbone
    if not (local_backbone / "config.json").is_file():
        raise FileNotFoundError("Local spatial construction assets missing; no network fallback")
    spatial = QUESTModel(**spatial_config)
    model = QUESTNativeDVGT(spatial, construct_native_core(repo))
    load_native_checkpoint_state(model, checkpoint)
    return model.to(device).eval()
