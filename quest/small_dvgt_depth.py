"""Scaled native DVGT attention/DPT with the existing shared frozen DINOv2 features."""

import sys
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from .dvgt_depth import DVGT_REVISION, native_working_directory, repository_revision


def validate_small_dvgt(options):
    width, rounds = options["geometry_dim"], options["geometry_layers"]
    if not isinstance(options["heads"], int) or options["heads"] <= 0:
        raise ValueError("Small DVGT attention heads must be a positive integer")
    if (not isinstance(width, int) or width < 32 or width > 384 or width % options["heads"]
            or (width // options["heads"]) % 4 or not isinstance(rounds, int) or rounds < 4
            or len(options["geometry_levels"]) != 4
            or sorted(set(options["geometry_levels"])) != list(options["geometry_levels"])
            or not 0 <= min(options["geometry_levels"]) <= max(options["geometry_levels"]) < rounds
            or options["temporal"]["frames"] != 2 or options["temporal"]["causal"] is not True
            or options["temporal"]["missing_history"] != "mask"
            or options["preserve_spatial_features"] is not True
            or options["image_geometry"] != "raw_opencv" or tuple(options["output_size"]) != (252, 448)
            or options["dpt_features"] < 8 or options["min_depth_m"] <= 0):
        raise ValueError("Invalid scaled DVGT width/rounds/levels/causal input profile")


def native_small_modules(options):
    """Reuse pinned native layers; never construct DINOv3 or load its weights."""
    from .utils import project_root

    repo = Path(options["native_repo"])
    repo = repo if repo.is_absolute() else project_root() / repo
    repository_revision(repo, DVGT_REVISION)
    existing = sys.modules.get("dvgt")
    if existing is not None and repo.resolve() not in Path(existing.__file__).resolve().parents:
        raise RuntimeError("Conflicting dvgt source package")
    sys.path.insert(0, str(repo.resolve()))
    # The native aggregator constructor asks for an image backbone. QUEST already
    # owns DINOv2, so this parameter-free placeholder is removed before any forward.
    with native_working_directory(repo), patch("torch.hub.load", return_value=nn.Identity()):
        from dvgt.models.backbones.dvgt1_aggregator import DVGT1Aggregator
        from dvgt.models.heads.dpt_head import DPTHead

        geometry = DVGT1Aggregator(patch_size=14, embed_dim=options["geometry_dim"],
            depth=options["geometry_layers"], num_heads=options["heads"], mlp_ratio=4,
            dino_v3_weight_path=None)
        del geometry.patch_embed
        decoder = DPTHead(dim_in=3 * options["geometry_dim"], patch_size=14,
            output_dim=2, activation="softplus", conf_activation="softplus1",
            features=options["dpt_features"], out_channels=[128, 256, 384, 384],
            intermediate_layer_idx=list(options["geometry_levels"]))
    nn.init.constant_(decoder.scratch.output_conv2[-1].bias[0], 10.)
    return geometry, decoder


class SmallDVGTDepthFrontend(nn.Module):
    def __init__(self, dim, options):
        super().__init__()
        validate_small_dvgt(options)
        self.options = dict(options)
        self.level_projections = nn.ModuleList([nn.Linear(dim, options["geometry_dim"])])
        self.ray_embedding = nn.Identity()  # No calibrated sampling is claimed in this native attention profile.
        self.geometry_blocks, self.depth_decoder = native_small_modules(options)

    @property
    def temporal_blocks(self):
        return self.geometry_blocks.cross_frame_blocks

    def forward(self, feature_levels, intrinsics, extrinsics, distortion, image_size,
                camera_mask=None, history=None):
        if (history is None or camera_mask is not None or feature_levels[-1].shape[0] != 1
                or history["valid"].shape != (1,) or history["valid"].dtype != torch.bool):
            raise ValueError("Small DVGT currently requires batch=1 and explicit causal history validity")
        current, previous = feature_levels[-1], history["feature_levels"][-1]
        if current.shape != previous.shape or current.shape[1] != 8:
            raise ValueError("Current/past DINOv2 feature shape mismatch")
        has_history = bool(history["valid"][0])
        dt = float(history["dt_s"][0])
        if ((has_history and not 0 < dt <= self.options["temporal"]["max_gap_s"])
                or (not has_history and dt != 0)):
            raise ValueError("Invalid causal history time interval")
        # Native attention has no missing-frame mask: use a REAL T=1 cold start.
        maps = torch.stack((previous, current), 1) if has_history else current[:, None]
        b, t, v, _, h, w = maps.shape
        geometry = self.geometry_blocks
        geometry.train(self.training and self.options["gradient_checkpointing"])
        c, p = self.options["geometry_dim"], h * w
        tokens = self.level_projections[0](maps.flatten(-2).transpose(-1, -2))
        tokens = tokens + geometry.temporal_pos_embed[:, :t, None, None]
        tokens = tokens.reshape(b * t * v, p, c)

        def special(value):
            first = value[:, :1].expand(b, v, *value.shape[2:])
            other = value[:, 1:].expand(b, (t - 1) * v, *value.shape[2:])
            return torch.cat((first, other), 1).reshape(b * t * v, *value.shape[2:])

        tokens = torch.cat((special(geometry.ego_pose_token), special(geometry.register_token), tokens), 1)
        p = tokens.shape[1]
        pos = geometry.position_getter(b * t * v, h, w, device=tokens.device) + 1
        pos = torch.cat((pos.new_zeros(b * t * v, geometry.patch_start_idx, 2), pos), 1)
        levels = []
        for index in range(self.options["geometry_layers"]):
            tokens, _, intra = geometry._process_intra_view_attention(tokens, b, t, v, p, c, index, pos)
            tokens, _, views = geometry._process_cross_view_attention(tokens, b, t, v, p, c, index, pos)
            tokens, _, frames = geometry._process_cross_frame_attention(tokens, b, t, v, p, c, index, pos)
            levels.append(torch.cat((frames[0], views[0], intra[0]), -1))
        # The native DPT reads only B/T/V/H/W from this image argument; RGB has
        # already been encoded by QUEST's one shared frozen DINOv2, not invented.
        shape_only = maps.new_zeros(b, t, v, 3, *image_size)
        with torch.autocast(device_type=maps.device.type, enabled=False):
            depth, _ = self.depth_decoder([value.float() for value in levels], shape_only,
                                         geometry.patch_start_idx, frames_chunk_size=1)
        depth = depth[:, -1].permute(0, 1, 4, 2, 3).float() + self.options["min_depth_m"]
        return {"depth": depth, "history_frame_fraction": depth.new_tensor(float(has_history)),
                "depth_temporal_coverage": depth.new_tensor(float(has_history))}
