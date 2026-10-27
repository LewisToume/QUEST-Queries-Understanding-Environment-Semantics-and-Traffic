"""Shared trainable image pyramid for joint dense/spatial student development.

This component is not enabled in legacy QUEST V3 or its training/checkpoint path.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


SHARED_IMAGE_FRONTEND_VERSION = "shared_image_pyramid_v1"


def conv_block(channels):
    return nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                         nn.GroupNorm(8, channels), nn.GELU())


class SharedDensePerception(nn.Module):
    """Fuse four DINO semantic depths; expose differentiable features to BEV.

    DINO inputs have a common patch grid. Resampled scales are constructed by
    this neck; they must not be described as native high-resolution DINO stages.
    """

    def __init__(self, hidden_dim=384, channels=128, num_classes=19,
                 output_size=(252, 448), min_depth_m=0.1):
        super().__init__()
        if channels < 8 or channels % 8 or num_classes != 19:
            raise ValueError("Use GroupNorm-compatible channels and native Cityscapes19")
        if (len(output_size) != 2 or any(not isinstance(value, int) or value <= 0 for value in output_size)
                or not math.isfinite(min_depth_m) or min_depth_m <= 0):
            raise ValueError("Invalid dense output dimensions/depth minimum")
        self.hidden_dim, self.channels = hidden_dim, channels
        self.output_size, self.min_depth_m = tuple(output_size), float(min_depth_m)
        self.lateral = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden_dim, channels, 1), nn.GroupNorm(8, channels), nn.GELU())
            for _ in range(4)
        ])
        self.refine = nn.ModuleList([conv_block(channels) for _ in range(4)])
        self.seg_decoder = nn.Sequential(conv_block(channels), nn.Conv2d(channels, 19, 1))
        self.camera_condition = nn.Sequential(nn.Linear(4, channels), nn.GELU(), nn.Linear(channels, channels * 2))
        self.depth_decoder = nn.Sequential(conv_block(channels), nn.Conv2d(channels, 1, 1))
        nn.init.constant_(self.depth_decoder[-1].bias, 10.0)
        self.bev_adapter = nn.Conv2d(channels, hidden_dim, 1)
        self.bev_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, feature_levels, intrinsics, image_size):
        if len(feature_levels) != 4 or feature_levels[-1].ndim != 5:
            raise ValueError("Need four normalized DINO maps [B,N_camera,C,H_patch,W_patch]")
        batch, cameras, hidden, patch_h, patch_w = feature_levels[-1].shape
        if hidden != self.hidden_dim or min(batch, cameras, patch_h, patch_w) <= 0:
            raise ValueError("Invalid DINO feature dimensions")
        if any(level.shape != feature_levels[-1].shape for level in feature_levels):
            raise ValueError("DINO intermediate block grids must agree")
        if (intrinsics.shape != (batch, cameras, 3, 3) or len(image_size) != 2 or min(image_size) <= 0
                or intrinsics.device != feature_levels[-1].device):
            raise ValueError("Scaled intrinsics must match current image and cameras")
        if not bool(torch.isfinite(intrinsics).all()) or not bool((intrinsics[..., (0, 1), (0, 1)] > 0).all()):
            raise ValueError("Intrinsic focal lengths must be finite and positive")
        scales = (4.0, 2.0, 1.0, 0.5)
        pyramid = [None] * 4
        previous = None
        for index in range(3, -1, -1):
            value = self.lateral[index](feature_levels[index].reshape(batch * cameras, hidden, patch_h, patch_w))
            size = (max(1, round(patch_h * scales[index])), max(1, round(patch_w * scales[index])))
            value = F.interpolate(value, size, mode="bilinear", align_corners=False)
            if previous is not None:
                value = value + F.interpolate(previous, size, mode="bilinear", align_corners=False)
            previous = self.refine[index](value)
            pyramid[index] = previous
        dense = pyramid[0]
        seg = F.interpolate(self.seg_decoder(dense), self.output_size, mode="bilinear", align_corners=False)
        height, width = image_size
        camera = torch.stack((
            (intrinsics[..., 0, 0] / width).log(), (intrinsics[..., 1, 1] / height).log(),
            intrinsics[..., 0, 2] / width, intrinsics[..., 1, 2] / height,
        ), -1).reshape(batch * cameras, 4).to(dtype=dense.dtype)
        scale, shift = self.camera_condition(camera).chunk(2, -1)
        conditioned = dense * (1 + 0.1 * scale.tanh()[..., None, None]) + shift[..., None, None]
        depth = F.interpolate(F.softplus(self.depth_decoder(conditioned)) + self.min_depth_m,
                              self.output_size, mode="bilinear", align_corners=False)
        residual = self.bev_adapter(F.interpolate(dense, (patch_h, patch_w), mode="bilinear", align_corners=False))
        # Small but nonzero gate permits shared-neck gradients from spatial tasks.
        spatial = feature_levels[-1] + (0.1 * self.bev_gate.tanh() * residual).reshape(
            batch, cameras, hidden, patch_h, patch_w)
        return {
            "seg_logits": seg.reshape(batch, cameras, 19, *self.output_size),
            "depth": depth.reshape(batch, cameras, 1, *self.output_size),
            "image_feature_maps": spatial,
            "image_pyramid": tuple(value.reshape(batch, cameras, self.channels, *value.shape[-2:]) for value in pyramid),
        }
