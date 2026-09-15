from __future__ import annotations

from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _PerCameraDenseDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_channels: int,
        output_size: Tuple[int, int],
        base_channels: int = 128,
    ) -> None:
        super().__init__()
        self.output_size = tuple(output_size)
        self.project = nn.Sequential(
            nn.Linear(input_dim, base_channels),
            nn.LayerNorm(base_channels),
            nn.GELU(),
        )
        self.decode = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels // 2, output_channels, 1),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.ndim != 4:
            raise ValueError(
                f"patch_tokens must be [B, N_cam, N_patch, C], got {tuple(patch_tokens.shape)}"
            )
        batch_size, num_cameras, num_patches, _ = patch_tokens.shape
        side = math.isqrt(num_patches)
        if side * side != num_patches:
            raise ValueError(f"DINO patch count must form a square grid, got {num_patches}")
        features = self.project(patch_tokens)
        features = features.permute(0, 1, 3, 2).reshape(
            batch_size * num_cameras, -1, side, side
        )
        dense = self.decode(features)
        dense = F.interpolate(
            dense, size=self.output_size, mode="bilinear", align_corners=False
        )
        return dense.reshape(batch_size, num_cameras, dense.shape[1], *self.output_size)


class SegHead(_PerCameraDenseDecoder):
    """Decode per-camera semantic logits from DINO patch features."""

    def __init__(
        self,
        input_dim: int,
        C_seg: int = 6,
        seg_size: Tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__(input_dim, C_seg, seg_size)


class DepthHead(_PerCameraDenseDecoder):
    """Decode positive per-camera depth predictions."""

    def __init__(
        self,
        input_dim: int,
        depth_size: Tuple[int, int] = (64, 64),
        min_depth: float = 1e-3,
    ) -> None:
        super().__init__(input_dim, 1, depth_size)
        self.min_depth = min_depth

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return F.softplus(super().forward(patch_tokens)) + self.min_depth


class AgentHead(nn.Module):
    """Predict single-frame object class, 3D box, and velocity."""

    def __init__(self, hidden_dim: int, C_agent: int = 10, D_box: int = 8) -> None:
        super().__init__()
        if D_box < 8:
            raise ValueError("D_box must be at least 8")
        self.D_box = D_box
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, C_agent + 1),
        )
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, D_box)
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3)
        )

    def forward(self, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_logits = self.cls_head(queries)
        raw_boxes = self.box_head(queries)
        velocity = self.velocity_head(queries)
        yaw = F.normalize(raw_boxes[..., 6:8], dim=-1, eps=1e-6)
        boxes = torch.cat([raw_boxes[..., :6].sigmoid(), yaw, raw_boxes[..., 8:]], dim=-1)
        return cls_logits, boxes, velocity


class MapHead(nn.Module):
    """Predict vector-map classes and polyline points."""

    def __init__(self, hidden_dim: int, C_map: int = 4, P: int = 20) -> None:
        super().__init__()
        self.P = P
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, C_map + 1),
        )
        self.point_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, P * 2)
        )

    def forward(self, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls_logits = self.cls_head(queries)
        points = self.point_head(queries).reshape(
            queries.shape[0], queries.shape[1], self.P, 2
        )
        return cls_logits, points.sigmoid()
