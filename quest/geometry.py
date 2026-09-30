from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryAwareBEVLift(nn.Module):
    """Lift calibrated multi-camera feature maps onto a metric BEV grid."""

    def __init__(
        self,
        hidden_dim: int = 384,
        bev_h: int = 32,
        bev_w: int = 32,
        x_range: tuple[float, float] = (-50.0, 50.0),
        y_range: tuple[float, float] = (-50.0, 50.0),
        z_anchors: Sequence[float] = (-1.0, 0.0, 1.0),
    ) -> None:
        super().__init__()
        if hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by 4 for metric 2D encoding")
        if bev_h <= 0 or bev_w <= 0:
            raise ValueError("BEV dimensions must be positive")
        if x_range[1] <= x_range[0] or y_range[1] <= y_range[0]:
            raise ValueError("BEV metric ranges must have positive extent")
        if not z_anchors:
            raise ValueError("at least one z anchor is required")

        self.hidden_dim = hidden_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.x_range = tuple(float(value) for value in x_range)
        self.y_range = tuple(float(value) for value in y_range)
        self.bev_embedding = nn.Parameter(
            torch.randn(bev_h * bev_w, hidden_dim) * 0.02
        )
        self.weight_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.ego_mlp = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        reference_points, metric_xy = self._make_reference_points(z_anchors)
        self.register_buffer("reference_points", reference_points, persistent=True)
        self.register_buffer(
            "metric_position", self._metric_position_encoding(metric_xy), persistent=True
        )

    def _make_reference_points(
        self, z_anchors: Sequence[float]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_step = (self.x_range[1] - self.x_range[0]) / self.bev_w
        y_step = (self.y_range[1] - self.y_range[0]) / self.bev_h
        xs = torch.linspace(
            self.x_range[0] + x_step / 2,
            self.x_range[1] - x_step / 2,
            self.bev_w,
        )
        ys = torch.linspace(
            self.y_range[0] + y_step / 2,
            self.y_range[1] - y_step / 2,
            self.bev_h,
        )
        try:
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        except TypeError:  # PyTorch 1.9
            grid_y, grid_x = torch.meshgrid(ys, xs)
        metric_xy = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
        z = torch.as_tensor(tuple(z_anchors), dtype=torch.float32)
        xy = metric_xy[:, None, :].expand(-1, z.numel(), -1)
        points = torch.cat(
            (xy, z.reshape(1, -1, 1).expand(xy.shape[0], -1, -1)), dim=-1
        )
        return points, metric_xy

    def _metric_position_encoding(self, metric_xy: torch.Tensor) -> torch.Tensor:
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(self.hidden_dim // 4, dtype=torch.float32)
            / max(self.hidden_dim // 4 - 1, 1)
        )
        x_phase = metric_xy[:, 0:1] * frequencies[None, :]
        y_phase = metric_xy[:, 1:2] * frequencies[None, :]
        return torch.cat(
            (x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()), dim=-1
        )

    def project_reference_points(
        self,
        intrinsics: torch.Tensor,
        sensor2lidar: torch.Tensor,
        image_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project LiDAR-frame BEV anchors into cameras for grid_sample."""

        if intrinsics.ndim != 4 or intrinsics.shape[-2:] != (3, 3):
            raise ValueError(f"intrinsics must be [B,N,3,3], got {tuple(intrinsics.shape)}")
        if sensor2lidar.ndim != 4 or sensor2lidar.shape[-2:] != (4, 4):
            raise ValueError(
                f"extrinsics must be sensor2lidar [B,N,4,4], got {tuple(sensor2lidar.shape)}"
            )
        if intrinsics.shape[:2] != sensor2lidar.shape[:2]:
            raise ValueError("intrinsics/extrinsics camera dimensions do not match")
        image_h, image_w = image_size
        if image_h <= 0 or image_w <= 0:
            raise ValueError("image dimensions must be positive")

        dtype = intrinsics.dtype
        device = intrinsics.device
        points = self.reference_points.to(device=device, dtype=dtype)
        homogeneous = torch.cat(
            (points, torch.ones_like(points[..., :1])), dim=-1
        )
        lidar2camera = torch.linalg.inv(sensor2lidar)
        camera_points = torch.einsum("bnij,qzj->bnqzi", lidar2camera, homogeneous)
        xyz = camera_points[..., :3]
        depth = xyz[..., 2]
        projected = torch.einsum("bnij,bnqzj->bnqzi", intrinsics, xyz)
        safe_depth = depth.clamp_min(torch.finfo(dtype).eps)
        pixel_x = projected[..., 0] / safe_depth
        pixel_y = projected[..., 1] / safe_depth
        visible = (
            (depth > 0)
            & (pixel_x >= 0)
            & (pixel_x < image_w)
            & (pixel_y >= 0)
            & (pixel_y < image_h)
            & torch.isfinite(pixel_x)
            & torch.isfinite(pixel_y)
        )
        grid_x = 2.0 * (pixel_x + 0.5) / image_w - 1.0
        grid_y = 2.0 * (pixel_y + 0.5) / image_h - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)
        return grid, visible, depth

    def forward(
        self,
        camera_features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        ego_state: torch.Tensor,
        image_size: tuple[int, int],
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if camera_features.ndim != 5:
            raise ValueError(
                "camera_features must be [B,N,C,H,W], got "
                f"{tuple(camera_features.shape)}"
            )
        batch_size, num_cameras, channels, feature_h, feature_w = camera_features.shape
        if channels != self.hidden_dim:
            raise ValueError(f"expected feature dim {self.hidden_dim}, got {channels}")
        if intrinsics.shape[:2] != (batch_size, num_cameras):
            raise ValueError("intrinsics do not match camera features")
        if extrinsics.shape[:2] != (batch_size, num_cameras):
            raise ValueError("extrinsics do not match camera features")
        if ego_state.shape != (batch_size, 9):
            raise ValueError(f"ego_state must be [B,9], got {tuple(ego_state.shape)}")

        dtype = camera_features.dtype
        projection_dtype = (
            torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        )
        intrinsics = intrinsics.to(
            device=camera_features.device, dtype=projection_dtype
        )
        extrinsics = extrinsics.to(
            device=camera_features.device, dtype=projection_dtype
        )
        grid, visible, _ = self.project_reference_points(
            intrinsics, extrinsics, image_size
        )
        sampling_features = (
            camera_features.float()
            if dtype in (torch.float16, torch.bfloat16)
            else camera_features
        )
        grid = grid.to(dtype=sampling_features.dtype)
        query_count, anchor_count = self.reference_points.shape[:2]
        sampled = F.grid_sample(
            sampling_features.reshape(
                batch_size * num_cameras, channels, feature_h, feature_w
            ),
            grid.reshape(batch_size * num_cameras, query_count * anchor_count, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.to(dtype=dtype).squeeze(-1).reshape(
            batch_size, num_cameras, channels, query_count, anchor_count
        )
        candidates = sampled.permute(0, 3, 1, 4, 2).contiguous()
        candidate_mask = visible.permute(0, 2, 1, 3).contiguous()
        logits = self.weight_mlp(candidates).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask, -1e4)
        weights = torch.softmax(logits.flatten(2), dim=-1).reshape_as(logits)
        weights = weights * candidate_mask.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        lifted = (candidates * weights.unsqueeze(-1)).sum(dim=(2, 3))

        ego = self.ego_mlp(ego_state.to(device=lifted.device, dtype=lifted.dtype))
        output = (
            lifted
            + self.bev_embedding.to(dtype=lifted.dtype).unsqueeze(0)
            + self.metric_position.to(dtype=lifted.dtype).unsqueeze(0)
            + ego.unsqueeze(1)
        )
        if not return_diagnostics:
            return output
        diagnostics = {
            "bev_visible_ratio": candidate_mask.flatten(2).any(dim=2).float().mean(dim=1),
            "camera_visible_ratio": visible.any(dim=-1).float().mean(dim=-1),
        }
        return output, diagnostics
