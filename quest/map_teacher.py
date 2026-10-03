from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


TEACHER_MAP_SCHEMA_VERSION = 3
TEACHER_COORDINATE_FRAME = "openscene_ego_xy"
TEACHER_ALIGNMENT_VERSION = "quest_lidar_to_navformer_ego_v1"
TEACHER_SCORE_KIND = "panseg_mask_score_clamped_0_1"
TEACHER_RAW_SCORE_SEMANTICS = "Pansegformer get_bboxes lane_score and score_list[-1] mask scores"
TEACHER_SCORE_TRANSFORM = "clamp(0.0, 1.0); no sigmoid or threshold"


def _validated_transform(value: Any, name: str) -> torch.Tensor:
    try:
        matrix = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, RuntimeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite 4x4 transform") from error
    if matrix.shape != (4, 4) or not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not torch.allclose(matrix[3], matrix.new_tensor([0, 0, 0, 1]), atol=1e-5, rtol=0):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if (not torch.allclose(rotation.T @ rotation, torch.eye(3, dtype=matrix.dtype,
                                                          device=matrix.device), atol=1e-3, rtol=0)
            or not bool(torch.det(rotation) > 0)):
        raise ValueError(f"{name} must contain a proper rigid rotation")
    return matrix


def resolve_lidar2ego(info: Mapping[str, Any]) -> torch.Tensor:
    explicit = (_validated_transform(info["lidar2ego"], "lidar2ego")
                if info.get("lidar2ego") is not None else None)
    derived = None
    if info.get("lidar2global") is not None and info.get("ego2global") is not None:
        lidar2global = _validated_transform(info["lidar2global"], "lidar2global")
        ego2global = _validated_transform(info["ego2global"], "ego2global")
        try:
            derived = torch.linalg.inv(ego2global) @ lidar2global
        except RuntimeError as error:
            raise ValueError("ego2global is not invertible") from error
        derived = _validated_transform(derived, "derived lidar2ego")
    if explicit is not None and derived is not None and not torch.allclose(
        explicit, derived, atol=1e-4, rtol=1e-4
    ):
        raise ValueError("explicit lidar2ego disagrees with ego2global^-1 @ lidar2global")
    if explicit is None and derived is None:
        raise ValueError("metadata needs lidar2ego or both lidar2global and ego2global")
    return explicit if explicit is not None else derived


def align_teacher_map_to_quest_bev(
    soft_scores: torch.Tensor,
    teacher_xy_range: Sequence[float],
    quest_xy_range: Sequence[float],
    quest_h: int,
    quest_w: int,
    row_axis: str,
    row_direction: int,
    col_direction: int,
    *,
    lidar2ego: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ego-frame teacher scores at LiDAR BEV cell centers and return validity."""
    if soft_scores.ndim != 3 or not torch.isfinite(soft_scores).all():
        raise ValueError("teacher soft scores must be finite [K,H,W]")
    if not bool(((soft_scores >= 0) & (soft_scores <= 1)).all()):
        raise ValueError("teacher soft scores must remain in [0,1]")
    if len(teacher_xy_range) != 4 or len(quest_xy_range) != 4:
        raise ValueError("metric XY ranges must be (xmin,ymin,xmax,ymax)")
    if row_axis not in ("x", "y") or row_direction not in (-1, 1) or col_direction not in (-1, 1):
        raise ValueError("teacher row axis and row/column directions must be audited")
    tx0, ty0, tx1, ty1 = (float(v) for v in teacher_xy_range)
    qx0, qy0, qx1, qy1 = (float(v) for v in quest_xy_range)
    if not (tx0 < tx1 and ty0 < ty1 and qx0 < qx1 and qy0 < qy1):
        raise ValueError("invalid metric range")
    transform = _validated_transform(lidar2ego, "lidar2ego").to(
        device=soft_scores.device, dtype=soft_scores.dtype
    )
    ys = qy0 + (torch.arange(quest_h, device=soft_scores.device, dtype=soft_scores.dtype) + 0.5) * (qy1 - qy0) / quest_h
    xs = qx0 + (torch.arange(quest_w, device=soft_scores.device, dtype=soft_scores.dtype) + 0.5) * (qx1 - qx0) / quest_w
    try:
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:
        yy, xx = torch.meshgrid(ys, xs)
    lidar_points = torch.stack((xx, yy, torch.zeros_like(xx), torch.ones_like(xx)), dim=-1)
    ego_points = lidar_points @ transform.T
    ego_x, ego_y = ego_points[..., 0], ego_points[..., 1]
    valid = ((ego_x >= tx0) & (ego_x <= tx1) &
             (ego_y >= ty0) & (ego_y <= ty1))
    row_value = ego_y if row_axis == "y" else ego_x
    col_value = ego_x if row_axis == "y" else ego_y
    row_min, row_max = (ty0, ty1) if row_axis == "y" else (tx0, tx1)
    col_min, col_max = (tx0, tx1) if row_axis == "y" else (ty0, ty1)
    row = (row_value - row_min) / (row_max - row_min)
    col = (col_value - col_min) / (col_max - col_min)
    if row_direction < 0:
        row = 1 - row
    if col_direction < 0:
        col = 1 - col
    grid = torch.stack((2 * col - 1, 2 * row - 1), dim=-1).unsqueeze(0)
    aligned = F.grid_sample(
        soft_scores.unsqueeze(0), grid, mode="bilinear",
        padding_mode="zeros", align_corners=False,
    ).squeeze(0)
    return aligned, valid


class MapRasterDistillHead(nn.Module):
    """Training-only dense branch; not a vector-map inference output."""

    def __init__(self, hidden_dim: int, channels: int, intermediate_dim: int = 128) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("at least one teacher channel is required")
        self.network = nn.Sequential(
            nn.Conv2d(hidden_dim, intermediate_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(intermediate_dim, channels, 1),
        )

    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        return self.network(bev_features)


def soft_map_distillation_loss(
    student_logits: torch.Tensor,
    teacher_soft_scores: torch.Tensor,
    valid_spatial_mask: torch.Tensor,
    support_mask: torch.Tensor,
    channel_weights: torch.Tensor,
) -> torch.Tensor:
    if student_logits.shape != teacher_soft_scores.shape or student_logits.ndim != 4:
        raise ValueError("student and teacher maps must be [B,K,H,W] with equal shapes")
    count = student_logits.shape[1]
    if support_mask.shape != (count,) or channel_weights.shape != (count,):
        raise ValueError("teacher support/weights must have one value per channel")
    if valid_spatial_mask.shape != (student_logits.shape[0], *student_logits.shape[2:]):
        raise ValueError("teacher valid spatial mask must be [B,H,W]")
    if valid_spatial_mask.dtype != torch.bool or not bool(valid_spatial_mask.any()):
        raise ValueError("teacher alignment has no valid spatial cells")
    if not bool(support_mask.any()):
        raise ValueError("no audited teacher channels are supported")
    if not bool(torch.isfinite(teacher_soft_scores).all()) or not bool(
        ((teacher_soft_scores >= 0) & (teacher_soft_scores <= 1)).all()
    ):
        raise ValueError("teacher map must contain finite soft scores in [0,1]")
    if not bool(torch.isfinite(channel_weights).all()) or bool((channel_weights < 0).any()):
        raise ValueError("channel weights must be finite and nonnegative")
    spatial_weight = valid_spatial_mask[:, None].to(student_logits.dtype)
    per_channel = (F.binary_cross_entropy_with_logits(
        student_logits, teacher_soft_scores, reduction="none"
    ) * spatial_weight).sum(dim=(0, 2, 3)) / spatial_weight.sum()
    weights = channel_weights * support_mask.to(channel_weights.dtype)
    if not bool(weights.sum() > 0):
        raise ValueError("supported teacher channel weights sum to zero")
    return (per_channel * weights).sum() / weights.sum()


def validate_teacher_record(record: Mapping[str, Any], token: str, sample_index: int) -> torch.Tensor:
    if record.get("schema_version") != TEACHER_MAP_SCHEMA_VERSION:
        raise ValueError("Navformer map teacher schema mismatch; regenerate old LiDAR-frame labels")
    if str(record.get("token")) != token or record.get("sample_index") != sample_index:
        raise ValueError(f"Navformer map teacher index/token mismatch for {token}")
    soft = record.get("teacher_map_soft")
    if not torch.is_tensor(soft) or soft.ndim != 3:
        raise ValueError("teacher_map_soft must be [K,H,W]")
    if tuple(record.get("teacher_map_shape", ())) != tuple(soft.shape):
        raise ValueError("teacher_map_shape does not match stored tensor")
    if len(record.get("teacher_channel_names_or_ids", ())) != soft.shape[0]:
        raise ValueError("teacher channel metadata does not match tensor")
    if len(record.get("teacher_pc_range", ())) != 4:
        raise ValueError("teacher metric pc_range is required")
    if record.get("teacher_coordinate_frame") != TEACHER_COORDINATE_FRAME:
        raise ValueError("Navformer map teacher must be in OpenScene EGO XY; regenerate old labels")
    if record.get("teacher_alignment_version") != TEACHER_ALIGNMENT_VERSION:
        raise ValueError("Navformer map teacher alignment version mismatch")
    _validated_transform(record.get("teacher_lidar2ego"), "teacher_lidar2ego")
    if record.get("teacher_score_kind") != TEACHER_SCORE_KIND:
        raise ValueError("Navformer teacher map score kind is not the audited Pansegformer mask score")
    if record.get("teacher_raw_score_semantics") != TEACHER_RAW_SCORE_SEMANTICS:
        raise ValueError("Navformer teacher raw score semantics mismatch")
    if record.get("teacher_score_transform") != TEACHER_SCORE_TRANSFORM:
        raise ValueError("Navformer teacher score transform mismatch")
    if record.get("temporal_mode") != "scene_start_to_target":
        raise ValueError("Navformer map teacher temporal initialization is not reproducible")
    if not record.get("temporal_history_sha256") or not record.get("temporal_scene_start_token"):
        raise ValueError("Navformer teacher temporal provenance is missing")
    if (not record.get("teacher_config") or not record.get("teacher_config_sha256")
            or not record.get("teacher_checkpoint")
            or record.get("teacher_checkpoint_size_bytes") is None
            or record.get("teacher_checkpoint_mtime_ns") is None):
        raise ValueError("Navformer teacher config/checkpoint provenance is missing")
    if not bool(torch.isfinite(soft).all()) or not bool(((soft >= 0) & (soft <= 1)).all()):
        raise ValueError("teacher map soft scores must be finite and bounded [0,1]")
    return soft.float()
