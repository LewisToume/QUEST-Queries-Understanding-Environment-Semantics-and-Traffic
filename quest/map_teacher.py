from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


TEACHER_MAP_SCHEMA_VERSION = 2
TEACHER_SCORE_KIND = "panseg_mask_score_clamped_0_1"
TEACHER_RAW_SCORE_SEMANTICS = "Pansegformer get_bboxes lane_score and score_list[-1] mask scores"
TEACHER_SCORE_TRANSFORM = "clamp(0.0, 1.0); no sigmoid or threshold"


def align_teacher_map_to_quest_bev(
    soft_scores: torch.Tensor,
    teacher_xy_range: Sequence[float],
    quest_xy_range: Sequence[float],
    quest_h: int,
    quest_w: int,
    row_axis: str,
    row_direction: int,
    col_direction: int,
) -> torch.Tensor:
    """Metric grid_sample; output rows grow with +Y and columns with +X."""
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
    if qx0 < tx0 or qx1 > tx1 or qy0 < ty0 or qy1 > ty1:
        raise ValueError("teacher metric range does not cover the QUEST ROI")
    ys = qy0 + (torch.arange(quest_h, device=soft_scores.device, dtype=soft_scores.dtype) + 0.5) * (qy1 - qy0) / quest_h
    xs = qx0 + (torch.arange(quest_w, device=soft_scores.device, dtype=soft_scores.dtype) + 0.5) * (qx1 - qx0) / quest_w
    try:
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:
        yy, xx = torch.meshgrid(ys, xs)
    row_value = yy if row_axis == "y" else xx
    col_value = xx if row_axis == "y" else yy
    row_min, row_max = (ty0, ty1) if row_axis == "y" else (tx0, tx1)
    col_min, col_max = (tx0, tx1) if row_axis == "y" else (ty0, ty1)
    row = (row_value - row_min) / (row_max - row_min)
    col = (col_value - col_min) / (col_max - col_min)
    if row_direction < 0:
        row = 1 - row
    if col_direction < 0:
        col = 1 - col
    grid = torch.stack((2 * col - 1, 2 * row - 1), dim=-1).unsqueeze(0)
    return F.grid_sample(
        soft_scores.unsqueeze(0), grid, mode="bilinear",
        padding_mode="zeros", align_corners=False,
    ).squeeze(0)


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
    support_mask: torch.Tensor,
    channel_weights: torch.Tensor,
) -> torch.Tensor:
    if student_logits.shape != teacher_soft_scores.shape or student_logits.ndim != 4:
        raise ValueError("student and teacher maps must be [B,K,H,W] with equal shapes")
    count = student_logits.shape[1]
    if support_mask.shape != (count,) or channel_weights.shape != (count,):
        raise ValueError("teacher support/weights must have one value per channel")
    if not bool(support_mask.any()):
        raise ValueError("no audited teacher channels are supported")
    if not bool(torch.isfinite(teacher_soft_scores).all()) or not bool(
        ((teacher_soft_scores >= 0) & (teacher_soft_scores <= 1)).all()
    ):
        raise ValueError("teacher map must contain finite soft scores in [0,1]")
    if not bool(torch.isfinite(channel_weights).all()) or bool((channel_weights < 0).any()):
        raise ValueError("channel weights must be finite and nonnegative")
    per_channel = F.binary_cross_entropy_with_logits(
        student_logits, teacher_soft_scores, reduction="none"
    ).mean(dim=(0, 2, 3))
    weights = channel_weights * support_mask.to(channel_weights.dtype)
    if not bool(weights.sum() > 0):
        raise ValueError("supported teacher channel weights sum to zero")
    return (per_channel * weights).sum() / weights.sum()


def validate_teacher_record(record: Mapping[str, Any], token: str, sample_index: int) -> torch.Tensor:
    if record.get("schema_version") != TEACHER_MAP_SCHEMA_VERSION:
        raise ValueError("Navformer map teacher schema mismatch")
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
    if record.get("teacher_coordinate_frame") != "openscene_lidar_xy":
        raise ValueError("teacher and QUEST coordinate frames are not confirmed equal")
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
