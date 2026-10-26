"""MapTR-inspired point decoding without a dependency on legacy OpenMMLab."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import GeometryAwareBEVLift


SPATIAL_MAP_VERSION = "spatial_map_v1"


def inverse_sigmoid(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.clamp(1e-4, 1 - 1e-4))


def _partition(value: torch.Tensor, size: int) -> torch.Tensor:
    b, h, w, c = value.shape
    return value.reshape(b, h // size, size, w // size, size, c).permute(
        0, 1, 3, 2, 4, 5
    ).reshape(-1, size * size, c)


def _unpartition(value: torch.Tensor, size: int, b: int, h: int, w: int) -> torch.Tensor:
    return value.reshape(b, h // size, w // size, size, size, -1).permute(
        0, 1, 3, 2, 4, 5
    ).reshape(b, h, w, -1)


class WindowBEVLayer(nn.Module):
    """Shifted local attention avoids quadratic attention over all 4096 cells."""

    def __init__(self, dim: int, h: int, w: int, heads: int, ffn: int,
                 window: int, shift: int, dropout: float) -> None:
        super().__init__()
        if h % window or w % window or not 0 <= shift < window:
            raise ValueError("map BEV must be divisible by the attention window")
        self.h, self.w, self.window, self.shift, self.heads = h, w, window, shift, heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn, dim))
        self.dropout = nn.Dropout(dropout)
        mask = None
        if shift:
            regions = torch.zeros(1, h, w, 1)
            slices = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
            region = 0
            for ys in slices:
                for xs in slices:
                    regions[:, ys, xs] = region
                    region += 1
            windows = _partition(regions, window).squeeze(-1)
            difference = windows[:, :, None] - windows[:, None, :]
            mask = torch.zeros_like(difference).masked_fill(difference != 0, float("-inf"))
        self.register_buffer("attention_mask", mask, persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = value.shape
        normalized = self.norm1(value)
        if self.shift:
            normalized = torch.roll(normalized, (-self.shift, -self.shift), (1, 2))
        windows = _partition(normalized, self.window)
        mask = (self.attention_mask.repeat(b, 1, 1).repeat_interleave(self.heads, dim=0)
                if self.attention_mask is not None else None)
        attended = self.attention(windows, windows, windows, attn_mask=mask,
                                  need_weights=False)[0]
        attended = _unpartition(attended, self.window, b, h, w)
        if self.shift:
            attended = torch.roll(attended, (self.shift, self.shift), (1, 2))
        value = value + self.dropout(attended)
        return value + self.dropout(self.ffn(self.norm2(value)))


class MapSpatialBranch(nn.Module):
    def __init__(self, dim: int, options: Mapping, x_range: Sequence[float],
                 y_range: Sequence[float], z_anchors: Sequence[float]) -> None:
        super().__init__()
        self.h, self.w = int(options["bev_h"]), int(options["bev_w"])
        self.num_levels = int(options["num_levels"])
        if self.num_levels < 1 or self.h % 2 ** (self.num_levels - 1) or self.w % 2 ** (self.num_levels - 1):
            raise ValueError("map pyramid dimensions must be divisible at every level")
        self.geometry_lift = GeometryAwareBEVLift(dim, self.h, self.w,
                                                 tuple(x_range), tuple(y_range), z_anchors)
        self.geometry_lift.ego_mlp.requires_grad_(False)
        self.coarse_projection = nn.Conv2d(dim, dim, 1)
        self.input_norm = nn.LayerNorm(dim)
        window = int(options["window_size"])
        self.layers = nn.ModuleList([
            WindowBEVLayer(dim, self.h, self.w, int(options["num_heads"]),
                           int(options["ffn_dim"]), window, (window // 2 if i % 2 else 0),
                           float(options["dropout"]))
            for i in range(int(options["encoder_layers"]))
        ])
        self.norm = nn.LayerNorm(dim)
        self.dense_head = nn.Sequential(nn.Conv2d(dim, 128, 3, padding=1),
                                         nn.GroupNorm(8, 128), nn.GELU(), nn.Conv2d(128, 3, 1))

    def initialize_lift(self, source: GeometryAwareBEVLift) -> None:
        """Copy only compatible visual weights; metric grids are regenerated, not loaded."""
        self.geometry_lift.weight_mlp.load_state_dict(source.weight_mlp.state_dict(), strict=True)
        with torch.no_grad():
            embedding = source.bev_embedding.T.reshape(1, source.hidden_dim, source.bev_h, source.bev_w)
            resized = F.interpolate(embedding, (self.h, self.w), mode="bilinear", align_corners=False)
            self.geometry_lift.bev_embedding.copy_(resized.flatten(2).transpose(1, 2)[0])

    def forward(self, camera_features: torch.Tensor, coarse_bev: torch.Tensor,
                intrinsics: torch.Tensor, extrinsics: torch.Tensor,
                ego_state: torch.Tensor, image_size: tuple[int, int], *,
                include_dense: bool = True) -> dict:
        tokens = self.geometry_lift(camera_features, intrinsics, extrinsics, ego_state,
                                    image_size, use_ego_state=False)
        # The independent lift resamples original DINO maps at fine metric locations.
        # Interpolation here supplies additional frozen coarse context, not new pixels.
        coarse = F.interpolate(self.coarse_projection(coarse_bev), (self.h, self.w),
                                mode="bilinear", align_corners=False)
        value = self.input_norm(tokens.reshape(tokens.shape[0], self.h, self.w, -1)
                                + coarse.permute(0, 2, 3, 1))
        for layer in self.layers:
            value = layer(value)
        features = self.norm(value).permute(0, 3, 1, 2).contiguous()
        pyramid = [features]
        for _ in range(1, self.num_levels):
            pyramid.append(F.avg_pool2d(pyramid[-1], 2))
        result = {"pyramid": pyramid, "features": features}
        if include_dense:
            result["dense_logits"] = self.dense_head(features)
        return result


class MultiScalePointAttention(nn.Module):
    """Reference-relative multi-head sampling, implemented with torch.grid_sample."""

    def __init__(self, dim: int, heads: int, levels: int, points: int) -> None:
        super().__init__()
        if dim % heads or min(heads, levels, points) <= 0:
            raise ValueError("invalid multi-scale point-attention dimensions")
        self.dim, self.heads, self.levels, self.points = dim, heads, levels, points
        self.offsets = nn.Linear(dim, heads * levels * points * 2)
        self.weights = nn.Linear(dim, heads * levels * points)
        self.value_projection = nn.Conv2d(dim, dim, 1)
        self.output_projection = nn.Linear(dim, dim)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)
        angles = torch.arange(heads, dtype=torch.float32) * (2 * math.pi / heads)
        directions = torch.stack((angles.cos(), angles.sin()), -1)
        directions /= directions.abs().amax(-1, keepdim=True).clamp_min(1e-6)
        radial = directions[:, None, None, :] * torch.arange(1, points + 1)[None, None, :, None]
        with torch.no_grad():
            self.offsets.bias.copy_(radial.expand(heads, levels, points, 2).reshape(-1))

    def forward(self, query: torch.Tensor, references: torch.Tensor,
                pyramid: list[torch.Tensor]) -> torch.Tensor:
        b, q, p, c = query.shape
        if len(pyramid) != self.levels or references.shape != (b, q, p, 2):
            raise ValueError("point attention reference/pyramid contract mismatch")
        n, d = q * p, c // self.heads
        flattened = query.reshape(b, n, c)
        offsets = self.offsets(flattened).reshape(b, n, self.heads, self.levels, self.points, 2)
        logits = self.weights(flattened).reshape(b, n, self.heads, self.levels, self.points)
        normalizers = query.new_tensor([(feature.shape[-1], feature.shape[-2]) for feature in pyramid])
        locations = references.reshape(b, n, 1, 1, 1, 2) + offsets / normalizers[None, None, None, :, None]
        valid = ((locations >= 0) & (locations <= 1)).all(-1) & torch.isfinite(locations).all(-1)
        weights = logits.masked_fill(~valid, -1e4).flatten(-2).softmax(-1).reshape_as(logits)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum((-1, -2), keepdim=True).clamp_min(1e-6)
        output = query.new_zeros(b, n, self.heads, d)
        for level, feature in enumerate(pyramid):
            value = self.value_projection(feature).reshape(b, self.heads, d, *feature.shape[-2:])
            safe_locations = locations[:, :, :, level].masked_fill(~valid[:, :, :, level, :, None], 2.0)
            grid = safe_locations.permute(0, 2, 1, 3, 4).reshape(b * self.heads, n, self.points, 2)
            sampled = F.grid_sample(value.reshape(b * self.heads, d, *feature.shape[-2:]),
                                     grid.mul(2).sub(1), mode="bilinear",
                                     padding_mode="zeros", align_corners=False)
            sampled = sampled.reshape(b, self.heads, d, n, self.points).permute(0, 3, 1, 4, 2)
            output = output + (sampled * weights[:, :, :, level, :, None]).sum(-2)
        projected = self.output_projection(output.reshape(b, q, p, c))
        any_valid = valid.flatten(2).any(-1).reshape(b, q, p, 1)
        return projected * any_valid.to(projected.dtype)


class PointMapDecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ffn: int, levels: int,
                 points: int, dropout: float) -> None:
        super().__init__()
        self.point_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.instance_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.spatial_attention = MultiScalePointAttention(dim, heads, levels, points)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(4)])
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor, position: torch.Tensor,
                references: torch.Tensor, pyramid: list[torch.Tensor]) -> torch.Tensor:
        b, q, p, c = value.shape
        normalized = self.norms[0](value)
        positioned = (normalized + position).reshape(b * q, p, c)
        attended = self.point_attention(positioned, positioned, normalized.reshape(b * q, p, c),
                                         need_weights=False)[0].reshape(b, q, p, c)
        value = value + self.dropout(attended)
        normalized = self.norms[1](value).mean(2)
        positioned = normalized + position.mean(2)
        attended = self.instance_attention(positioned, positioned, normalized, need_weights=False)[0]
        value = value + self.dropout(attended[:, :, None])
        value = value + self.dropout(self.spatial_attention(self.norms[2](value) + position,
                                                           references, pyramid))
        return value + self.dropout(self.ffn(self.norms[3](value)))


class SpatialMapHead(nn.Module):
    def __init__(self, dim: int, classes: int, layers: int) -> None:
        super().__init__()
        self.cls_branches = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, classes + 1))
            for _ in range(layers)
        ])
        self.point_branches = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2))
            for _ in range(layers)
        ])
        for branch in self.point_branches:
            nn.init.normal_(branch[-1].weight, std=0.001)
            nn.init.zeros_(branch[-1].bias)

    def forward(self, features: torch.Tensor, references: torch.Tensor,
                layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.cls_branches[layer_index](features.mean(2))
        points = (inverse_sigmoid(references) + self.point_branches[layer_index](features)).sigmoid()
        return logits, points


class SpatialMapDecoder(nn.Module):
    def __init__(self, dim: int, queries: int, points: int, options: Mapping) -> None:
        super().__init__()
        self.num_queries, self.num_points = queries, points
        side = math.ceil(math.sqrt(queries))
        # Spread instance references across the entire ROI; do not initialize horizontal lines.
        ids = torch.linspace(0, side * side - 1, queries).round().long()
        centers = torch.stack(((ids % side + 0.5) / side, (ids // side + 0.5) / side), -1)
        self.reference_logits = nn.Parameter(inverse_sigmoid(centers))
        self.instance_embedding = nn.Parameter(torch.randn(queries, dim) * 0.02)
        self.point_embedding = nn.Parameter(torch.randn(points, dim) * 0.02)
        angles = torch.arange(points, dtype=torch.float32) * (2 * math.pi / points)
        self.point_offsets = nn.Parameter(torch.stack((angles.cos(), angles.sin()), -1) * 0.02)
        self.reference_mlp = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.input_norm = nn.LayerNorm(dim)
        self.output_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(int(options["decoder_layers"]))])
        self.layers = nn.ModuleList([
            PointMapDecoderLayer(dim, int(options["num_heads"]), int(options["ffn_dim"]),
                                 int(options["num_levels"]), int(options["sampling_points"]),
                                 float(options["dropout"]))
            for _ in self.output_norms
        ])

    def forward(self, pyramid: list[torch.Tensor], head: SpatialMapHead) -> dict[str, torch.Tensor]:
        b = pyramid[0].shape[0]
        centers = self.reference_logits.sigmoid()
        references = (centers[:, None] + self.point_offsets[None]).clamp(1e-4, 1 - 1e-4)
        references = references.unsqueeze(0).expand(b, -1, -1, -1)
        local = F.grid_sample(pyramid[0], centers[None, :, None].expand(b, -1, -1, -1).mul(2).sub(1),
                              mode="bilinear", padding_mode="zeros", align_corners=False)
        instances = local.squeeze(-1).transpose(1, 2) + self.instance_embedding[None]
        value = self.input_norm(instances[:, :, None] + self.point_embedding[None, None])
        logits_layers, point_layers = [], []
        for index, layer in enumerate(self.layers):
            value = layer(value, self.reference_mlp(references), references, pyramid)
            logits, points = head(self.output_norms[index](value), references, index)
            logits_layers.append(logits)
            point_layers.append(points)
            references = points.detach()
        return {"map_cls_logits_layers": torch.stack(logits_layers),
                "map_points_layers": torch.stack(point_layers),
                "map_cls_logits": logits_layers[-1], "map_points": point_layers[-1]}
