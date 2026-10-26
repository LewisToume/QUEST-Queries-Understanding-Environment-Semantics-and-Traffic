"""Metric vector regression and auxiliary raster supervision from real vector GT."""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch.nn import functional as F


SPATIAL_MAP_OBJECTIVE = "metric_focal_dense_v3"


def map_class_weights(reference: torch.Tensor, config: Mapping) -> torch.Tensor:
    foreground = tuple(float(v) for v in config["class_weights"])
    background = float(config["background_weight"])
    if len(foreground) != 3 or any(not math.isfinite(v) or v <= 0 for v in (*foreground, background)):
        raise ValueError("map class weights require three positive foreground weights and background")
    return reference.new_tensor((*foreground, background))


def softmax_focal_terms(logits: torch.Tensor, gamma: float) -> torch.Tensor:
    """Keep the canonical 3-foreground + background softmax output contract."""
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("map focal_gamma must be finite and nonnegative")
    log_probability = logits.log_softmax(-1)
    return -(1 - log_probability.exp()).pow(gamma) * log_probability


def focal_matching_cost(logits: torch.Tensor, labels: torch.Tensor, config: Mapping) -> torch.Tensor:
    terms = softmax_focal_terms(logits, float(config["focal_gamma"]))
    terms = terms * map_class_weights(logits, config)
    # Assignment changes a query's label from background to the proposed GT class.
    # Use the same focal objective as classification, not a bounded -probability proxy.
    return terms[:, labels.to(device=logits.device, dtype=torch.long)] - terms[:, -1:]


def focal_classification_loss(logits: torch.Tensor, labels: torch.Tensor,
                              config: Mapping) -> torch.Tensor:
    weights = map_class_weights(logits, config)[labels]
    losses = softmax_focal_terms(logits, float(config["focal_gamma"])).gather(1, labels[:, None]).squeeze(1)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


@torch.no_grad()
def rasterize_vector_targets(targets: list[Mapping], h: int, w: int,
                             xy_range: tuple[float, float, float, float],
                             reference: torch.Tensor, line_width_m: float) -> torch.Tensor:
    """Distance-to-segment rasterization; retain open/closed semantics and every GT."""
    if line_width_m <= 0 or not math.isfinite(line_width_m):
        raise ValueError("dense_line_width_m must be finite and positive")
    x0, y0, x1, y1 = xy_range
    xs = x0 + (torch.arange(w, device=reference.device, dtype=reference.dtype) + 0.5) * (x1 - x0) / w
    ys = y0 + (torch.arange(h, device=reference.device, dtype=reference.dtype) + 0.5) * (y1 - y0) / h
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack((xx, yy), -1).reshape(-1, 2)
    # Include the cell footprint so a line falling between cell centers is not lost.
    radius = line_width_m / 2 + math.hypot((x1 - x0) / w, (y1 - y0) / h) / 2
    result = reference.new_zeros(len(targets), 3, h * w)
    for batch_index, target in enumerate(targets):
        lines = target["points_xy_m"].to(reference)
        labels = target["class_ids"].to(device=reference.device, dtype=torch.long)
        closed = target["is_closed"].to(device=reference.device, dtype=torch.bool)
        # Bound intermediate memory by chunking, never by truncating target instances.
        for start in range(0, len(lines), 16):
            chunk = lines[start:start + 16]
            a, z = chunk, torch.roll(chunk, -1, dims=1)
            direction = z - a
            delta = centers[None, None] - a[:, :, None]
            fraction = (delta * direction[:, :, None]).sum(-1) / direction.square().sum(-1).clamp_min(1e-8)[:, :, None]
            closest = a[:, :, None] + fraction.clamp(0, 1)[..., None] * direction[:, :, None]
            squared = (centers[None, None] - closest).square().sum(-1)
            # Closing the final segment is legal only for a genuinely closed contour.
            squared[:, -1] = squared[:, -1].masked_fill(~closed[start:start + len(chunk), None], float("inf"))
            hit = squared.amin(1) <= radius * radius
            for class_id in range(3):
                selected = hit[labels[start:start + len(chunk)] == class_id]
                if selected.numel():
                    result[batch_index, class_id] = torch.maximum(
                        result[batch_index, class_id], selected.any(0).to(result.dtype))
    return result.reshape(len(targets), 3, h, w)


def dense_map_gt_loss(logits: torch.Tensor, targets: list[Mapping], config: Mapping,
                      xy_range: tuple[float, float, float, float]) -> dict[str, torch.Tensor]:
    if logits.ndim != 4 or logits.shape[:2] != (len(targets), 3):
        raise ValueError("dense map GT head must produce [B,3,H,W]")
    target = rasterize_vector_targets(targets, *logits.shape[-2:], xy_range, logits,
                                      float(config["dense_line_width_m"]))
    probability = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    correct_probability = torch.where(target.bool(), probability, 1 - probability)
    focal = bce * (1 - correct_probability).pow(float(config["focal_gamma"]))
    positive, negative = target.bool(), ~target.bool()
    pos_count, neg_count = positive.sum((-1, -2)), negative.sum((-1, -2))
    pos_loss = (focal * positive).sum((-1, -2)) / pos_count.clamp_min(1)
    neg_loss = (focal * negative).sum((-1, -2)) / neg_count.clamp_min(1)
    per_class_focal = (pos_loss + float(config["dense_negative_weight"]) * neg_loss).mean(0)
    intersection = (probability * target).sum((-1, -2))
    dice = 1 - (2 * intersection + 1) / (probability.sum((-1, -2)) + target.sum((-1, -2)) + 1)
    weights = map_class_weights(logits, config)[:3]
    return {"map_dense_focal_loss": (per_class_focal * weights).sum() / weights.sum(),
            "map_dense_dice_loss": (dice.mean(0) * weights).sum() / weights.sum()}
