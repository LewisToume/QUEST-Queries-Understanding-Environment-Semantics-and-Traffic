from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


PROPOSAL_PRETRAIN_STAGE = "proposal_pretrain"


def configure_proposal_pretraining(model: nn.Module) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.agent_proposal_head.requires_grad_(True)
    trainable = [
        parameter
        for parameter in model.agent_proposal_head.parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("AgentProposalHead has no trainable parameters")
    return trainable


def rasterize_proposal_targets(
    agent_target: Mapping[str, torch.Tensor],
    x_range: Sequence[float],
    y_range: Sequence[float],
    bev_h: int,
    bev_w: int,
) -> dict[str, Any]:
    required = {"labels", "boxes_metric", "scores", "valid_mask", "class_support_mask"}
    if not required.issubset(agent_target):
        raise ValueError(f"canonical Agent target requires {sorted(required)}")
    labels = agent_target["labels"]
    boxes = agent_target["boxes_metric"]
    scores = agent_target["scores"]
    valid = agent_target["valid_mask"].bool()
    support = agent_target["class_support_mask"].bool()
    if labels.ndim != 2:
        raise ValueError("labels must be [B,N]")
    batch_size, instance_count = labels.shape
    if boxes.shape != (batch_size, instance_count, 7):
        raise ValueError("boxes_metric must be [B,N,7]")
    if scores.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError("scores and valid_mask must match labels")
    if support.shape != (batch_size, 4):
        raise ValueError("class_support_mask must be [B,4]")
    if bev_h <= 0 or bev_w <= 0:
        raise ValueError("BEV dimensions must be positive")
    x_min, x_max = (float(value) for value in x_range)
    y_min, y_max = (float(value) for value in y_range)
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("BEV metric ranges must have positive extent")

    cell_count = bev_h * bev_w
    objectness = boxes.new_zeros((batch_size, cell_count))
    offsets = boxes.new_zeros((batch_size, cell_count, 2))
    weights = boxes.new_zeros((batch_size, cell_count))
    positive = torch.zeros(
        (batch_size, cell_count), dtype=torch.bool, device=boxes.device
    )
    winning_scores = boxes.new_full((batch_size, cell_count), -float("inf"))
    centers_by_sample: list[torch.Tensor] = []
    collision_count = 0

    for batch_index in range(batch_size):
        valid_centers = []
        for target_index in range(instance_count):
            if not bool(valid[batch_index, target_index]):
                continue
            class_id = int(labels[batch_index, target_index])
            if class_id not in (0, 1) or not bool(support[batch_index, class_id]):
                continue
            xy = boxes[batch_index, target_index, :2]
            score = scores[batch_index, target_index]
            if not bool(torch.isfinite(xy).all() & torch.isfinite(score)):
                continue
            x, y = xy
            if not bool((x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)):
                continue
            valid_centers.append(xy)
            grid_x = (x - x_min) / (x_max - x_min) * bev_w
            grid_y = (y - y_min) / (y_max - y_min) * bev_h
            column = min(int(torch.floor(grid_x)), bev_w - 1)
            row = min(int(torch.floor(grid_y)), bev_h - 1)
            flat_index = row * bev_w + column
            if bool(positive[batch_index, flat_index]):
                collision_count += 1
            positive[batch_index, flat_index] = True
            objectness[batch_index, flat_index] = 1.0
            if score > winning_scores[batch_index, flat_index]:
                winning_scores[batch_index, flat_index] = score
                weights[batch_index, flat_index] = score.clamp(0.0, 1.0).clamp_min(1e-3)
                offsets[batch_index, flat_index, 0] = (
                    grid_x - column - 0.5
                ).clamp(-0.5, 0.5)
                offsets[batch_index, flat_index, 1] = (
                    grid_y - row - 0.5
                ).clamp(-0.5, 0.5)
        centers_by_sample.append(
            torch.stack(valid_centers) if valid_centers else boxes.new_empty((0, 2))
        )

    return {
        "objectness_target": objectness,
        "offset_target": offsets,
        "positive_mask": positive,
        "positive_weight": weights,
        "target_centers_metric": centers_by_sample,
        "positive_cell_count": int(positive.sum()),
        "collision_count": collision_count,
    }


def compute_proposal_pretraining_loss(
    objectness_logits: torch.Tensor,
    predicted_offsets: torch.Tensor,
    targets: Mapping[str, Any],
    negative_loss_weight: float = 3.0,
    lambda_objectness: float = 2.0,
    lambda_offset: float = 1.0,
) -> dict[str, torch.Tensor]:
    if any(
        not math.isfinite(weight) or weight < 0
        for weight in (negative_loss_weight, lambda_objectness, lambda_offset)
    ):
        raise ValueError("proposal loss weights must be finite and non-negative")
    objectness_target = targets["objectness_target"]
    offset_target = targets["offset_target"]
    positive_mask = targets["positive_mask"].bool()
    positive_weight = targets["positive_weight"]
    if objectness_logits.ndim != 2 or objectness_logits.shape != objectness_target.shape:
        raise ValueError("objectness logits and target must be [B,H*W]")
    if predicted_offsets.shape != (*objectness_logits.shape, 2):
        raise ValueError("predicted offsets must be [B,H*W,2]")
    if offset_target.shape != predicted_offsets.shape:
        raise ValueError("offset target shape does not match predictions")
    if (
        positive_mask.shape != objectness_logits.shape
        or positive_weight.shape != objectness_logits.shape
    ):
        raise ValueError("positive mask and weights must match objectness")

    per_cell = F.binary_cross_entropy_with_logits(
        objectness_logits, objectness_target, reduction="none"
    )
    probabilities = objectness_logits.sigmoid()
    positive_losses = []
    negative_losses = []
    offset_losses = []
    positive_probabilities = []
    negative_probabilities = []
    for batch_index in range(objectness_logits.shape[0]):
        pos = positive_mask[batch_index]
        neg = ~pos
        zero = objectness_logits[batch_index].sum() * 0.0
        if bool(pos.any()):
            weight = positive_weight[batch_index, pos]
            positive_losses.append(
                (per_cell[batch_index, pos] * weight).sum() / weight.sum()
            )
            raw_offset = F.smooth_l1_loss(
                predicted_offsets[batch_index, pos],
                offset_target[batch_index, pos],
                reduction="none",
            ).mean(dim=-1)
            offset_losses.append((raw_offset * weight).sum() / weight.sum())
            positive_probabilities.append(probabilities[batch_index, pos].mean())
        else:
            positive_losses.append(zero)
            offset_losses.append(predicted_offsets[batch_index].sum() * 0.0)
            positive_probabilities.append(zero)
        negative_losses.append(
            per_cell[batch_index, neg].mean() if bool(neg.any()) else zero
        )
        negative_probabilities.append(
            probabilities[batch_index, neg].mean() if bool(neg.any()) else zero
        )

    positive_loss = torch.stack(positive_losses).mean()
    negative_loss = torch.stack(negative_losses).mean()
    offset_loss = torch.stack(offset_losses).mean()
    objectness_loss = positive_loss + negative_loss_weight * negative_loss
    proposal_loss = lambda_objectness * objectness_loss + lambda_offset * offset_loss
    return {
        "proposal_loss": proposal_loss,
        "objectness_loss": objectness_loss,
        "positive_objectness_loss": positive_loss,
        "negative_objectness_loss": negative_loss,
        "offset_loss": offset_loss,
        "mean_positive_objectness_probability": torch.stack(positive_probabilities).mean(),
        "mean_negative_objectness_probability": torch.stack(negative_probabilities).mean(),
    }


def proposal_center_hits(
    objectness_logits: torch.Tensor,
    predicted_offsets: torch.Tensor,
    target_centers_metric: Sequence[torch.Tensor],
    top_k: int,
    distance_threshold_m: float,
    x_range: Sequence[float],
    y_range: Sequence[float],
    bev_h: int,
    bev_w: int,
) -> tuple[int, int]:
    if top_k <= 0 or not math.isfinite(distance_threshold_m) or distance_threshold_m < 0:
        raise ValueError("top_k must be positive and distance threshold non-negative")
    if objectness_logits.ndim != 2 or objectness_logits.shape[1] != bev_h * bev_w:
        raise ValueError("objectness logits do not match BEV grid")
    if predicted_offsets.shape != (*objectness_logits.shape, 2):
        raise ValueError("predicted offsets do not match objectness logits")
    if len(target_centers_metric) != objectness_logits.shape[0]:
        raise ValueError("target centers do not match batch size")

    x_min, x_max = (float(value) for value in x_range)
    y_min, y_max = (float(value) for value in y_range)
    step_x = (x_max - x_min) / bev_w
    step_y = (y_max - y_min) / bev_h
    hits = 0
    target_count = 0
    for batch_index, target_xy in enumerate(target_centers_metric):
        target_count += int(target_xy.shape[0])
        if not target_xy.numel():
            continue
        selected = objectness_logits[batch_index].topk(
            min(top_k, bev_h * bev_w)
        ).indices
        columns = (selected % bev_w).to(predicted_offsets.dtype)
        rows = torch.div(selected, bev_w, rounding_mode="floor").to(predicted_offsets.dtype)
        offsets = predicted_offsets[batch_index, selected]
        centers = torch.stack(
            (
                x_min + (columns + 0.5 + offsets[:, 0]) * step_x,
                y_min + (rows + 0.5 + offsets[:, 1]) * step_y,
            ),
            dim=-1,
        )
        distances = torch.cdist(target_xy.unsqueeze(0), centers.unsqueeze(0))[0]
        hits += int((distances <= distance_threshold_m).any(dim=1).sum())
    return hits, target_count


def save_proposal_pretrain_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    training_config: Mapping[str, Any],
) -> None:
    from .model import QUEST_ARCHITECTURE_VERSION

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(
        {
            "architecture_version": QUEST_ARCHITECTURE_VERSION,
            "stage": PROPOSAL_PRETRAIN_STAGE,
            "epoch": int(epoch),
            "bev_use_ego_state": False,
            "geometry_aware_bev_lift_state_dict": model.geometry_lift.state_dict(),
            "bev_encoder_state_dict": model.bev_encoder.state_dict(),
            "agent_proposal_head_state_dict": model.agent_proposal_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_config": dict(training_config),
        },
        temporary,
    )
    temporary.replace(output)


def load_proposal_pretrain_checkpoint(
    model: nn.Module, checkpoint: Mapping[str, Any]
) -> int:
    from .model import QUEST_ARCHITECTURE_VERSION

    if checkpoint.get("architecture_version") != QUEST_ARCHITECTURE_VERSION:
        raise ValueError("proposal checkpoint architecture version mismatch")
    if checkpoint.get("stage") != PROPOSAL_PRETRAIN_STAGE:
        raise ValueError("checkpoint is not a QUEST proposal pretraining checkpoint")
    if checkpoint.get("bev_use_ego_state") is not False:
        raise ValueError("proposal checkpoint must use BEV without absolute ego state")
    model.geometry_lift.load_state_dict(
        checkpoint["geometry_aware_bev_lift_state_dict"], strict=True
    )
    model.bev_encoder.load_state_dict(
        checkpoint["bev_encoder_state_dict"], strict=True
    )
    model.agent_proposal_head.load_state_dict(
        checkpoint["agent_proposal_head_state_dict"], strict=True
    )
    return int(checkpoint["epoch"])
