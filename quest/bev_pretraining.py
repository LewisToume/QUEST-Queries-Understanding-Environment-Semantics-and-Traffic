from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


BEV_PRETRAIN_STAGE = "bev_pretrain"
BEV_PRETRAIN_CLASS_NAMES = ("vehicle", "pedestrian")


class BEVAuxiliaryHead(nn.Module):
    """Small dense head used only to supervise the encoded BEV during Stage 1."""

    def __init__(self, hidden_dim: int = 384, intermediate_dim: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(hidden_dim, intermediate_dim, kernel_size=1),
            nn.GELU(),
        )
        self.foreground_head = nn.Conv2d(intermediate_dim, 1, kernel_size=1)
        self.class_head = nn.Conv2d(intermediate_dim, 2, kernel_size=1)

    def forward(self, bev_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if bev_features.ndim != 4:
            raise ValueError(
                f"bev_features must be [B,C,H,W], got {tuple(bev_features.shape)}"
            )
        features = self.trunk(bev_features)
        foreground = self.foreground_head(features).squeeze(1)
        classes = self.class_head(features).permute(0, 2, 3, 1).contiguous()
        return {"foreground_logits": foreground, "class_logits": classes}


def extract_canonical_agent_batch(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor] | None:
    soft_labels = batch.get("soft_labels", {})
    if isinstance(soft_labels, list):
        agents = [item.get("agent") for item in soft_labels if isinstance(item, Mapping)]
        if len(agents) != len(soft_labels) or any(agent is None for agent in agents):
            return None
        return {
            key: torch.stack([agent[key] for agent in agents]).to(device)
            for key in agents[0]
        }
    if not isinstance(soft_labels, Mapping) or "agent" not in soft_labels:
        return None
    agent = soft_labels["agent"]
    if not isinstance(agent, Mapping):
        raise ValueError("soft_labels.agent must be a canonical target mapping")
    return {key: value.to(device) for key, value in agent.items()}


def configure_bev_pretraining(
    model: nn.Module, auxiliary_head: BEVAuxiliaryHead
) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.geometry_lift, model.bev_encoder, auxiliary_head):
        module.requires_grad_(True)
    ego_mlp = getattr(model.geometry_lift, "ego_mlp", None)
    if ego_mlp is not None:
        ego_mlp.requires_grad_(False)
    model.backbone.requires_grad_(False)
    trainable = [
        parameter
        for module in (model.geometry_lift, model.bev_encoder, auxiliary_head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("BEV pretraining has no trainable parameters")
    return trainable


def rasterize_agent_centers(
    agent_target: Mapping[str, torch.Tensor],
    x_range: Sequence[float],
    y_range: Sequence[float],
    bev_h: int,
    bev_w: int,
) -> dict[str, torch.Tensor]:
    """Rasterize supported canonical Agent centers onto the existing metric BEV."""

    required = {
        "labels", "boxes_metric", "scores", "class_support_mask", "valid_mask"
    }
    if not required.issubset(agent_target):
        raise ValueError(f"canonical Agent target requires {sorted(required)}")
    labels = agent_target["labels"]
    boxes = agent_target["boxes_metric"]
    scores = agent_target["scores"]
    valid_mask = agent_target["valid_mask"].bool()
    support = agent_target["class_support_mask"].bool()
    if labels.ndim != 2:
        raise ValueError("batched Agent labels must be [B,N]")
    batch_size, instance_count = labels.shape
    if boxes.shape != (batch_size, instance_count, 7):
        raise ValueError("boxes_metric must be [B,N,7]")
    if scores.shape != labels.shape or valid_mask.shape != labels.shape:
        raise ValueError("Agent labels, scores, and valid_mask must have equal shape")
    if support.shape != (batch_size, 4):
        raise ValueError("class_support_mask must be [B,4]")
    if bev_h <= 0 or bev_w <= 0:
        raise ValueError("BEV dimensions must be positive")
    x_min, x_max = (float(value) for value in x_range)
    y_min, y_max = (float(value) for value in y_range)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("BEV spatial ranges must have positive extent")

    device = boxes.device
    foreground = boxes.new_zeros((batch_size, bev_h, bev_w))
    class_target = torch.full(
        (batch_size, bev_h, bev_w), -1, dtype=torch.long, device=device
    )
    winning_score = boxes.new_full((batch_size, bev_h, bev_w), -float("inf"))
    collisions = torch.zeros((), dtype=torch.long, device=device)
    class_counts = torch.zeros(2, dtype=torch.long, device=device)

    for batch_index in range(batch_size):
        for target_index in range(instance_count):
            if not bool(valid_mask[batch_index, target_index]):
                continue
            class_id = int(labels[batch_index, target_index])
            if class_id not in (0, 1) or not bool(support[batch_index, class_id]):
                continue
            x = boxes[batch_index, target_index, 0]
            y = boxes[batch_index, target_index, 1]
            if not bool(torch.isfinite(x) & torch.isfinite(y)):
                continue
            if not bool((x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)):
                continue
            class_counts[class_id] += 1
            column = min(int(torch.floor((x - x_min) / (x_max - x_min) * bev_w)), bev_w - 1)
            row = min(int(torch.floor((y - y_min) / (y_max - y_min) * bev_h)), bev_h - 1)
            score = scores[batch_index, target_index]
            if foreground[batch_index, row, column] > 0:
                collisions += 1
            else:
                foreground[batch_index, row, column] = 1.0
            if score > winning_score[batch_index, row, column]:
                winning_score[batch_index, row, column] = score
                class_target[batch_index, row, column] = class_id

    positive = foreground.bool()
    return {
        "foreground_target": foreground,
        "class_target": class_target,
        "positive_mask": positive,
        "occupied_bev_cells": positive.sum(),
        "target_collision_count": collisions,
        "vehicle_target_count": class_counts[0],
        "pedestrian_target_count": class_counts[1],
    }


def compute_bev_pretraining_loss(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    lambda_class: float = 1.0,
) -> dict[str, torch.Tensor]:
    foreground_logits = predictions["foreground_logits"]
    class_logits = predictions["class_logits"]
    foreground_target = targets["foreground_target"].to(foreground_logits.dtype)
    positive_mask = targets["positive_mask"].bool()
    if foreground_logits.shape != foreground_target.shape:
        raise ValueError("foreground prediction and target shapes do not match")
    if class_logits.shape != (*foreground_logits.shape, 2):
        raise ValueError("class_logits must be [B,H,W,2]")

    per_cell = F.binary_cross_entropy_with_logits(
        foreground_logits, foreground_target, reduction="none"
    )
    positive_losses = []
    negative_losses = []
    positive_probabilities = []
    negative_probabilities = []
    probabilities = foreground_logits.sigmoid()
    for batch_index in range(foreground_logits.shape[0]):
        positive = positive_mask[batch_index]
        negative = ~positive
        positive_losses.append(
            per_cell[batch_index][positive].mean()
            if positive.any()
            else foreground_logits[batch_index].sum() * 0.0
        )
        negative_losses.append(
            per_cell[batch_index][negative].mean()
            if negative.any()
            else foreground_logits[batch_index].sum() * 0.0
        )
        positive_probabilities.append(
            probabilities[batch_index][positive].mean()
            if positive.any()
            else probabilities[batch_index].sum() * 0.0
        )
        negative_probabilities.append(
            probabilities[batch_index][negative].mean()
            if negative.any()
            else probabilities[batch_index].sum() * 0.0
        )
    positive_loss = torch.stack(positive_losses).mean()
    negative_loss = torch.stack(negative_losses).mean()
    foreground_loss = positive_loss + negative_loss
    if positive_mask.any():
        class_loss = F.cross_entropy(
            class_logits[positive_mask], targets["class_target"][positive_mask]
        )
    else:
        class_loss = class_logits.sum() * 0.0
    total = foreground_loss + float(lambda_class) * class_loss
    return {
        "total_loss": total,
        "foreground_loss": foreground_loss,
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
        "class_loss": class_loss,
        "mean_positive_foreground_probability": torch.stack(
            positive_probabilities
        ).mean(),
        "mean_negative_foreground_probability": torch.stack(
            negative_probabilities
        ).mean(),
    }


def gradient_rms(module: nn.Module) -> float:
    squared_sum = 0.0
    element_count = 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        squared_sum += float(gradient.square().sum())
        element_count += gradient.numel()
    return math.sqrt(squared_sum / element_count) if element_count else 0.0


def foreground_counts(
    foreground_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, int]:
    predicted = foreground_logits.sigmoid() >= float(threshold)
    target = positive_mask.bool()
    return {
        "true_positive": int((predicted & target).sum()),
        "false_positive": int((predicted & ~target).sum()),
        "false_negative": int((~predicted & target).sum()),
    }


def topk_center_hits(
    foreground_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    top_k: int,
    tolerance_cells: int = 1,
) -> tuple[int, int]:
    if top_k <= 0 or tolerance_cells < 0:
        raise ValueError("top_k must be positive and tolerance_cells non-negative")
    batch_size, height, width = foreground_logits.shape
    hit_count = 0
    target_count = 0
    for batch_index in range(batch_size):
        flat = foreground_logits[batch_index].flatten()
        selected = flat.topk(min(top_k, flat.numel())).indices
        selected_rows = torch.div(selected, width, rounding_mode="floor")
        selected_columns = selected % width
        targets = torch.nonzero(positive_mask[batch_index], as_tuple=False)
        target_count += int(targets.shape[0])
        for row, column in targets:
            near = (
                (selected_rows - row).abs() <= tolerance_cells
            ) & ((selected_columns - column).abs() <= tolerance_cells)
            hit_count += int(bool(near.any()))
    return hit_count, target_count


def class_accuracy_counts(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive = targets["positive_mask"].bool()
    predicted_positive = predictions["foreground_logits"].sigmoid() >= threshold
    matched = positive & predicted_positive
    predicted_class = predictions["class_logits"].argmax(dim=-1)
    correct = torch.zeros(2, dtype=torch.long, device=positive.device)
    total = torch.zeros(2, dtype=torch.long, device=positive.device)
    for class_id in range(2):
        selected = matched & (targets["class_target"] == class_id)
        total[class_id] = selected.sum()
        correct[class_id] = (selected & (predicted_class == class_id)).sum()
    return correct, total


def save_bev_pretrain_checkpoint(
    path: str | Path,
    model: nn.Module,
    auxiliary_head: BEVAuxiliaryHead,
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
            "stage": BEV_PRETRAIN_STAGE,
            "epoch": int(epoch),
            "geometry_aware_bev_lift_state_dict": model.geometry_lift.state_dict(),
            "bev_encoder_state_dict": model.bev_encoder.state_dict(),
            "bev_auxiliary_head_state_dict": auxiliary_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_config": dict(training_config),
        },
        temporary,
    )
    temporary.replace(output)


def load_bev_pretrain_checkpoint(
    model: nn.Module,
    checkpoint: Mapping[str, Any],
    auxiliary_head: BEVAuxiliaryHead | None = None,
) -> int:
    from .model import QUEST_ARCHITECTURE_VERSION

    if checkpoint.get("architecture_version") != QUEST_ARCHITECTURE_VERSION:
        raise ValueError("BEV pretraining checkpoint architecture version mismatch")
    if checkpoint.get("stage") != BEV_PRETRAIN_STAGE:
        raise ValueError("checkpoint is not a QUEST BEV pretraining checkpoint")
    model.geometry_lift.load_state_dict(
        checkpoint["geometry_aware_bev_lift_state_dict"], strict=True
    )
    model.bev_encoder.load_state_dict(checkpoint["bev_encoder_state_dict"], strict=True)
    if auxiliary_head is not None:
        auxiliary_head.load_state_dict(
            checkpoint["bev_auxiliary_head_state_dict"], strict=True
        )
    return int(checkpoint.get("epoch", 0))
