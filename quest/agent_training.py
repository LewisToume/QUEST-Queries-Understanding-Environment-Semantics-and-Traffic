from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from .bev_pretraining import extract_canonical_agent_batch
from .losses import compute_agent_decoder_loss
from .proposal_pretraining import (
    compute_proposal_pretraining_loss,
    rasterize_proposal_targets,
)
from .teacher_adapters import NAVFORMER_CLASS_SUPPORT


AGENT_END_TO_END_STAGE = "agent_end_to_end"
TRAINABLE_AGENT_MODULES = (
    "geometry_lift", "bev_encoder", "agent_proposal_head", "agent_decoder", "agent_head"
)


def load_checkpoint_cpu(path: str | Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def configure_agent_training(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    model.requires_grad_(False)
    for name in TRAINABLE_AGENT_MODULES:
        getattr(model, name).requires_grad_(True)
    model.geometry_lift.ego_mlp.requires_grad_(False)
    bev_parameters = [
        parameter
        for name in ("geometry_lift", "bev_encoder")
        for parameter in getattr(model, name).parameters()
        if parameter.requires_grad
    ]
    agent_parameters = [
        parameter
        for name in ("agent_proposal_head", "agent_decoder", "agent_head")
        for parameter in getattr(model, name).parameters()
        if parameter.requires_grad
    ]
    if not bev_parameters or not agent_parameters:
        raise RuntimeError("Stage 2 Agent optimizer groups must both be nonempty")
    return bev_parameters, agent_parameters


def prepare_navformer_agent_batch(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    target = extract_canonical_agent_batch(batch, device)
    if target is None:
        raise RuntimeError(f"missing Navformer Agent labels for {batch['sample_token']}")
    labels = target["labels"]
    valid = target["valid_mask"].bool()
    support = target["class_support_mask"].bool()
    if labels.ndim != 2 or valid.shape != labels.shape or support.shape != (labels.shape[0], 4):
        raise ValueError("Navformer canonical Agent batch has invalid shapes")
    expected = NAVFORMER_CLASS_SUPPORT.to(device=device)
    if not bool((support[:, :2] == expected[:2]).all()):
        raise ValueError("Navformer teacher must declare vehicle and pedestrian support")
    supported_labels = (labels == 0) | (labels == 1)
    return {
        **target,
        "valid_mask": valid & supported_labels,
        "class_support_mask": expected.unsqueeze(0).expand(labels.shape[0], -1),
    }


def forward_agent_end_to_end(
    model: nn.Module, batch: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    encoded = model.encode_image(
        batch["images"].to(device),
        batch["intrinsics"].to(device),
        batch["extrinsics"].to(device),
        batch["ego_state"].to(device),
        use_ego_state=False,
    )
    return model.forward_agent(encoded["bev_tokens"], encoded["bev_features"])


def compute_stage2_agent_loss(
    model: nn.Module,
    predictions: Mapping[str, torch.Tensor],
    agent_target: Mapping[str, torch.Tensor],
    train_config: Mapping[str, Any],
    agent_loss_config: Mapping[str, Any],
) -> dict[str, Any]:
    proposal_targets = rasterize_proposal_targets(
        agent_target,
        model.geometry_lift.x_range,
        model.geometry_lift.y_range,
        model.bev_h,
        model.bev_w,
    )
    proposal = compute_proposal_pretraining_loss(
        predictions["proposal_objectness_logits"],
        predictions["proposal_xy_offsets"],
        proposal_targets,
        negative_loss_weight=float(train_config["negative_loss_weight"]),
        lambda_objectness=float(train_config["lambda_proposal_objectness"]),
        lambda_offset=float(train_config["lambda_proposal_offset"]),
    )
    decoder_settings = {
        **agent_loss_config,
        "background_weight": float(train_config["background_weight"]),
    }
    if not decoder_settings.get("teacher_confidence_weighting", False):
        raise ValueError("Stage 2 Agent requires teacher confidence weighted regression")
    if tuple(float(value) for value in decoder_settings["xy_range"]) != model.geometry_lift.x_range:
        raise ValueError("Agent loss xy_range must match the metric BEV range")
    if model.geometry_lift.x_range != model.geometry_lift.y_range:
        raise ValueError("Agent box decoder requires identical x and y ranges")
    decoder = compute_agent_decoder_loss(predictions, agent_target, decoder_settings)
    return {
        **proposal,
        **decoder,
        "total_agent_loss": proposal["proposal_loss"] + decoder["decoder_agent_loss"],
        "positive_cell_count": proposal_targets["positive_cell_count"],
        "collision_count": proposal_targets["collision_count"],
    }


def save_agent_stage2_checkpoint(
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
            "stage": AGENT_END_TO_END_STAGE,
            "epoch": int(epoch),
            "bev_use_ego_state": False,
            "trained_class_support_mask": NAVFORMER_CLASS_SUPPORT.clone(),
            "geometry_aware_bev_lift_state_dict": model.geometry_lift.state_dict(),
            "bev_encoder_state_dict": model.bev_encoder.state_dict(),
            "agent_proposal_head_state_dict": model.agent_proposal_head.state_dict(),
            "agent_decoder_state_dict": model.agent_decoder.state_dict(),
            "agent_head_state_dict": model.agent_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_config": dict(training_config),
        },
        temporary,
    )
    temporary.replace(output)


def load_agent_stage2_checkpoint(
    model: nn.Module, checkpoint: Mapping[str, Any]
) -> int:
    from .model import QUEST_ARCHITECTURE_VERSION

    if checkpoint.get("architecture_version") != QUEST_ARCHITECTURE_VERSION:
        raise ValueError("Stage 2 Agent checkpoint architecture version mismatch")
    if checkpoint.get("stage") != AGENT_END_TO_END_STAGE:
        raise ValueError("checkpoint is not an end-to-end Agent checkpoint")
    if checkpoint.get("bev_use_ego_state") is not False:
        raise ValueError("Stage 2 Agent checkpoint must disable absolute ego state")
    support = checkpoint.get("trained_class_support_mask")
    if not torch.is_tensor(support) or support.shape != (4,) or not torch.equal(
        support.cpu().bool(), NAVFORMER_CLASS_SUPPORT
    ):
        raise ValueError("Stage 2 Agent checkpoint must support vehicle and pedestrian only")
    for name, key in (
        ("geometry_lift", "geometry_aware_bev_lift_state_dict"),
        ("bev_encoder", "bev_encoder_state_dict"),
        ("agent_proposal_head", "agent_proposal_head_state_dict"),
        ("agent_decoder", "agent_decoder_state_dict"),
        ("agent_head", "agent_head_state_dict"),
    ):
        getattr(model, name).load_state_dict(checkpoint[key], strict=True)
    return int(checkpoint["epoch"])


def decode_supported_agent_predictions(
    predictions: Mapping[str, torch.Tensor],
    support_mask: torch.Tensor,
    confidence_threshold: float,
    x_range: Sequence[float],
    y_range: Sequence[float],
    z_range: Sequence[float],
    size_norm: Sequence[float],
) -> dict[str, torch.Tensor]:
    logits = predictions["agent_cls_logits"]
    boxes = predictions["agent_boxes"]
    velocity = predictions["agent_velocity"]
    if logits.ndim != 2 or logits.shape[-1] != 5 or boxes.shape != (logits.shape[0], 8):
        raise ValueError("Agent predictions must be [Q,5] logits and [Q,8] boxes")
    if velocity.shape != (logits.shape[0], 3):
        raise ValueError("Agent velocity must be [Q,3]")
    support = support_mask.to(device=logits.device, dtype=torch.bool)
    if support.shape != (4,):
        raise ValueError("Agent class support mask must be [4]")
    masked_logits = logits.clone()
    masked_logits[:, :4] = masked_logits[:, :4].masked_fill(~support[None, :], float("-inf"))
    probabilities = masked_logits.softmax(dim=-1)
    scores, labels = probabilities.max(dim=-1)
    keep = (labels != 4) & (scores >= confidence_threshold)
    boxes = boxes[keep]
    x_min, x_max = (float(value) for value in x_range)
    y_min, y_max = (float(value) for value in y_range)
    z_min, z_max = (float(value) for value in z_range)
    size_scale = boxes.new_tensor(size_norm)
    centers = torch.stack((
        x_min + boxes[:, 0] * (x_max - x_min),
        y_min + boxes[:, 1] * (y_max - y_min),
        z_min + boxes[:, 2] * (z_max - z_min),
    ), dim=-1)
    return {
        "labels": labels[keep],
        "scores": scores[keep],
        "centers": centers,
        "sizes": boxes[:, 3:6] * size_scale,
        "yaw": torch.atan2(boxes[:, 6], boxes[:, 7]),
        "velocity": velocity[keep],
    }


def class_aware_center_matches(
    predicted_centers: torch.Tensor,
    predicted_labels: torch.Tensor,
    target_centers: torch.Tensor,
    target_labels: torch.Tensor,
    distance_threshold_m: float,
) -> list[tuple[int, int, float]]:
    if predicted_centers.numel() == 0 or target_centers.numel() == 0:
        return []
    matches = []
    for class_id in (0, 1):
        pred_indices = torch.nonzero(predicted_labels == class_id).flatten()
        gt_indices = torch.nonzero(target_labels == class_id).flatten()
        if not pred_indices.numel() or not gt_indices.numel():
            continue
        distances = torch.cdist(
            predicted_centers[pred_indices].float(),
            target_centers[gt_indices].float(),
            p=2,
        )
        permitted = distances <= distance_threshold_m
        cost = distances.masked_fill(~permitted, 1e6)
        row_indices, column_indices = linear_sum_assignment(cost.detach().cpu().numpy())
        for row, column in zip(row_indices, column_indices):
            if bool(permitted[row, column]):
                matches.append((
                    int(pred_indices[row]),
                    int(gt_indices[column]),
                    float(distances[row, column]),
                ))
    return matches
