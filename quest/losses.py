from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _available_mask(value: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    if value is None:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    if torch.is_tensor(value):
        mask = value.to(device=device).bool()
        if mask.ndim == 0:
            return mask.expand(batch_size)
        if mask.shape[0] != batch_size:
            raise ValueError(f"availability batch mismatch: {tuple(mask.shape)}")
        return mask.reshape(batch_size, -1).any(dim=1)
    return torch.full((batch_size,), bool(value), dtype=torch.bool, device=device)


def _select_batch(
    values: Mapping[str, torch.Tensor], mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {key: value[mask] for key, value in values.items()}


def _hungarian(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if cost.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=cost.device)
        return empty, empty
    safe = torch.nan_to_num(cost.detach(), nan=1e6, posinf=1e6, neginf=-1e6)
    rows, cols = linear_sum_assignment(safe.cpu().numpy())
    return (
        torch.as_tensor(rows, dtype=torch.long, device=cost.device),
        torch.as_tensor(cols, dtype=torch.long, device=cost.device),
    )


def _focal_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    alpha: float,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    ce = F.nll_loss(log_probs, targets, reduction="none")
    pt = probs.gather(1, targets[:, None]).squeeze(1)
    return (alpha * (1.0 - pt).pow(gamma) * ce).mean()


def compute_seg_loss(
    seg_logits: torch.Tensor,
    seg_gt: torch.Tensor,
    config: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    settings = {"ignore_index": 255}
    settings.update(config or {})
    if seg_gt.ndim != 4 or seg_logits.ndim != 5:
        raise ValueError("seg logits/GT must be [B,N,C,H,W] and [B,N,H,W]")
    if seg_logits.shape[:2] != seg_gt.shape[:2] or seg_logits.shape[-2:] != seg_gt.shape[-2:]:
        raise ValueError(
            f"segmentation shape mismatch: {tuple(seg_logits.shape)} vs {tuple(seg_gt.shape)}"
        )
    logits = seg_logits.flatten(0, 1)
    target = seg_gt.long().flatten(0, 1)
    if not (target != int(settings["ignore_index"])).any():
        return {"seg_loss": _zero(seg_logits)}
    return {
        "seg_loss": F.cross_entropy(
            logits, target, ignore_index=int(settings["ignore_index"])
        )
    }


def compute_depth_loss(
    depth: torch.Tensor,
    depth_gt: torch.Tensor,
    config: Mapping[str, Any] | None = None,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    settings = {"eps": 1e-3, "loss_type": "smooth_l1"}
    settings.update(config or {})
    target = depth_gt.to(device=depth.device, dtype=depth.dtype)
    if target.ndim == 4:
        target = target.unsqueeze(2)
    if target.shape != depth.shape:
        raise ValueError(f"depth shape mismatch: {tuple(depth.shape)} vs {tuple(target.shape)}")
    valid = torch.isfinite(target) & (target > 0)
    if valid_mask is not None:
        supplied = valid_mask.to(device=depth.device).bool()
        if supplied.ndim == 4:
            supplied = supplied.unsqueeze(2)
        if supplied.shape != depth.shape:
            supplied = supplied.expand_as(depth)
        valid &= supplied
    if not valid.any():
        return {"depth_loss": _zero(depth)}
    eps = float(settings["eps"])
    pred_log = torch.log(depth[valid].clamp_min(eps))
    target_log = torch.log(target[valid].clamp_min(eps))
    if settings["loss_type"] == "l1":
        loss = F.l1_loss(pred_log, target_log)
    else:
        loss = F.smooth_l1_loss(pred_log, target_log)
    return {"depth_loss": loss}


def compute_agent_loss(
    cls_logits: torch.Tensor,
    boxes: torch.Tensor,
    velocity: torch.Tensor,
    agent_gt: Mapping[str, torch.Tensor],
    config: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    settings = {
        "cls_cost_weight": 2.0,
        "bbox_cost_weight": 0.25,
        "cls_gamma": 2.0,
        "cls_alpha": 0.25,
        "lambda_box": 0.25,
        "lambda_velocity": 0.2,
        "box_loss_type": "l1",
    }
    settings.update(config or {})
    background = cls_logits.shape[-1] - 1
    cls_losses: list[torch.Tensor] = []
    box_losses: list[torch.Tensor] = []
    velocity_losses: list[torch.Tensor] = []
    for batch_index in range(cls_logits.shape[0]):
        pred_cls = cls_logits[batch_index]
        pred_boxes = boxes[batch_index]
        pred_velocity = velocity[batch_index]
        valid = agent_gt["labels"][batch_index] >= 0
        gt_labels = agent_gt["labels"][batch_index][valid].long()
        gt_boxes = agent_gt["boxes"][batch_index][valid]
        gt_velocity = agent_gt["velocity"][batch_index][valid]
        targets = torch.full(
            (pred_cls.shape[0],), background, dtype=torch.long, device=pred_cls.device
        )
        if gt_labels.numel():
            class_cost = -F.log_softmax(pred_cls, dim=-1)[:, gt_labels]
            box_cost = torch.cdist(pred_boxes, gt_boxes, p=1) / pred_boxes.shape[-1]
            pred_indices, gt_indices = _hungarian(
                float(settings["cls_cost_weight"]) * class_cost
                + float(settings["bbox_cost_weight"]) * box_cost
            )
            targets[pred_indices] = gt_labels[gt_indices]
            regression = F.smooth_l1_loss if settings["box_loss_type"] == "smooth_l1" else F.l1_loss
            box_loss = regression(pred_boxes[pred_indices], gt_boxes[gt_indices])
            velocity_loss = F.smooth_l1_loss(
                pred_velocity[pred_indices], gt_velocity[gt_indices]
            )
        else:
            box_loss = _zero(pred_boxes)
            velocity_loss = _zero(pred_velocity)
        cls_losses.append(
            _focal_classification_loss(
                pred_cls,
                targets,
                float(settings["cls_gamma"]),
                float(settings["cls_alpha"]),
            )
        )
        box_losses.append(box_loss)
        velocity_losses.append(velocity_loss)
    cls_loss = torch.stack(cls_losses).mean()
    box_loss = torch.stack(box_losses).mean()
    velocity_loss = torch.stack(velocity_losses).mean()
    total = (
        cls_loss
        + float(settings["lambda_box"]) * box_loss
        + float(settings["lambda_velocity"]) * velocity_loss
    )
    return {
        "agent_cls_loss": cls_loss,
        "agent_box_loss": box_loss,
        "agent_velocity_loss": velocity_loss,
        "agent_loss": total,
    }


def compute_map_loss(
    cls_logits: torch.Tensor,
    points: torch.Tensor,
    map_gt: Mapping[str, torch.Tensor],
    config: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    settings = {
        "cls_cost_weight": 2.0,
        "pts_cost_weight": 5.0,
        "cls_gamma": 2.0,
        "cls_alpha": 0.25,
        "lambda_cls": 2.0,
        "lambda_pts": 5.0,
        "lambda_dir": 0.005,
    }
    settings.update(config or {})
    background = cls_logits.shape[-1] - 1
    cls_losses: list[torch.Tensor] = []
    point_losses: list[torch.Tensor] = []
    direction_losses: list[torch.Tensor] = []
    for batch_index in range(cls_logits.shape[0]):
        pred_cls = cls_logits[batch_index]
        pred_points = points[batch_index]
        valid = map_gt["labels"][batch_index] >= 0
        gt_labels = map_gt["labels"][batch_index][valid].long()
        gt_points = map_gt["points"][batch_index][valid]
        targets = torch.full(
            (pred_cls.shape[0],), background, dtype=torch.long, device=pred_cls.device
        )
        if gt_labels.numel():
            class_cost = -F.log_softmax(pred_cls, dim=-1)[:, gt_labels]
            point_cost = torch.cdist(
                pred_points.flatten(1), gt_points.flatten(1), p=1
            ) / pred_points[0].numel()
            pred_indices, gt_indices = _hungarian(
                float(settings["cls_cost_weight"]) * class_cost
                + float(settings["pts_cost_weight"]) * point_cost
            )
            targets[pred_indices] = gt_labels[gt_indices]
            matched_pred = pred_points[pred_indices]
            matched_gt = gt_points[gt_indices]
            point_loss = F.l1_loss(matched_pred, matched_gt)
            pred_direction = F.normalize(
                matched_pred[:, 1:] - matched_pred[:, :-1], dim=-1
            )
            gt_direction = F.normalize(
                matched_gt[:, 1:] - matched_gt[:, :-1], dim=-1
            )
            direction_loss = (1.0 - (pred_direction * gt_direction).sum(-1)).mean()
        else:
            point_loss = _zero(pred_points)
            direction_loss = _zero(pred_points)
        cls_losses.append(
            _focal_classification_loss(
                pred_cls,
                targets,
                float(settings["cls_gamma"]),
                float(settings["cls_alpha"]),
            )
        )
        point_losses.append(point_loss)
        direction_losses.append(direction_loss)
    cls_loss = torch.stack(cls_losses).mean()
    point_loss = torch.stack(point_losses).mean()
    direction_loss = torch.stack(direction_losses).mean()
    total = (
        float(settings["lambda_cls"]) * cls_loss
        + float(settings["lambda_pts"]) * point_loss
        + float(settings["lambda_dir"]) * direction_loss
    )
    return {
        "map_cls_loss": cls_loss,
        "map_pts_loss": point_loss,
        "map_dir_loss": direction_loss,
        "map_loss": total,
    }


def compute_total_loss(
    preds: Mapping[str, torch.Tensor],
    gts: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    settings: dict[str, Any] = {
        "tasks": {"seg": False, "depth": False, "agent": True, "map": False},
        "task_weights": {"seg": 1.0, "depth": 1.0, "agent": 2.0, "map": 1.5},
        "seg": {},
        "depth": {},
        "agent": {},
        "map": {},
    }
    for key, value in (config or {}).items():
        if isinstance(value, Mapping) and key in settings:
            settings[key] = {**settings[key], **value}
        else:
            settings[key] = value
    batch_size = preds["agent_cls_logits"].shape[0]
    device = preds["agent_cls_logits"].device
    tasks = settings["tasks"]

    seg_available = _available_mask(gts.get("seg_valid"), batch_size, device)
    if tasks.get("seg") and seg_available.any() and "seg_gt" in gts:
        seg_losses = compute_seg_loss(
            preds["seg_logits"][seg_available],
            gts["seg_gt"][seg_available],
            settings["seg"],
        )
    else:
        seg_losses = {"seg_loss": _zero(preds["seg_logits"])}

    depth_available = _available_mask(gts.get("depth_valid"), batch_size, device)
    if tasks.get("depth") and depth_available.any() and "depth_gt" in gts:
        depth_mask = gts.get("depth_mask")
        if torch.is_tensor(depth_mask) and depth_mask.shape[0] == batch_size:
            depth_mask = depth_mask[depth_available]
        depth_losses = compute_depth_loss(
            preds["depth"][depth_available],
            gts["depth_gt"][depth_available],
            settings["depth"],
            depth_mask,
        )
    else:
        depth_losses = {"depth_loss": _zero(preds["depth"])}

    agent_available = _available_mask(gts.get("agent_valid"), batch_size, device)
    if tasks.get("agent") and agent_available.any() and "agent_gt" in gts:
        agent_losses = compute_agent_loss(
            preds["agent_cls_logits"][agent_available],
            preds["agent_boxes"][agent_available],
            preds["agent_velocity"][agent_available],
            _select_batch(gts["agent_gt"], agent_available),
            settings["agent"],
        )
    else:
        agent_losses = {
            "agent_cls_loss": _zero(preds["agent_cls_logits"]),
            "agent_box_loss": _zero(preds["agent_boxes"]),
            "agent_velocity_loss": _zero(preds["agent_velocity"]),
            "agent_loss": _zero(preds["agent_cls_logits"]),
        }

    map_available = _available_mask(gts.get("map_valid"), batch_size, device)
    if tasks.get("map") and map_available.any() and "map_gt" in gts:
        map_losses = compute_map_loss(
            preds["map_cls_logits"][map_available],
            preds["map_points"][map_available],
            _select_batch(gts["map_gt"], map_available),
            settings["map"],
        )
    else:
        map_losses = {
            "map_cls_loss": _zero(preds["map_cls_logits"]),
            "map_pts_loss": _zero(preds["map_points"]),
            "map_dir_loss": _zero(preds["map_points"]),
            "map_loss": _zero(preds["map_cls_logits"]),
        }

    losses = {**seg_losses, **depth_losses, **agent_losses, **map_losses}
    weights = settings["task_weights"]
    losses["total_loss"] = (
        float(weights.get("seg", 0.0)) * losses["seg_loss"]
        + float(weights.get("depth", 0.0)) * losses["depth_loss"]
        + float(weights.get("agent", 0.0)) * losses["agent_loss"]
        + float(weights.get("map", 0.0)) * losses["map_loss"]
    )
    return losses
