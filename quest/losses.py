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


def _decode_agent_boxes(
    boxes: torch.Tensor, settings: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xy_min, xy_max = (float(value) for value in settings["xy_range"])
    z_min, z_max = (float(value) for value in settings["z_range"])
    center = torch.stack(
        (
            xy_min + boxes[..., 0] * (xy_max - xy_min),
            xy_min + boxes[..., 1] * (xy_max - xy_min),
            z_min + boxes[..., 2] * (z_max - z_min),
        ),
        dim=-1,
    )
    size_scale = boxes.new_tensor(settings["size_norm"])
    size = boxes[..., 3:6] * size_scale
    yaw = F.normalize(boxes[..., 6:8], dim=-1)
    return center, size, yaw


def _supported_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_support_mask: torch.Tensor,
    background_weight: float,
) -> torch.Tensor:
    foreground = torch.nonzero(class_support_mask.bool(), as_tuple=False).flatten()
    background = logits.shape[-1] - 1
    columns = torch.cat((foreground, foreground.new_tensor([background])))
    reduced_logits = logits[:, columns]
    remapped = torch.full_like(targets, foreground.numel())
    for reduced_index, class_index in enumerate(foreground.tolist()):
        remapped[targets == class_index] = reduced_index
    invalid = (targets != background) & (remapped == foreground.numel())
    if bool(invalid.any()):
        raise ValueError("Agent target uses a class unsupported by its teacher")
    weights = logits.new_ones(foreground.numel() + 1)
    weights[-1] = float(background_weight)
    return F.cross_entropy(reduced_logits, remapped, weight=weights)


def _match_agent_layer(
    cls_logits: torch.Tensor,
    boxes: torch.Tensor,
    velocity: torch.Tensor,
    target: Mapping[str, torch.Tensor],
    settings: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    center_scale = boxes.new_tensor((10.0, 10.0, 2.0))
    size_scale = boxes.new_tensor(settings["size_norm"])
    background = cls_logits.shape[-1] - 1
    component_losses: dict[str, list[torch.Tensor]] = {
        "cls": [], "center": [], "size": [], "yaw": [], "velocity": []
    }
    for batch_index in range(cls_logits.shape[0]):
        pred_cls = cls_logits[batch_index]
        pred_center, pred_size, pred_yaw = _decode_agent_boxes(
            boxes[batch_index], settings
        )
        pred_velocity = velocity[batch_index]
        valid = target["valid_mask"][batch_index].bool()
        gt_labels = target["labels"][batch_index][valid].long()
        gt_boxes = target["boxes_metric"][batch_index][valid].to(boxes.dtype)
        gt_velocity = target["velocity_mps"][batch_index][valid].to(velocity.dtype)
        gt_scores = target["scores"][batch_index][valid].to(boxes.dtype)
        support = target["class_support_mask"][batch_index].bool()
        cls_targets = torch.full(
            (pred_cls.shape[0],), background, dtype=torch.long, device=pred_cls.device
        )
        if gt_labels.numel():
            if not bool(support[gt_labels].all()):
                raise ValueError("valid Agent targets include unsupported classes")
            supported = torch.nonzero(support, as_tuple=False).flatten()
            columns = torch.cat((supported, supported.new_tensor([background])))
            class_log_probs = F.log_softmax(pred_cls[:, columns], dim=-1)
            class_lookup = torch.full(
                (background,), -1, dtype=torch.long, device=pred_cls.device
            )
            class_lookup[supported] = torch.arange(
                supported.numel(), device=pred_cls.device
            )
            class_cost = -class_log_probs[:, class_lookup[gt_labels]]
            gt_center = gt_boxes[:, :3]
            gt_size = gt_boxes[:, 3:6]
            gt_yaw = torch.stack(
                (torch.sin(gt_boxes[:, 6]), torch.cos(gt_boxes[:, 6])), dim=-1
            )
            center_cost = torch.cdist(
                pred_center / center_scale, gt_center / center_scale, p=1
            )
            size_cost = torch.cdist(
                pred_size / size_scale, gt_size / size_scale, p=1
            )
            yaw_cost = 1.0 - pred_yaw @ gt_yaw.transpose(0, 1)
            pred_indices, gt_indices = _hungarian(
                float(settings["match_cls_weight"]) * class_cost
                + float(settings["match_center_weight"]) * center_cost
                + float(settings["match_size_weight"]) * size_cost
                + float(settings["match_yaw_weight"]) * yaw_cost
            )
            cls_targets[pred_indices] = gt_labels[gt_indices]
            matched_scores = gt_scores[gt_indices].clamp(0.0, 1.0).clamp_min(1e-3)

            def reduce_regression(per_match: torch.Tensor) -> torch.Tensor:
                if settings.get("teacher_confidence_weighting", False):
                    return (per_match * matched_scores).sum() / matched_scores.sum()
                return per_match.mean()

            component_losses["center"].append(
                reduce_regression(F.smooth_l1_loss(
                    pred_center[pred_indices] / center_scale,
                    gt_center[gt_indices] / center_scale,
                    reduction="none",
                ).mean(dim=-1))
            )
            component_losses["size"].append(
                reduce_regression(F.smooth_l1_loss(
                    pred_size[pred_indices] / size_scale,
                    gt_size[gt_indices] / size_scale,
                    reduction="none",
                ).mean(dim=-1))
            )
            component_losses["yaw"].append(
                reduce_regression(
                    1.0 - (pred_yaw[pred_indices] * gt_yaw[gt_indices]).sum(-1)
                )
            )
            component_losses["velocity"].append(
                reduce_regression(F.smooth_l1_loss(
                    pred_velocity[pred_indices],
                    gt_velocity[gt_indices],
                    reduction="none",
                ).mean(dim=-1))
            )
        else:
            for key in ("center", "size", "yaw", "velocity"):
                component_losses[key].append(_zero(boxes[batch_index]))
        component_losses["cls"].append(
            _supported_cross_entropy(
                pred_cls,
                cls_targets,
                support,
                float(settings["background_weight"]),
            )
        )
    return {
        key: torch.stack(values).mean() for key, values in component_losses.items()
    }


def _proposal_targets(
    objectness: torch.Tensor,
    offsets: torch.Tensor,
    target: Mapping[str, torch.Tensor],
    settings: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, cell_count = objectness.shape
    bev_h, bev_w = int(settings["bev_h"]), int(settings["bev_w"])
    if cell_count != bev_h * bev_w:
        raise ValueError("proposal objectness does not match configured BEV grid")
    xy_min, xy_max = (float(value) for value in settings["xy_range"])
    object_target = torch.zeros_like(objectness)
    offset_target = torch.zeros_like(offsets)
    positive_mask = torch.zeros_like(objectness, dtype=torch.bool)
    positive_weight = torch.zeros_like(objectness)
    for batch_index in range(batch_size):
        valid = target["valid_mask"][batch_index].bool()
        centers = target["boxes_metric"][batch_index, valid, :2].to(objectness.dtype)
        scores = target["scores"][batch_index, valid].to(objectness.dtype).clamp(0.0, 1.0)
        if not centers.numel():
            continue
        normalized = ((centers - xy_min) / (xy_max - xy_min)).clamp(0, 1 - 1e-6)
        cell_x = torch.floor(normalized[:, 0] * bev_w).long()
        cell_y = torch.floor(normalized[:, 1] * bev_h).long()
        flat = cell_y * bev_w + cell_x
        desired = torch.stack(
            (
                normalized[:, 0] * bev_w - (cell_x.to(objectness.dtype) + 0.5),
                normalized[:, 1] * bev_h - (cell_y.to(objectness.dtype) + 0.5),
            ),
            dim=-1,
        ).clamp(-0.5, 0.5)
        for item_index, flat_index in enumerate(flat.tolist()):
            if not positive_mask[batch_index, flat_index] or scores[item_index] > positive_weight[batch_index, flat_index]:
                positive_mask[batch_index, flat_index] = True
                positive_weight[batch_index, flat_index] = scores[item_index]
                object_target[batch_index, flat_index] = scores[item_index]
                offset_target[batch_index, flat_index] = desired[item_index]
    object_loss = F.binary_cross_entropy_with_logits(objectness, object_target)
    if positive_mask.any():
        raw_offset = F.smooth_l1_loss(
            offsets[positive_mask], offset_target[positive_mask], reduction="none"
        ).mean(dim=-1)
        offset_loss = (
            raw_offset * positive_weight[positive_mask].clamp_min(1e-3)
        ).sum() / positive_weight[positive_mask].clamp_min(1e-3).sum()
    else:
        offset_loss = _zero(offsets)
    return object_loss, offset_loss


def compute_agent_loss(
    predictions: Mapping[str, torch.Tensor],
    agent_gt: Mapping[str, torch.Tensor],
    config: Mapping[str, Any] | None = None,
    decoder_enabled: bool = True,
) -> dict[str, torch.Tensor]:
    settings: dict[str, Any] = {
        "xy_range": (-50.0, 50.0),
        "z_range": (-5.0, 5.0),
        "size_norm": (20.0, 10.0, 8.0),
        "bev_h": 32,
        "bev_w": 32,
        "match_cls_weight": 1.0,
        "match_center_weight": 5.0,
        "match_size_weight": 2.0,
        "match_yaw_weight": 1.0,
        "background_weight": 0.1,
        "lambda_cls": 1.0,
        "lambda_center": 5.0,
        "lambda_size": 2.0,
        "lambda_yaw": 1.0,
        "lambda_velocity": 0.5,
        "lambda_proposal_objectness": 2.0,
        "lambda_proposal_offset": 1.0,
        "aux_layer_weights": (0.25, 0.5, 0.75, 1.0),
    }
    settings.update(config or {})
    objectness_loss, offset_loss = _proposal_targets(
        predictions["proposal_objectness_logits"],
        predictions["proposal_xy_offsets"],
        agent_gt,
        settings,
    )
    proposal_loss = (
        float(settings["lambda_proposal_objectness"]) * objectness_loss
        + float(settings["lambda_proposal_offset"]) * offset_loss
    )
    cls_layers = predictions.get(
        "agent_cls_logits_layers", predictions["agent_cls_logits"].unsqueeze(0)
    )
    box_layers = predictions.get(
        "agent_boxes_layers", predictions["agent_boxes"].unsqueeze(0)
    )
    velocity_layers = predictions.get(
        "agent_velocity_layers", predictions["agent_velocity"].unsqueeze(0)
    )
    weights = tuple(float(value) for value in settings["aux_layer_weights"])
    if len(weights) != cls_layers.shape[0]:
        if cls_layers.shape[0] == 1:
            weights = (1.0,)
        else:
            raise ValueError("aux_layer_weights must match Agent decoder layers")
    accumulated = {
        key: _zero(cls_layers) for key in ("cls", "center", "size", "yaw", "velocity")
    }
    normalizer = max(sum(weights), 1e-6)
    if decoder_enabled:
        for layer_index, layer_weight in enumerate(weights):
            layer_losses = _match_agent_layer(
                cls_layers[layer_index],
                box_layers[layer_index],
                velocity_layers[layer_index],
                agent_gt,
                settings,
            )
            for key in accumulated:
                accumulated[key] = accumulated[key] + layer_weight * layer_losses[key]
        accumulated = {key: value / normalizer for key, value in accumulated.items()}
    decoder_loss = (
        float(settings["lambda_cls"]) * accumulated["cls"]
        + float(settings["lambda_center"]) * accumulated["center"]
        + float(settings["lambda_size"]) * accumulated["size"]
        + float(settings["lambda_yaw"]) * accumulated["yaw"]
        + float(settings["lambda_velocity"]) * accumulated["velocity"]
    )
    return {
        "proposal_objectness_loss": objectness_loss,
        "proposal_offset_loss": offset_loss,
        "proposal_loss": proposal_loss,
        "agent_cls_loss": accumulated["cls"],
        "agent_center_loss": accumulated["center"],
        "agent_size_loss": accumulated["size"],
        "agent_yaw_loss": accumulated["yaw"],
        "agent_velocity_loss": accumulated["velocity"],
        "decoder_agent_loss": decoder_loss,
        "agent_loss": proposal_loss + decoder_loss,
    }


def compute_agent_decoder_loss(
    predictions: Mapping[str, torch.Tensor],
    agent_gt: Mapping[str, torch.Tensor],
    settings: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Match and supervise every Agent decoder layer without proposal BCE."""

    cls_layers = predictions["agent_cls_logits_layers"]
    box_layers = predictions["agent_boxes_layers"]
    velocity_layers = predictions["agent_velocity_layers"]
    weights = tuple(float(value) for value in settings["aux_layer_weights"])
    if len(weights) != cls_layers.shape[0] or sum(weights) <= 0:
        raise ValueError("positive aux_layer_weights must match Agent decoder layers")
    if box_layers.shape[:2] != cls_layers.shape[:2] or velocity_layers.shape[:2] != cls_layers.shape[:2]:
        raise ValueError("Agent decoder layer shapes do not match")

    accumulated = {
        key: cls_layers.sum() * 0.0
        for key in ("cls", "center", "size", "yaw", "velocity")
    }
    for layer_index, layer_weight in enumerate(weights):
        layer_losses = _match_agent_layer(
            cls_layers[layer_index],
            box_layers[layer_index],
            velocity_layers[layer_index],
            agent_gt,
            settings,
        )
        for key in accumulated:
            accumulated[key] = accumulated[key] + layer_weight * layer_losses[key]
    accumulated = {key: value / sum(weights) for key, value in accumulated.items()}
    total = (
        float(settings["lambda_cls"]) * accumulated["cls"]
        + float(settings["lambda_center"]) * accumulated["center"]
        + float(settings["lambda_size"]) * accumulated["size"]
        + float(settings["lambda_yaw"]) * accumulated["yaw"]
        + float(settings["lambda_velocity"]) * accumulated["velocity"]
    )
    return {
        "agent_cls_loss": accumulated["cls"],
        "agent_center_loss": accumulated["center"],
        "agent_size_loss": accumulated["size"],
        "agent_yaw_loss": accumulated["yaw"],
        "agent_velocity_loss": accumulated["velocity"],
        "decoder_agent_loss": total,
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
    agent_decoder_enabled: bool = True,
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
            {
                key: value[:, agent_available] if key.endswith("_layers") else value[agent_available]
                for key, value in preds.items()
                if key.startswith("agent_") or key.startswith("proposal_")
            },
            _select_batch(gts["agent_gt"], agent_available),
            settings["agent"],
            decoder_enabled=agent_decoder_enabled,
        )
    else:
        agent_losses = {
            "proposal_objectness_loss": _zero(preds["proposal_objectness_logits"]),
            "proposal_offset_loss": _zero(preds["proposal_xy_offsets"]),
            "proposal_loss": _zero(preds["proposal_objectness_logits"]),
            "agent_cls_loss": _zero(preds["agent_cls_logits"]),
            "agent_center_loss": _zero(preds["agent_boxes"]),
            "agent_size_loss": _zero(preds["agent_boxes"]),
            "agent_yaw_loss": _zero(preds["agent_boxes"]),
            "agent_velocity_loss": _zero(preds["agent_velocity"]),
            "decoder_agent_loss": _zero(preds["agent_cls_logits"]),
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
