from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _as_tensor_or_none(
    values: list[float] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if values is None:
        return None
    return torch.tensor(values, device=device, dtype=dtype)


def _softmax_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    class_weight: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    valid_mask = targets != ignore_index
    if not valid_mask.any():
        return _zero_like(logits)

    valid_logits = logits[valid_mask]
    valid_targets = targets[valid_mask]
    log_probs = F.log_softmax(valid_logits, dim=-1)
    probs = log_probs.exp()
    ce_loss = F.nll_loss(
        log_probs,
        valid_targets,
        reduction="none",
        weight=class_weight,
    )
    pt = probs.gather(dim=1, index=valid_targets.unsqueeze(1)).squeeze(1)
    focal_weight = (1.0 - pt).pow(gamma)
    if alpha is not None:
        focal_weight = focal_weight * alpha
    return (focal_weight * ce_loss).mean()


def _classification_cost(logits: torch.Tensor, gt_labels: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1).clamp_min(1e-8)
    return -torch.log(probs[:, gt_labels])


def _l1_cost(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0 or target.numel() == 0:
        return pred.new_zeros((pred.shape[0], target.shape[0]))
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    return torch.cdist(pred_flat, target_flat, p=1) / pred_flat.shape[-1]


def _hungarian_match(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_pred, num_gt = cost.shape
    if num_pred == 0 or num_gt == 0:
        empty = cost.new_zeros((0,), dtype=torch.long)
        return empty, empty
    safe_cost = torch.nan_to_num(
        cost.detach(),
        nan=1e6,
        posinf=1e6,
        neginf=-1e6,
    ).cpu()
    row_ind, col_ind = linear_sum_assignment(safe_cost.numpy())
    return (
        torch.as_tensor(row_ind, device=cost.device, dtype=torch.long),
        torch.as_tensor(col_ind, device=cost.device, dtype=torch.long),
    )


def compute_seg_loss(
    seg_logits: torch.Tensor,
    seg_gt: torch.Tensor,
    config: Mapping[str, Any] | None = None,
) -> Dict[str, torch.Tensor]:
    seg_config = {
        "ignore_index": 255,
        "class_weight": None,
    }
    if config is not None:
        seg_config.update(config)

    class_weight = _as_tensor_or_none(
        seg_config.get("class_weight"),
        device=seg_logits.device,
        dtype=seg_logits.dtype,
    )
    seg_loss = F.cross_entropy(
        seg_logits,
        seg_gt.long(),
        weight=class_weight,
        ignore_index=int(seg_config["ignore_index"]),
    )
    return {"seg_loss": seg_loss}


def compute_agent_loss(
    agent_cls_logits: torch.Tensor,
    agent_boxes: torch.Tensor,
    agent_gt: Mapping[str, torch.Tensor],
    config: Mapping[str, Any] | None = None,
) -> Dict[str, torch.Tensor]:
    agent_config = {
        "cls_cost_weight": 2.0,
        "bbox_cost_weight": 0.25,
        "cls_gamma": 2.0,
        "cls_alpha": 0.25,
        "class_weight": None,
        "lambda_box": 0.25,
        "lambda_dn": 0.0,
        "box_loss_type": "l1",
        "dn_enabled": False,
    }
    if config is not None:
        agent_config.update(config)

    batch_size, num_queries, num_classes_with_bg = agent_cls_logits.shape
    no_object_index = num_classes_with_bg - 1
    class_weight = _as_tensor_or_none(
        agent_config.get("class_weight"),
        device=agent_cls_logits.device,
        dtype=agent_cls_logits.dtype,
    )

    cls_losses: list[torch.Tensor] = []
    box_losses: list[torch.Tensor] = []

    for batch_index in range(batch_size):
        pred_cls = agent_cls_logits[batch_index]
        pred_boxes = agent_boxes[batch_index]
        gt_labels = agent_gt["labels"][batch_index]
        gt_boxes = agent_gt["boxes"][batch_index]
        valid_mask = gt_labels >= 0
        valid_labels = gt_labels[valid_mask]
        valid_boxes = gt_boxes[valid_mask]

        target_labels = torch.full(
            (num_queries,),
            no_object_index,
            device=pred_cls.device,
            dtype=torch.long,
        )

        if valid_labels.numel() > 0:
            cls_cost = _classification_cost(pred_cls, valid_labels)
            reg_cost = _l1_cost(pred_boxes, valid_boxes)
            total_cost = (
                float(agent_config["cls_cost_weight"]) * cls_cost
                + float(agent_config["bbox_cost_weight"]) * reg_cost
            )
            matched_pred, matched_gt = _hungarian_match(total_cost)
            if matched_pred.numel() > 0:
                target_labels[matched_pred] = valid_labels[matched_gt]
                matched_pred_boxes = pred_boxes[matched_pred]
                matched_gt_boxes = valid_boxes[matched_gt]
                if agent_config["box_loss_type"] == "smooth_l1":
                    box_loss = F.smooth_l1_loss(
                        matched_pred_boxes,
                        matched_gt_boxes,
                        reduction="mean",
                    )
                else:
                    box_loss = F.l1_loss(
                        matched_pred_boxes,
                        matched_gt_boxes,
                        reduction="mean",
                    )
            else:
                box_loss = _zero_like(pred_boxes)
        else:
            box_loss = _zero_like(pred_boxes)

        cls_loss = _softmax_focal_loss(
            pred_cls,
            target_labels,
            gamma=float(agent_config["cls_gamma"]),
            alpha=float(agent_config["cls_alpha"]),
            class_weight=class_weight,
        )
        cls_losses.append(cls_loss)
        box_losses.append(box_loss)

    agent_cls_loss = torch.stack(cls_losses).mean() if cls_losses else _zero_like(agent_cls_logits)
    agent_box_loss = torch.stack(box_losses).mean() if box_losses else _zero_like(agent_boxes)
    agent_dn_loss = _zero_like(agent_cls_logits)
    agent_loss = (
        agent_cls_loss
        + float(agent_config["lambda_box"]) * agent_box_loss
        + float(agent_config["lambda_dn"]) * agent_dn_loss
    )
    return {
        "agent_cls_loss": agent_cls_loss,
        "agent_box_loss": agent_box_loss,
        "agent_dn_loss": agent_dn_loss,
        "agent_loss": agent_loss,
    }


def _direction_loss(pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    if pred_points.shape[1] < 2:
        return _zero_like(pred_points)
    pred_dir = pred_points[:, 1:, :] - pred_points[:, :-1, :]
    target_dir = target_points[:, 1:, :] - target_points[:, :-1, :]
    pred_dir = F.normalize(pred_dir, dim=-1)
    target_dir = F.normalize(target_dir, dim=-1)
    cosine = (pred_dir * target_dir).sum(dim=-1)
    return (1.0 - cosine).mean()


def compute_map_loss(
    map_cls_logits: torch.Tensor,
    map_points: torch.Tensor,
    map_gt: Mapping[str, torch.Tensor],
    config: Mapping[str, Any] | None = None,
) -> Dict[str, torch.Tensor]:
    map_config = {
        "cls_cost_weight": 2.0,
        "pts_cost_weight": 5.0,
        "cls_gamma": 2.0,
        "cls_alpha": 0.25,
        "class_weight": None,
        "lambda_cls": 2.0,
        "lambda_pts": 5.0,
        "lambda_dir": 0.005,
    }
    if config is not None:
        map_config.update(config)

    batch_size, num_queries, num_classes_with_bg = map_cls_logits.shape
    no_object_index = num_classes_with_bg - 1
    class_weight = _as_tensor_or_none(
        map_config.get("class_weight"),
        device=map_cls_logits.device,
        dtype=map_cls_logits.dtype,
    )

    cls_losses: list[torch.Tensor] = []
    pts_losses: list[torch.Tensor] = []
    dir_losses: list[torch.Tensor] = []

    for batch_index in range(batch_size):
        pred_cls = map_cls_logits[batch_index]
        pred_points = map_points[batch_index]
        gt_labels = map_gt["labels"][batch_index]
        gt_points = map_gt["points"][batch_index]
        valid_mask = gt_labels >= 0
        valid_labels = gt_labels[valid_mask]
        valid_points = gt_points[valid_mask]

        target_labels = torch.full(
            (num_queries,),
            no_object_index,
            device=pred_cls.device,
            dtype=torch.long,
        )

        if valid_labels.numel() > 0:
            cls_cost = _classification_cost(pred_cls, valid_labels)
            pts_cost = _l1_cost(pred_points, valid_points)
            total_cost = (
                float(map_config["cls_cost_weight"]) * cls_cost
                + float(map_config["pts_cost_weight"]) * pts_cost
            )
            matched_pred, matched_gt = _hungarian_match(total_cost)
            if matched_pred.numel() > 0:
                target_labels[matched_pred] = valid_labels[matched_gt]
                matched_pred_points = pred_points[matched_pred]
                matched_gt_points = valid_points[matched_gt]
                pts_loss = F.l1_loss(
                    matched_pred_points,
                    matched_gt_points,
                    reduction="mean",
                )
                dir_loss = _direction_loss(matched_pred_points, matched_gt_points)
            else:
                pts_loss = _zero_like(pred_points)
                dir_loss = _zero_like(pred_points)
        else:
            pts_loss = _zero_like(pred_points)
            dir_loss = _zero_like(pred_points)

        cls_loss = _softmax_focal_loss(
            pred_cls,
            target_labels,
            gamma=float(map_config["cls_gamma"]),
            alpha=float(map_config["cls_alpha"]),
            class_weight=class_weight,
        )
        cls_losses.append(cls_loss)
        pts_losses.append(pts_loss)
        dir_losses.append(dir_loss)

    map_cls_loss = torch.stack(cls_losses).mean() if cls_losses else _zero_like(map_cls_logits)
    map_pts_loss = torch.stack(pts_losses).mean() if pts_losses else _zero_like(map_points)
    map_dir_loss = torch.stack(dir_losses).mean() if dir_losses else _zero_like(map_points)
    map_loss = (
        float(map_config["lambda_cls"]) * map_cls_loss
        + float(map_config["lambda_pts"]) * map_pts_loss
        + float(map_config["lambda_dir"]) * map_dir_loss
    )
    return {
        "map_cls_loss": map_cls_loss,
        "map_pts_loss": map_pts_loss,
        "map_dir_loss": map_dir_loss,
        "map_loss": map_loss,
    }


def compute_occ_loss(
    occ_logits: torch.Tensor,
    occ_gt: torch.Tensor,
    config: Mapping[str, Any] | None = None,
    mask_camera: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    occ_config = {
        "class_weight": None,
        "use_camera_mask": False,
        "lambda_sem": 0.0,
        "lambda_geo": 0.0,
        "lambda_lovasz": 0.0,
        "enable_sem_scal_loss": False,
        "enable_geo_scal_loss": False,
        "enable_lovasz_loss": False,
    }
    if config is not None:
        occ_config.update(config)

    C_occ = occ_logits.shape[1]
    class_weight = _as_tensor_or_none(
        occ_config.get("class_weight"),
        device=occ_logits.device,
        dtype=occ_logits.dtype,
    )

    if C_occ == 1:
        target = occ_gt.float()
        logits = occ_logits.squeeze(1)
        if bool(occ_config["use_camera_mask"]) and mask_camera is not None:
            valid = mask_camera.bool()
            logits = logits[valid]
            target = target[valid]
        if logits.numel() == 0:
            occ_main_loss = _zero_like(occ_logits)
        else:
            occ_main_loss = F.binary_cross_entropy_with_logits(logits, target)
    else:
        target = occ_gt.long()
        if bool(occ_config["use_camera_mask"]) and mask_camera is not None:
            valid = mask_camera.bool().view(-1)
            logits = occ_logits.permute(0, 2, 3, 4, 1).reshape(-1, C_occ)[valid]
            target = target.reshape(-1)[valid]
            if logits.numel() == 0:
                occ_main_loss = _zero_like(occ_logits)
            else:
                occ_main_loss = F.cross_entropy(logits, target, weight=class_weight)
        else:
            occ_main_loss = F.cross_entropy(occ_logits, target, weight=class_weight)

    occ_sem_scal_loss = _zero_like(occ_logits)
    occ_geo_scal_loss = _zero_like(occ_logits)
    occ_lovasz_loss = _zero_like(occ_logits)
    occ_loss = (
        occ_main_loss
        + float(occ_config["lambda_sem"]) * occ_sem_scal_loss
        + float(occ_config["lambda_geo"]) * occ_geo_scal_loss
        + float(occ_config["lambda_lovasz"]) * occ_lovasz_loss
    )
    return {
        "occ_main_loss": occ_main_loss,
        "occ_sem_scal_loss": occ_sem_scal_loss,
        "occ_geo_scal_loss": occ_geo_scal_loss,
        "occ_lovasz_loss": occ_lovasz_loss,
        "occ_loss": occ_loss,
    }


def compute_total_loss(
    preds: Mapping[str, torch.Tensor],
    gts: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> Dict[str, torch.Tensor]:
    total_config = {
        "task_weights": {
            "seg": 1.0,
            "agent": 2.0,
            "map": 1.5,
            "occ": 1.0,
        },
        "seg": {},
        "agent": {},
        "map": {},
        "occ": {},
    }
    if config is not None:
        for key, value in config.items():
            if isinstance(value, dict) and key in total_config:
                merged = dict(total_config[key])
                merged.update(value)
                total_config[key] = merged
            else:
                total_config[key] = value

    seg_losses = compute_seg_loss(
        preds["seg_logits"],
        gts["seg_gt"],
        total_config["seg"],
    )
    agent_losses = compute_agent_loss(
        preds["agent_cls_logits"],
        preds["agent_boxes"],
        gts["agent_gt"],
        total_config["agent"],
    )
    map_losses = compute_map_loss(
        preds["map_cls_logits"],
        preds["map_points"],
        gts["map_gt"],
        total_config["map"],
    )
    occ_losses = compute_occ_loss(
        preds["occ_logits"],
        gts["occ_gt"],
        total_config["occ"],
        mask_camera=gts.get("mask_camera"),
    )

    task_weights = total_config["task_weights"]
    total_loss = (
        float(task_weights["seg"]) * seg_losses["seg_loss"]
        + float(task_weights["agent"]) * agent_losses["agent_loss"]
        + float(task_weights["map"]) * map_losses["map_loss"]
        + float(task_weights["occ"]) * occ_losses["occ_loss"]
    )

    output = {
        **seg_losses,
        **agent_losses,
        **map_losses,
        **occ_losses,
        "total_loss": total_loss,
    }
    return output
