from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .agent_training import compute_stage2_agent_loss, load_agent_stage2_checkpoint
from .map_teacher import (
    TEACHER_ALIGNMENT_VERSION, TEACHER_COORDINATE_FRAME, TEACHER_MAP_SCHEMA_VERSION,
    MapRasterDistillHead, soft_map_distillation_loss,
)
from .model import QUEST_ARCHITECTURE_VERSION
from .nuplan_relation_audit import BASELINE_RELATION_AUDIT_VERSION, validate_relation_audit
from .vector_map_labels import (
    COORDINATE_FRAME, MAP_CLASS_NAMES, MAP_HEIGHT_REFERENCE, VECTOR_GT_SCHEMA_VERSION,
    VECTOR_SEMANTICS_VERSION, require_lidar2global,
)


STAGE3_NAME = "map_hybrid_distillation"
VECTOR_PROVENANCE_KEYS = (
    "num_points", "min_length_m", "map_version", "vector_semantics_version",
    "map_height_reference",
    "map_cast_audit_version",
)
STAGE3_MODULES = (
    "geometry_lift", "bev_encoder", "agent_proposal_head", "agent_decoder",
    "agent_head", "map_decoder", "map_head",
)


def validate_vector_record(record: Mapping[str, Any], token: str, sample_index: int,
                           xy_range: tuple[float, float, float, float], *,
                           expected_min_length_m: float | None = None,
                           expected_map_version: str | None = None,
                           expected_info: Mapping[str, Any] | None = None,
                           expected_map_location: str | None = None) -> None:
    if record.get("schema_version") != VECTOR_GT_SCHEMA_VERSION:
        raise ValueError("nuPlan vector GT schema mismatch; regenerate old vector labels")
    if record.get("vector_semantics_version") != VECTOR_SEMANTICS_VERSION:
        raise ValueError("nuPlan vector GT geometry semantics mismatch; regenerate vector labels")
    if record.get("map_cast_audit_version") != BASELINE_RELATION_AUDIT_VERSION:
        raise ValueError("nuPlan vector GT invalid-cast audit version mismatch; regenerate labels")
    audit = record.get("map_cast_diagnostics")
    if not isinstance(audit, Mapping):
        raise ValueError("nuPlan vector GT has no baseline relation audit")
    validate_relation_audit(audit)
    if record.get("num_points") != 20:
        raise ValueError("nuPlan vector GT num_points must be 20")
    minimum = record.get("min_length_m")
    if not isinstance(minimum, (float, int)) or not math.isfinite(minimum) or minimum < 0:
        raise ValueError("nuPlan vector GT min_length_m is invalid")
    version = record.get("map_version")
    if not isinstance(version, str) or not version:
        raise ValueError("nuPlan vector GT map_version is missing")
    if expected_min_length_m is not None and minimum != expected_min_length_m:
        raise ValueError("nuPlan vector GT min_length_m differs from this export")
    if expected_map_version is not None and version != expected_map_version:
        raise ValueError("nuPlan vector GT map_version differs from this export")
    if record.get("map_height_reference") != MAP_HEIGHT_REFERENCE:
        raise ValueError("nuPlan vector GT map height convention mismatch; regenerate labels")
    reference_z = record.get("map_reference_global_z_m")
    if not isinstance(reference_z, (float, int)) or not math.isfinite(reference_z):
        raise ValueError("nuPlan vector GT global reference height is invalid")
    if not isinstance(record.get("map_location"), str) or not record["map_location"]:
        raise ValueError("nuPlan vector GT resolved map_location is missing")
    if expected_map_location is not None and record["map_location"] != expected_map_location:
        raise ValueError("nuPlan vector GT map_location differs from projected city match")
    if expected_info is not None:
        if str(record.get("scene_token")) != str(expected_info.get("scene_token")):
            raise ValueError("nuPlan vector GT scene_token differs from metadata")
        current_z = float(require_lidar2global(expected_info)[2, 3])
        if not math.isclose(reference_z, current_z, rel_tol=0, abs_tol=1e-4):
            raise ValueError("nuPlan vector GT reference height differs from metadata")
    if str(record.get("token")) != token or record.get("sample_index") != sample_index:
        raise ValueError(f"nuPlan vector GT index/token mismatch for {token}")
    if record.get("coordinate_frame") != COORDINATE_FRAME:
        raise ValueError("nuPlan vector GT is not in the QUEST LiDAR frame")
    if tuple(record.get("xy_range_m", ())) != tuple(xy_range):
        raise ValueError("nuPlan vector GT ROI differs from QUEST BEV ROI")
    classes, points = record.get("class_ids"), record.get("points_xy_m")
    closed, length = record.get("is_closed"), record.get("length_m")
    if not all(torch.is_tensor(x) for x in (classes, points, closed, length)):
        raise ValueError("nuPlan vector GT fields must be tensors")
    n = classes.numel()
    if classes.shape != (n,) or points.shape != (n, 20, 2) or closed.shape != (n,) or length.shape != (n,):
        raise ValueError("nuPlan vector GT tensor shapes are inconsistent")
    if bool(((classes < 0) | (classes >= len(MAP_CLASS_NAMES))).any()):
        raise ValueError("nuPlan vector GT has an unsupported class")
    if not bool(torch.isfinite(points).all()) or not bool(torch.isfinite(length).all()):
        raise ValueError("nuPlan vector GT has NaN/Inf")
    x0, y0, x1, y1 = xy_range
    if bool(((points[..., 0] < x0 - 1e-3) | (points[..., 0] > x1 + 1e-3)
             | (points[..., 1] < y0 - 1e-3) | (points[..., 1] > y1 + 1e-3)).any()):
        raise ValueError("nuPlan vector GT contains points outside ROI")


def load_vector_capacity_audit(path: str | Path, config: Mapping[str, Any]) -> tuple[int, dict]:
    with Path(path).open(encoding="utf-8") as stream:
        audit = json.load(stream)
    if (audit.get("capacity_certified") is not True
            or audit.get("schema_version") != VECTOR_GT_SCHEMA_VERSION
            or audit.get("vector_semantics_version") != VECTOR_SEMANTICS_VERSION):
        raise ValueError("full train/validation Vector GT capacity audit is missing or obsolete")
    for split, section in (("train", config["train"]), ("eval", config["eval"])):
        actual = audit.get("splits", {}).get(split, {})
        if (actual.get("start_index") != section["start_index"]
                or actual.get("num_samples") != section["num_samples"]
                or actual.get("available") != section["num_samples"]):
            raise ValueError(f"Vector GT capacity audit does not cover full {split} split")
    provenance = audit.get("vector_gt_provenance")
    if (not isinstance(provenance, dict)
            or provenance.get("vector_semantics_version") != VECTOR_SEMANTICS_VERSION
            or provenance.get("map_height_reference") != MAP_HEIGHT_REFERENCE
            or provenance.get("map_cast_audit_version") != BASELINE_RELATION_AUDIT_VERSION
            or provenance.get("num_points") != 20
            or not provenance.get("map_version")):
        raise ValueError("Vector GT capacity audit provenance is incompatible")
    count = audit.get("recommended_map_query_count")
    if not isinstance(count, int) or count <= 0:
        raise ValueError("Vector GT audit has no valid query capacity")
    if count < max(audit["splits"][split]["total"]["max"] for split in ("train", "eval")):
        raise ValueError("Vector GT audit query capacity would truncate targets")
    configured = config["map"].get("map_query_count")
    if configured is not None and int(configured) != count:
        raise ValueError("configured map_query_count differs from full Vector GT audit")
    return count, provenance


def equivalent_point_orders(points: torch.Tensor, closed: bool) -> torch.Tensor:
    if points.shape != (20, 2):
        raise ValueError("vector instance must have 20 XY points")
    if not closed:
        return torch.stack((points, points.flip(0)), dim=0)
    variants = [torch.roll(sequence, shift, dims=0)
                for sequence in (points, points.flip(0)) for shift in range(20)]
    return torch.stack(variants, dim=0)


def normalize_points(points: torch.Tensor, xy_range: tuple[float, float, float, float]) -> torch.Tensor:
    x0, y0, x1, y1 = xy_range
    lower = points.new_tensor((x0, y0))
    extent = points.new_tensor((x1 - x0, y1 - y0))
    return (points - lower) / extent


def denormalize_points(points: torch.Tensor, xy_range: tuple[float, float, float, float]) -> torch.Tensor:
    x0, y0, x1, y1 = xy_range
    return points * points.new_tensor((x1 - x0, y1 - y0)) + points.new_tensor((x0, y0))


def match_vector_queries(pred_logits: torch.Tensor, pred_points: torch.Tensor,
                         target: Mapping[str, Any], xy_range: tuple[float, float, float, float],
                         cls_cost_weight: float, point_cost_weight: float
                         ) -> list[tuple[int, int, torch.Tensor]]:
    labels = target["class_ids"]
    if not labels.numel():
        return []
    if labels.numel() > pred_logits.shape[0]:
        raise ValueError(f"map GT has {labels.numel()} instances but only {pred_logits.shape[0]} queries")
    normalized = normalize_points(target["points_xy_m"].to(pred_points), xy_range)
    orders = [equivalent_point_orders(normalized[index], bool(target["is_closed"][index]))
              for index in range(len(labels))]
    geometry = torch.stack([
        (pred_points[:, None] - variants[None]).abs().mean(dim=(-1, -2)).min(dim=1).values
        for variants in orders
    ], dim=1)
    cls_cost = -pred_logits.softmax(-1)[:, labels.long().to(pred_logits.device)]
    cost = cls_cost_weight * cls_cost + point_cost_weight * geometry
    rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
    matches = []
    for row, column in zip(rows, columns):
        variants = orders[column]
        order = (pred_points[row].detach()[None] - variants).abs().mean(dim=(-1, -2)).argmin()
        matches.append((int(row), int(column), variants[order]))
    return matches


def vector_map_loss(pred_logits: torch.Tensor, pred_points: torch.Tensor,
                    targets: list[Mapping[str, Any]], config: Mapping[str, Any],
                    xy_range: tuple[float, float, float, float]) -> dict[str, torch.Tensor]:
    if pred_logits.ndim != 3 or pred_points.shape != (*pred_logits.shape[:2], 20, 2):
        raise ValueError("MapHead must produce [B,Q,C+1] and [B,Q,20,2]")
    if pred_logits.shape[-1] != len(MAP_CLASS_NAMES) + 1 or len(targets) != pred_logits.shape[0]:
        raise ValueError("MapHead class count or map target batch differs")
    cls_losses, point_losses, direction_losses = [], [], []
    background = len(MAP_CLASS_NAMES)
    for batch_index, target in enumerate(targets):
        logits, points = pred_logits[batch_index], pred_points[batch_index]
        matches = match_vector_queries(
            logits, points, target, xy_range,
            float(config["map_cls_cost_weight"]), float(config["map_point_cost_weight"]),
        )
        cls_target = torch.full((logits.shape[0],), background, dtype=torch.long, device=logits.device)
        weights = logits.new_ones(background + 1)
        weights[-1] = float(config["background_weight"])
        if matches:
            pred_indices = [row for row, _, _ in matches]
            gt_indices = [column for _, column, _ in matches]
            cls_target[pred_indices] = target["class_ids"][gt_indices].to(logits.device)
            chosen = torch.stack([variant for _, _, variant in matches])
            selected = points[pred_indices]
            point_losses.append(F.smooth_l1_loss(selected, chosen))
            per_instance_direction = []
            for match_index, gt_index in enumerate(gt_indices):
                predicted = selected[match_index]
                expected = chosen[match_index]
                if bool(target["is_closed"][gt_index]):
                    predicted = torch.cat((predicted, predicted[:1]), dim=0)
                    expected = torch.cat((expected, expected[:1]), dim=0)
                predicted_segments = predicted[1:] - predicted[:-1]
                expected_segments = expected[1:] - expected[:-1]
                per_instance_direction.append(
                    (1 - F.cosine_similarity(predicted_segments, expected_segments, dim=-1)).mean()
                )
            direction_losses.append(torch.stack(per_instance_direction).mean())
        else:
            point_losses.append(points.sum() * 0)
            direction_losses.append(points.sum() * 0)
        cls_losses.append(F.cross_entropy(logits, cls_target, weight=weights))
    classification = torch.stack(cls_losses).mean()
    point = torch.stack(point_losses).mean()
    direction = torch.stack(direction_losses).mean()
    total = (float(config["map_cls_loss_weight"]) * classification
             + float(config["map_point_loss_weight"]) * point
             + float(config["map_direction_loss_weight"]) * direction)
    return {"vector_map_gt_loss": total, "map_cls_loss": classification,
            "map_point_loss": point, "map_direction_loss": direction}


def class_aware_vector_matches(pred_labels: torch.Tensor, pred_points_m: torch.Tensor,
                               target: Mapping[str, Any], distance_threshold_m: float
                               ) -> list[tuple[int, int, float, float]]:
    matches = []
    for class_id in range(len(MAP_CLASS_NAMES)):
        pred_indices = torch.nonzero(pred_labels == class_id).flatten()
        gt_indices = torch.nonzero(target["class_ids"] == class_id).flatten()
        if not pred_indices.numel() or not gt_indices.numel():
            continue
        costs = torch.stack([
            torch.stack([
                torch.linalg.vector_norm(
                    pred_points_m[pred_index][None] - variant, dim=-1
                ).mean(dim=-1).min()
                for gt_index in gt_indices
                for variant in [equivalent_point_orders(
                    target["points_xy_m"][gt_index].to(pred_points_m),
                    bool(target["is_closed"][gt_index]),
                )]
            ]) for pred_index in pred_indices
        ])
        gated = costs.masked_fill(costs > distance_threshold_m, 1e6)
        rows, columns = linear_sum_assignment(gated.detach().cpu().numpy())
        for row, column in zip(rows, columns):
            if not bool(costs[row, column] <= distance_threshold_m):
                continue
            pred_index, gt_index = int(pred_indices[row]), int(gt_indices[column])
            gt_points = target["points_xy_m"][gt_index].to(pred_points_m)
            pairwise = torch.cdist(pred_points_m[pred_index][None], gt_points[None])[0]
            chamfer = (pairwise.min(dim=0).values.mean() + pairwise.min(dim=1).values.mean()) / 2
            matches.append((pred_index, gt_index, float(costs[row, column]), float(chamfer)))
    return matches


def configure_stage3(model: nn.Module, raster_head: MapRasterDistillHead,
                     config: Mapping[str, Any]) -> torch.optim.Optimizer:
    model.requires_grad_(False)
    raster_head.requires_grad_(True)
    for name in STAGE3_MODULES:
        getattr(model, name).requires_grad_(True)
    model.geometry_lift.ego_mlp.requires_grad_(False)
    groups = []
    for names, lr in (("geometry_lift bev_encoder", "bev_lr"),
                      ("map_decoder map_head", "map_lr"),
                      ("agent_proposal_head agent_decoder agent_head", "agent_lr")):
        params = [parameter for name in names.split() for parameter in getattr(model, name).parameters()
                  if parameter.requires_grad]
        if not params:
            raise ValueError(f"empty Stage 3 optimizer group: {names}")
        groups.append({"params": params, "lr": float(config[lr])})
    groups.append({"params": list(raster_head.parameters()), "lr": float(config["map_lr"])})
    return torch.optim.AdamW(groups, weight_decay=float(config["weight_decay"]))


def stage3_forward(model: nn.Module, raster_head: MapRasterDistillHead,
                   batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    encoded = model.encode_image(
        batch["images"].to(device), batch["intrinsics"].to(device),
        batch["extrinsics"].to(device), batch["ego_state"].to(device),
        use_ego_state=False,
    )
    predictions = model.forward_agent(encoded["bev_tokens"], encoded["bev_features"])
    queries = model.map_decoder(encoded["bev_tokens"])
    predictions["map_cls_logits"], predictions["map_points"] = model.map_head(queries)
    predictions["student_map_raster_logits"] = raster_head(encoded["bev_features"])
    return predictions


def mixed_stage3_loss(model: nn.Module, predictions: Mapping[str, torch.Tensor],
                      vector_targets: list[Mapping[str, Any]], teacher_soft_scores: torch.Tensor,
                      teacher_valid_spatial: torch.Tensor,
                      support_mask: torch.Tensor, channel_weights: torch.Tensor,
                      agent_target: Mapping[str, torch.Tensor], config: Mapping[str, Any]
                      ) -> dict[str, torch.Tensor]:
    xy_range = (*model.geometry_lift.x_range[:1], *model.geometry_lift.y_range[:1],
                model.geometry_lift.x_range[1], model.geometry_lift.y_range[1])
    vector = vector_map_loss(predictions["map_cls_logits"], predictions["map_points"],
                             vector_targets, config["map_loss"], xy_range)
    kd = soft_map_distillation_loss(predictions["student_map_raster_logits"],
                                    teacher_soft_scores, teacher_valid_spatial,
                                    support_mask, channel_weights)
    agent = compute_stage2_agent_loss(model, predictions, agent_target,
                                      config["agent_train"], config["agent_loss"])
    weights = config["loss_weights"]
    weighted_map = float(weights["lambda_map_gt"]) * vector["vector_map_gt_loss"]
    weighted_kd = float(weights["lambda_map_kd"]) * kd
    weighted_agent = float(weights["lambda_agent"]) * agent["total_agent_loss"]
    return {
        **vector, "map_kd_loss": kd, "existing_agent_loss": agent["total_agent_loss"],
        "raw_map_gt_loss": vector["vector_map_gt_loss"], "raw_map_kd_loss": kd,
        "raw_agent_loss": agent["total_agent_loss"],
        "weighted_map_gt_loss": weighted_map, "weighted_map_kd_loss": weighted_kd,
        "weighted_agent_loss": weighted_agent,
        "total_loss": weighted_map + weighted_kd + weighted_agent,
    }


def save_stage3_checkpoint(path: str | Path, model: nn.Module, raster_head: MapRasterDistillHead,
                           optimizer: torch.optim.Optimizer, epoch: int, config: Mapping[str, Any],
                           teacher_metadata: Mapping[str, Any],
                           vector_metadata: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture_version": QUEST_ARCHITECTURE_VERSION,
        "stage": STAGE3_NAME, "epoch": int(epoch), "bev_use_ego_state": False,
        "trained_class_support_mask": torch.tensor([True, True, False, False]),
        "training_config": dict(config), "map_class_names": MAP_CLASS_NAMES,
        "teacher_channel_metadata": dict(teacher_metadata),
        "teacher_channel_support_mask": list(teacher_metadata["teacher_channel_support_mask"]),
        "vector_gt_schema_version": VECTOR_GT_SCHEMA_VERSION,
        "vector_gt_provenance": dict(vector_metadata),
        "teacher_schema_version": teacher_metadata["teacher_schema_version"],
        "optimizer_state_dict": optimizer.state_dict(),
        "map_raster_distill_head_state_dict": raster_head.state_dict(),
    }
    keys = {
        "geometry_lift": "geometry_aware_bev_lift_state_dict", "bev_encoder": "bev_encoder_state_dict",
        "agent_proposal_head": "agent_proposal_head_state_dict", "agent_decoder": "agent_decoder_state_dict",
        "agent_head": "agent_head_state_dict", "map_decoder": "map_decoder_state_dict",
        "map_head": "map_head_state_dict",
    }
    checkpoint.update({key: getattr(model, name).state_dict() for name, key in keys.items()})
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)


def load_stage3_checkpoint(model: nn.Module, raster_head: MapRasterDistillHead,
                           checkpoint: Mapping[str, Any]) -> int:
    if checkpoint.get("stage") != STAGE3_NAME or checkpoint.get("architecture_version") != QUEST_ARCHITECTURE_VERSION:
        raise ValueError("not a compatible QUEST Stage 3 checkpoint")
    if checkpoint.get("bev_use_ego_state") is not False:
        raise ValueError("Stage 3 checkpoint must disable absolute ego state")
    if tuple(checkpoint.get("map_class_names", ())) != MAP_CLASS_NAMES:
        raise ValueError("Stage 3 checkpoint map taxonomy mismatch")
    if checkpoint.get("vector_gt_schema_version") != VECTOR_GT_SCHEMA_VERSION:
        raise ValueError("Stage 3 checkpoint vector GT schema mismatch")
    provenance = checkpoint.get("vector_gt_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("vector_semantics_version") != VECTOR_SEMANTICS_VERSION:
        raise ValueError("Stage 3 checkpoint vector GT semantics mismatch")
    if (provenance.get("num_points") != 20 or not provenance.get("map_version")
            or provenance.get("map_height_reference") != MAP_HEIGHT_REFERENCE
            or provenance.get("map_cast_audit_version") != BASELINE_RELATION_AUDIT_VERSION
            or not isinstance(provenance.get("min_length_m"), (float, int))):
        raise ValueError("Stage 3 checkpoint vector GT provenance is incomplete")
    if checkpoint.get("teacher_schema_version") != TEACHER_MAP_SCHEMA_VERSION:
        raise ValueError("Stage 3 checkpoint teacher schema mismatch")
    teacher_metadata = checkpoint.get("teacher_channel_metadata")
    if (not isinstance(teacher_metadata, Mapping)
            or teacher_metadata.get("teacher_alignment_version") != TEACHER_ALIGNMENT_VERSION
            or teacher_metadata.get("teacher_coordinate_frame") != TEACHER_COORDINATE_FRAME):
        raise ValueError("Stage 3 checkpoint teacher coordinate alignment mismatch")
    support = checkpoint.get("trained_class_support_mask")
    if not torch.is_tensor(support) or not torch.equal(
        support.cpu().bool(), torch.tensor([True, True, False, False]),
    ):
        raise ValueError("Stage 3 checkpoint Agent support mask mismatch")
    keys = {
        "geometry_lift": "geometry_aware_bev_lift_state_dict", "bev_encoder": "bev_encoder_state_dict",
        "agent_proposal_head": "agent_proposal_head_state_dict", "agent_decoder": "agent_decoder_state_dict",
        "agent_head": "agent_head_state_dict", "map_decoder": "map_decoder_state_dict",
        "map_head": "map_head_state_dict",
    }
    for name, key in keys.items():
        getattr(model, name).load_state_dict(checkpoint[key], strict=True)
    raster_head.load_state_dict(checkpoint["map_raster_distill_head_state_dict"], strict=True)
    return int(checkpoint["epoch"])


def load_stage2_for_stage3(model: nn.Module, checkpoint: Mapping[str, Any]) -> int:
    return load_agent_stage2_checkpoint(model, checkpoint)
