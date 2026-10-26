from __future__ import annotations

from collections.abc import Mapping
import hashlib
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
from .map_objectives import (SPATIAL_MAP_OBJECTIVE, dense_map_gt_loss,
                             focal_classification_loss, focal_matching_cost, map_class_weights)
from .spatial_map import SPATIAL_MAP_VERSION
from .nuplan_map_locator import MAP_LAYER_AUDIT_VERSION, validate_map_layer_audit
from .nuplan_relation_audit import BASELINE_RELATION_AUDIT_VERSION, validate_relation_audit
from .vector_map_labels import (
    COORDINATE_FRAME, MAP_CLASS_NAMES, MAP_HEIGHT_REFERENCE, VECTOR_GT_SCHEMA_VERSION,
    VECTOR_SEMANTICS_VERSION, require_lidar2global,
)


STAGE3_NAME = "map_hybrid_distillation"
STAGE3_GT_ONLY_NAME = "map_gt_agent"
MAP_OBJECTIVE_VERSION = "ordered_l1_sum_v2"
VECTOR_PROVENANCE_KEYS = (
    "num_points", "min_length_m", "map_version", "vector_semantics_version",
    "map_height_reference",
    "map_cast_audit_version",
    "map_layer_audit_version",
)
STAGE3_MODULES = (
    "geometry_lift", "bev_encoder", "agent_proposal_head", "agent_decoder",
    "agent_head", "map_decoder", "map_head",
)


def stage3_kd_enabled(config: Mapping[str, Any]) -> bool:
    weights = config["loss_weights"]
    for name in ("lambda_map_gt", "lambda_map_kd", "lambda_agent"):
        value = float(weights[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Stage 3 {name} must be finite and nonnegative")
    if float(weights["lambda_map_gt"]) <= 0:
        raise ValueError("Stage 3 requires vector GT supervision")
    frozen_agent = bool(config.get("train", {}).get("freeze_agent_and_bev", False))
    if frozen_agent != (float(weights["lambda_agent"]) == 0):
        raise ValueError("lambda_agent must be zero only when Agent and shared BEV are frozen")
    enabled = float(weights["lambda_map_kd"]) > 0
    expected_stage = STAGE3_NAME if enabled else STAGE3_GT_ONLY_NAME
    if config.get("stage") != expected_stage:
        raise ValueError(f"Stage 3 stage/lambda_map_kd mismatch: expected {expected_stage}")
    objective = config.get("map", {}).get("objective_version", "legacy_v1")
    if objective not in ("legacy_v1", MAP_OBJECTIVE_VERSION, SPATIAL_MAP_OBJECTIVE):
        raise ValueError("unknown Stage 3 map objective version")
    if objective == MAP_OBJECTIVE_VERSION:
        settings = config["map"]
        if settings.get("point_cost_reduction") != "sum" or settings.get("point_loss_type") != "l1_sum":
            raise ValueError("ordered_l1_sum_v2 requires sum matching and L1 sum-per-instance loss")
    spatial = config.get("map_architecture")
    spatial_enabled = bool(config.get("train", {}).get("map_spatial_enabled", False))
    if spatial_enabled != (spatial is not None):
        raise ValueError("map_spatial_enabled must agree with map_architecture")
    if spatial is not None:
        if not isinstance(spatial, Mapping) or spatial.get("version") != SPATIAL_MAP_VERSION:
            raise ValueError("unknown spatial Map architecture")
        if not frozen_agent or enabled or objective != SPATIAL_MAP_OBJECTIVE:
            raise ValueError("spatial Map requires frozen Agent/shared BEV and GT-only metric_focal_dense_v3")
        required = ("bev_h", "bev_w", "encoder_layers", "window_size", "decoder_layers",
                    "num_heads", "ffn_dim", "num_levels", "sampling_points")
        if any(not isinstance(spatial.get(key), int) or spatial[key] <= 0 for key in required):
            raise ValueError("spatial Map dimensions and layer counts must be positive integers")
        divisor = 2 ** (spatial["num_levels"] - 1)
        if any(spatial[key] % spatial["window_size"] or spatial[key] % divisor for key in ("bev_h", "bev_w")):
            raise ValueError("spatial Map grid is incompatible with window/pyramid sizes")
        settings = config["map"]
        if settings.get("point_loss_type") != "metric_l1" or settings.get("classification_loss") != "softmax_focal":
            raise ValueError("spatial Map requires metric L1 and matching softmax Focal classification")
        map_class_weights(torch.empty(0), settings)
        for key in ("map_cls_cost_weight", "map_point_cost_weight", "map_cls_loss_weight",
                    "map_point_loss_weight", "map_direction_loss_weight", "dense_focal_weight",
                    "dense_dice_weight", "dense_negative_weight", "focal_gamma"):
            value = float(settings[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid spatial Map loss setting: {key}")
        if (not 0 <= float(spatial["dropout"]) < 1
                or float(settings["dense_line_width_m"]) <= 0
                or not math.isfinite(float(settings["dense_line_width_m"]))):
            raise ValueError("invalid spatial Map dropout or dense line width")
        if float(settings["dense_focal_weight"]) <= 0 or float(settings["map_point_loss_weight"]) <= 0:
            raise ValueError("spatial Map requires direct dense and vector GT supervision")
    elif objective == SPATIAL_MAP_OBJECTIVE:
        raise ValueError("metric_focal_dense_v3 requires the spatial Map branch")
    return enabled


def stage3_model_config(base: Mapping[str, Any], config: Mapping[str, Any],
                        checkpoint: Mapping[str, Any] | None = None) -> dict:
    """Evaluation reconstructs the saved map branch, not the latest YAML defaults."""
    result = dict(base)
    architecture = (checkpoint.get("map_architecture") if checkpoint is not None
                    else config.get("map_architecture"))
    result.pop("map_architecture", None)
    if architecture is not None:
        if not isinstance(architecture, Mapping) or architecture.get("version") != SPATIAL_MAP_VERSION:
            raise ValueError("checkpoint/config has an unknown Map architecture")
        if int(result["hidden_dim"]) % int(architecture["num_heads"]):
            raise ValueError("Map hidden_dim must be divisible by num_heads")
        result["map_architecture"] = dict(architecture)
        result["map_decoder_layers"] = int(architecture["decoder_layers"])
    return result


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
    if record.get("map_layer_audit_version") != MAP_LAYER_AUDIT_VERSION:
        raise ValueError("nuPlan map-layer audit version mismatch; regenerate labels")
    layer_audit = record.get("map_layer_diagnostics")
    if not isinstance(layer_audit, Mapping):
        raise ValueError("nuPlan vector GT has no per-frame map-layer audit")
    validate_map_layer_audit(layer_audit)
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


def load_vector_capacity_audit(path: str | Path, config: Mapping[str, Any],
                               split_manifest: Mapping[str, Any] | None = None) -> tuple[int, dict]:
    with Path(path).open(encoding="utf-8") as stream:
        audit = json.load(stream)
    if (audit.get("capacity_certified") is not True
            or audit.get("schema_version") != VECTOR_GT_SCHEMA_VERSION
            or audit.get("vector_semantics_version") != VECTOR_SEMANTICS_VERSION):
        raise ValueError("full train/validation Vector GT capacity audit is missing or obsolete")
    if split_manifest is not None:
        if (audit.get("split_sha256") != split_manifest.get("split_sha256")
                or audit.get("metadata_sha256") != split_manifest.get("metadata_sha256")):
            raise ValueError("Vector GT capacity audit was built for a different Stage 3 frame list")
    elif config.get("paths", {}).get("split_manifest_path"):
        raise ValueError("Stage 3 split manifest is required to certify Vector GT capacity")
    for split, section in (("train", config["train"]), ("eval", config["eval"])):
        actual = audit.get("splits", {}).get(split, {})
        if split_manifest is not None:
            name = "train" if split == "train" else "validation"
            expected = split_manifest[name]["frames"]
            if (actual.get("indices") != [row["index"] for row in expected]
                    or actual.get("tokens") != [row["token"] for row in expected]
                    or actual.get("num_samples") != len(expected)
                    or actual.get("available") != len(expected)):
                raise ValueError(f"Vector GT capacity audit does not cover exact {split} indices/tokens")
        elif (actual.get("start_index") != section["start_index"]
              or actual.get("num_samples") != section["num_samples"]
              or actual.get("available") != section["num_samples"]):
            raise ValueError(f"Vector GT capacity audit does not cover full {split} split")
    provenance = audit.get("vector_gt_provenance")
    if (not isinstance(provenance, dict)
            or provenance.get("vector_semantics_version") != VECTOR_SEMANTICS_VERSION
            or provenance.get("map_height_reference") != MAP_HEIGHT_REFERENCE
            or provenance.get("map_cast_audit_version") != BASELINE_RELATION_AUDIT_VERSION
            or provenance.get("map_layer_audit_version") != MAP_LAYER_AUDIT_VERSION
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


@torch.no_grad()
def match_vector_queries(pred_logits: torch.Tensor, pred_points: torch.Tensor,
                         target: Mapping[str, Any], xy_range: tuple[float, float, float, float],
                         cls_cost_weight: float, point_cost_weight: float,
                         point_cost_reduction: str = "mean",
                         objective_config: Mapping[str, Any] | None = None,
                         ) -> list[tuple[int, int, torch.Tensor]]:
    labels = target["class_ids"]
    if not labels.numel():
        return []
    if labels.numel() > pred_logits.shape[0]:
        raise ValueError(f"map GT has {labels.numel()} instances but only {pred_logits.shape[0]} queries")
    normalized = normalize_points(target["points_xy_m"].to(pred_points), xy_range)
    closed_flags = target["is_closed"].detach().cpu().tolist()
    orders = [equivalent_point_orders(normalized[index], bool(closed_flags[index]))
              for index in range(len(labels))]
    if point_cost_reduction not in ("mean", "sum"):
        raise ValueError("point_cost_reduction must be mean or sum")
    metric = objective_config is not None and objective_config.get("point_loss_type") == "metric_l1"
    predicted_for_cost = denormalize_points(pred_points, xy_range) if metric else pred_points
    cost_orders = [denormalize_points(variants, xy_range) if metric else variants for variants in orders]
    divisor = (pred_points.shape[-2] if metric else
               (pred_points.shape[-2] * pred_points.shape[-1] if point_cost_reduction == "mean" else 1))
    geometry = torch.stack([
        ((predicted_for_cost[:, None] - variants[None]).abs().sum(dim=(-1, -2))
         / divisor
         ).min(dim=1).values
        for variants in cost_orders
    ], dim=1)
    if metric:
        geometry = geometry * map_class_weights(pred_logits, objective_config)[labels.to(pred_logits.device).long()][None]
    cls_cost = (focal_matching_cost(pred_logits, labels, objective_config) if metric
                else -pred_logits.softmax(-1)[:, labels.long().to(pred_logits.device)])
    cost = cls_cost_weight * cls_cost + point_cost_weight * geometry
    rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
    matches = []
    for row, column in zip(rows, columns):
        variants = cost_orders[column]
        order = (predicted_for_cost[row].detach()[None] - variants).abs().mean(dim=(-1, -2)).argmin()
        matches.append((int(row), int(column), orders[column][order]))
    return matches


def vector_map_loss(pred_logits: torch.Tensor, pred_points: torch.Tensor,
                    targets: list[Mapping[str, Any]], config: Mapping[str, Any],
                    xy_range: tuple[float, float, float, float]) -> dict[str, torch.Tensor]:
    if pred_logits.ndim != 3 or pred_points.shape != (*pred_logits.shape[:2], 20, 2):
        raise ValueError("MapHead must produce [B,Q,C+1] and [B,Q,20,2]")
    if pred_logits.shape[-1] != len(MAP_CLASS_NAMES) + 1 or len(targets) != pred_logits.shape[0]:
        raise ValueError("MapHead class count or map target batch differs")
    cls_losses, point_losses, direction_losses, metric_errors = [], [], [], []
    per_class_errors, per_class_counts = [[] for _ in MAP_CLASS_NAMES], [0] * len(MAP_CLASS_NAMES)
    background = len(MAP_CLASS_NAMES)
    for batch_index, target in enumerate(targets):
        logits, points = pred_logits[batch_index], pred_points[batch_index]
        matches = match_vector_queries(
            logits, points, target, xy_range,
            float(config["map_cls_cost_weight"]), float(config["map_point_cost_weight"]),
            point_cost_reduction=config.get("point_cost_reduction", "mean"),
            objective_config=config,
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
            point_type = config.get("point_loss_type", "smooth_l1_mean")
            if point_type == "l1_sum":
                point_losses.append((selected - chosen).abs().sum(dim=(-1, -2)).mean())
            elif point_type == "smooth_l1_mean":
                point_losses.append(F.smooth_l1_loss(selected, chosen))
            elif point_type == "metric_l1":
                labels = target["class_ids"][gt_indices].to(device=logits.device, dtype=torch.long)
                instance_weights = map_class_weights(logits, config)[labels]
                metric_distance = (denormalize_points(selected, xy_range)
                                   - denormalize_points(chosen, xy_range)).abs().sum(-1).mean(-1)
                point_losses.append((metric_distance * instance_weights).sum() / instance_weights.sum())
            else:
                raise ValueError("unknown map point_loss_type")
            selected_m = denormalize_points(selected, xy_range)
            chosen_m = denormalize_points(chosen, xy_range)
            metric_errors.append(torch.linalg.vector_norm(selected_m - chosen_m, dim=-1).mean().detach())
            for class_id in range(len(MAP_CLASS_NAMES)):
                class_selection = target["class_ids"][gt_indices].to(logits.device) == class_id
                if bool(class_selection.any()):
                    errors = torch.linalg.vector_norm(selected_m - chosen_m, dim=-1).mean(-1)
                    per_class_errors[class_id].append(errors[class_selection].detach())
                    per_class_counts[class_id] += int(class_selection.sum())
            predicted_segments = torch.roll(selected_m, -1, dims=1) - selected_m
            expected_segments = torch.roll(chosen_m, -1, dims=1) - chosen_m
            segment_valid = torch.ones_like(predicted_segments[..., 0])
            segment_valid[:, -1] = target["is_closed"][gt_indices].to(segment_valid)
            direction_error = 1 - F.cosine_similarity(predicted_segments, expected_segments, dim=-1)
            per_direction = (direction_error * segment_valid).sum(-1) / segment_valid.sum(-1)
            direction_losses.append((per_direction * instance_weights).sum() / instance_weights.sum()
                                    if point_type == "metric_l1" else per_direction.mean())
        else:
            point_losses.append(points.sum() * 0)
            direction_losses.append(points.sum() * 0)
            metric_errors.append(points.detach().sum() * 0)
        cls_losses.append(focal_classification_loss(logits, cls_target, config)
                          if config.get("classification_loss") == "softmax_focal"
                          else F.cross_entropy(logits, cls_target, weight=weights))
    classification = torch.stack(cls_losses).mean()
    point = torch.stack(point_losses).mean()
    direction = torch.stack(direction_losses).mean()
    total = (float(config["map_cls_loss_weight"]) * classification
             + float(config["map_point_loss_weight"]) * point
             + float(config["map_direction_loss_weight"]) * direction)
    result = {"vector_map_gt_loss": total, "map_cls_loss": classification,
            "map_point_loss": point, "map_direction_loss": direction,
            "map_matched_point_error_m": torch.stack(metric_errors).mean()}
    for class_id, name in enumerate(MAP_CLASS_NAMES):
        result[f"map_{name}_point_error_sum_m"] = (torch.cat(per_class_errors[class_id]).sum()
                                                   if per_class_errors[class_id] else total.detach() * 0)
        result[f"map_{name}_matched_count"] = total.detach().new_tensor(per_class_counts[class_id])
    return result


def pairwise_vector_distances(pred_points_m: torch.Tensor, target: Mapping[str, Any],
                              metric: str = "ordered") -> torch.Tensor:
    """Custom instance distances in meters; Chamfer is separate from ordered error."""
    if metric not in ("ordered", "chamfer"):
        raise ValueError("vector distance metric must be ordered or chamfer")
    if not pred_points_m.shape[0] or not target["class_ids"].numel():
        return pred_points_m.new_empty((pred_points_m.shape[0], target["class_ids"].numel()))
    columns = []
    for gt_index, points in enumerate(target["points_xy_m"]):
        points = points.to(pred_points_m)
        if metric == "ordered":
            orders = equivalent_point_orders(points, bool(target["is_closed"][gt_index]))
            distance = torch.linalg.vector_norm(pred_points_m[:, None] - orders[None], dim=-1).mean(-1).min(-1).values
        else:
            pairwise = torch.cdist(pred_points_m, points[None].expand(pred_points_m.shape[0], -1, -1))
            distance = (pairwise.min(-1).values.mean(-1) + pairwise.min(-2).values.mean(-1)) / 2
        columns.append(distance)
    return torch.stack(columns, dim=1)


def class_aware_vector_matches(pred_labels: torch.Tensor, pred_points_m: torch.Tensor,
                               target: Mapping[str, Any], distance_threshold_m: float,
                               distance_metric: str = "ordered",
                               ) -> list[tuple[int, int, float, float]]:
    matches = []
    for class_id in range(len(MAP_CLASS_NAMES)):
        pred_indices = torch.nonzero(pred_labels == class_id).flatten()
        gt_indices = torch.nonzero(target["class_ids"] == class_id).flatten()
        if not pred_indices.numel() or not gt_indices.numel():
            continue
        subset = {"class_ids": target["class_ids"][gt_indices],
                  "points_xy_m": target["points_xy_m"][gt_indices],
                  "is_closed": target["is_closed"][gt_indices]}
        costs = pairwise_vector_distances(pred_points_m[pred_indices], subset, distance_metric)
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


def stage3_trainable_modules(config: Mapping[str, Any]) -> tuple[str, ...]:
    if config.get("map_spatial_enabled", False):
        return ("map_spatial", "map_decoder", "map_head")
    return ("map_decoder", "map_head") if config.get("freeze_agent_and_bev", False) else STAGE3_MODULES


def frozen_agent_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name in STAGE3_MODULES:
        if name in ("map_decoder", "map_head"):
            continue
        for key, value in sorted(getattr(model, name).state_dict().items()):
            digest.update(f"{name}.{key}:{value.dtype}:{tuple(value.shape)}".encode("ascii"))
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def configure_stage3(model: nn.Module, raster_head: MapRasterDistillHead | None,
                     config: Mapping[str, Any]) -> torch.optim.Optimizer:
    model.requires_grad_(False)
    if raster_head is not None:
        raster_head.requires_grad_(True)
    trainable = stage3_trainable_modules(config)
    for name in trainable:
        getattr(model, name).requires_grad_(True)
    model.geometry_lift.ego_mlp.requires_grad_(False)
    if getattr(model, "map_spatial", None) is not None:
        model.map_spatial.geometry_lift.ego_mlp.requires_grad_(False)
    groups = []
    for names, lr in (("geometry_lift bev_encoder", "bev_lr"),
                      ("map_spatial map_decoder map_head", "map_lr"),
                      ("agent_proposal_head agent_decoder agent_head", "agent_lr")):
        params = [parameter for name in names.split() if name in trainable
                  for parameter in getattr(model, name).parameters()
                  if parameter.requires_grad]
        if params:
            groups.append({"params": params, "lr": float(config[lr])})
    if raster_head is not None:
        groups.append({"params": list(raster_head.parameters()), "lr": float(config["map_lr"])})
    return torch.optim.AdamW(groups, weight_decay=float(config["weight_decay"]))


def stage3_forward(model: nn.Module, raster_head: MapRasterDistillHead | None,
                   batch: Mapping[str, torch.Tensor], device: torch.device, *,
                   include_dense: bool | None = None) -> dict[str, torch.Tensor]:
    encoded = model.encode_image(
        batch["images"].to(device), batch["intrinsics"].to(device),
        batch["extrinsics"].to(device), batch["ego_state"].to(device),
        use_ego_state=False,
    )
    predictions = model.forward_agent(encoded["bev_tokens"], encoded["bev_features"])
    if hasattr(model, "forward_map"):
        predictions.update(model.forward_map(
            encoded, batch["intrinsics"].to(device), batch["extrinsics"].to(device),
            batch["ego_state"].to(device), tuple(batch["images"].shape[-2:]),
            include_dense=(getattr(model, "map_spatial", None) is not None and model.map_spatial.training
                           if include_dense is None else include_dense),
        ))
    elif hasattr(model.map_decoder, "forward_layers"):
        layers = [model.map_head(query) for query in model.map_decoder.forward_layers(encoded["bev_tokens"])]
        predictions["map_cls_logits_layers"] = torch.stack([layer[0] for layer in layers])
        predictions["map_points_layers"] = torch.stack([layer[1] for layer in layers])
        predictions["map_cls_logits"], predictions["map_points"] = layers[-1]
    else:
        queries = model.map_decoder(encoded["bev_tokens"])
        predictions["map_cls_logits"], predictions["map_points"] = model.map_head(queries)
    if raster_head is not None:
        predictions["student_map_raster_logits"] = raster_head(encoded["bev_features"])
    return predictions


def mixed_stage3_loss(model: nn.Module, predictions: Mapping[str, torch.Tensor],
                      vector_targets: list[Mapping[str, Any]], teacher_soft_scores: torch.Tensor | None,
                      teacher_valid_spatial: torch.Tensor | None,
                      support_mask: torch.Tensor | None, channel_weights: torch.Tensor | None,
                      agent_target: Mapping[str, torch.Tensor], config: Mapping[str, Any]
                      ) -> dict[str, torch.Tensor]:
    xy_range = (*model.geometry_lift.x_range[:1], *model.geometry_lift.y_range[:1],
                model.geometry_lift.x_range[1], model.geometry_lift.y_range[1])
    vector = vector_map_loss(predictions["map_cls_logits"], predictions["map_points"],
                             vector_targets, config["map_loss"], xy_range)
    auxiliary_weights = config["map_loss"].get("aux_layer_weights")
    if auxiliary_weights is not None:
        logits_layers, point_layers = predictions["map_cls_logits_layers"], predictions["map_points_layers"]
        if len(auxiliary_weights) != logits_layers.shape[0] or float(auxiliary_weights[-1]) != 1.0:
            raise ValueError("map auxiliary weights must cover decoder layers and end in 1.0")
        auxiliary = vector["vector_map_gt_loss"].new_zeros(())
        for index, weight in enumerate(auxiliary_weights[:-1]):
            if not math.isfinite(float(weight)) or float(weight) < 0:
                raise ValueError("invalid map auxiliary layer weight")
            auxiliary = auxiliary + float(weight) * vector_map_loss(
                logits_layers[index], point_layers[index], vector_targets, config["map_loss"], xy_range
            )["vector_map_gt_loss"]
        vector["map_final_gt_loss"] = vector["vector_map_gt_loss"]
        vector["map_aux_loss"] = auxiliary
        vector["vector_map_gt_loss"] = vector["vector_map_gt_loss"] + auxiliary
    if config["map_loss"].get("objective_version") == SPATIAL_MAP_OBJECTIVE:
        if "map_dense_gt_logits" not in predictions:
            raise ValueError("spatial Map training requires its real-GT dense auxiliary head")
        dense = dense_map_gt_loss(predictions["map_dense_gt_logits"], vector_targets,
                                  config["map_loss"], xy_range)
        vector.update(dense)
        vector["map_dense_gt_loss"] = (
            float(config["map_loss"]["dense_focal_weight"]) * dense["map_dense_focal_loss"]
            + float(config["map_loss"]["dense_dice_weight"]) * dense["map_dense_dice_loss"])
        vector["vector_map_gt_loss"] = vector["vector_map_gt_loss"] + vector["map_dense_gt_loss"]
    kd_enabled = stage3_kd_enabled(config)
    if kd_enabled:
        if (teacher_soft_scores is None or teacher_valid_spatial is None
                or support_mask is None or channel_weights is None
                or "student_map_raster_logits" not in predictions):
            raise ValueError("map KD requires a raster head and audited teacher targets")
        kd = soft_map_distillation_loss(predictions["student_map_raster_logits"],
                                        teacher_soft_scores, teacher_valid_spatial,
                                        support_mask, channel_weights)
    else:
        if "student_map_raster_logits" in predictions or any(
            value is not None for value in (teacher_soft_scores, teacher_valid_spatial,
                                            support_mask, channel_weights)
        ):
            raise ValueError("GT-only Stage 3 must not receive map teacher targets")
        kd = vector["vector_map_gt_loss"].new_zeros(())
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


def save_stage3_checkpoint(path: str | Path, model: nn.Module, raster_head: MapRasterDistillHead | None,
                           optimizer: torch.optim.Optimizer, epoch: int, config: Mapping[str, Any],
                           teacher_metadata: Mapping[str, Any] | None,
                           vector_metadata: Mapping[str, Any], *, scheduler=None) -> None:
    kd_enabled = stage3_kd_enabled(config)
    if kd_enabled != (raster_head is not None) or kd_enabled != (teacher_metadata is not None):
        raise ValueError("Stage 3 checkpoint raster head/audit do not match KD mode")
    frozen_agent = bool(config.get("train", {}).get("freeze_agent_and_bev", False))
    if getattr(model, "map_architecture", None) != config.get("map_architecture"):
        raise ValueError("refusing checkpoint with different model/config Map architecture")
    if frozen_agent and config.get("agent_baseline_sha256") != frozen_agent_fingerprint(model):
        raise ValueError("frozen shared BEV/Agent differs from Stage 2 initialization; refusing checkpoint")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture_version": QUEST_ARCHITECTURE_VERSION,
        "stage": STAGE3_NAME if kd_enabled else STAGE3_GT_ONLY_NAME,
        "map_kd_enabled": kd_enabled,
        "map_objective_version": config.get("map", {}).get("objective_version", "legacy_v1"),
        "map_architecture": getattr(model, "map_architecture", None),
        "agent_and_bev_frozen": bool(config.get("train", {}).get("freeze_agent_and_bev", False)),
        "agent_baseline_sha256": config.get("agent_baseline_sha256"),
        "agent_preservation_verified": frozen_agent,
        "epoch": int(epoch), "bev_use_ego_state": False,
        "trained_class_support_mask": torch.tensor([True, True, False, False]),
        "training_config": dict(config), "map_class_names": MAP_CLASS_NAMES,
        "vector_gt_schema_version": VECTOR_GT_SCHEMA_VERSION,
        "vector_gt_provenance": dict(vector_metadata),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if kd_enabled:
        checkpoint.update({
            "teacher_channel_metadata": dict(teacher_metadata),
            "teacher_channel_support_mask": list(teacher_metadata["teacher_channel_support_mask"]),
            "teacher_schema_version": teacher_metadata["teacher_schema_version"],
            "map_raster_distill_head_state_dict": raster_head.state_dict(),
        })
    keys = {
        "geometry_lift": "geometry_aware_bev_lift_state_dict", "bev_encoder": "bev_encoder_state_dict",
        "agent_proposal_head": "agent_proposal_head_state_dict", "agent_decoder": "agent_decoder_state_dict",
        "agent_head": "agent_head_state_dict", "map_decoder": "map_decoder_state_dict",
        "map_head": "map_head_state_dict",
    }
    checkpoint.update({key: getattr(model, name).state_dict() for name, key in keys.items()})
    if getattr(model, "map_spatial", None) is not None:
        checkpoint["map_spatial_state_dict"] = model.map_spatial.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)


def load_stage3_checkpoint(model: nn.Module, raster_head: MapRasterDistillHead | None,
                           checkpoint: Mapping[str, Any], *,
                           expected_map_kd_enabled: bool | None = None) -> int:
    stage = checkpoint.get("stage")
    if stage not in (STAGE3_NAME, STAGE3_GT_ONLY_NAME) or checkpoint.get("architecture_version") != QUEST_ARCHITECTURE_VERSION:
        raise ValueError("not a compatible QUEST Stage 3 checkpoint")
    kd_enabled = stage == STAGE3_NAME
    if checkpoint.get("map_kd_enabled", kd_enabled) is not kd_enabled:
        raise ValueError("Stage 3 checkpoint KD mode metadata is inconsistent")
    if expected_map_kd_enabled is not None and kd_enabled != expected_map_kd_enabled:
        raise ValueError("Stage 3 checkpoint KD mode differs from evaluation config")
    if kd_enabled != (raster_head is not None):
        raise ValueError("Stage 3 checkpoint raster head differs from KD mode")
    if checkpoint.get("bev_use_ego_state") is not False:
        raise ValueError("Stage 3 checkpoint must disable absolute ego state")
    if checkpoint.get("map_objective_version", "legacy_v1") not in ("legacy_v1", MAP_OBJECTIVE_VERSION, SPATIAL_MAP_OBJECTIVE):
        raise ValueError("unknown Stage 3 checkpoint map objective version")
    saved_architecture = checkpoint.get("map_architecture")
    if saved_architecture != getattr(model, "map_architecture", None):
        raise ValueError("Stage 3 Map architecture mismatch; reconstruct from checkpoint metadata; no automatic conversion")
    if (saved_architecture is not None) != ("map_spatial_state_dict" in checkpoint):
        raise ValueError("Stage 3 checkpoint spatial Map state/architecture mismatch")
    if saved_architecture is not None:
        if (saved_architecture.get("version") != SPATIAL_MAP_VERSION
                or checkpoint.get("map_objective_version") != SPATIAL_MAP_OBJECTIVE
                or not checkpoint.get("agent_and_bev_frozen") or kd_enabled):
            raise ValueError("spatial Map checkpoint supervision or Agent preservation metadata mismatch")
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
            or provenance.get("map_layer_audit_version") != MAP_LAYER_AUDIT_VERSION
            or not isinstance(provenance.get("min_length_m"), (float, int))):
        raise ValueError("Stage 3 checkpoint vector GT provenance is incomplete")
    if kd_enabled:
        if checkpoint.get("teacher_schema_version") != TEACHER_MAP_SCHEMA_VERSION:
            raise ValueError("Stage 3 checkpoint teacher schema mismatch")
        teacher_metadata = checkpoint.get("teacher_channel_metadata")
        if (not isinstance(teacher_metadata, Mapping)
                or teacher_metadata.get("teacher_alignment_version") != TEACHER_ALIGNMENT_VERSION
                or teacher_metadata.get("teacher_coordinate_frame") != TEACHER_COORDINATE_FRAME):
            raise ValueError("Stage 3 checkpoint teacher coordinate alignment mismatch")
    elif any(key in checkpoint for key in ("teacher_schema_version", "teacher_channel_metadata",
                                           "map_raster_distill_head_state_dict")):
        raise ValueError("GT-only Stage 3 checkpoint unexpectedly contains map teacher state")
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
    if saved_architecture is not None:
        model.map_spatial.load_state_dict(checkpoint["map_spatial_state_dict"], strict=True)
    if kd_enabled:
        raster_head.load_state_dict(checkpoint["map_raster_distill_head_state_dict"], strict=True)
    if checkpoint.get("agent_and_bev_frozen", False):
        if (checkpoint.get("agent_preservation_verified") is not True
                or checkpoint.get("agent_baseline_sha256") != frozen_agent_fingerprint(model)):
            raise ValueError("Stage 3 frozen Agent preservation checksum mismatch")
    return int(checkpoint["epoch"])


def load_stage2_for_stage3(model: nn.Module, checkpoint: Mapping[str, Any]) -> int:
    epoch = load_agent_stage2_checkpoint(model, checkpoint)
    if getattr(model, "map_spatial", None) is not None:
        model.map_spatial.initialize_lift(model.geometry_lift)
    return epoch
