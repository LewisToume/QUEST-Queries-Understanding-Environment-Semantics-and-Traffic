"""Four-task losses, gradient-interference diagnostics, and independent quality gates."""

import math

import torch

from .agent_training import compute_stage2_agent_loss, prepare_navformer_agent_batch
from .map_objectives import dense_map_gt_loss
from .map_training import vector_map_loss
from .seg_depth_losses import compute_seg_depth_loss


TASKS = ("agent", "map", "seg", "depth")


def four_task_losses(model, output, batch, settings, device):
    target = prepare_navformer_agent_batch(batch, device)
    agent = compute_stage2_agent_loss(model, output, target, settings["agent"]["train"], settings["agent"]["agent_loss"])
    xy = (model.geometry_lift.x_range[0], model.geometry_lift.y_range[0],
          model.geometry_lift.x_range[1], model.geometry_lift.y_range[1])
    vectors = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in row.items()}
               for row in batch["vector_targets"]]
    weights = settings["map"]["aux_layer_weights"]
    if len(weights) != output["map_cls_logits_layers"].shape[0] or weights[-1] != 1:
        raise ValueError("Map auxiliary loss weights must match shared query layers")
    maps = [vector_map_loss(logits.float(), points.float(), vectors, settings["map"], xy)
            for logits, points in zip(output["map_cls_logits_layers"], output["map_points_layers"])]
    dense_map = dense_map_gt_loss(output["map_dense_gt_logits"], vectors, settings["map"], xy)
    map_loss = sum(float(weight) * row["vector_map_gt_loss"] for weight, row in zip(weights, maps))
    map_loss = map_loss + settings["map"]["dense_focal_weight"] * dense_map["map_dense_focal_loss"]
    map_loss = map_loss + settings["map"]["dense_dice_weight"] * dense_map["map_dense_dice_loss"]
    dense = compute_seg_depth_loss(output, batch["dense_targets"], settings["dense"], ("seg", "depth"))
    if min(dense["seg_pixels"], dense["depth_pixels"]) <= 0:
        raise ValueError("Every joint batch must have real, valid Seg and Depth pseudo supervision")
    dw = settings["dense"]["weights"]
    raw = {"agent": agent["total_agent_loss"], "map": map_loss, "seg": dense["seg_ce"],
           "depth": sum(float(dw[key]) * dense[key] for key in ("depth_log", "depth_relative", "depth_gradient"))}
    if not all(bool(torch.isfinite(value)) and float(value.detach()) >= 0 for value in raw.values()):
        raise RuntimeError("Nonfinite/negative four-task loss")
    return raw, {"map_point_error_m": maps[-1]["map_matched_point_error_m"], "depth_absrel": dense["depth_absrel"]}


def scaled_task_losses(raw, scales, weights):
    if set(raw) != set(TASKS) or set(scales) != set(TASKS) or set(weights) != set(TASKS):
        raise ValueError("All four tasks require explicit nonzero loss scales/weights")
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in (*scales.values(), *weights.values())):
        raise ValueError("Task loss scales and weights must be finite and positive")
    return {name: raw[name] * float(weights[name]) / float(scales[name]) for name in TASKS}


def shared_gradient_audit(model, task_losses):
    """Sample actual shared parameters; output cosine signs, not accuracy promises."""
    probes = {
        "geometry": model.trunk.geometry.level_projections[-1].weight,
        "pyramid": model.trunk.pyramid_fusion.scratch[-1].weight,
        "bev": model.trunk.bev_layers[-1].ffn[-1].weight,
        "query": model.trunk.query_layers[-1].ffn[-1].weight,
    }
    gradients, norms = {}, {}
    task_names = tuple(task_losses)
    for name in task_names:
        values = torch.autograd.grad(task_losses[name], tuple(probes.values()), retain_graph=True, allow_unused=True)
        norms[name] = {}
        for (group, parameter), value in zip(probes.items(), values):
            norm = float(value.detach().float().norm()) if value is not None else 0.
            if not math.isfinite(norm) or norm <= 0:
                raise RuntimeError(f"Task {name} has no finite/nonzero gradient to shared {group}")
            norms[name][group] = norm
        gradients[name] = torch.cat([value.detach().float().flatten() for value in values])
    cosines = {}
    for i, name in enumerate(task_names):
        for other in task_names[i + 1:]:
            a, b = gradients[name], gradients[other]
            cosines[name + ":" + other] = float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-20))
    return {"task_to_shared_gradient_norms": norms, "sampled_shared_gradient_cosines": cosines,
            "negative_pairs": sum(value < 0 for value in cosines.values()), "probe_only": True}


def pcgrad_vectors(vectors, generator):
    """PCGrad Eq.2: project each task against ORIGINAL other-task gradients."""
    if vectors.ndim != 2 or not bool(torch.isfinite(vectors).all()):
        raise ValueError("PCGrad requires finite [tasks,shared_parameters] vectors")
    projected = []
    for task_index, vector in enumerate(vectors):
        value = vector.clone()
        for index in torch.randperm(len(vectors), generator=generator).tolist():
            if index == task_index:
                continue
            other = vectors[index]
            dot = value @ other
            if float(dot) < 0:
                value = value - dot / other.square().sum().clamp_min(1e-20) * other
        projected.append(value)
    # Sum matches the default summed-loss gradient when there is no conflict.
    return torch.stack(projected).sum(0)


def joint_backward(model, losses, *, mode="sum", generator=None):
    if mode not in ("sum", "pcgrad"):
        raise ValueError("Unknown joint gradient mode")
    if mode == "sum":
        sum(losses.values()).backward()
        return
    if generator is None:
        raise ValueError("PCGrad requires an explicitly seeded/resumable generator")
    shared = [p for p in model.trunk.parameters() if p.requires_grad]
    vectors = []
    for loss in losses.values():
        gradients = torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
        vectors.append(torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().flatten()
                                  for p, g in zip(shared, gradients)]))
    # Private output heads keep their ordinary summed-loss gradients.
    sum(losses.values()).backward()
    merged = pcgrad_vectors(torch.stack(vectors), generator)
    start = 0
    for parameter in shared:
        end = start + parameter.numel()
        parameter.grad = merged[start:end].reshape_as(parameter).to(parameter.dtype)
        start = end


def quality_gate(current, baseline, limits):
    """All monitored metrics must pass; a good mean cannot conceal a failed task."""
    checks = {}

    def at_least(name, value, reference, tolerance):
        checks[name] = bool(value is not None and reference is not None
                            and math.isfinite(value) and value >= reference - tolerance)

    at_least("seg_mIoU", current["dense"]["seg_teacher_mIoU"], baseline["dense"]["seg_teacher_mIoU"], limits["seg_miou_drop"])
    value, reference = current["dense"]["depth_teacher_absrel"], baseline["dense"]["depth_teacher_absrel"]
    checks["depth_absrel"] = bool(value is not None and reference is not None and math.isfinite(value)
                                   and value <= reference * (1 + limits["depth_absrel_increase_ratio"]))
    at_least("depth_delta1", current["dense"]["depth_teacher_delta1"], baseline["dense"]["depth_teacher_delta1"], limits["depth_delta1_drop"])
    for task, names in (("agent", ("vehicle", "pedestrian")), ("map", ("centerline", "ped_crossing", "road_boundary"))):
        for name in names:
            for metric in ("precision", "recall"):
                at_least(task + ":" + name + ":" + metric,
                    current[task][name][metric], baseline[task][name][metric], limits[task + "_precision_recall_drop"])
    if "seg_class_iou_drop" in limits:
        for index, reference in enumerate(baseline["seg_per_class"]):
            if reference["teacher_pixels"] >= limits["seg_min_teacher_pixels"]:
                at_least("seg_class:" + reference["class"], current["seg_per_class"][index]["iou"],
                         reference["iou"], limits["seg_class_iou_drop"])
    if "depth_far_absrel_increase_ratio" in limits:
        value, reference = current["depth_distance_bins"][-1], baseline["depth_distance_bins"][-1]
        checks["depth_far_absrel"] = bool(value["absrel"] is not None and reference["absrel"] is not None
            and math.isfinite(value["absrel"]) and value["absrel"] <= reference["absrel"] * (1 + limits["depth_far_absrel_increase_ratio"]))
    return {"all_pass": all(checks.values()), "checks": checks, "limits": dict(limits),
            "meaning": "NO_REGRESSION_TO_SAME_SPLIT_BASELINE_NOT_REAL_GT_QUALIFICATION"}
