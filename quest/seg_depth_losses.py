"""Validity-masked dense pseudo supervision and teacher-agreement metrics."""

import math

import torch
import torch.nn.functional as F

from .seg_teacher import CITYSCAPES_CLASSES


def segmentation_class_weights(settings, reference):
    values = settings.get("seg", {}).get("class_weights")
    if values is None:
        return None
    if tuple(settings["seg"].get("class_names", CITYSCAPES_CLASSES)) != CITYSCAPES_CLASSES:
        raise ValueError("Seg class weight order must be canonical Cityscapes19")
    weights = torch.as_tensor(values, dtype=reference.dtype, device=reference.device)
    if weights.shape != (19,) or not bool((torch.isfinite(weights) & (weights > 0)).all()):
        raise ValueError("Seg class_weights must contain 19 finite positive values")
    return weights


def depth_valid_mask(target, seg_target, settings):
    value, valid = target["depth_m"], target["valid_mask"].clone()
    minimum, maximum = float(settings["min_depth_m"]), float(settings["max_depth_m"])
    if not (math.isfinite(minimum) and math.isfinite(maximum) and 0 < minimum < maximum):
        raise ValueError("Invalid depth supervision range")
    valid &= torch.isfinite(value) & (value >= minimum) & (value <= maximum)
    if settings["exclude_sky"]:
        if seg_target is None:
            raise ValueError("Sky exclusion requires token-aligned Seg teacher labels")
        valid &= (seg_target["valid_mask"] & (seg_target["labels"] != 10)).unsqueeze(1)
    return valid


def compute_seg_depth_loss(predictions, targets, settings, tasks=("seg", "depth")):
    seg, depth = predictions["seg_logits"], predictions["depth"]
    if seg.ndim != 5 or seg.shape[2] != 19 or depth.shape != (*seg.shape[:2], 1, *seg.shape[-2:]):
        raise ValueError("Require [B,N,19,H,W] segmentation and [B,N,1,H,W] camera-Z")
    if len(targets) != seg.shape[0] or not bool(torch.isfinite(seg).all() & torch.isfinite(depth).all()) or not bool((depth > 0).all()):
        raise ValueError("Nonfinite/invalid dense predictions or target batch mismatch")
    weights = settings["weights"]
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in weights.values()):
        raise ValueError("Dense loss weights must be finite and nonnegative")
    zero = seg.sum() * 0 + depth.sum() * 0
    class_weights = segmentation_class_weights(settings, seg)
    seg_normalizer = seg.new_zeros(())
    sums = {key: zero for key in ("seg_ce", "depth_log", "depth_relative", "depth_gradient")}
    counts = {"seg_pixels": 0, "depth_pixels": 0, "depth_edges": 0}
    for index, target in enumerate(targets):
        semantic = target.get("seg")
        if "seg" in tasks and semantic is not None:
            labels, valid = semantic["labels"].to(seg.device).long(), semantic["valid_mask"].to(seg.device)
            if labels.shape != (seg.shape[1], *seg.shape[-2:]) or valid.shape != labels.shape:
                raise ValueError("Seg target grid differs from Student")
            if bool(((labels[valid] < 0) | (labels[valid] >= 19)).any()):
                raise ValueError("Unsupported Cityscapes ID in valid target")
            safe_labels = torch.where(valid, labels, 0)
            pixels = F.cross_entropy(seg[index], safe_labels, weight=class_weights, reduction="none")
            sums["seg_ce"] = sums["seg_ce"] + pixels[valid].sum()
            if class_weights is not None:
                seg_normalizer = seg_normalizer + class_weights[safe_labels[valid]].sum()
            counts["seg_pixels"] += int(valid.sum())
        metric = target.get("depth")
        if "depth" in tasks and metric is not None:
            valid = depth_valid_mask(metric, semantic, settings["depth"]).to(depth.device)
            truth = metric["depth_m"].to(depth.device)
            if truth.shape != depth[index].shape or valid.shape != truth.shape:
                raise ValueError("Depth target grid differs from Student")
            safe_truth = torch.where(valid, truth, torch.ones_like(truth))
            predicted_log, target_log = depth[index].log(), safe_truth.log()
            sums["depth_log"] = sums["depth_log"] + F.smooth_l1_loss(
                predicted_log[valid], target_log[valid], reduction="sum")
            sums["depth_relative"] = sums["depth_relative"] + F.smooth_l1_loss(
                ((depth[index] - safe_truth) / safe_truth)[valid], torch.zeros_like(truth[valid]), reduction="sum")
            counts["depth_pixels"] += int(valid.sum())
            for axis in (-2, -1):
                edge_valid = valid.narrow(axis, 1, valid.shape[axis] - 1) & valid.narrow(axis, 0, valid.shape[axis] - 1)
                delta = predicted_log.diff(dim=axis) - target_log.diff(dim=axis)
                sums["depth_gradient"] = sums["depth_gradient"] + delta[edge_valid].abs().sum()
                counts["depth_edges"] += int(edge_valid.sum())
    # Match weighted CrossEntropyLoss(mean): normalize by target weights, not
    # raw pixel count, so changing the balance does not arbitrarily rescale Seg/Depth.
    denominator = (seg_normalizer.clamp_min(torch.finfo(seg.dtype).eps) if class_weights is not None
                   else max(counts["seg_pixels"], 1))
    losses = {"seg_ce": sums["seg_ce"] / denominator,
              "depth_log": sums["depth_log"] / max(counts["depth_pixels"], 1),
              "depth_relative": sums["depth_relative"] / max(counts["depth_pixels"], 1),
              "depth_gradient": sums["depth_gradient"] / max(counts["depth_edges"], 1)}
    losses["total_loss"] = sum(float(weights[key]) * losses[key] for key in losses)
    return {**losses, **counts}


def empty_dense_metrics():
    return {"confusion": torch.zeros(19, 19, dtype=torch.int64), "depth_pixels": 0,
            "absrel_sum": 0.0, "squared_error_sum": 0.0, "delta1_sum": 0}


@torch.no_grad()
def update_dense_metrics(metrics, predictions, targets, settings, tasks):
    for index, target in enumerate(targets):
        semantic = target.get("seg")
        if "seg" in tasks and semantic is not None:
            labels = semantic["labels"].long()
            valid = semantic["valid_mask"]
            predicted = predictions["seg_logits"][index].argmax(1).cpu()
            pairs = labels[valid] * 19 + predicted[valid]
            metrics["confusion"] += torch.bincount(pairs, minlength=19 * 19).reshape(19, 19)
        metric = target.get("depth")
        if "depth" in tasks and metric is not None:
            valid = depth_valid_mask(metric, semantic, settings["depth"])
            truth, predicted = metric["depth_m"][valid], predictions["depth"][index].detach().cpu()[valid]
            metrics["depth_pixels"] += truth.numel()
            metrics["absrel_sum"] += float(((predicted - truth).abs() / truth).sum())
            metrics["squared_error_sum"] += float((predicted - truth).square().sum())
            metrics["delta1_sum"] += int((torch.maximum(predicted / truth, truth / predicted) < 1.25).sum())


def summarize_dense_metrics(metrics):
    confusion = metrics["confusion"].double()
    union = confusion.sum(0) + confusion.sum(1) - confusion.diag()
    present, pixels, depth_pixels = union > 0, float(confusion.sum()), metrics["depth_pixels"]
    return {"seg_teacher_pixel_accuracy": float(confusion.diag().sum()) / pixels if pixels else None,
            "seg_teacher_mIoU": float((confusion.diag()[present] / union[present]).mean()) if bool(present.any()) else None,
            "depth_teacher_absrel": metrics["absrel_sum"] / depth_pixels if depth_pixels else None,
            "depth_teacher_rmse_m": math.sqrt(metrics["squared_error_sum"] / depth_pixels) if depth_pixels else None,
            "depth_teacher_delta1": metrics["delta1_sum"] / depth_pixels if depth_pixels else None,
            "reference": "OFFLINE_TEACHER_PSEUDO_NOT_REAL_GT"}
