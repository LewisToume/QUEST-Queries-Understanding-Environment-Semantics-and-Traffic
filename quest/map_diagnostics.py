from __future__ import annotations

from typing import Any, Mapping

import torch

from .map_training import (
    class_aware_vector_matches, denormalize_points, normalize_points, pairwise_vector_distances,
)
from .vector_map_labels import MAP_CLASS_NAMES


def oracle_check(record: Mapping[str, Any], xy_range: tuple) -> None:
    points = record["points_xy_m"]
    recovered = denormalize_points(normalize_points(points, xy_range), xy_range)
    if not torch.allclose(points, recovered, atol=1e-4, rtol=0):
        raise ValueError("Vector GT normalization round-trip failed")
    for metric in ("ordered", "chamfer"):
        matches = class_aware_vector_matches(record["class_ids"], recovered, record, 0.01, metric)
        if len(matches) != len(points):
            raise ValueError(f"GT-as-prediction oracle failed for {metric}")


def frame_geometry_stats(labels: torch.Tensor, points: torch.Tensor, record: Mapping[str, Any]) -> dict:
    result = {}
    for class_id, name in enumerate(MAP_CLASS_NAMES):
        predicted = points[labels == class_id]
        chosen = record["class_ids"] == class_id
        target = {key: record[key][chosen] for key in ("class_ids", "points_xy_m", "is_closed")}
        item = {"gt": int(chosen.sum()), "predictions": len(predicted)}
        for metric in ("ordered", "chamfer"):
            matches = class_aware_vector_matches(
                labels[labels == class_id], predicted, target, 1.5, metric)
            costs = pairwise_vector_distances(predicted, target, metric)
            item[f"{metric}_matched_1_5m"] = len(matches)
            item[f"{metric}_nearest_mean_m"] = (
                float(costs.min(-1).values.mean()) if costs.shape[0] and costs.shape[1] else None)
        item["predicted_mean_length_m"] = (float(torch.linalg.vector_norm(
            predicted[:, 1:] - predicted[:, :-1], dim=-1).sum(-1).mean()) if len(predicted) else None)
        target_points = target["points_xy_m"]
        item["gt_mean_length_m"] = (float(record["length_m"][chosen].mean()) if len(target_points) else None)
        item["predicted_center_std_xy_m"] = (predicted.mean(1).std(0, unbiased=False).tolist()
                                               if len(predicted) else None)
        result[name] = item
    return result


def evenly_spaced_indices(frames: list[Mapping[str, Any]], count: int) -> list[int]:
    if count <= 0 or count > len(frames):
        raise ValueError("diagnostic sample count must fit each manifest split")
    positions = [round(index * (len(frames) - 1) / max(count - 1, 1)) for index in range(count)]
    return [int(frames[position]["index"]) for position in positions]
