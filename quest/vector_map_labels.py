from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


VECTOR_GT_SCHEMA_VERSION = 1
MAP_CLASS_NAMES = ("centerline", "ped_crossing", "road_boundary")
COORDINATE_FRAME = "openscene_lidar_xy"


def require_lidar2global(info: Mapping[str, Any]) -> np.ndarray:
    matrix = np.asarray(info.get("lidar2global"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("metadata lidar2global must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("lidar2global has an invalid homogeneous row")
    if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-3):
        raise ValueError("lidar2global rotation is not orthonormal")
    inverse = np.linalg.inv(matrix)
    if np.max(np.abs(inverse[:2, 2])) > 1e-3:
        raise ValueError("tilted lidar frame needs a 3D map height transform")
    return matrix


def global_to_local_geometry(geometry: Any, lidar2global: np.ndarray) -> Any:
    from shapely.affinity import affine_transform

    inverse = np.linalg.inv(lidar2global)
    return affine_transform(
        geometry,
        [inverse[0, 0], inverse[0, 1], inverse[1, 0], inverse[1, 1],
         inverse[0, 3], inverse[1, 3]],
    )


def _line_parts(geometry: Any):
    if geometry.is_empty:
        return
    if geometry.geom_type in ("LineString", "LinearRing"):
        yield geometry
    elif geometry.geom_type == "Polygon":
        yield geometry.exterior
        for interior in geometry.interiors:
            yield interior
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _line_parts(part)


def resample_polyline(coords: np.ndarray, num_points: int, closed: bool) -> np.ndarray:
    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or num_points < 2:
        raise ValueError("polyline requires [N,2+] coordinates and at least two samples")
    points = points[:, :2]
    if closed and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if len(np.unique(points, axis=0)) < 2:
        raise ValueError("polyline has fewer than two distinct points")
    if closed:
        points = np.concatenate((points, points[:1]), axis=0)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    keep = np.concatenate(([True], np.diff(cumulative) > 1e-9))
    cumulative, points = cumulative[keep], points[keep]
    if cumulative[-1] <= 0:
        raise ValueError("polyline has zero physical length")
    positions = np.linspace(0, cumulative[-1], num_points, endpoint=not closed)
    return np.stack([np.interp(positions, cumulative, points[:, axis]) for axis in (0, 1)], axis=-1).astype(np.float32)


def process_geometry(
    geometry: Any,
    lidar2global: np.ndarray,
    xy_range: tuple[float, float, float, float],
    num_points: int = 20,
    min_length_m: float = 1.0,
) -> list[tuple[np.ndarray, bool, float]]:
    from shapely.geometry import box

    if geometry is None or geometry.is_empty or not geometry.is_valid:
        return []
    local = global_to_local_geometry(geometry, lidar2global)
    clipped = local.intersection(box(*xy_range))
    if clipped.is_empty or not clipped.is_valid:
        return []
    results = []
    for line in _line_parts(clipped):
        if line.is_empty or not line.is_valid or line.length < min_length_m:
            continue
        coords = np.asarray(line.coords, dtype=np.float64)
        if len(np.unique(coords[:, :2], axis=0)) < 2:
            continue
        closed = bool(line.is_ring)
        results.append((resample_polyline(coords, num_points, closed), closed, float(line.length)))
    return results


def _baseline_geometry(map_object: Any) -> Any:
    from shapely.geometry import LineString

    path = getattr(map_object, "baseline_path", None)
    if path is None:
        raise ValueError("LANE/LANE_CONNECTOR object has no baseline_path")
    geometry = getattr(path, "linestring", None)
    if geometry is not None:
        return geometry
    discrete = getattr(path, "discrete_path", None)
    if discrete is None:
        raise ValueError("baseline_path has neither linestring nor discrete_path")
    return LineString([(state.x, state.y) for state in discrete])


def extract_vector_map(
    info: Mapping[str, Any], map_api: Any, sample_index: int,
    xy_range: tuple[float, float, float, float], num_points: int = 20,
    min_length_m: float = 1.0,
) -> dict[str, Any]:
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.common.actor_state.state_representation import Point2D

    if num_points != 20:
        raise ValueError("QUEST MapHead requires exactly 20 points")
    matrix = require_lidar2global(info)
    layer_names = {
        0: ("LANE", "LANE_CONNECTOR"),
        1: ("CROSSWALK",),
        2: ("ROADBLOCK", "INTERSECTION", "CARPARK_AREA"),
    }
    layers = []
    for names in layer_names.values():
        for name in names:
            layer = getattr(SemanticMapLayer, name, None)
            if layer is None:
                raise RuntimeError(f"nuPlan SemanticMapLayer.{name} is unavailable")
            layers.append(layer)
    x0, y0, x1, y1 = xy_range
    radius = max(abs(x0), abs(x1), abs(y0), abs(y1)) * 2**0.5 + 10.0
    origin = matrix[:2, 3]
    objects = map_api.get_proximal_map_objects(Point2D(float(origin[0]), float(origin[1])), radius, layers)
    classes, points, closed_flags, lengths = [], [], [], []
    for class_id, names in layer_names.items():
        for name in names:
            for map_object in objects.get(getattr(SemanticMapLayer, name), []):
                geometry = _baseline_geometry(map_object) if class_id == 0 else getattr(map_object, "polygon", None)
                if geometry is None:
                    raise ValueError(f"nuPlan {name} object has no usable geometry")
                for sampled, closed, length in process_geometry(
                    geometry, matrix, xy_range, num_points, min_length_m
                ):
                    classes.append(class_id)
                    points.append(sampled)
                    closed_flags.append(closed)
                    lengths.append(length)
    return {
        "sample_index": int(sample_index), "token": str(info["token"]),
        "class_ids": torch.tensor(classes, dtype=torch.int64),
        "points_xy_m": torch.tensor(np.stack(points) if points else np.empty((0, num_points, 2)), dtype=torch.float32),
        "is_closed": torch.tensor(closed_flags, dtype=torch.bool),
        "length_m": torch.tensor(lengths, dtype=torch.float32),
        "xy_range_m": tuple(float(v) for v in xy_range),
        "coordinate_frame": COORDINATE_FRAME,
        "schema_version": VECTOR_GT_SCHEMA_VERSION,
    }
