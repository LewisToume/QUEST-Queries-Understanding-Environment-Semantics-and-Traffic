from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SPLIT_SCHEMA_VERSION = 2
MINIMUM_SPATIAL_DISTANCE_M = 200.0


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_hash(train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]) -> str:
    identity = {"schema_version": SPLIT_SCHEMA_VERSION, "splits": {
        name: [(int(row["index"]), str(row["token"]), str(row["scene_token"]), str(row["city"]))
               for row in rows]
        for name, rows in (("train", train), ("validation", validation))
    }}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_split(manifest: Mapping[str, Any], infos: Sequence[Mapping[str, Any]],
                   metadata_path: str | Path) -> None:
    import numpy as np
    from scipy.spatial import cKDTree

    from .vector_map_labels import require_lidar2global

    if manifest.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError("Stage 3 split schema is unsupported")
    if manifest.get("metadata_sha256") != file_sha256(metadata_path):
        raise ValueError("Stage 3 split metadata SHA256 mismatch")
    if manifest.get("metadata_frame_count") != len(infos):
        raise ValueError("Stage 3 split metadata frame count mismatch")
    train_minimum = float(manifest.get("minimum_train_distance_m", -1))
    scene_minimum = float(manifest.get("minimum_validation_scene_distance_m", -1))
    if (not np.isfinite(train_minimum) or train_minimum < MINIMUM_SPATIAL_DISTANCE_M
            or not np.isfinite(scene_minimum) or scene_minimum < MINIMUM_SPATIAL_DISTANCE_M):
        raise ValueError("Stage 3 split cannot lower either 200 m spatial constraint")
    train = manifest.get("train", {}).get("frames")
    validation = manifest.get("validation", {}).get("frames")
    if not isinstance(train, list) or not isinstance(validation, list):
        raise ValueError("Stage 3 split lacks full frame lists")
    if not validation:
        raise ValueError("Stage 3 split has no validation frames")
    if [row.get("index") for row in train] != list(range(5000)):
        raise ValueError("Stage 3 train split must preserve original indices 0-4999")
    indices: set[int] = set()
    tokens: set[str] = set()
    train_scenes = {str(row["scene_token"]) for row in train}
    train_by_city: dict[str, list[list[float]]] = {}
    validation_by_scene: dict[str, list[Mapping[str, Any]]] = {}
    for split_name, rows in (("train", train), ("validation", validation)):
        for row in rows:
            index = row.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(infos) or index in indices:
                raise ValueError(f"Stage 3 {split_name} has duplicate/out-of-range index {index}")
            info = infos[index]
            token = str(info["token"])
            if token != row.get("token") or str(info["scene_token"]) != row.get("scene_token"):
                raise ValueError(f"Stage 3 split index/token/scene mismatch at {index}")
            if not isinstance(row.get("city"), str) or not row["city"]:
                raise ValueError(f"Stage 3 split has no city at {index}")
            if token in tokens:
                raise ValueError(f"Stage 3 split has duplicate token {token}")
            indices.add(index)
            tokens.add(token)
            point = require_lidar2global(info)[:2, 3].tolist()
            if split_name == "validation":
                if index < 5000 or str(info["scene_token"]) in train_scenes:
                    raise ValueError(f"Stage 3 validation overlaps training scene/index at {index}")
                validation_by_scene.setdefault(row["scene_token"], []).append(row)
            else:
                train_by_city.setdefault(row["city"], []).append(point)
    scenes = manifest.get("validation", {}).get("scenes")
    if (not isinstance(scenes, list) or len(scenes) != 3
            or len(validation_by_scene) != 3
            or {scene.get("scene_token") for scene in scenes} != set(validation_by_scene)):
        raise ValueError("Stage 3 validation must contain exactly three described scenes")
    scene_cities = {scene.get("city") for scene in scenes}
    if len(scene_cities) != 1 or next(iter(scene_cities)) not in train_by_city:
        raise ValueError("Stage 3 validation scenes must share one training city")
    tree = cKDTree(np.asarray(train_by_city[next(iter(scene_cities))], dtype=np.float64))
    scene_points = {}
    all_scene_indices: dict[str, list[int]] = {scene: [] for scene in validation_by_scene}
    for index, info in enumerate(infos):
        scene = str(info["scene_token"])
        if scene in all_scene_indices:
            all_scene_indices[scene].append(index)
    for scene in scenes:
        name = scene["scene_token"]
        rows = validation_by_scene[name]
        if any(row["city"] != scene["city"] for row in rows):
            raise ValueError(f"Stage 3 scene/frame city mismatch for {name}")
        expected_indices = sorted(all_scene_indices[name],
                                  key=lambda index: (float(infos[index]["timestamp"]), index))
        listed_indices = list(scene.get("indices", []))
        if (listed_indices != expected_indices or set(listed_indices) != {row["index"] for row in rows}
                or scene.get("frame_count") != len(rows)):
            raise ValueError(f"Stage 3 scene description does not match complete frame list for {name}")
        points = np.asarray([require_lidar2global(infos[index])[:2, 3]
                             for index in listed_indices], dtype=np.float64)
        distances = tree.query(points)[0]
        if not np.all(np.isfinite(distances) & (distances >= train_minimum)):
            raise ValueError(f"Stage 3 scene {name} is not isolated from same-city training frames")
        by_index = {row["index"]: row for row in rows}
        for index, distance in zip(listed_indices, distances):
            recorded = float(by_index[index].get("nearest_train_m", -1))
            if not np.isfinite(recorded) or abs(recorded - distance) > 1e-3:
                raise ValueError(f"Stage 3 training distance is stale at index {index}")
        span = float(np.linalg.norm(points[:, None] - points[None, :], axis=-1).max())
        for key, actual in (("motion_extent_m", span), ("nearest_train_min_m", float(distances.min())),
                            ("nearest_train_median_m", float(np.median(distances)))):
            recorded = float(scene.get(key, -1))
            if not np.isfinite(recorded) or abs(recorded - actual) > 1e-3:
                raise ValueError(f"Stage 3 scene {name} has stale {key}")
        scene_points[name] = points
    for scene in scenes:
        name = scene["scene_token"]
        nearest = min(float(cKDTree(scene_points[other]).query(scene_points[name])[0].min())
                      for other in scene_points if other != name)
        if nearest < scene_minimum:
            raise ValueError(f"Stage 3 validation scenes overlap within {scene_minimum:g} m: {name}")
        recorded = float(scene.get("nearest_selected_scene_m", -1))
        if not np.isfinite(recorded) or abs(recorded - nearest) > 1e-3:
            raise ValueError(f"Stage 3 scene {name} has stale inter-scene distance")
    if manifest.get("split_sha256") != split_hash(train, validation):
        raise ValueError("Stage 3 split frame-list hash mismatch")


def load_stage3_split(path: str | Path, infos: Sequence[Mapping[str, Any]],
                      metadata_path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    validate_split(manifest, infos, metadata_path)
    return manifest


def selected_frames(manifest: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    if split not in ("train", "validation"):
        raise ValueError(f"unknown Stage 3 split: {split}")
    return list(manifest[split]["frames"])
