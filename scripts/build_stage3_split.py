from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.nuplan_map_locator import NuPlanMapLocator
from quest.stage3_split import (
    MINIMUM_SPATIAL_DISTANCE_M, SPLIT_SCHEMA_VERSION, file_sha256, split_hash,
    validate_split,
)
from quest.vector_map_labels import require_lidar2global
from scripts.run_navformer_openscene_teacher import load_infos


LOG_KEYS = ("log_token", "log_name", "log_id", "logfile", "log_path")


def log_identity(info: dict) -> str | None:
    for key in LOG_KEYS:
        value = info.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{value}"
    return None


def nearest_same_city(train_by_city: dict[str, np.ndarray], cities: list[str],
                      points: np.ndarray) -> np.ndarray:
    """Never query a KD-tree containing positions from another map projection."""
    if len(cities) != len(points):
        raise ValueError("city and position counts differ")
    result = np.full(len(points), np.inf)
    for city in set(cities):
        if city not in train_by_city:
            continue
        query = np.flatnonzero(np.asarray(cities) == city)
        result[query] = cKDTree(train_by_city[city]).query(points[query])[0]
    return result


def motion_extent(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1).max())


def scene_distance_m(left: dict, right: dict) -> float:
    if left["city"] != right["city"]:
        raise ValueError("cannot compare scene coordinates across cities")
    return float(cKDTree(right["points"]).query(left["points"])[0].min())


def scene_priority(item: dict, train_logs: set[str], seed: int) -> tuple:
    count = len(item["indices"])
    tie = hashlib.sha256(f"{seed}:{item['city']}:{item['scene']}".encode()).hexdigest()
    return (count == 40 and item["motion_m"] > 30, count == 40,
            item["motion_m"] > 30, -abs(count - 40),
            item["log"] is not None and item["log"] not in train_logs,
            float(item["distances"].min()), item["motion_m"], tie)


def _first_triangle(vertices: list[int], edges: dict[int, set[int]]) -> tuple[int, int, int] | None:
    """Search every compatible pair; a single greedy path cannot prove absence."""
    for position, first in enumerate(vertices):
        for second_position in range(position + 1, len(vertices)):
            second = vertices[second_position]
            if second not in edges[first]:
                continue
            shared = edges[first] & edges[second]
            for third in vertices[second_position + 1:]:
                if third in shared:
                    return first, second, third
    return None


def find_scene_group(candidates: list[dict], train_logs: set[str], seed: int = 42,
                     minimum_distance_m: float = MINIMUM_SPATIAL_DISTANCE_M,
                     ) -> tuple[list[dict], dict[str, dict]]:
    if minimum_distance_m < MINIMUM_SPATIAL_DISTANCE_M:
        raise ValueError("validation scene separation cannot be below 200 m")
    by_city: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        by_city[item["city"]].append(item)
    city_reports: dict[str, dict] = {}
    feasible = []
    for city, items in sorted(by_city.items()):
        ordered = sorted(items, key=lambda item: scene_priority(item, train_logs, seed), reverse=True)
        edges = {index: set() for index in range(len(ordered))}
        for first in range(len(ordered)):
            for second in range(first + 1, len(ordered)):
                if scene_distance_m(ordered[first], ordered[second]) >= minimum_distance_m:
                    edges[first].add(second)
                    edges[second].add(first)
        vertices = list(range(len(ordered)))
        preferred = [index for index in vertices
                     if len(ordered[index]["indices"]) == 40 and ordered[index]["motion_m"] > 30]
        triangle = _first_triangle(preferred, edges) or _first_triangle(vertices, edges)
        pair = next(((first, second) for first in vertices for second in vertices
                     if second > first and second in edges[first]), None)
        maximum = 3 if triangle else (2 if pair else (1 if ordered else 0))
        best_indices = triangle or pair or ((0,) if ordered else ())
        city_reports[city] = {
            "eligible_scenes": len(ordered), "maximum_compatible_scenes": maximum,
            "best_candidate_scenes": [ordered[index]["scene"] for index in best_indices],
            "top_candidates": [
                {"scene": item["scene"], "frames": len(item["indices"]),
                 "motion_extent_m": item["motion_m"],
                 "nearest_train_min_m": float(item["distances"].min())}
                for item in ordered[:10]
            ],
        }
        if triangle:
            group = [ordered[index] for index in triangle]
            quality = tuple(scene_priority(item, train_logs, seed) for item in group)
            feasible.append((quality, city, group))
    if not feasible:
        return [], city_reports
    return max(feasible, key=lambda result: (result[0], result[1]))[2], city_reports


def choose_scenes(candidates: list[dict], train_logs: set[str], seed: int = 42,
                  minimum_distance_m: float = MINIMUM_SPATIAL_DISTANCE_M) -> list[dict]:
    return find_scene_group(candidates, train_logs, seed, minimum_distance_m)[0]


def build_split(infos: list[dict], locator: NuPlanMapLocator, metadata_path: Path,
                minimum_distance_m: float = 200.0, seed: int = 42) -> dict:
    if len(infos) <= 5000:
        raise ValueError("OpenScene metadata has no frames beyond Stage 2 train indices 0-4999")
    if minimum_distance_m < MINIMUM_SPATIAL_DISTANCE_M:
        raise ValueError("Stage 3 isolation distance cannot be lowered below 200 m")
    cities: list[str] = []
    positions = np.empty((len(infos), 2), dtype=np.float64)
    by_scene: dict[str, list[int]] = defaultdict(list)
    for index, info in enumerate(infos):
        city = locator.resolve(info)
        cities.append(city)
        positions[index] = require_lidar2global(info)[:2, 3]
        by_scene[str(info["scene_token"])].append(index)
    train_scenes = {str(infos[index]["scene_token"]) for index in range(5000)}
    train_by_city = {city: positions[[index for index in range(5000) if cities[index] == city]]
                     for city in set(cities[:5000])}
    train_logs = {value for index in range(5000) if (value := log_identity(infos[index])) is not None}
    candidates = []
    rejected = Counter()
    near_misses = []
    for scene, indices in by_scene.items():
        if scene in train_scenes:
            rejected["shared_training_scene"] += 1
            continue
        if not 30 <= len(indices) <= 50:
            rejected["scene_not_near_40_frames"] += 1
            continue
        if any(index < 5000 for index in indices):
            raise ValueError(f"scene {scene} unexpectedly crosses training boundary")
        scene_cities = {cities[index] for index in indices}
        if len(scene_cities) != 1:
            raise ValueError(f"scene {scene} crosses cities/projections")
        city = next(iter(scene_cities))
        if city not in train_by_city:
            rejected["city_not_in_training"] += 1
            near_misses.append({"scene": scene, "city": city, "frames": len(indices),
                                "reason": "city_not_in_training", "nearest_train_m": None})
            continue
        ordered = sorted(indices, key=lambda index: (float(infos[index]["timestamp"]), index))
        timestamps = [float(infos[index]["timestamp"]) for index in ordered]
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            rejected["nonmonotonic_scene"] += 1
            continue
        scene_points = positions[ordered]
        distances = nearest_same_city(train_by_city, [city] * len(ordered), scene_points)
        if not bool(np.all(np.isfinite(distances) & (distances >= minimum_distance_m))):
            rejected["within_200m_of_training"] += 1
            near_misses.append({"scene": scene, "city": city, "frames": len(indices),
                                "reason": "within_200m_of_training",
                                "nearest_train_m": float(distances.min())})
            continue
        logs = {log_identity(infos[index]) for index in ordered}
        logs.discard(None)
        if len(logs) > 1:
            rejected["inconsistent_log_identity"] += 1
            continue
        candidates.append({"scene": scene, "city": city, "indices": ordered,
                           "points": scene_points, "distances": distances,
                           "motion_m": motion_extent(scene_points),
                           "log": next(iter(logs)) if logs else None})
    selected, city_reports = find_scene_group(candidates, train_logs, seed, minimum_distance_m)
    if len(selected) != 3:
        near_misses.sort(key=lambda row: (row["nearest_train_m"] is not None,
                                          row["nearest_train_m"] or -1), reverse=True)
        maximum = max((report["maximum_compatible_scenes"] for report in city_reports.values()), default=0)
        raise RuntimeError(f"no same-city triple has pairwise separation >= {minimum_distance_m:g} m; "
                           f"maximum_compatible_scenes={maximum} city_candidates={city_reports} "
                           f"rejected={dict(rejected)} nearest_rejected={near_misses[:10]}")
    train = [{"index": index, "token": str(infos[index]["token"]),
              "scene_token": str(infos[index]["scene_token"]), "city": cities[index]}
             for index in range(5000)]
    validation = []
    scenes = []
    for item in selected:
        other_scene_distance = min(scene_distance_m(item, other)
                                   for other in selected if other is not item)
        scenes.append({"scene_token": item["scene"], "city": item["city"],
                       "indices": item["indices"], "frame_count": len(item["indices"]),
                       "motion_extent_m": item["motion_m"],
                       "nearest_train_min_m": float(item["distances"].min()),
                       "nearest_train_median_m": float(np.median(item["distances"])),
                       "nearest_selected_scene_m": other_scene_distance,
                       "log_identity": item["log"],
                       "log_shared_with_training": item["log"] in train_logs if item["log"] else None})
        for index, distance in zip(item["indices"], item["distances"]):
            validation.append({"index": index, "token": str(infos[index]["token"]),
                               "scene_token": item["scene"], "city": item["city"],
                               "nearest_train_m": float(distance)})
    validation.sort(key=lambda row: row["index"])
    manifest = {"schema_version": SPLIT_SCHEMA_VERSION, "seed": seed,
                "metadata_sha256": file_sha256(metadata_path), "metadata_frame_count": len(infos),
                "minimum_train_distance_m": minimum_distance_m,
                "minimum_validation_scene_distance_m": minimum_distance_m,
                "train": {"frames": train}, "validation": {"frames": validation, "scenes": scenes},
                "rejected_scene_counts": dict(rejected), "city_candidate_summary": city_reports,
                "log_identity_confirmed": all(item["log"] is not None for item in selected)}
    manifest["split_sha256"] = split_hash(train, validation)
    validate_split(manifest, infos, metadata_path)
    return manifest


def save_split_manifest(output: Path, manifest: dict) -> Path | None:
    """Archive an existing v1 split byte-for-byte before replacing it with v2."""
    output.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if output.exists():
        old_bytes = output.read_bytes()
        existing = json.loads(old_bytes)
        if existing == manifest:
            return None
        if existing.get("schema_version") != 1:
            raise ValueError(f"existing non-v1 split differs; refusing to overwrite {output}")
        backup = output.with_name(f"{output.stem}.v1.backup{output.suffix}")
        if backup.exists():
            if backup.read_bytes() != old_bytes:
                raise ValueError(f"backup differs from existing split; refusing to overwrite {output}")
        else:
            shutil.copy2(output, backup)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Build isolated Stage 3 split from original OpenScene indices")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/openscene/meta_datas/meta_data_mini.pkl")
    parser.add_argument("--map-root", type=Path, default=os.getenv("NUPLAN_MAPS_ROOT"))
    parser.add_argument("--map-version", default=os.getenv("NUPLAN_MAP_VERSION"))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/stage3_split.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vector-dir", type=Path, default=PROJECT_ROOT / "data/vector_map_gt")
    parser.add_argument("--agent-dir", type=Path, default=PROJECT_ROOT / "data/soft_labels_navformer")
    parser.add_argument("--map-teacher-dir", type=Path, default=PROJECT_ROOT / "data/navformer_map_soft")
    args = parser.parse_args()
    if args.map_root is None or not args.map_version:
        raise ValueError("nuPlan map root/version required for city-specific projected coordinates")
    from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB

    infos = load_infos(args.metadata)
    db = GPKGMapsDB(map_version=args.map_version, map_root=str(args.map_root))
    manifest = build_split(infos, NuPlanMapLocator(db), args.metadata, seed=args.seed)
    backup = save_split_manifest(args.output, manifest)
    print(f"train_frames={len(manifest['train']['frames'])} validation_frames={len(manifest['validation']['frames'])}")
    print(f"validation_city_counts={dict(Counter(row['city'] for row in manifest['validation']['frames']))}")
    nearest = np.asarray([row["nearest_train_m"] for row in manifest["validation"]["frames"]])
    print(f"validation_nearest_train_min_m={nearest.min():.3f} "
          f"validation_nearest_train_median_m={np.median(nearest):.3f}")
    for scene in manifest["validation"]["scenes"]:
        print(json.dumps(scene, sort_keys=True))
    for split_name in ("train", "validation"):
        for name, directory in (("vector_gt", args.vector_dir), ("agent", args.agent_dir),
                                ("map_teacher", args.map_teacher_dir)):
            missing = [row["token"] for row in manifest[split_name]["frames"]
                       if not (directory / f"{row['token']}.pt").is_file()]
            print(f"missing_{split_name}_{name}={len(missing)} examples={missing[:10]}")
    print(f"split_sha256={manifest['split_sha256']} log_identity_confirmed={manifest['log_identity_confirmed']}")
    if backup is not None:
        print(f"previous_split_backup={backup}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
