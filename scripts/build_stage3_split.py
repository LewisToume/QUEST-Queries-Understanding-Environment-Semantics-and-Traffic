from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.nuplan_map_locator import NuPlanMapLocator
from quest.stage3_split import SPLIT_SCHEMA_VERSION, file_sha256, split_hash, validate_split
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


def choose_scenes(candidates: list[dict], train_logs: set[str], seed: int = 42,
                  target_frames: int = 120, target_scenes: int = 3) -> list[dict]:
    selected: list[dict] = []
    available = list(candidates)
    while available and len(selected) < target_scenes:
        ranked = []
        for item in available:
            separation = min(
                float(cKDTree(other["points"]).query(item["points"])[0].min())
                for other in selected if other["city"] == item["city"]
            ) if any(other["city"] == item["city"] for other in selected) else float("inf")
            new_city = item["city"] not in {other["city"] for other in selected}
            log_clean = item["log"] is not None and item["log"] not in train_logs
            log_clean &= item["log"] not in {other["log"] for other in selected}
            # The final component is a fixed-seed, stable tie-breaker, never Python's random hash.
            import hashlib
            tie = hashlib.sha256(f"{seed}:{item['scene']}".encode()).hexdigest()
            key = (len(item["indices"]) == 40, -abs(len(item["indices"]) - 40),
                   item["motion_m"] > 30,
                   log_clean, new_city, min(separation, 1000.0),
                   -abs(target_frames - (sum(len(x["indices"]) for x in selected) + len(item["indices"]))),
                   item["motion_m"], tie)
            ranked.append((key, item))
        chosen = max(ranked, key=lambda pair: pair[0])[1]
        selected.append(chosen)
        available.remove(chosen)
    return selected


def build_split(infos: list[dict], locator: NuPlanMapLocator, metadata_path: Path,
                minimum_distance_m: float = 200.0, seed: int = 42) -> dict:
    if len(infos) <= 5000:
        raise ValueError("OpenScene metadata has no frames beyond Stage 2 train indices 0-4999")
    if minimum_distance_m < 200:
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
    selected = choose_scenes(candidates, train_logs, seed)
    if len(selected) != 3:
        ranked_candidates = sorted(
            candidates,
            key=lambda item: (len(item["indices"]) == 40, item["motion_m"] > 30,
                              float(item["distances"].min()), item["scene"]), reverse=True,
        )
        best = [(item["scene"], item["city"], len(item["indices"]),
                 round(float(item["distances"].min()), 1)) for item in ranked_candidates[:10]]
        near_misses.sort(key=lambda row: (row["nearest_train_m"] is not None,
                                          row["nearest_train_m"] or -1), reverse=True)
        raise RuntimeError(f"only {len(selected)} isolated scenes found; rejected={dict(rejected)} "
                           f"best_eligible={best} nearest_rejected={near_misses[:10]}")
    train = [{"index": index, "token": str(infos[index]["token"]),
              "scene_token": str(infos[index]["scene_token"]), "city": cities[index]}
             for index in range(5000)]
    validation = []
    scenes = []
    for item in selected:
        other_scene_distance = min(
            float(cKDTree(other["points"]).query(item["points"])[0].min())
            for other in selected if other is not item and other["city"] == item["city"]
        ) if any(other is not item and other["city"] == item["city"] for other in selected) else None
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
                "train": {"frames": train}, "validation": {"frames": validation, "scenes": scenes},
                "rejected_scene_counts": dict(rejected),
                "log_identity_confirmed": all(item["log"] is not None for item in selected)}
    manifest["split_sha256"] = split_hash(train, validation)
    validate_split(manifest, infos, metadata_path)
    return manifest


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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        with args.output.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != manifest:
            raise ValueError(f"existing split differs; refusing to overwrite {args.output}")
    else:
        args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
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
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
