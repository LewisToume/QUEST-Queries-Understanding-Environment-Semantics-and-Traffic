from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.nuplan_map_locator import NuPlanMapLocator
from quest.stage3_split import (
    SPLIT_SCHEMA_VERSION, file_sha256, split_hash, validate_split,
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


def motion_extent(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1).max())


def choose_scenes(candidates: list[dict], train_cities: set[str], seed: int = 42) -> list[dict]:
    """Sample complete scenes, favoring training cities and one high-motion scene."""
    ordered = sorted(candidates, key=lambda item: (item["city"], item["scene"]))
    if len(ordered) < 3:
        raise ValueError(f"need three complete 40-frame scenes, found {len(ordered)}")
    rng = random.Random(seed)
    preferred = [item for item in ordered if item["city"] in train_cities]
    moving_pool = sorted(ordered, key=lambda item: (-item["motion_m"], item["city"], item["scene"]))
    top_quarter = max(1, (len(moving_pool) + 3) // 4)
    moving_pool = moving_pool[:top_quarter]
    preferred_moving = [item for item in moving_pool if item["city"] in train_cities]
    selected = [rng.choice(preferred_moving or moving_pool)]
    selected_scenes = {selected[0]["scene"]}
    remaining_preferred = [item for item in preferred if item["scene"] not in selected_scenes]
    selected.extend(rng.sample(remaining_preferred, min(2, len(remaining_preferred))))
    if len(selected) < 3:
        selected_scenes = {item["scene"] for item in selected}
        remaining = [item for item in ordered if item["scene"] not in selected_scenes]
        selected.extend(rng.sample(remaining, 3 - len(selected)))
    return selected


def build_split(infos: list[dict], locator: NuPlanMapLocator, metadata_path: Path,
                seed: int = 42) -> dict:
    if len(infos) <= 5000:
        raise ValueError("OpenScene metadata has no frames beyond Stage 2 train indices 0-4999")
    cities: list[str] = []
    positions = np.empty((len(infos), 2), dtype=np.float64)
    by_scene: dict[str, list[int]] = defaultdict(list)
    for index, info in enumerate(infos):
        city = locator.resolve(info)
        cities.append(city)
        positions[index] = require_lidar2global(info)[:2, 3]
        by_scene[str(info["scene_token"])].append(index)
    train_scenes = {str(infos[index]["scene_token"]) for index in range(5000)}
    train_cities = set(cities[:5000])
    train_logs = {value for index in range(5000) if (value := log_identity(infos[index])) is not None}
    candidates = []
    rejected = Counter()
    for scene, indices in by_scene.items():
        if scene in train_scenes:
            rejected["shared_training_scene"] += 1
            continue
        if len(indices) != 40:
            rejected["scene_not_40_frames"] += 1
            continue
        if any(index < 5000 for index in indices):
            raise ValueError(f"scene {scene} unexpectedly crosses training boundary")
        scene_cities = {cities[index] for index in indices}
        if len(scene_cities) != 1:
            raise ValueError(f"scene {scene} crosses cities/projections")
        city = next(iter(scene_cities))
        ordered = sorted(indices, key=lambda index: (float(infos[index]["timestamp"]), index))
        timestamps = [float(infos[index]["timestamp"]) for index in ordered]
        if (not all(np.isfinite(timestamps))
                or any(right <= left for left, right in zip(timestamps, timestamps[1:]))):
            rejected["nonmonotonic_scene"] += 1
            continue
        scene_points = positions[ordered]
        logs = {log_identity(infos[index]) for index in ordered}
        logs.discard(None)
        if len(logs) > 1:
            rejected["inconsistent_log_identity"] += 1
            continue
        candidates.append({"scene": scene, "city": city, "indices": ordered,
                           "motion_m": motion_extent(scene_points),
                           "log": next(iter(logs)) if logs else None})
    if len(candidates) < 3:
        raise RuntimeError(f"need three complete 40-frame scenes outside training; "
                           f"eligible={len(candidates)} rejected={dict(rejected)}")
    selected = choose_scenes(candidates, train_cities, seed)
    train = [{"index": index, "token": str(infos[index]["token"]),
              "scene_token": str(infos[index]["scene_token"]), "city": cities[index]}
             for index in range(5000)]
    validation = []
    scenes = []
    for item in selected:
        scenes.append({"scene_token": item["scene"], "city": item["city"],
                       "indices": item["indices"], "frame_count": len(item["indices"]),
                       "motion_extent_m": item["motion_m"],
                       "log_identity": item["log"],
                       "log_shared_with_training": item["log"] in train_logs if item["log"] else None})
        for index in item["indices"]:
            validation.append({"index": index, "token": str(infos[index]["token"]),
                               "scene_token": item["scene"], "city": item["city"]})
    validation.sort(key=lambda row: row["index"])
    manifest = {"schema_version": SPLIT_SCHEMA_VERSION, "seed": seed,
                "metadata_sha256": file_sha256(metadata_path), "metadata_frame_count": len(infos),
                "train": {"frames": train}, "validation": {"frames": validation, "scenes": scenes},
                "rejected_scene_counts": dict(rejected),
                "log_identity_confirmed": all(item["log"] is not None for item in selected)}
    manifest["split_sha256"] = split_hash(train, validation)
    validate_split(manifest, infos, metadata_path)
    return manifest


def save_split_manifest(output: Path, manifest: dict) -> Path | None:
    """Archive an older split byte-for-byte before replacing it with v3."""
    output.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if output.exists():
        old_bytes = output.read_bytes()
        existing = json.loads(old_bytes)
        if existing == manifest:
            return None
        old_version = existing.get("schema_version")
        if type(old_version) is not int or old_version not in (1, 2):
            raise ValueError(f"existing split is not v1/v2; refusing to overwrite {output}")
        backup = output.with_name(f"{output.stem}.v{old_version}.backup{output.suffix}")
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
    parser = argparse.ArgumentParser(description="Build deterministic Stage 3 split from original OpenScene indices")
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
