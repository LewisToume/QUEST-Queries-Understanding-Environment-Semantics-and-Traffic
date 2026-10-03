from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.vector_map_labels import extract_vector_map
from quest.map_training import validate_vector_record
from quest.stage3_dataset import load_record
from quest.utils import load_yaml_config
from scripts.run_navformer_openscene_teacher import load_infos, select_infos


def main() -> None:
    parser = argparse.ArgumentParser(description="Export nuPlan hard vector GT for QUEST")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/vector_map_gt")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/openscene/meta_datas/meta_data_mini.pkl")
    parser.add_argument("--map-root", type=Path, default=os.getenv("NUPLAN_MAPS_ROOT"))
    parser.add_argument("--map-version", default=os.getenv("NUPLAN_MAP_VERSION"))
    parser.add_argument("--num-points", type=int, default=20)
    parser.add_argument("--min-length-m", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.map_root is None or not args.map_version:
        raise ValueError("--map-root and --map-version (or NUPLAN_MAPS_ROOT/NUPLAN_MAP_VERSION) are required")
    if args.min_length_m < 0:
        raise ValueError("--min-length-m must be nonnegative")
    from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
    from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory

    factory = NuPlanMapFactory(GPKGMapsDB(str(args.map_root), args.map_version))
    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    x0, x1 = model_config["x_range"]
    y0, y1 = model_config["y_range"]
    quest_range = (float(x0), float(y0), float(x1), float(y1))
    infos = select_infos(load_infos(args.metadata), args.sample_index, args.num_frames)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    maps = {}
    for sample_index, info in enumerate(infos, start=args.sample_index):
        path = output / f"{info['token']}.pt"
        if path.exists() and not args.overwrite:
            try:
                existing = load_record(path)
                validate_vector_record(
                    existing, str(info["token"]), sample_index, quest_range,
                    expected_min_length_m=args.min_length_m,
                    expected_map_version=args.map_version,
                )
                if existing["num_points"] != args.num_points:
                    raise ValueError("existing vector num_points differs from --num-points")
            except Exception as error:
                raise ValueError(f"existing vector GT is invalid: {path}: {error}; use --overwrite to regenerate") from error
            print(f"index={sample_index} token={info['token']} skipped_existing=true")
            continue
        location = info.get("map_location")
        if not location:
            raise KeyError(f"OpenScene frame {sample_index} has no map_location")
        if location not in maps:
            maps[location] = factory.build_map_from_name(location)
        record = extract_vector_map(
            info, maps[location], sample_index, quest_range,
            args.num_points, args.min_length_m, map_version=args.map_version,
        )
        temporary = path.with_suffix(".pt.tmp")
        torch.save(record, temporary)
        temporary.replace(path)
        print(f"index={sample_index} token={info['token']} vectors={len(record['class_ids'])}")


if __name__ == "__main__":
    main()
