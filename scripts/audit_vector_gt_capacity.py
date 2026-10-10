from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.map_training import VECTOR_PROVENANCE_KEYS, validate_vector_record
from quest.stage3_split import load_stage3_split
from quest.stage3_dataset import load_record
from quest.utils import load_yaml_config
from quest.vector_map_labels import MAP_CLASS_NAMES, VECTOR_GT_SCHEMA_VERSION, VECTOR_SEMANTICS_VERSION
from scripts.run_navformer_openscene_teacher import load_infos


def summary(values: list[int]) -> dict:
    array = np.asarray(values, dtype=np.int64)
    return {
        "frames": len(values), "total_instances": int(array.sum()) if len(array) else 0,
        "p50": float(np.percentile(array, 50)) if len(array) else 0.0,
        "p95": float(np.percentile(array, 95)) if len(array) else 0.0,
        "p99": float(np.percentile(array, 99)) if len(array) else 0.0,
        "max": int(array.max()) if len(array) else 0,
        "frames_exceeding": {str(n): int((array > n).sum()) for n in (50, 64, 100, 128)},
    }


def recommended_capacity(max_instances: int) -> int:
    for slots in (50, 64, 100, 128):
        if max_instances <= slots:
            return slots
    return ((max_instances + 31) // 32) * 32


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit nuPlan Vector GT counts without truncation")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/openscene/meta_datas/meta_data_mini.pkl")
    parser.add_argument("--vector-dir", type=Path, default=PROJECT_ROOT / "data/vector_map_gt")
    parser.add_argument("--train-start", type=int)
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--eval-start", type=int)
    parser.add_argument("--eval-count", type=int)
    parser.add_argument("--show-vectors", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    args = parser.parse_args()
    config = load_yaml_config(PROJECT_ROOT / "configs/stage3_map.yaml")
    model = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    xy_range = (float(model["x_range"][0]), float(model["y_range"][0]),
                float(model["x_range"][1]), float(model["y_range"][1]))
    infos = load_infos(args.metadata)
    manifest_path = args.split_manifest or PROJECT_ROOT / config["paths"]["split_manifest_path"]
    manifest = load_stage3_split(manifest_path, infos, args.metadata)
    if any(value is not None for value in (args.train_start, args.train_count, args.eval_start, args.eval_count)):
        raise ValueError("range overrides cannot certify the formal Stage 3 frame-list split")
    split_rows = (("train", manifest["train"]["frames"]),
                  ("eval", manifest["validation"]["frames"]))
    provenance = None
    scene_locations = {}
    results = {}
    missing = []
    for split, rows in split_rows:
        totals = []
        cast_warning_frames = cast_warning_events = 0
        preexisting_invalid_outside_roi_frames = 0
        by_class = {name: [] for name in MAP_CLASS_NAMES}
        for position, row in enumerate(rows):
            index = row["index"]
            info = infos[index]
            token = str(info["token"])
            path = args.vector_dir / f"{token}.pt"
            if not path.is_file():
                missing.append(f"{split} index={index} token={token}")
                continue
            record = load_record(path)
            validate_vector_record(record, token, index, xy_range, expected_info=info)
            scene = str(info["scene_token"])
            previous_location = scene_locations.setdefault(scene, record["map_location"])
            if previous_location != record["map_location"]:
                raise ValueError(f"mixed vector GT cities within scene={scene}")
            current = {key: record[key] for key in VECTOR_PROVENANCE_KEYS}
            if provenance is None:
                provenance = current
            elif current != provenance:
                raise ValueError(f"mixed vector GT provenance at {split} index={index} token={token}")
            classes = record["class_ids"]
            cast_warning_frames += bool(record["map_cast_diagnostics"]["invalid_cast_warning_count"])
            cast_warning_events += record["map_cast_diagnostics"]["invalid_cast_warning_count"]
            preexisting_invalid_outside_roi_frames += any(
                item["preexisting_invalid_outside_roi"]
                for item in record["map_layer_diagnostics"]["per_layer"].values()
            )
            totals.append(len(classes))
            for class_id, name in enumerate(MAP_CLASS_NAMES):
                by_class[name].append(int((classes == class_id).sum()))
            if args.show_vectors and position == 0:
                for class_id, points in zip(classes[:args.show_vectors], record["points_xy_m"][:args.show_vectors]):
                    print(f"vector token={token} class={MAP_CLASS_NAMES[int(class_id)]} "
                          f"first_xy_m={points[0].tolist()} last_xy_m={points[-1].tolist()}")
        results[split] = {
            "indices": [row["index"] for row in rows],
            "tokens": [row["token"] for row in rows],
            "num_samples": len(rows), "available": len(totals),
            "total": summary(totals),
            "per_class": {name: summary(values) for name, values in by_class.items()},
            "map_cast_warning_frames": cast_warning_frames,
            "map_cast_warning_events": cast_warning_events,
            "frames_with_preexisting_invalid_geometry_outside_roi": preexisting_invalid_outside_roi_frames,
        }
    max_count = max(results["train"]["total"]["max"], results["eval"]["total"]["max"])
    certified = not missing and bool(manifest["validation"]["frames"])
    report = {
        "schema_version": VECTOR_GT_SCHEMA_VERSION, "vector_semantics_version": VECTOR_SEMANTICS_VERSION,
        "split_sha256": manifest["split_sha256"], "metadata_sha256": manifest["metadata_sha256"],
        "vector_gt_provenance": provenance, "capacity_certified": certified,
        "recommended_map_query_count": recommended_capacity(max_count) if certified else None,
        "splits": results, "missing_count": len(missing), "missing_examples": missing[:20],
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        certified_path = PROJECT_ROOT / config["paths"]["vector_capacity_audit_path"]
        if not certified and args.output.resolve() == certified_path.resolve():
            raise ValueError("partial audit cannot overwrite the certified capacity report")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    elif certified:
        destination = PROJECT_ROOT / config["paths"]["vector_capacity_audit_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"capacity_audit_saved={destination}")
    if missing:
        print(f"missing_vector_gt={len(missing)}; capacity is not certified")


if __name__ == "__main__":
    main()
