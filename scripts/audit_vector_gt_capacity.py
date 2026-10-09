from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.map_training import VECTOR_PROVENANCE_KEYS, validate_vector_record
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
    args = parser.parse_args()
    config = load_yaml_config(PROJECT_ROOT / "configs/stage3_map.yaml")
    model = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    xy_range = (float(model["x_range"][0]), float(model["y_range"][0]),
                float(model["x_range"][1]), float(model["y_range"][1]))
    train_start = args.train_start if args.train_start is not None else int(config["train"]["start_index"])
    train_count = args.train_count if args.train_count is not None else int(config["train"]["num_samples"])
    eval_start = args.eval_start if args.eval_start is not None else int(config["eval"]["start_index"])
    eval_count = args.eval_count if args.eval_count is not None else int(config["eval"]["num_samples"])
    if min(train_start, train_count, eval_start, eval_count) < 0 or train_count + eval_count == 0:
        raise ValueError("invalid train/eval ranges")
    if set(range(train_start, train_start + train_count)) & set(range(eval_start, eval_start + eval_count)):
        raise ValueError("train/eval ranges overlap")
    infos = load_infos(args.metadata)
    if max(train_start + train_count, eval_start + eval_count) > len(infos):
        raise ValueError("requested range exceeds metadata")
    provenance = None
    scene_locations = {}
    results = {}
    missing = []
    for split, start, count in (("train", train_start, train_count), ("eval", eval_start, eval_count)):
        totals = []
        by_class = {name: [] for name in MAP_CLASS_NAMES}
        for index in range(start, start + count):
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
            totals.append(len(classes))
            for class_id, name in enumerate(MAP_CLASS_NAMES):
                by_class[name].append(int((classes == class_id).sum()))
            if args.show_vectors and index == start:
                for class_id, points in zip(classes[:args.show_vectors], record["points_xy_m"][:args.show_vectors]):
                    print(f"vector token={token} class={MAP_CLASS_NAMES[int(class_id)]} "
                          f"first_xy_m={points[0].tolist()} last_xy_m={points[-1].tolist()}")
        results[split] = {
            "start_index": start, "num_samples": count, "available": len(totals),
            "total": summary(totals),
            "per_class": {name: summary(values) for name, values in by_class.items()},
        }
    max_count = max(results["train"]["total"]["max"], results["eval"]["total"]["max"])
    certified = (not missing and train_start == int(config["train"]["start_index"])
                 and train_count == int(config["train"]["num_samples"])
                 and eval_start == int(config["eval"]["start_index"])
                 and eval_count == int(config["eval"]["num_samples"]))
    report = {
        "schema_version": VECTOR_GT_SCHEMA_VERSION, "vector_semantics_version": VECTOR_SEMANTICS_VERSION,
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
