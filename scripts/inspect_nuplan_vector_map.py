from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.map_training import validate_vector_record
from quest.stage3_dataset import load_record
from quest.utils import load_yaml_config
from quest.vector_map_labels import MAP_CLASS_NAMES
from scripts.run_navformer_openscene_teacher import load_infos, select_infos


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect direct nuPlan vector-map GT coverage")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=500)
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/openscene/meta_datas/meta_data_mini.pkl")
    parser.add_argument("--vector-dir", type=Path, default=PROJECT_ROOT / "data/vector_map_gt")
    args = parser.parse_args()
    infos = select_infos(load_infos(args.metadata), args.sample_index, args.num_frames)
    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    stage3 = load_yaml_config(PROJECT_ROOT / "configs/stage3_map.yaml")
    x0, x1 = model_config["x_range"]
    y0, y1 = model_config["y_range"]
    xy_range = (float(x0), float(y0), float(x1), float(y1))
    counts, lengths = [], []
    class_counts = {name: 0 for name in MAP_CLASS_NAMES}
    missing = empty = closed = open_count = nonfinite = outside = 0
    invalid_records = []
    exceed = {limit: 0 for limit in (32, 50, 64, 100, 128)}
    for sample_index, info in enumerate(infos, start=args.sample_index):
        token = str(info["token"])
        path = args.vector_dir / f"{token}.pt"
        if not path.is_file():
            missing += 1
            continue
        record = load_record(path)
        points = record.get("points_xy_m")
        if torch.is_tensor(points) and points.ndim == 3 and points.shape[-1] == 2:
            nonfinite += int((~torch.isfinite(points)).sum())
            outside += int(((points[..., 0] < x0) | (points[..., 0] > x1)
                            | (points[..., 1] < y0) | (points[..., 1] > y1)).sum())
        try:
            validate_vector_record(record, token, sample_index, xy_range)
        except (ValueError, TypeError, KeyError) as error:
            invalid_records.append(f"index={sample_index} token={token}: {error}")
            continue
        n = len(record["class_ids"])
        counts.append(n)
        empty += n == 0
        closed += int(record["is_closed"].sum())
        open_count += n - int(record["is_closed"].sum())
        lengths.extend(record["length_m"].tolist())
        for class_id, name in enumerate(MAP_CLASS_NAMES):
            class_counts[name] += int((record["class_ids"] == class_id).sum())
        for limit in exceed:
            exceed[limit] += n > limit
    print(f"total_frames={len(infos)} missing_frames={missing} empty_map_frames={empty}")
    if counts:
        values = np.asarray(counts)
        print(f"instance_count mean={values.mean():.3f} P50={np.percentile(values, 50):.1f} "
              f"P90={np.percentile(values, 90):.1f} P95={np.percentile(values, 95):.1f} "
              f"P99={np.percentile(values, 99):.1f} max={values.max()}")
    print(f"per_class_counts={class_counts}")
    if lengths:
        values = np.asarray(lengths)
        print(f"polyline_length_m mean={values.mean():.3f} min={values.min():.3f} "
              f"max={values.max():.3f}")
    print(f"closed={closed} open={open_count} closed_ratio={closed / (closed + open_count) if closed + open_count else 0:.6f}")
    print(f"NaN_or_Inf_values={nonfinite} outside_ROI_points={outside}")
    print(f"invalid_records={len(invalid_records)}")
    for error in invalid_records[:20]:
        print(error)
    print(f"frames_exceeding_instance_limits={exceed}")
    query_count = int(stage3["map"]["map_query_count"])
    if any(value > query_count for value in counts):
        raise ValueError(f"GT exceeds configured N_map={query_count}; increase query count before training")
    if invalid_records or missing:
        raise ValueError("vector GT coverage or validation failed; see report above")


if __name__ == "__main__":
    main()
