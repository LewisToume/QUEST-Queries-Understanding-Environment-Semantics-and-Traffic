from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.map_training import load_vector_capacity_audit, validate_vector_record
from quest.stage3_dataset import load_record
from quest.utils import load_yaml_config
from quest.vector_map_labels import MAP_CLASS_NAMES
from scripts.run_navformer_openscene_teacher import load_infos, select_infos


def distribution(values: list[float]) -> str:
    if not values:
        return "mean=NA P50=NA P90=NA P95=NA P99=NA max=NA"
    array = np.asarray(values, dtype=np.float64)
    return (f"mean={array.mean():.3f} P50={np.percentile(array, 50):.3f} "
            f"P90={np.percentile(array, 90):.3f} P95={np.percentile(array, 95):.3f} "
            f"P99={np.percentile(array, 99):.3f} max={array.max():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect direct nuPlan vector-map GT coverage")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=5000)
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
    per_class = {name: {"counts": [], "lengths": [], "open": 0, "closed": 0}
                 for name in MAP_CLASS_NAMES}
    missing = empty = closed = open_count = nonfinite = outside = 0
    found = degenerate = duplicate_consecutive = zero_length_segments = 0
    invalid_records = []
    road_geometry_totals = {}
    missing_road_diagnostics = 0
    cast_warning_frames = cast_warning_events = 0
    invalid_relation_max = {"lane_fid": 0, "lane_connector_fid": 0}
    exceed = {limit: 0 for limit in (32, 50, 64, 100, 128)}
    for sample_index, info in enumerate(infos, start=args.sample_index):
        token = str(info["token"])
        path = args.vector_dir / f"{token}.pt"
        if not path.is_file():
            missing += 1
            continue
        found += 1
        try:
            record = load_record(path)
        except Exception as error:
            invalid_records.append(f"index={sample_index} token={token}: unreadable file: {error}")
            continue
        points = record.get("points_xy_m")
        degenerate_mask = None
        if torch.is_tensor(points) and points.ndim == 3 and points.shape[-1] == 2:
            degenerate_mask = torch.zeros(points.shape[0], dtype=torch.bool)
            nonfinite += int((~torch.isfinite(points)).sum())
            outside += int(((points[..., 0] < x0) | (points[..., 0] > x1)
                            | (points[..., 1] < y0) | (points[..., 1] > y1)).sum())
            if points.shape[1] >= 2:
                segments = points[:, 1:] - points[:, :-1]
                closed_flags = record.get("is_closed")
                if torch.is_tensor(closed_flags) and closed_flags.shape == (points.shape[0],):
                    closing = (points[:, :1] - points[:, -1:])[closed_flags.bool()]
                    segments = torch.cat((segments.reshape(-1, 2), closing.reshape(-1, 2)), dim=0)
                duplicate_consecutive += int((segments == 0).all(dim=-1).sum())
                zero_length_segments += int((torch.linalg.vector_norm(segments, dim=-1) <= 1e-6).sum())
            for line_index, line in enumerate(points):
                if not bool(torch.isfinite(line).all()) or len(torch.unique(line, dim=0)) < 2:
                    degenerate_mask[line_index] = True
        length_values = record.get("length_m")
        if torch.is_tensor(length_values):
            nonfinite += int((~torch.isfinite(length_values)).sum())
            if degenerate_mask is not None and length_values.shape == degenerate_mask.shape:
                degenerate_mask |= length_values <= 1e-6
        if degenerate_mask is not None:
            degenerate += int(degenerate_mask.sum())
        try:
            validate_vector_record(record, token, sample_index, xy_range, expected_info=info)
        except (ValueError, TypeError, KeyError) as error:
            invalid_records.append(f"index={sample_index} token={token}: {error}")
            continue
        n = len(record["class_ids"])
        cast = record["map_cast_diagnostics"]
        cast_warning_frames += bool(cast["invalid_cast_warning_count"])
        cast_warning_events += cast["invalid_cast_warning_count"]
        for column in invalid_relation_max:
            invalid_relation_max[column] = max(
                invalid_relation_max[column], cast["fields"][column]["invalid_non_null_rows"]
            )
        road_geometry = record.get("geometry_diagnostics", {}).get("road_area")
        if isinstance(road_geometry, dict):
            for key, value in road_geometry.items():
                road_geometry_totals[key] = road_geometry_totals.get(key, 0) + int(value)
        else:
            missing_road_diagnostics += 1
        counts.append(n)
        empty += n == 0
        closed += int(record["is_closed"].sum())
        open_count += n - int(record["is_closed"].sum())
        lengths.extend(record["length_m"].tolist())
        for class_id, name in enumerate(MAP_CLASS_NAMES):
            selected = record["class_ids"] == class_id
            per_class[name]["counts"].append(int(selected.sum()))
            per_class[name]["lengths"].extend(record["length_m"][selected].tolist())
            per_class[name]["closed"] += int(record["is_closed"][selected].sum())
            per_class[name]["open"] += int(selected.sum()) - int(record["is_closed"][selected].sum())
        for limit in exceed:
            exceed[limit] += n > limit
    print(f"requested={len(infos)} found={found} missing={missing} invalid={len(invalid_records)} "
          f"valid={len(counts)} empty={empty}")
    print(f"overall_instances_per_valid_frame {distribution(counts)}")
    for name in MAP_CLASS_NAMES:
        item = per_class[name]
        print(f"class={name} total_count={sum(item['counts'])} instances_per_valid_frame "
              f"{distribution(item['counts'])}")
        print(f"class={name} polyline_length_m {distribution(item['lengths'])} "
              f"open={item['open']} closed={item['closed']}")
    if lengths:
        print(f"overall_polyline_length_m {distribution(lengths)}")
    print(f"closed={closed} open={open_count} closed_ratio={closed / (closed + open_count) if closed + open_count else 0:.6f}")
    print(f"NaN_or_Inf_values={nonfinite} outside_ROI_points={outside} "
          f"degenerate_lines={degenerate} duplicate_consecutive_points={duplicate_consecutive} "
          f"zero_length_consecutive_segments={zero_length_segments}")
    print(f"road_geometry_totals={road_geometry_totals} "
          f"frames_missing_road_diagnostics={missing_road_diagnostics}")
    print(f"map_cast_warning_frames={cast_warning_frames} warning_events={cast_warning_events} "
          f"invalid_non_null_relation_max_per_city={invalid_relation_max}")
    print(f"invalid_records={len(invalid_records)}")
    for error in invalid_records[:20]:
        print(error)
    print(f"frames_exceeding_instance_limits={exceed}")
    capacity_path = PROJECT_ROOT / stage3["paths"]["vector_capacity_audit_path"]
    if capacity_path.is_file():
        query_count, _ = load_vector_capacity_audit(capacity_path, stage3)
        print(f"certified_N_map={query_count}")
        if any(value > query_count for value in counts):
            raise ValueError(f"GT exceeds certified N_map={query_count}")
    else:
        print("certified_N_map=pending_full_train_eval_audit")
    if invalid_records or missing:
        raise ValueError("vector GT coverage or validation failed; see report above")


if __name__ == "__main__":
    main()
