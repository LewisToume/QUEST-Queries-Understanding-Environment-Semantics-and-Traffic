from __future__ import print_function

import argparse
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def first_existing(paths, label):
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "{} not found; checked: {}".format(label, ", ".join(str(p) for p in paths))
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect OpenScene metadata sequence order")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=first_existing(
            (
                ROOT / "data/openscene/meta_datas/meta_data_mini.pkl",
                ROOT
                / "data/openscene/meta_datas/openscene-v1.0/meta_datas/meta_data_mini.pkl",
            ),
            "OpenScene metadata",
        ),
    )
    return parser.parse_args()


def scene_and_timestamp(info):
    if "scene_token" not in info:
        raise KeyError("OpenScene info lacks scene_token")
    if "timestamp" not in info:
        raise KeyError("OpenScene info lacks timestamp")
    return str(info["scene_token"]), float(info["timestamp"])


def sort_infos_temporally(infos):
    return sorted(infos, key=lambda info: scene_and_timestamp(info))


def select_range(items, start, count):
    if start < 0:
        raise ValueError("start must be non-negative, got {}".format(start))
    if count <= 0:
        raise ValueError("count must be positive, got {}".format(count))
    end = start + count
    if end > len(items):
        raise IndexError(
            "requested samples [{}, {}) exceed metadata size {}".format(
                start, end, len(items)
            )
        )
    return items[start:end]


def analyze_sequence(infos):
    scene_positions = defaultdict(list)
    scene_timestamps = defaultdict(list)
    for index, info in enumerate(infos):
        scene_token, timestamp = scene_and_timestamp(info)
        scene_positions[scene_token].append(index)
        scene_timestamps[scene_token].append(timestamp)

    fragmented_scenes = []
    non_monotonic_scenes = []
    deltas = []
    scene_summaries = {}
    for scene_token, positions in scene_positions.items():
        timestamps = scene_timestamps[scene_token]
        if positions[-1] - positions[0] + 1 != len(positions):
            fragmented_scenes.append(scene_token)
        scene_deltas = [
            current - previous
            for previous, current in zip(timestamps[:-1], timestamps[1:])
        ]
        if any(delta <= 0 for delta in scene_deltas):
            non_monotonic_scenes.append(scene_token)
        deltas.extend(scene_deltas)
        scene_summaries[scene_token] = {
            "frame_count": len(timestamps),
            "first_timestamp": min(timestamps),
            "last_timestamp": max(timestamps),
        }

    delta_summary = {
        "count": len(deltas),
        "mean": statistics.mean(deltas) if deltas else None,
        "median": statistics.median(deltas) if deltas else None,
        "min": min(deltas) if deltas else None,
        "max": max(deltas) if deltas else None,
    }
    return {
        "total_samples": len(infos),
        "total_scenes": len(scene_positions),
        "infos_already_grouped_by_scene": not fragmented_scenes,
        "infos_already_time_sorted_within_scene": not non_monotonic_scenes,
        "fragmented_scenes": sorted(fragmented_scenes),
        "non_monotonic_scenes": sorted(non_monotonic_scenes),
        "scene_summaries": scene_summaries,
        "delta_summary": delta_summary,
    }


def previous_same_scene_timestamps(infos):
    previous_by_scene = {}
    result = []
    for info in infos:
        scene_token, timestamp = scene_and_timestamp(info)
        result.append(previous_by_scene.get(scene_token))
        previous_by_scene[scene_token] = timestamp
    return result


def main():
    args = parse_args()
    with args.metadata.open("rb") as stream:
        metadata = pickle.load(stream)
    infos = metadata.get("infos")
    if not isinstance(infos, list):
        raise ValueError("OpenScene metadata does not contain an infos list")
    selected = select_range(infos, args.start, args.count)
    previous_timestamps = previous_same_scene_timestamps(infos)

    for offset, info in enumerate(selected):
        original_index = args.start + offset
        scene_token, timestamp = scene_and_timestamp(info)
        previous_timestamp = previous_timestamps[original_index]
        same_scene_as_previous = (
            original_index > 0
            and str(infos[original_index - 1].get("scene_token")) == scene_token
        )
        delta_t = (
            timestamp - previous_timestamp if previous_timestamp is not None else None
        )
        print("original_index: {}".format(original_index))
        print("token: {}".format(info.get("token")))
        print("scene_token: {}".format(scene_token))
        print("timestamp: {}".format(timestamp))
        print("previous_timestamp_same_scene: {}".format(previous_timestamp))
        print("delta_t: {}".format(delta_t))
        print("same_scene_as_previous_item: {}".format(same_scene_as_previous))

    analysis = analyze_sequence(infos)
    print("scene_summaries:")
    for scene_token in sorted(analysis["scene_summaries"]):
        summary = analysis["scene_summaries"][scene_token]
        print(
            "  {}: frame_count={} first_timestamp={} last_timestamp={}".format(
                scene_token,
                summary["frame_count"],
                summary["first_timestamp"],
                summary["last_timestamp"],
            )
        )
    delta = analysis["delta_summary"]
    print("delta_t_summary:")
    print("  count: {}".format(delta["count"]))
    print("  mean: {}".format(delta["mean"]))
    print("  median: {}".format(delta["median"]))
    print("  min: {}".format(delta["min"]))
    print("  max: {}".format(delta["max"]))
    print("summary:")
    for key in (
        "total_samples",
        "total_scenes",
        "infos_already_grouped_by_scene",
        "infos_already_time_sorted_within_scene",
        "fragmented_scenes",
        "non_monotonic_scenes",
    ):
        print("  {}: {}".format(key, analysis[key]))

    sorted_infos = sort_infos_temporally(infos)
    print("recommended_temporal_order_count: {}".format(len(sorted_infos)))
    if sorted_infos:
        print("recommended_first_token: {}".format(sorted_infos[0].get("token")))
        print("recommended_last_token: {}".format(sorted_infos[-1].get("token")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
