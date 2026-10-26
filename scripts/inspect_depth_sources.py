"""Read-only depth-source discovery; never loads a model or scans directories."""

import argparse
import pickle
from collections.abc import Mapping
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMERAS = (
    "CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1",
    "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0",
)


def path_candidates(raw, data_root):
    """Check bounded layout variants of a recorded path, not guessed filenames."""
    normalized = str(raw).replace("\\", "/")
    path = Path(normalized)
    candidates = [path] if path.is_absolute() else [ROOT / path, data_root / path]
    parts = normalized.split("/")
    for marker in ("sensor_blobs", "sensor_blobs_mini", "lidar", "lidar_blobs"):
        if marker not in parts:
            continue
        suffix = parts[parts.index(marker) + 1:]
        candidates.append(data_root.joinpath(marker, *suffix))
        if marker == "sensor_blobs":
            if suffix and suffix[0] == "mini":
                suffix = suffix[1:]
            candidates.extend((
                data_root.joinpath("sensor_blobs_mini", *suffix),
                data_root.joinpath("sensor_blobs_mini", "sensor_blobs", "mini", *suffix),
            ))
    return list(dict.fromkeys(candidates))


def describe(value):
    shape = getattr(value, "shape", None)
    if shape is not None:
        return "shape={} dtype={}".format(tuple(shape), getattr(value, "dtype", "unknown"))
    if isinstance(value, Mapping):
        return "keys={}".format(list(value))
    if isinstance(value, (list, tuple)):
        return "{} len={}".format(type(value).__name__, len(value))
    return repr(value)


def source_fields(value, prefix="", depth=0):
    """Inspect metadata only; bound nested sweeps and disclose the bound."""
    if depth > 5:
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = "{}.{}".format(prefix, key) if prefix else str(key)
            if str(key) == "cams":
                continue
            if isinstance(item, (Mapping, list, tuple)):
                yield from source_fields(item, name, depth + 1)
            elif any(word in name.lower() for word in ("lidar", "depth", "point_cloud")):
                yield name, item
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value[:2]):
            yield from source_fields(item, "{}[{}]".format(prefix, index), depth + 1)


def print_source(name, value, data_root):
    print("  {}: {}".format(name, describe(value)))
    if not isinstance(value, str):
        return
    is_path = any(word in name.lower() for word in ("path", "file"))
    if not is_path and Path(value).suffix.lower() not in (".pcd", ".bin", ".npz", ".npy"):
        return
    candidates = path_candidates(value, data_root)
    found = [path for path in candidates if path.is_file()]
    if found:
        for path in found:
            print("    EXISTS {} bytes={}".format(path, path.stat().st_size))
    else:
        print("    MISSING checked={}".format([str(path) for path in candidates]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path,
                        default=ROOT / "data/openscene/meta_datas/meta_data_mini.pkl")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/openscene")
    parser.add_argument("--start", type=int, default=0, help="Original metadata index")
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    if args.start < 0 or args.count <= 0:
        parser.error("start must be non-negative and count must be positive")
    # Only use the user's trusted local OpenScene pickle; pickle is not a safe interchange format.
    with args.metadata.open("rb") as handle:
        metadata = pickle.load(handle)
    infos = metadata.get("infos") if isinstance(metadata, Mapping) else metadata
    if not isinstance(infos, (list, tuple)):
        raise TypeError("Expected metadata infos list or top-level frame list")
    if args.start + args.count > len(infos):
        raise ValueError("Requested original-index range exceeds metadata length")
    print("READ_ONLY=True GPU=False directory_scan=False point_cloud_read=False")
    print("metadata_type={} frames={} selected=[{},{})".format(
        type(metadata).__name__, len(infos), args.start, args.start + args.count))
    print("Nested inspection limit: depth=5, first 2 items per sequence; no completeness claim")
    for index in range(args.start, args.start + args.count):
        info = infos[index]
        if not isinstance(info, Mapping):
            raise TypeError("Frame {} is not a mapping".format(index))
        print("\nindex={} token={} scene={} timestamp={}".format(
            index, info.get("token"), info.get("scene_token"), info.get("timestamp")))
        if index == args.start:
            print("frame_keys={}".format(list(info)))
        fields = list(source_fields(info))
        for name, value in fields:
            print_source(name, value, args.data_root)
        if not fields:
            print("  LIDAR_OR_DEPTH_FIELDS=MISSING")
        for key in ("lidar2ego", "lidar2global", "ego2global"):
            value = info.get(key)
            print("  {}={}".format(key, "MISSING" if value is None else describe(value)))
        cams = info.get("cams", {})
        for camera in CAMERAS:
            entry = cams.get(camera) if isinstance(cams, Mapping) else None
            if not isinstance(entry, Mapping):
                print("  {}=MISSING".format(camera))
                continue
            print("  {} keys={}".format(camera, list(entry)))
            for key in ("cam_intrinsic", "sensor2lidar_rotation", "sensor2lidar_translation"):
                print("    {}={}".format(key, describe(entry[key]) if key in entry else "MISSING"))
            for key, value in entry.items():
                if "depth" in str(key).lower():
                    print_source("{}.{}".format(camera, key), value, args.data_root)
    print("\nDEPTH_SUPERVISION_READY=UNCONFIRMED; file presence alone does not verify format, units or synchronization")


if __name__ == "__main__":
    main()
