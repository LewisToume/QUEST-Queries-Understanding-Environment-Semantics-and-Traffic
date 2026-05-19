from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CAMS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract OpenScene multi-view camera images into sample folders.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "openscene_first_test_100"))
    parser.add_argument("--sensor-archive", default=str(PROJECT_ROOT / "data" / "openscene" / "sensor_blobs_mini.tar.gz"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cams", nargs="*", default=list(DEFAULT_CAMS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    archive_path = Path(args.sensor_archive)
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    if args.limit is not None:
        manifest = manifest[: args.limit]

    targets: dict[str, Path] = {}
    for item in manifest:
        sample_dir = data_root / item["sample_id"]
        with (sample_dir / "metadata.pkl").open("rb") as f:
            metadata = pickle.load(f)
        cam_dir = sample_dir / "cams"
        cam_dir.mkdir(exist_ok=True)
        for cam_name in args.cams:
            cam_info = metadata["cams"].get(cam_name)
            if cam_info is None:
                continue
            archive_member = "openscene-v1.0/sensor_blobs/mini/" + cam_info["data_path"]
            targets[archive_member] = cam_dir / f"{cam_name}.jpg"

    existing = {member for member, path in targets.items() if path.exists()}
    missing = set(targets) - existing
    print("=" * 72)
    print("OpenScene multi-view extraction")
    print(f"data_root        : {data_root}")
    print(f"samples          : {len(manifest)}")
    print(f"requested images : {len(targets)}")
    print(f"already exists   : {len(existing)}")
    print(f"to extract       : {len(missing)}")
    print("=" * 72)
    if not missing:
        return

    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf:
            if member.name not in missing:
                continue
            with tf.extractfile(member) as src, targets[member.name].open("wb") as dst:
                shutil.copyfileobj(src, dst)
            missing.remove(member.name)
            if len(missing) % 25 == 0 or not missing:
                print(f"remaining: {len(missing)}")
            if not missing:
                break

    if missing:
        raise RuntimeError(f"Missing {len(missing)} images, first: {next(iter(missing))}")
    print("multi-view extraction finished")


if __name__ == "__main__":
    main()
