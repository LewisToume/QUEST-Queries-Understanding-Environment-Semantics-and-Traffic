from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.openscene_dataset import OPENSCENE_CAMERA_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one OpenScene QUEST sample.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "openscene_first_test_100"))
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-path", default=str(PROJECT_ROOT / "debug_vis" / "openscene_inspect_preview.jpg"))
    return parser.parse_args()


def camera_path(sample_dir: Path, cam_name: str) -> Path:
    path = sample_dir / "cams" / f"{cam_name}.jpg"
    if path.exists():
        return path
    if cam_name == "CAM_F0":
        return sample_dir / "cam_f0.jpg"
    return path


def print_array(name: str, value: object, max_rows: int = 3) -> None:
    array = np.asarray(value)
    print(f"{name}: shape={array.shape} dtype={array.dtype}")
    if array.ndim == 0:
        print(f"  {array.item()}")
    elif array.size:
        print(array[:max_rows])


def build_preview(sample_dir: Path, output_path: Path) -> None:
    thumbs: list[Image.Image] = []
    thumb_w, thumb_h = 320, 180
    for cam_name in OPENSCENE_CAMERA_NAMES:
        image = Image.open(camera_path(sample_dir, cam_name)).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.BILINEAR)
        canvas = Image.new("RGB", (thumb_w, thumb_h + 24), "white")
        x = (thumb_w - image.width) // 2
        canvas.paste(image, (x, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, thumb_h + 5), cam_name, fill=(0, 0, 0))
        thumbs.append(canvas)

    grid = Image.new("RGB", (thumb_w * 4, (thumb_h + 24) * 2), "white")
    for idx, thumb in enumerate(thumbs):
        x = (idx % 4) * thumb_w
        y = (idx // 4) * (thumb_h + 24)
        grid.paste(thumb, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path, quality=92)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest:
        raise RuntimeError(f"empty manifest: {data_root / 'manifest.json'}")

    if args.index is None:
        rng = random.Random(args.seed)
        index = rng.randrange(len(manifest))
    else:
        index = args.index
    item = manifest[index]
    sample_dir = data_root / item["sample_id"]
    with (sample_dir / "metadata.pkl").open("rb") as f:
        metadata = pickle.load(f)

    print("=" * 80)
    print("OpenScene QUEST sample inspection")
    print(f"data_root      : {data_root}")
    print(f"index          : {index}")
    print(f"sample_id      : {item['sample_id']}")
    print(f"token          : {metadata.get('token')}")
    print(f"log_name       : {metadata.get('log_name')}")
    print(f"scene_name     : {metadata.get('scene_name')}")
    print("=" * 80)

    print("8 camera paths and image shapes:")
    for cam_name in OPENSCENE_CAMERA_NAMES:
        path = camera_path(sample_dir, cam_name)
        with Image.open(path) as image:
            print(f"  {cam_name:<7} path={path} shape=({image.height}, {image.width}, {len(image.getbands())})")

    print("-" * 80)
    for cam_name in OPENSCENE_CAMERA_NAMES:
        cam_info = metadata["cams"][cam_name]
        print(f"{cam_name} intrinsic:")
        print(np.asarray(cam_info["cam_intrinsic"]))
        print(f"{cam_name} extrinsic sensor2lidar:")
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = np.asarray(cam_info["sensor2lidar_rotation"], dtype=np.float32)
        extrinsic[:3, 3] = np.asarray(cam_info["sensor2lidar_translation"], dtype=np.float32)
        print(extrinsic)

    print("-" * 80)
    print_array("ego state / can_bus", metadata.get("can_bus", []))
    print_array("ego pose / ego2global", metadata.get("ego2global", []), max_rows=4)
    print_array("ego dynamic state", metadata.get("ego_dynamic_state", []))

    anns = metadata.get("anns") or {}
    print("-" * 80)
    print_array("GT boxes", anns.get("gt_boxes", []))
    print_array("GT classes", anns.get("gt_names", []))
    print_array("GT velocity", anns.get("gt_velocity_3d", []))

    occ_path = sample_dir / "occ_gt_final.npy"
    occ = np.load(occ_path)
    print("-" * 80)
    print(f"occupancy path : {occ_path}")
    print_array("occupancy sparse GT", occ)

    flow_candidates = [sample_dir / "flow_gt_final.npy", sample_dir / "flow.npy"]
    flow_path = next((path for path in flow_candidates if path.exists()), None)
    if flow_path is None:
        print("flow shape     : unavailable; loss must remain masked")
    else:
        print(f"flow path      : {flow_path}")
        print_array("flow GT", np.load(flow_path))

    print("-" * 80)
    print(f"map_location   : {metadata.get('map_location')}")
    print(f"roadblock_ids  : {metadata.get('roadblock_ids')}")
    print(f"prev sample    : {metadata.get('sample_prev')}")
    print(f"next sample    : {metadata.get('sample_next')}")
    print("map vector GT  : unavailable in current extracted sample; loss must remain masked")

    preview_path = Path(args.preview_path)
    build_preview(sample_dir, preview_path)
    print("-" * 80)
    print(f"preview image  : {preview_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
