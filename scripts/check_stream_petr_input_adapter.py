from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


OPENSCENE_TO_STREAMPETR = {
    "vehicle": 0,  # approximate: StreamPETR "car"
    "pedestrian": 8,
    "traffic_cone": 9,
}
DEFAULT_CAMS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0")
IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def camera_to_lidar_matrix(cam_info: dict) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = np.asarray(cam_info["sensor2lidar_rotation"], dtype=np.float32)
    mat[:3, 3] = np.asarray(cam_info["sensor2lidar_translation"], dtype=np.float32)
    return mat


def load_and_normalize_image(path: Path, size_hw: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    ori_w, ori_h = image.size
    out_h, out_w = size_hw
    image = image.resize((out_w, out_h), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    array = (array - IMG_MEAN) / IMG_STD
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor, (ori_h, ori_w)


def build_adapter(sample_dir: Path, cams: tuple[str, ...], image_size: tuple[int, int]) -> dict:
    with (sample_dir / "metadata.pkl").open("rb") as f:
        metadata = pickle.load(f)

    images = []
    intrinsics = []
    extrinsics = []
    lidar2imgs = []
    filenames = []
    ori_shapes = []
    for cam_name in cams:
        cam_info = metadata["cams"][cam_name]
        image_path = sample_dir / "cams" / f"{cam_name}.jpg"
        image, ori_shape = load_and_normalize_image(image_path, image_size)
        ori_h, ori_w = ori_shape
        out_h, out_w = image_size

        intrinsic = np.asarray(cam_info["cam_intrinsic"], dtype=np.float32).copy()
        intrinsic[0, :] *= out_w / ori_w
        intrinsic[1, :] *= out_h / ori_h
        viewpad = np.eye(4, dtype=np.float32)
        viewpad[:3, :3] = intrinsic

        cam2lidar = camera_to_lidar_matrix(cam_info)
        lidar2cam = np.linalg.inv(cam2lidar)
        lidar2img = viewpad @ lidar2cam

        images.append(image)
        intrinsics.append(viewpad)
        extrinsics.append(lidar2cam)
        lidar2imgs.append(lidar2img)
        filenames.append(str(image_path))
        ori_shapes.append(ori_shape)

    anns = metadata["anns"]
    gt_boxes = np.asarray(anns["gt_boxes"], dtype=np.float32)
    gt_names = np.asarray(anns["gt_names"])
    gt_velocity = np.asarray(anns.get("gt_velocity_3d", np.zeros((len(gt_boxes), 3))), dtype=np.float32)
    labels = []
    boxes = []
    skipped_names: dict[str, int] = {}
    for box, name, velocity in zip(gt_boxes, gt_names, gt_velocity):
        name = str(name)
        if name not in OPENSCENE_TO_STREAMPETR:
            skipped_names[name] = skipped_names.get(name, 0) + 1
            continue
        # Approximate StreamPETR 10D code target: x,y,z,dx,dy,dz,yaw,vx,vy,unused.
        boxes.append([*box[:7].tolist(), float(velocity[0]), float(velocity[1]), 0.0])
        labels.append(OPENSCENE_TO_STREAMPETR[name])

    return {
        "img": torch.stack(images),
        "lidar2img": torch.from_numpy(np.stack(lidar2imgs)),
        "intrinsics": torch.from_numpy(np.stack(intrinsics)),
        "extrinsics": torch.from_numpy(np.stack(extrinsics)),
        "timestamp": float(metadata["timestamp"]) / 1e6,
        "img_timestamp": torch.full((len(cams),), float(metadata["timestamp"]) / 1e6),
        "ego_pose": torch.from_numpy(np.asarray(metadata["lidar2global"], dtype=np.float32)),
        "ego_pose_inv": torch.from_numpy(np.linalg.inv(np.asarray(metadata["lidar2global"], dtype=np.float32))),
        "gt_bboxes_3d_approx": torch.tensor(boxes, dtype=torch.float32),
        "gt_labels_3d_approx": torch.tensor(labels, dtype=torch.long),
        "meta": {
            "sample": sample_dir.name,
            "cams": list(cams),
            "filenames": filenames,
            "ori_shapes": ori_shapes,
            "note": "Adapter-level tensor check only. Original StreamPETR still requires MMDet3D box classes, pipeline wrappers, checkpoint, and installed mmcv/mmdet3d.",
            "class_mapping": OPENSCENE_TO_STREAMPETR,
            "skipped_names": skipped_names,
        },
    }


def tensor_summary(value: torch.Tensor) -> dict:
    x = value.detach().float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(x.min()) if x.numel() else None,
        "max": float(x.max()) if x.numel() else None,
        "mean": float(x.mean()) if x.numel() else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a StreamPETR-like input dict from an OpenScene sample.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "openscene_first_test_100"))
    parser.add_argument("--sample-id", default="sample_000")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "debug_vis" / "expert_input_check"))
    parser.add_argument("--cams", nargs="*", default=list(DEFAULT_CAMS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.data_root) / args.sample_id
    adapter = build_adapter(sample_dir, tuple(args.cams), image_size=(256, 704))
    report = {
        "expert": "StreamPETR",
        "status": "adapter_ok_model_not_run",
        "reason_model_not_run": "mmcv/mmdet/mmdet3d and StreamPETR checkpoint are not installed/present in this QUEST environment.",
        "fields": {key: tensor_summary(value) for key, value in adapter.items() if isinstance(value, torch.Tensor)},
        "meta": adapter["meta"],
    }
    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S") / "streampetr"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "input_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({k: v for k, v in adapter.items() if isinstance(v, torch.Tensor)}, out_dir / "input_tensors.pt")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved_to: {out_dir}")


if __name__ == "__main__":
    main()
