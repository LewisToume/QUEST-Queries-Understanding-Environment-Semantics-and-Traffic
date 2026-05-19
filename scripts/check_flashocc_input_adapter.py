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


DEFAULT_CAMS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0")
IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def camera_to_ego_matrix(cam_info: dict) -> np.ndarray:
    # OpenScene metadata provides sensor2lidar. In this extracted log, lidar2ego
    # is identity, so sensor2lidar is a practical sensor2ego approximation.
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = np.asarray(cam_info["sensor2lidar_rotation"], dtype=np.float32)
    mat[:3, 3] = np.asarray(cam_info["sensor2lidar_translation"], dtype=np.float32)
    return mat


def load_image(path: Path, size_hw: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    ori_w, ori_h = image.size
    out_h, out_w = size_hw
    image = image.resize((out_w, out_h), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    array = (array - IMG_MEAN) / IMG_STD
    return torch.from_numpy(array).permute(2, 0, 1).contiguous(), (ori_h, ori_w)


def load_dense_occ(sample_dir: Path, occ_size: tuple[int, int, int], num_classes: int) -> torch.Tensor:
    sparse_occ = np.load(sample_dir / "occ_gt_final.npy")
    dense = torch.zeros(occ_size, dtype=torch.long)
    if sparse_occ.size == 0:
        return dense
    flat_indices = torch.from_numpy(sparse_occ[:, 0].astype(np.int64))
    class_ids = torch.from_numpy(sparse_occ[:, 1].astype(np.int64)).clamp(0, num_classes - 1)
    valid = (flat_indices >= 0) & (flat_indices < dense.numel())
    dense.view(-1)[flat_indices[valid]] = class_ids[valid]
    return dense


def build_adapter(sample_dir: Path, cams: tuple[str, ...], image_size: tuple[int, int]) -> dict:
    with (sample_dir / "metadata.pkl").open("rb") as f:
        metadata = pickle.load(f)

    imgs = []
    sensor2egos = []
    ego2globals = []
    intrins = []
    post_rots = []
    post_trans = []
    filenames = []
    ori_shapes = []
    for cam_name in cams:
        cam_info = metadata["cams"][cam_name]
        image_path = sample_dir / "cams" / f"{cam_name}.jpg"
        image, ori_shape = load_image(image_path, image_size)
        ori_h, ori_w = ori_shape
        out_h, out_w = image_size

        intrinsic = np.asarray(cam_info["cam_intrinsic"], dtype=np.float32).copy()
        intrinsic[0, :] *= out_w / ori_w
        intrinsic[1, :] *= out_h / ori_h
        post_rot = np.eye(3, dtype=np.float32)
        post_rot[0, 0] = out_w / ori_w
        post_rot[1, 1] = out_h / ori_h

        imgs.append(image)
        sensor2egos.append(camera_to_ego_matrix(cam_info))
        ego2globals.append(np.asarray(metadata["ego2global"], dtype=np.float32))
        intrins.append(intrinsic)
        post_rots.append(post_rot)
        post_trans.append(np.zeros(3, dtype=np.float32))
        filenames.append(str(image_path))
        ori_shapes.append(ori_shape)

    voxel_semantics = load_dense_occ(sample_dir, occ_size=(200, 200, 16), num_classes=18)
    # FlashOCC expects visibility masks from official labels.npz. The OpenScene
    # sparse file here does not include them, so this adapter uses all-true masks
    # only for input-shape smoke testing.
    mask_lidar = torch.ones_like(voxel_semantics, dtype=torch.bool)
    mask_camera = torch.ones_like(voxel_semantics, dtype=torch.bool)

    return {
        "img_inputs": (
            torch.stack(imgs),
            torch.from_numpy(np.stack(sensor2egos)),
            torch.from_numpy(np.stack(ego2globals)),
            torch.from_numpy(np.stack(intrins)),
            torch.from_numpy(np.stack(post_rots)),
            torch.from_numpy(np.stack(post_trans)),
            torch.eye(3, dtype=torch.float32),
        ),
        "voxel_semantics": voxel_semantics,
        "mask_lidar": mask_lidar,
        "mask_camera": mask_camera,
        "meta": {
            "sample": sample_dir.name,
            "cams": list(cams),
            "filenames": filenames,
            "ori_shapes": ori_shapes,
            "note": "Adapter-level tensor check only. Original FlashOCC still requires MMDet3D/MMCV, checkpoint, and official mask_camera/mask_lidar for faithful training/eval.",
            "warning": "FlashOCC default configs use 6 NuScenes cameras and num_classes=18; this adapter keeps OpenScene 8-view input for smoke testing.",
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
    parser = argparse.ArgumentParser(description="Build a FlashOCC-like input dict from an OpenScene sample.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "openscene_first_test_100"))
    parser.add_argument("--sample-id", default="sample_000")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "debug_vis" / "expert_input_check"))
    parser.add_argument("--cams", nargs="*", default=list(DEFAULT_CAMS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.data_root) / args.sample_id
    adapter = build_adapter(sample_dir, tuple(args.cams), image_size=(256, 704))
    img_inputs = adapter["img_inputs"]
    report = {
        "expert": "FlashOCC",
        "status": "adapter_ok_model_not_run",
        "reason_model_not_run": "mmcv/mmdet/mmdet3d and FlashOCC checkpoint are not installed/present in this QUEST environment.",
        "img_inputs": {
            "imgs": tensor_summary(img_inputs[0]),
            "sensor2egos": tensor_summary(img_inputs[1]),
            "ego2globals": tensor_summary(img_inputs[2]),
            "intrins": tensor_summary(img_inputs[3]),
            "post_rots": tensor_summary(img_inputs[4]),
            "post_trans": tensor_summary(img_inputs[5]),
            "bda_rot": tensor_summary(img_inputs[6]),
        },
        "targets": {
            "voxel_semantics": tensor_summary(adapter["voxel_semantics"]),
            "mask_lidar": tensor_summary(adapter["mask_lidar"]),
            "mask_camera": tensor_summary(adapter["mask_camera"]),
            "unique_occ_classes": sorted(torch.unique(adapter["voxel_semantics"]).cpu().tolist()),
        },
        "meta": adapter["meta"],
    }
    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S") / "flashocc"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "input_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(
        {
            "img_inputs": adapter["img_inputs"],
            "voxel_semantics": adapter["voxel_semantics"],
            "mask_lidar": adapter["mask_lidar"],
            "mask_camera": adapter["mask_camera"],
        },
        out_dir / "input_tensors.pt",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved_to: {out_dir}")


if __name__ == "__main__":
    main()
