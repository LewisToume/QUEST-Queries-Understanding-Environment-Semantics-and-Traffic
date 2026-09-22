from __future__ import print_function

import argparse
import importlib
import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ("CAM_F0", "CAM_R0", "CAM_R2", "CAM_B0", "CAM_L2", "CAM_L0")
CONFIG_RELATIVE = Path(
    "projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py"
)
CHECKPOINT_NAME = "stream_petr_r50_flash_704_bs2_seq_90e.pth"


def default_stream_petr_root():
    for candidate in (
        ROOT / "third_party/StreamPETR",
        ROOT / "third_party/StreamPETR/StreamPETR-main",
        ROOT / "third_party/StreamPETR-main",
    ):
        if (candidate / CONFIG_RELATIVE).is_file():
            return candidate
    return ROOT / "third_party/StreamPETR"


STREAM_PETR_ROOT = default_stream_petr_root()


def first_existing(paths, label):
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "{} not found; checked: {}".format(label, ", ".join(str(p) for p in paths))
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="One-frame OpenScene inference with the original StreamPETR model"
    )
    parser.add_argument("--sample-index", type=int, default=0)
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
    parser.add_argument(
        "--camera-root", type=Path, default=ROOT / "data/openscene/sensor_blobs_mini"
    )
    parser.add_argument("--stream-petr-root", type=Path, default=STREAM_PETR_ROOT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=first_existing(
            (
                ROOT / "third_party/StreamPETR-main/ckpts" / CHECKPOINT_NAME,
                STREAM_PETR_ROOT / "ckpts" / CHECKPOINT_NAME,
                ROOT / "third_party/StreamPETR/StreamPETR-main/ckpts" / CHECKPOINT_NAME,
            ),
            "StreamPETR checkpoint",
        ),
    )
    parser.add_argument(
        "--mmdet3d-config-root",
        type=Path,
        default=None,
        help="Directory containing configs/_base_ if StreamPETR has no sibling mmdetection3d checkout",
    )
    return parser.parse_args()


def resolve_camera_path(raw_path, camera_root):
    parts = list(Path(raw_path.replace("\\", "/")).parts)
    if parts and parts[0].lower() == "dataset":
        parts.pop(0)
    candidate = camera_root.joinpath(*parts)
    if candidate.is_file():
        return candidate
    try:
        sensor_index = parts.index("sensor_blobs") + 1
    except ValueError:
        raise ValueError("camera path lacks sensor_blobs: {}".format(raw_path))
    if sensor_index >= len(parts) or parts[sensor_index] != "mini":
        parts.insert(sensor_index, "mini")
    candidate = camera_root.joinpath(*parts)
    if not candidate.is_file():
        raise FileNotFoundError("OpenScene JPEG missing: {}".format(candidate))
    return candidate


def load_frame(path, index):
    with path.open("rb") as stream:
        infos = pickle.load(stream)["infos"]
    if index < 0 or index >= len(infos):
        raise IndexError("sample-index {} outside [0, {})".format(index, len(infos)))
    return infos[index]


def make_input(info, camera_root):
    filenames = []
    intrinsics = []
    extrinsics = []
    lidar2img = []
    image_timestamps = []
    for name in CAMERAS:
        camera = info["cams"][name]
        filename = resolve_camera_path(camera["data_path"], camera_root)
        cam2lidar = np.eye(4, dtype=np.float32)
        cam2lidar[:3, :3] = np.asarray(
            camera["sensor2lidar_rotation"], dtype=np.float32
        )
        cam2lidar[:3, 3] = np.asarray(
            camera["sensor2lidar_translation"], dtype=np.float32
        )
        lidar2cam = np.linalg.inv(cam2lidar).astype(np.float32)
        intrinsic = np.eye(4, dtype=np.float32)
        intrinsic[:3, :3] = np.asarray(camera["cam_intrinsic"], dtype=np.float32)
        filenames.append(str(filename))
        intrinsics.append(intrinsic)
        extrinsics.append(lidar2cam)
        lidar2img.append(intrinsic @ lidar2cam)
        image_timestamps.append(
            float(camera.get("timestamp", info["timestamp"])) / 1e6
        )

    ego_pose = np.asarray(info["lidar2global"], dtype=np.float64)
    if ego_pose.shape != (4, 4):
        raise ValueError("lidar2global must be 4x4, got {}".format(ego_pose.shape))
    if not np.isfinite(ego_pose).all():
        raise ValueError("ego_pose contains NaN or Inf")
    if not all(
        np.isfinite(matrix).all()
        for group in (intrinsics, extrinsics, lidar2img)
        for matrix in group
    ):
        raise ValueError("camera geometry contains NaN or Inf")
    return {
        "sample_idx": str(info["token"]),
        "scene_token": str(info["scene_token"]),
        "img_filename": filenames,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "lidar2img": lidar2img,
        "ego_pose": ego_pose,
        "ego_pose_inv": np.linalg.inv(ego_pose),
        "timestamp": float(info["timestamp"]) / 1e6,
        "img_timestamp": image_timestamps,
        "prev_exists": False,
    }


def find_config_base(args, relative_path):
    candidates = []
    if args.mmdet3d_config_root is not None:
        candidates.append(args.mmdet3d_config_root / relative_path)
    candidates.append(args.stream_petr_root / "mmdetection3d" / relative_path)
    try:
        import mmdet3d

        candidates.append(Path(mmdet3d.__file__).resolve().parent.parent / relative_path)
    except ImportError:
        pass
    candidates.extend(
        (
            ROOT / "third_party/mmdetection3d-v1.0.0rc6" / relative_path,
            ROOT / "third_party/mmdetection3d-v0.17.1" / relative_path,
        )
    )
    return first_existing(candidates, "MMDetection3D base config")


def load_native_config(args, mmcv):
    config_path = first_existing(
        (args.stream_petr_root / CONFIG_RELATIVE,), "StreamPETR config"
    )
    source = config_path.read_text(encoding="utf-8")
    for name in ("datasets/nus-3d.py", "default_runtime.py"):
        relative = Path("configs/_base_") / name
        base = find_config_base(args, relative)
        original = "../../../mmdetection3d/" + relative.as_posix()
        if original not in source:
            raise ValueError("StreamPETR config lacks expected base: {}".format(original))
        source = source.replace(original, base.resolve().as_posix())
    return mmcv.Config.fromstring(source, file_format=".py")


def main():
    args = parse_args()
    info = load_frame(args.metadata, args.sample_index)
    raw = make_input(info, args.camera_root)
    print("sample token:", raw["sample_idx"])
    print("camera order:", " ".join(CAMERAS))
    for name, filename in zip(CAMERAS, raw["img_filename"]):
        print("  {}: {}".format(name, filename))
    print("intrinsics:", np.stack(raw["intrinsics"]).shape)
    print("extrinsics (lidar2cam):", np.stack(raw["extrinsics"]).shape)
    print("lidar2img:", np.stack(raw["lidar2img"]).shape)
    print("ego_pose:", raw["ego_pose"].shape)
    print("ego_pose_inv:", raw["ego_pose_inv"].shape)
    print("timestamp:", raw["timestamp"])
    print("img_timestamp:", raw["img_timestamp"])
    print("prev_exists:", raw["prev_exists"])

    import torch
    import mmcv
    import mmdet
    import mmdet3d
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmdet.datasets.pipelines import Compose
    from mmdet3d.models import build_model

    if not torch.cuda.is_available():
        raise RuntimeError("StreamPETR requires CUDA in the Teacher runtime")
    print(
        "runtime:",
        "torch={}".format(torch.__version__),
        "mmcv={}".format(mmcv.__version__),
        "mmdet={}".format(mmdet.__version__),
        "mmdet3d={}".format(mmdet3d.__version__),
        "gpu={}".format(torch.cuda.get_device_name(0)),
    )

    cfg = load_native_config(args, mmcv)
    sys.path.insert(0, str(args.stream_petr_root.resolve()))
    importlib.import_module("projects.mmdet3d_plugin")
    pipeline = Compose(cfg.test_pipeline)
    processed = pipeline(raw)
    if processed is None:
        raise RuntimeError("StreamPETR native test pipeline rejected the sample")
    data = scatter(collate([processed], samples_per_gpu=1), [0])[0]
    # The native detector resets temporal memory for a new scene; retain the
    # explicit first-frame flag in its test-time input contract as well.
    data["prev_exists"] = [[torch.tensor(0.0, device="cuda")]]
    print("native test image:", tuple(data["img"][0].shape))
    print("native test metadata:", sorted(data["img_metas"][0][0]))

    checkpoint_path = first_existing((args.checkpoint,), "StreamPETR checkpoint")
    cfg.model.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    print("checkpoint loaded:", str(checkpoint_path))
    print("checkpoint meta keys:", sorted(checkpoint.get("meta", {})))
    model = model.cuda().eval()
    with torch.no_grad():
        outputs = model(return_loss=False, rescale=True, **data)

    if not isinstance(outputs, list) or len(outputs) != 1:
        raise RuntimeError("unexpected StreamPETR output: {}".format(type(outputs)))
    pts_bbox = outputs[0]["pts_bbox"]
    boxes = pts_bbox["boxes_3d"]
    scores = pts_bbox["scores_3d"]
    labels = pts_bbox["labels_3d"]
    print("raw pts_bbox keys:", sorted(pts_bbox))
    print("boxes_3d:", boxes)
    print("boxes_3d tensor shape:", tuple(boxes.tensor.shape))
    print("scores_3d:", scores)
    print("scores_3d shape:", tuple(scores.shape))
    print("labels_3d:", labels)
    print("labels_3d shape:", tuple(labels.shape))
    return 0


if __name__ == "__main__":
    sys.exit(main())
