from __future__ import print_function

import argparse
import copy
import importlib
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


NAVFORMER_ROOT = Path("/home/user/DataDisk/QUEST_WORK/Navformer")
OPENSCENE_ROOT = Path("/home/user/DataDisk/QUEST_WORK/data/openscene/openscene-v1.0")
DEFAULT_CONFIG = NAVFORMER_ROOT / "configs/navformer/track_map_nuplan_r50_navtrain.py"
DEFAULT_CHECKPOINT = (
    NAVFORMER_ROOT
    / "data/alg_engine/ckpts/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth"
)
DEFAULT_METADATA = OPENSCENE_ROOT / "meta_datas/meta_data_mini.pkl"
DEFAULT_IMAGE_ROOT = Path(
    "/home/user/DataDisk/QUEST_WORK/QUEST/data/openscene/sensor_blobs_mini"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Navformer track+map inference directly on one OpenScene frame"
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--navformer-root", type=Path, default=NAVFORMER_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    return parser.parse_args()


def require_file(path, label):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("{} not found: {}".format(label, path))
    return path


def require_directory(path, label):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError("{} not found: {}".format(label, path))
    return path


def load_info(metadata_path, sample_index):
    with metadata_path.open("rb") as stream:
        metadata = pickle.load(stream)
    if isinstance(metadata, dict):
        infos = metadata.get("infos")
    elif isinstance(metadata, list):
        infos = metadata
    else:
        infos = None
    if not isinstance(infos, list):
        raise ValueError("OpenScene metadata does not contain an infos list")
    if sample_index < 0 or sample_index >= len(infos):
        raise IndexError(
            "sample-index {} outside [0, {})".format(sample_index, len(infos))
        )
    info = infos[sample_index]
    if not isinstance(info, dict):
        raise TypeError("OpenScene info must be a dict, got {}".format(type(info)))
    return info


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


def require_matrix(value, shape, label, dtype=np.float64):
    matrix = np.asarray(value, dtype=dtype)
    if matrix.shape != shape:
        raise ValueError("{} must have shape {}, got {}".format(label, shape, matrix.shape))
    if not np.isfinite(matrix).all():
        raise ValueError("{} contains NaN or Inf".format(label))
    return matrix


def rotation_matrix_to_quaternion(matrix):
    """Return a scalar-first (w, x, y, z) unit quaternion."""
    matrix = require_matrix(matrix, (3, 3), "rotation matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
             (matrix[0, 2] - matrix[2, 0]) / scale,
             (matrix[1, 0] - matrix[0, 1]) / scale]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale]
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [(matrix[0, 2] - matrix[2, 0]) / scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [(matrix[1, 0] - matrix[0, 1]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale]
            )
    return quaternion / np.linalg.norm(quaternion)


def make_can_bus(info, ego2global):
    if "can_bus" not in info or info["can_bus"] is None:
        raise KeyError("can_bus is required for real Navformer inference")
    can_bus = np.asarray(info["can_bus"], dtype=np.float64).reshape(-1).copy()
    if can_bus.size < 18:
        raise ValueError("can_bus must contain at least 18 values, got {}".format(can_bus.size))
    if not np.isfinite(can_bus).all():
        raise ValueError("can_bus contains NaN or Inf")

    rotation = ego2global[:3, :3]
    can_bus[:3] = ego2global[:3, 3]
    can_bus[3:7] = rotation_matrix_to_quaternion(rotation)
    yaw_radians = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    yaw_degrees = float(np.degrees(yaw_radians))
    if yaw_degrees < 0.0:
        yaw_degrees += 360.0
    can_bus[-2] = np.radians(yaw_degrees)
    can_bus[-1] = yaw_degrees
    return can_bus.astype(np.float32)


def build_camera_geometry(info, image_root, cv2):
    cameras = info.get("cams")
    if not isinstance(cameras, dict) or not cameras:
        raise KeyError("OpenScene info has no cams mapping")
    camera_names = list(cameras.keys())
    if len(camera_names) != 8:
        raise ValueError(
            "Navformer requires all 8 metadata cameras, got {}: {}".format(
                len(camera_names), camera_names
            )
        )

    filenames = []
    lidar2img = []
    lidar2cam = []
    cam_intrinsic = []
    cam_distortion = []
    cam_optim_intrinsic = []
    for camera_name in camera_names:
        camera = cameras[camera_name]
        for field in (
            "data_path",
            "sensor2lidar_rotation",
            "sensor2lidar_translation",
            "cam_intrinsic",
            "distortion",
        ):
            if field not in camera:
                raise KeyError("cams.{}.{} is required".format(camera_name, field))

        filename = resolve_camera_path(camera["data_path"], image_root)
        sensor2lidar_rotation = require_matrix(
            camera["sensor2lidar_rotation"], (3, 3),
            "cams.{}.sensor2lidar_rotation".format(camera_name)
        )
        sensor2lidar_translation = require_matrix(
            camera["sensor2lidar_translation"], (3,),
            "cams.{}.sensor2lidar_translation".format(camera_name)
        )
        intrinsic = require_matrix(
            camera["cam_intrinsic"], (3, 3),
            "cams.{}.cam_intrinsic".format(camera_name)
        )
        distortion = np.asarray(camera["distortion"], dtype=np.float64).reshape(-1)
        if not np.isfinite(distortion).all():
            raise ValueError("cams.{}.distortion contains NaN or Inf".format(camera_name))

        # This is the exact convention used by NavSimOpenSceneE2E.update_sensor().
        lidar2cam_rotation = sensor2lidar_rotation.T
        lidar2cam_translation = -lidar2cam_rotation @ sensor2lidar_translation
        lidar2cam_matrix = np.eye(4, dtype=np.float64)
        lidar2cam_matrix[:3, :3] = lidar2cam_rotation
        lidar2cam_matrix[:3, 3] = lidar2cam_translation
        viewpad = np.eye(4, dtype=np.float64)
        viewpad[:3, :3] = intrinsic

        filenames.append(str(filename))
        lidar2cam.append(lidar2cam_matrix.astype(np.float32))
        lidar2img.append((viewpad @ lidar2cam_matrix).astype(np.float32))
        cam_intrinsic.append(intrinsic.astype(np.float32))
        cam_distortion.append(distortion.astype(np.float32))
        optimal, _ = cv2.getOptimalNewCameraMatrix(
            intrinsic, distortion, (1920, 1080), 1
        )
        cam_optim_intrinsic.append(optimal.astype(np.float32))

    return OrderedDict(
        camera_names=camera_names,
        filenames=filenames,
        lidar2img=lidar2img,
        lidar2cam=lidar2cam,
        cam_intrinsic=cam_intrinsic,
        cam_distortion=cam_distortion,
        cam_optim_intrinsic=cam_optim_intrinsic,
    )


def find_transform(config_value, transform_type):
    matches = []

    def visit(value):
        if isinstance(value, dict):
            if value.get("type") == transform_type:
                matches.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(config_value)
    if len(matches) != 1:
        raise ValueError(
            "expected one {} in test_pipeline, found {}".format(
                transform_type, len(matches)
            )
        )
    return copy.deepcopy(dict(matches[0]))


def build_preprocess_transforms(cfg, build_from_cfg, pipeline_registry):
    normalize_cfg = find_transform(cfg.test_pipeline, "NormalizeMultiviewImage")
    pad_cfg = find_transform(cfg.test_pipeline, "PadMultiViewImage")
    scale_cfg = find_transform(cfg.test_pipeline, "RandomScaleImageMultiViewImage")
    if pad_cfg.get("size_divisor") != 32:
        raise ValueError("Navformer test pipeline pad divisor is not 32")
    if list(scale_cfg.get("scales", [])) != [0.5]:
        raise ValueError("Navformer test pipeline scale is not exactly [0.5]")
    configs = (normalize_cfg, scale_cfg, pad_cfg)
    transforms = [build_from_cfg(item, pipeline_registry) for item in configs]
    return transforms, configs


def preprocess_images(geometry, transforms, mmcv):
    images = []
    source_shapes = []
    for filename in geometry["filenames"]:
        image = mmcv.imread(filename, flag="color")
        if image is None:
            raise RuntimeError("failed to decode OpenScene image: {}".format(filename))
        image = image.astype(np.float32)
        images.append(image)
        source_shapes.append(image.shape)
    if len(set(source_shapes)) != 1:
        raise ValueError("8 camera image shapes differ: {}".format(source_shapes))

    results = {
        "img": images,
        "lidar2img": [matrix.copy() for matrix in geometry["lidar2img"]],
    }
    for transform in transforms:
        results = transform(results)
        if results is None:
            raise RuntimeError("Navformer preprocessing transform returned None")
    final_shapes = [image.shape for image in results["img"]]
    if len(set(final_shapes)) != 1:
        raise ValueError("preprocessed camera image shapes differ: {}".format(final_shapes))
    image_tensor = torch.from_numpy(
        np.stack([image.transpose(2, 0, 1) for image in results["img"]])
    ).contiguous()
    return results, image_tensor


def build_model_and_load_checkpoint(cfg, checkpoint_path, build_model):
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    print("built model class: {}".format(type(model).__name__))
    if cfg.model.get("type") == "UniAD" and type(model).__name__ != "UniAD":
        raise RuntimeError(
            "configured UniAD built as unexpected class {}".format(type(model).__name__)
        )
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint has no state_dict mapping")
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key)[7:]: value for key, value in state_dict.items()}
    incompatible = model.load_state_dict(state_dict, strict=False)
    print("checkpoint missing keys ({}): {}".format(
        len(incompatible.missing_keys), incompatible.missing_keys
    ))
    print("checkpoint unexpected keys ({}): {}".format(
        len(incompatible.unexpected_keys), incompatible.unexpected_keys
    ))
    return model


def prepare_model_inputs(info, geometry, processed, image_tensor, get_box_type, device):
    for field in ("token", "scene_token", "timestamp", "frame_idx", "lidar2global", "ego2global"):
        if field not in info:
            raise KeyError("{} is required for real Navformer inference".format(field))
    lidar2global = require_matrix(info["lidar2global"], (4, 4), "lidar2global")
    ego2global = require_matrix(info["ego2global"], (4, 4), "ego2global")
    can_bus = make_can_bus(info, ego2global)
    box_type_3d, box_mode_3d = get_box_type("LiDAR")
    timestamp_seconds = float(info["timestamp"]) / 1e6
    token = str(info["token"])

    meta = {
        "filename": geometry["filenames"],
        "frame_idx": int(info["frame_idx"]),
        "ori_shape": processed["ori_shape"],
        "img_shape": processed["img_shape"],
        "pad_shape": processed["pad_shape"],
        "lidar2img": processed["lidar2img"],
        "lidar2cam": geometry["lidar2cam"],
        "cam_intrinsic": geometry["cam_intrinsic"],
        "cam_distortion": geometry["cam_distortion"],
        "cam_optim_intrinsic": geometry["cam_optim_intrinsic"],
        "lidar2global_rotation": lidar2global[:3, :3].astype(np.float32),
        "scale_factor": 1.0,
        "flip": False,
        "box_type_3d": box_type_3d,
        "box_mode_3d": box_mode_3d,
        "img_norm_cfg": processed["img_norm_cfg"],
        "sample_idx": token,
        "prev_idx": info.get("sample_prev", info.get("prev")),
        "next_idx": info.get("sample_next", info.get("next")),
        "scene_token": str(info["scene_token"]),
        "can_bus": can_bus,
        # Panseg only uses this value as a display identifier when decoding maps.
        "pts_filename": str(info.get("lidar_path", token)),
    }
    return {
        "img": image_tensor.unsqueeze(0).to(device),
        "img_metas": [meta],
        "l2g_t": torch.as_tensor(
            lidar2global[:3, 3], dtype=torch.float32, device=device
        ).unsqueeze(0),
        "l2g_r_mat": torch.as_tensor(
            lidar2global[:3, :3], dtype=torch.float32, device=device
        ).unsqueeze(0),
        "timestamp": torch.tensor([timestamp_seconds], dtype=torch.float32, device=device),
    }


def run_track_and_map_without_gt(model, model_inputs):
    """Run the forward_test prediction path without its GT-only map IoU branch."""
    track_results = model.simple_test_track(
        model_inputs["img"],
        model_inputs["l2g_t"],
        model_inputs["l2g_r_mat"],
        model_inputs["img_metas"],
        model_inputs["timestamp"],
    )
    track_results[0] = model.upsample_bev_if_tiny(track_results[0])
    if not getattr(model, "with_seg_head", False):
        raise RuntimeError("configured Navformer model has no segmentation/map head")
    bev_embed = track_results[0]["bev_embed"]
    prediction = model.seg_head(bev_embed)
    map_results = model.seg_head.get_bboxes(
        prediction["outputs_classes"],
        prediction["outputs_coords"],
        prediction["enc_outputs_class"],
        prediction["enc_outputs_coord"],
        prediction["args_tuple"],
        prediction["reference"],
        model_inputs["img_metas"],
        rescale=True,
    )
    return track_results[0], map_results[0]


def as_cpu(value):
    if hasattr(value, "tensor"):
        value = value.tensor
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def print_outputs(track, mapping):
    boxes = as_cpu(track.get("boxes_3d", track.get("track_bbox_results")))
    scores = as_cpu(track.get("scores_3d", track.get("track_scores")))
    labels = as_cpu(track.get("labels_3d"))
    track_ids = as_cpu(track.get("track_ids"))
    drivable = as_cpu(mapping["score_list"][-1])
    lanes = as_cpu(mapping["lane_score"])
    print("\n=== Navformer outputs ===")
    print("track boxes: {}".format(boxes))
    print("track scores: {}".format(scores))
    print("track labels: {}".format(labels))
    print("track ids: {}".format(track_ids))
    print("map soft drivable shape: {}".format(tuple(drivable.shape)))
    print("map soft lanes shape: {}".format(tuple(lanes.shape)))


def main():
    args = parse_args()
    navformer_root = require_directory(args.navformer_root, "Navformer root")
    config_path = require_file(args.config, "Navformer config")
    checkpoint_path = require_file(args.checkpoint, "Navformer checkpoint")
    metadata_path = require_file(args.metadata, "OpenScene metadata")
    image_root = require_directory(args.image_root, "OpenScene image root")
    if str(navformer_root) not in sys.path:
        sys.path.insert(0, str(navformer_root))

    import cv2
    import mmcv
    from mmcv import Config
    from mmcv.utils import build_from_cfg
    from mmdet3d.core.bbox import get_box_type
    from mmdet.datasets.builder import PIPELINES
    from mmdet3d.models import build_model

    custom_imports = Config.fromfile(str(config_path)).get("custom_imports", {})
    for module_name in custom_imports.get("imports", ["mmdet3d_plugin"]):
        importlib.import_module(module_name)
    cfg = Config.fromfile(str(config_path))
    print("model type: {}".format(cfg.model.get("type", "MISSING")))
    info = load_info(metadata_path, args.sample_index)
    geometry = build_camera_geometry(info, image_root, cv2)
    print("camera names (metadata order): {}".format(geometry["camera_names"]))

    transforms, transform_configs = build_preprocess_transforms(
        cfg, build_from_cfg, PIPELINES
    )
    print("test preprocessing: {}".format(
        [item["type"] for item in transform_configs]
    ))
    print("normalize config: {}".format(transform_configs[0]))
    print("scale config: {}".format(transform_configs[1]))
    print("pad config: {}".format(transform_configs[2]))
    processed, image_tensor = preprocess_images(geometry, transforms, mmcv)
    print("preprocessed img shape: {}".format(tuple(image_tensor.shape)))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Navformer teacher inference")
    device = torch.device("cuda:0")
    model = build_model_and_load_checkpoint(cfg, checkpoint_path, build_model)
    model.to(device).eval()
    model_inputs = prepare_model_inputs(
        info, geometry, processed, image_tensor, get_box_type, device
    )
    print("sample_idx: {}".format(model_inputs["img_metas"][0]["sample_idx"]))
    print("scene_token: {}".format(model_inputs["img_metas"][0]["scene_token"]))
    print("frame_idx: {}".format(model_inputs["img_metas"][0]["frame_idx"]))
    print("timestamp: {}".format(model_inputs["timestamp"].detach().cpu().tolist()))
    print("l2g_t shape: {}".format(tuple(model_inputs["l2g_t"].shape)))
    print("l2g_r_mat shape: {}".format(tuple(model_inputs["l2g_r_mat"].shape)))

    with torch.no_grad():
        track, mapping = run_track_and_map_without_gt(model, model_inputs)
    print_outputs(track, mapping)
    print("NAVFORMER_OPENSCENE_INFERENCE = PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("NAVFORMER_OPENSCENE_INFERENCE = FAIL", file=sys.stderr)
        print("{}: {}".format(type(error).__name__, error), file=sys.stderr)
        raise
