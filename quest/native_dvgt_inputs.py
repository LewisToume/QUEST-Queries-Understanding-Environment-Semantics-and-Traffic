"""One original JPEG packet -> preserved spatial pixels and native DVGT pixels."""

import numpy as np
import torch
from PIL import Image

from .depth_teacher import scale_intrinsic
from .dvgt_depth import DVGT_TEACHER_CAMERAS, camera_from_first_ego, prepare_image
from .native_dvgt import raw_to_native_grid
from .openscene_dataset import OPENSCENE_CAMERA_NAMES


def causal_native_window(plan):
    """Use native T=1 at a scene start, never duplicate current or borrow the future."""
    index, previous = plan["source_index"], plan["history_index"]
    if previous is None:
        if plan["reason"] not in ("scene_start", "gap") or plan["dt_s"] != 0:
            raise ValueError("Invalid cold-start history plan")
        return [index]
    if (plan["reason"] != "available" or not plan["history_timestamp"] < plan["timestamp"]
            or not plan["dt_s"] > 0 or previous == index):
        raise ValueError("Native context must be an actual strictly previous frame")
    return [previous, index]


def prepare_native_packet(infos, plan, cameras_by_index, helpers):
    indices = causal_native_window(plan)
    current = indices[-1]
    native_frames, current_rgb, native_ks = [], {}, {}
    for index in indices:
        cameras = {row["name"]: row for row in cameras_by_index[index]}
        views = []
        for name in DVGT_TEACHER_CAMERAS:
            camera = cameras[name]
            with Image.open(camera["path"]) as stream:
                rgb = np.asarray(stream.convert("RGB")).copy()
            if tuple(rgb.shape[:2]) != tuple(camera["source_hw"]):
                raise ValueError("JPEG dimensions changed after camera inspection")
            image, native_k = prepare_image(rgb, camera["intrinsic"], camera["distortion"],
                                            "raw_opencv", helpers)
            views.append(image)
            if index == current:
                current_rgb[name], native_ks[name] = rgb, native_k
        native_frames.append(torch.stack(views))
    cameras = {row["name"]: row for row in cameras_by_index[current]}
    images, intrinsics, extrinsics, transforms, ks, grids = [], [], [], [], [], []
    for name in OPENSCENE_CAMERA_NAMES:
        camera, rgb = cameras[name], current_rgb[name]
        resized = Image.fromarray(rgb).resize((448, 252), Image.Resampling.BILINEAR)
        image = torch.from_numpy(np.array(resized, dtype=np.float32) / 255.).permute(2, 0, 1)
        images.append((image - image.new_tensor([.485, .456, .406])[:, None, None])
                      / image.new_tensor([.229, .224, .225])[:, None, None])
        intrinsics.append(torch.as_tensor(scale_intrinsic(camera["intrinsic"], camera["source_hw"], (252, 448))))
        extrinsics.append(torch.as_tensor(camera["sensor2lidar"]))
        transforms.append(torch.as_tensor(camera_from_first_ego(infos[indices[0]], infos[current],
                                                                 camera["sensor2lidar"])))
        ks.append(torch.as_tensor(native_ks[name]))
        grids.append(raw_to_native_grid(camera["intrinsic"], camera["distortion"], native_ks[name], camera["source_hw"]))
    packet = {"images": torch.stack(images)[None].contiguous(),
              "intrinsics": torch.stack(intrinsics)[None].float(),
              "extrinsics": torch.stack(extrinsics)[None].float(),
              "ego_state": torch.as_tensor(infos[current]["can_bus"][:9], dtype=torch.float32)[None],
              "native_images": torch.stack(native_frames)[None],
              "restoration": {"camera_from_first_ego": torch.stack(transforms)[None].float(),
                              "native_intrinsics": torch.stack(ks)[None].float(),
                              "raw_native_uv": torch.stack(grids)[None]}}
    return packet


def packet_to_device(packet, device):
    return {key: {name: tensor.to(device) for name, tensor in value.items()} if isinstance(value, dict)
            else value.to(device) for key, value in packet.items()}
