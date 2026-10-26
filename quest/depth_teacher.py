"""Metric3Dv2 adaptation, without importing QUEST models or training modules."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


DEPTH_SCHEMA_VERSION = 1
DEPTH_ALIGNMENT_VERSION = "metric3dv2_camera_z_original_pixels_v1"
IMAGE_GEOMETRIES = ("unverified", "rectified_pinhole", "raw_opencv")


def validate_intrinsic(value: Any) -> np.ndarray:
    intrinsic = np.asarray(value, dtype=np.float64)
    if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-6)
            or abs(intrinsic[0, 1]) > 1e-6 or abs(intrinsic[1, 0]) > 1e-6):
        raise ValueError("Expected finite, zero-skew pinhole cam_intrinsic with positive fx/fy")
    return intrinsic


def sensor_to_lidar(camera: Mapping[str, Any]) -> np.ndarray:
    rotation = np.asarray(camera.get("sensor2lidar_rotation"), dtype=np.float64)
    translation = np.asarray(camera.get("sensor2lidar_translation"), dtype=np.float64)
    if (rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
        raise ValueError("Invalid camera sensor2lidar rigid transform")
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, translation
    return matrix


def validate_image_geometry(camera: Mapping[str, Any], geometry: str) -> np.ndarray:
    if geometry not in IMAGE_GEOMETRIES or geometry == "unverified":
        raise ValueError("IMAGE_GEOMETRY_UNVERIFIED: confirm raw_opencv or rectified_pinhole first")
    if "distortion" not in camera:
        raise ValueError("MISSING distortion: do not silently assume zero distortion")
    distortion = np.asarray(camera["distortion"], dtype=np.float64).reshape(-1)
    if not np.isfinite(distortion).all():
        raise ValueError("distortion contains NaN/Inf")
    if geometry == "rectified_pinhole":
        if np.any(np.abs(distortion) > 1e-10):
            raise ValueError("Rectified images with nonzero metadata distortion need verified corrected K; "
                             "this adapter will not silently reuse raw calibration")
    elif len(distortion) not in (4, 5, 8, 12, 14):
        raise ValueError("raw_opencv requires documented OpenCV distortion ordering, not fisheye coefficients")
    return distortion


def scale_intrinsic(intrinsic: np.ndarray, source_hw: tuple[int, int],
                    target_hw: tuple[int, int]) -> np.ndarray:
    result = validate_intrinsic(intrinsic).copy()
    if min(*source_hw, *target_hw) <= 0:
        raise ValueError("Image dimensions must be positive")
    # Match the existing OpenSceneMetadataDataset resize convention.
    result[0] *= target_hw[1] / source_hw[1]
    result[1] *= target_hw[0] / source_hw[0]
    return result


def prepare_metric3d_input(rgb: np.ndarray, intrinsic: np.ndarray,
                           input_hw: tuple[int, int], canonical_focal: float
                           ) -> tuple[torch.Tensor, dict[str, Any]]:
    """Follow official hubconf.py: RGB, aspect resize, mean padding, normalization."""
    import cv2

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Teacher input must be original RGB uint8 JPEG pixels")
    intrinsic = validate_intrinsic(intrinsic)
    if canonical_focal <= 0 or not np.isfinite(canonical_focal):
        raise ValueError("canonical_focal must be finite and positive")
    source_h, source_w = rgb.shape[:2]
    input_h, input_w = input_hw
    if min(input_hw) <= 0 or input_h % 28 or input_w % 28:
        raise ValueError("Metric3Dv2 ViT input dimensions must be positive multiples of 28")
    ratio = min(input_h / source_h, input_w / source_w)
    resized_h, resized_w = int(source_h * ratio), int(source_w * ratio)
    if min(resized_h, resized_w) <= 0:
        raise ValueError("Teacher resize produced an empty image")
    resized = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    top, left = (input_h - resized_h) // 2, (input_w - resized_w) // 2
    bottom, right = input_h - resized_h - top, input_w - resized_w - left
    mean = [123.675, 116.28, 103.53]
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=mean)
    image = torch.from_numpy(padded.transpose(2, 0, 1).copy()).float()
    image = (image - torch.tensor(mean)[:, None, None]) / torch.tensor(
        [58.395, 57.12, 57.375])[:, None, None]
    # The official ViT hub example uses fx * uniform resize ratio / canonical focal.
    metadata = {"source_hw": (source_h, source_w), "resized_hw": (resized_h, resized_w),
                "input_hw": tuple(input_hw), "padding_tblr": (top, bottom, left, right),
                "resize_ratio": ratio, "metric_scale": float(intrinsic[0, 0] * ratio / canonical_focal)}
    return image.unsqueeze(0), metadata


def restore_metric_depth(prediction: torch.Tensor, resize: Mapping[str, Any]) -> np.ndarray:
    if prediction.ndim != 4 or tuple(prediction.shape[:2]) != (1, 1):
        raise ValueError(f"Unexpected Metric3D depth shape {tuple(prediction.shape)}; expected [1,1,H,W]")
    if tuple(prediction.shape[-2:]) != tuple(resize["input_hw"]):
        raise ValueError("Native depth resolution differs from input; refuse unverified padding removal")
    if not prediction.is_floating_point() or not torch.isfinite(prediction).all():
        raise ValueError("Native teacher output contains NaN/Inf")
    top, _, left, _ = resize["padding_tblr"]
    height, width = resize["resized_hw"]
    cropped = prediction[..., top:top + height, left:left + width]
    depth = F.interpolate(cropped.float(), size=resize["source_hw"], mode="bilinear",
                          align_corners=False)[0, 0] * float(resize["metric_scale"])
    # No per-image median scaling, sigmoid, clipping, or confidence fabrication.
    return depth.detach().cpu().numpy()


def rectified_teacher_image(rgb: np.ndarray, intrinsic: np.ndarray,
                            distortion: np.ndarray, geometry: str) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    if geometry == "rectified_pinhole":
        return rgb, np.ones(rgb.shape[:2], dtype=bool)
    if geometry != "raw_opencv":
        raise ValueError("Image geometry is unverified")
    height, width = rgb.shape[:2]
    map_x, map_y = cv2.initUndistortRectifyMap(
        intrinsic, distortion, np.eye(3), intrinsic, (width, height), cv2.CV_32FC1)
    valid = ((map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
             & np.isfinite(map_x) & np.isfinite(map_y))
    rectified = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(123.675, 116.28, 103.53))
    return rectified, valid


def depth_to_original_pixels(depth: np.ndarray, valid: np.ndarray, intrinsic: np.ndarray,
                             distortion: np.ndarray, geometry: str
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Return camera Z at original JPEG pixels; rectification uses R=I, so Z is unchanged."""
    import cv2

    if geometry == "rectified_pinhole":
        return depth.copy(), valid.copy()
    if geometry != "raw_opencv":
        raise ValueError("Image geometry is unverified")
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    raw_pixels = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
    rectified_uv = cv2.undistortPoints(raw_pixels, intrinsic, distortion, P=intrinsic)
    maps = rectified_uv.reshape(height, width, 2).astype(np.float32)
    inside = (np.isfinite(maps).all(axis=-1) & (maps[..., 0] >= 0) & (maps[..., 0] <= width - 1)
              & (maps[..., 1] >= 0) & (maps[..., 1] <= height - 1))
    safe_maps = np.where(inside[..., None], maps, -1).astype(np.float32)
    restored = cv2.remap(np.where(valid, depth, 0).astype(np.float32),
                         safe_maps[..., 0], safe_maps[..., 1], cv2.INTER_LINEAR)
    coverage = cv2.remap(valid.astype(np.float32), safe_maps[..., 0], safe_maps[..., 1],
                         cv2.INTER_LINEAR)
    mask = inside & (coverage >= 1 - 1e-6)
    return np.where(mask, restored, 0), mask


def resize_depth_label(depth: np.ndarray, mask: np.ndarray, output_hw: tuple[int, int],
                       min_depth: float, max_depth: float) -> tuple[torch.Tensor, torch.Tensor]:
    import cv2

    if not (0 < min_depth < max_depth and np.isfinite([min_depth, max_depth]).all()):
        raise ValueError("Invalid depth range")
    mask = mask & np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    size = (output_hw[1], output_hw[0])
    resized = cv2.resize(np.where(mask, depth, 0).astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    coverage = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    valid = (coverage >= 1 - 1e-6) & (resized >= min_depth) & (resized <= max_depth)
    return (torch.from_numpy(np.where(valid, resized, 0).astype(np.float32)).unsqueeze(0),
            torch.from_numpy(valid).unsqueeze(0))


def pixel_rays(uv: np.ndarray, intrinsic: np.ndarray, distortion: np.ndarray,
               geometry: str) -> np.ndarray:
    import cv2

    if geometry == "raw_opencv":
        xy = cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), intrinsic,
                                  distortion).reshape(-1, 2)
        return np.column_stack((xy, np.ones(len(xy))))
    if geometry != "rectified_pinhole":
        raise ValueError("Image geometry is unverified")
    return np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(intrinsic).T


def cross_view_audit(depth: np.ndarray, valid: np.ndarray, intrinsics: np.ndarray,
                     transforms: np.ndarray, distortions: list[np.ndarray],
                     names: list[str], geometry: str, stride: int = 8) -> list[dict[str, Any]]:
    """Reprojection diagnostic only: occlusion/dynamics also cause disagreement."""
    import cv2

    if stride <= 0:
        raise ValueError("overlap_stride must be positive")
    _, height, width = depth.shape
    yy, xx = np.mgrid[0:height:stride, 0:width:stride]
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    results = []
    for source in range(len(names)):
        take = valid[source, yy, xx].ravel()
        if not take.any():
            continue
        uv = pixels[take]
        camera_xyz = pixel_rays(uv, intrinsics[source], distortions[source], geometry)
        camera_xyz *= depth[source, uv[:, 1], uv[:, 0], None]
        lidar_xyz = camera_xyz @ transforms[source, :3, :3].T + transforms[source, :3, 3]
        for target in range(source + 1, len(names)):
            inverse = np.linalg.inv(transforms[target])
            target_xyz = lidar_xyz @ inverse[:3, :3].T + inverse[:3, 3]
            target_xyz = target_xyz[np.isfinite(target_xyz).all(axis=1) & (target_xyz[:, 2] > 0)]
            if not len(target_xyz):
                continue
            coefficients = distortions[target] if geometry == "raw_opencv" else None
            projected, _ = cv2.projectPoints(target_xyz, np.zeros(3), np.zeros(3),
                                             intrinsics[target], coefficients)
            projected = projected.reshape(-1, 2)
            inside = (np.isfinite(projected).all(axis=1) & (projected[:, 0] >= 0)
                      & (projected[:, 0] <= width - 1) & (projected[:, 1] >= 0)
                      & (projected[:, 1] <= height - 1))
            uv_target = np.rint(projected[inside]).astype(np.int64)
            z = target_xyz[inside, 2]
            keep = valid[target, uv_target[:, 1], uv_target[:, 0]]
            observed = depth[target, uv_target[keep, 1], uv_target[keep, 0]]
            error = np.abs(z[keep] - observed)
            if len(error):
                results.append({"source": names[source], "target": names[target],
                                "compared_pixels": len(error), "median_abs_z_error_m": float(np.median(error)),
                                "median_relative_z_error": float(np.median(error / observed))})
    return results


def validate_depth_record(record: Mapping[str, Any], token: str, source_index: int,
                           camera_names: tuple[str, ...]) -> None:
    if (record.get("schema_version") != DEPTH_SCHEMA_VERSION
            or record.get("alignment_version") != DEPTH_ALIGNMENT_VERSION):
        raise ValueError("Unsupported depth label schema/alignment version")
    if record.get("token") != token or record.get("source_index") != source_index:
        raise ValueError("Depth label original-index/token mismatch")
    if tuple(record.get("camera_names", ())) != tuple(camera_names):
        raise ValueError("Depth label camera order mismatch")
    if record.get("adaptation_verified") is not False:
        raise ValueError("This export schema only represents unaudited teacher adaptation outputs")
    if (not isinstance(record.get("metadata_sha256"), str) or len(record["metadata_sha256"]) != 64
            or not isinstance(record.get("provenance"), Mapping)
            or not record["provenance"].get("checkpoint_sha256")
            or not record["provenance"].get("geometry_evidence")):
        raise ValueError("Missing depth metadata/checkpoint/geometry provenance")
    label = record["depth"]
    if label.get("units") != "m" or label.get("definition") != "camera_z":
        raise ValueError("Depth must be camera Z in meters, not radial distance or relative depth")
    depth, mask = label["depth_m"], label["valid_mask"]
    if (depth.dtype != torch.float32 or depth.ndim != 4
            or tuple(depth.shape) != (len(camera_names), 1, 252, 448)
            or mask.shape != depth.shape or mask.dtype != torch.bool
            or not torch.isfinite(depth).all() or not (depth[mask] > 0).all()
            or (depth[~mask] != 0).any() or not mask.flatten(1).any(dim=1).all()):
        raise ValueError("Invalid canonical depth values/shape/mask or camera with no valid pixels")
    if label.get("image_geometry") not in IMAGE_GEOMETRIES[1:]:
        raise ValueError("Depth image geometry must be explicit")
    lower, upper = float(label["min_depth_m"]), float(label["max_depth_m"])
    if (not np.isfinite([lower, upper]).all() or not 0 < lower < upper
            or (depth[mask] < lower).any() or (depth[mask] > upper).any()):
        raise ValueError("Depth outside declared valid metric range")
    intrinsics, transforms, distortions = label["intrinsics"], label["sensor2lidar"], label["distortion"]
    if (tuple(intrinsics.shape) != (len(camera_names), 3, 3)
            or tuple(transforms.shape) != (len(camera_names), 4, 4)
            or len(distortions) != len(camera_names)):
        raise ValueError("Depth calibration camera shape mismatch")
    for intrinsic, transform in zip(intrinsics, transforms):
        validate_intrinsic(intrinsic.cpu().numpy())
        if not torch.isfinite(transform).all() or not torch.allclose(
                transform[3], torch.tensor([0., 0., 0., 1.], device=transform.device, dtype=transform.dtype)):
            raise ValueError("Invalid saved sensor2lidar homogeneous matrix")
        sensor_to_lidar({"sensor2lidar_rotation": transform[:3, :3].cpu().numpy(),
                         "sensor2lidar_translation": transform[:3, 3].cpu().numpy()})
    for distortion in distortions:
        validate_image_geometry({"distortion": distortion.cpu().numpy()}, label["image_geometry"])
