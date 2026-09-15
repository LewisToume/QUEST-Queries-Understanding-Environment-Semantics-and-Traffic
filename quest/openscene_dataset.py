from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

OPENSCENE_CAMERA_NAMES = (
    "CAM_F0",
    "CAM_B0",
    "CAM_L0",
    "CAM_L1",
    "CAM_L2",
    "CAM_R0",
    "CAM_R1",
    "CAM_R2",
)

OPENSCENE_AGENT_CLASS_TO_ID = {
    "vehicle": 0,
    "pedestrian": 1,
    "traffic_cone": 2,
    "generic_object": 3,
}

SOFT_LABEL_TASKS = frozenset({"seg", "depth", "agent", "map"})


def resolve_openscene_camera_path(raw_path: str, camera_root: str | Path) -> Path:
    """Map an official OpenScene camera path to the extracted mini archive."""
    parts = list(Path(raw_path.replace("\\", "/")).parts)
    if parts and parts[0].lower() == "dataset":
        parts = parts[1:]
    try:
        sensor_index = parts.index("sensor_blobs") + 1
    except ValueError as error:
        raise ValueError(f"camera path does not contain sensor_blobs: {raw_path}") from error
    if sensor_index >= len(parts) or parts[sensor_index] != "mini":
        parts.insert(sensor_index, "mini")
    return Path(camera_root).joinpath(*parts)


class OpenSceneMetadataDataset(Dataset):
    """OpenScene 8-camera frames loaded directly from the official metadata."""

    def __init__(
        self,
        metadata_path: str | Path,
        camera_root: str | Path,
        max_samples: int = 0,
        image_size: tuple[int, int] = (224, 224),
        max_agent_instances: int = 64,
        camera_names: Sequence[str] = OPENSCENE_CAMERA_NAMES,
        xy_range: tuple[float, float] = (-50.0, 50.0),
        z_range: tuple[float, float] = (-5.0, 5.0),
        size_norm: tuple[float, float, float] = (20.0, 10.0, 8.0),
        velocity_norm: float = 20.0,
        soft_labels_root: str | Path | None = None,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.camera_root = Path(camera_root)
        self.image_size = tuple(image_size)
        self.max_agent_instances = max_agent_instances
        self.camera_names = tuple(camera_names)
        self.xy_range = tuple(xy_range)
        self.z_range = tuple(z_range)
        self.size_norm = torch.tensor(size_norm, dtype=torch.float32)
        self.velocity_norm = float(velocity_norm)
        self.soft_labels_root = (
            Path(soft_labels_root) if soft_labels_root is not None else None
        )

        with self.metadata_path.open("rb") as file:
            metadata = pickle.load(file)
        infos = metadata.get("infos")
        if not isinstance(infos, list):
            raise ValueError(f"OpenScene metadata has no infos list: {self.metadata_path}")

        self.infos: list[dict[str, Any]] = []
        for info in infos:
            if self._is_complete_frame(info):
                self.infos.append(info)
                if max_samples > 0 and len(self.infos) >= max_samples:
                    break
        if not self.infos:
            raise RuntimeError(f"no complete 8-camera frames found in {self.metadata_path}")
        if max_samples > 0 and len(self.infos) < max_samples:
            raise RuntimeError(
                f"requested {max_samples} complete frames, found {len(self.infos)}"
            )

    def __len__(self) -> int:
        return len(self.infos)

    def _camera_metadata_path(self, info: Mapping[str, Any], camera_name: str) -> Path:
        return resolve_openscene_camera_path(
            info["cams"][camera_name]["data_path"], self.camera_root
        )

    def _is_complete_frame(self, info: Mapping[str, Any]) -> bool:
        cameras = info.get("cams") or {}
        if any(camera_name not in cameras for camera_name in self.camera_names):
            return False
        for camera_name in self.camera_names:
            camera = cameras[camera_name]
            intrinsic = np.asarray(camera.get("cam_intrinsic"))
            rotation = np.asarray(camera.get("sensor2lidar_rotation"))
            translation = np.asarray(camera.get("sensor2lidar_translation"))
            if (
                intrinsic.shape != (3, 3)
                or rotation.shape != (3, 3)
                or translation.shape != (3,)
            ):
                return False
            if not all(np.isfinite(value).all() for value in (intrinsic, rotation, translation)):
                return False
        ego_pose = np.asarray(info.get("ego2global"))
        can_bus = np.asarray(info.get("can_bus"))
        if ego_pose.shape != (4, 4) or can_bus.size < 9:
            return False
        if not np.isfinite(ego_pose).all() or not np.isfinite(can_bus[:9]).all():
            return False
        try:
            paths = [
                self._camera_metadata_path(info, camera_name)
                for camera_name in self.camera_names
            ]
        except (KeyError, TypeError, ValueError):
            return False
        return all(path.exists() for path in paths)

    def _load_image(self, path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
        with Image.open(path) as image:
            image = image.convert("RGB")
            source_size = image.size
            image = image.resize(
                (self.image_size[1], self.image_size[0]), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return (tensor - mean) / std, source_size

    def _load_multiview(self, info: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        images: list[torch.Tensor] = []
        intrinsics: list[torch.Tensor] = []
        extrinsics: list[torch.Tensor] = []
        for camera_name in self.camera_names:
            camera = info["cams"][camera_name]
            image, (source_width, source_height) = self._load_image(
                self._camera_metadata_path(info, camera_name)
            )
            intrinsic = torch.as_tensor(
                camera["cam_intrinsic"], dtype=torch.float32
            ).clone()
            intrinsic[0, :] *= self.image_size[1] / source_width
            intrinsic[1, :] *= self.image_size[0] / source_height
            extrinsic = torch.eye(4, dtype=torch.float32)
            extrinsic[:3, :3] = torch.as_tensor(
                camera["sensor2lidar_rotation"], dtype=torch.float32
            )
            extrinsic[:3, 3] = torch.as_tensor(
                camera["sensor2lidar_translation"], dtype=torch.float32
            )
            images.append(image)
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)
        return {
            "images": torch.stack(images),
            "intrinsics": torch.stack(intrinsics),
            "extrinsics": torch.stack(extrinsics),
        }

    def _load_agent_gt(
        self, info: Mapping[str, Any]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        labels = torch.full((self.max_agent_instances,), -1, dtype=torch.long)
        boxes = torch.zeros((self.max_agent_instances, 8), dtype=torch.float32)
        velocity = torch.zeros((self.max_agent_instances, 3), dtype=torch.float32)
        raw_boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float32)
        raw_names = np.asarray(info.get("gt_names", []))
        raw_velocity = np.asarray(
            info.get("gt_velocity_3d", info.get("gt_velocity", [])),
            dtype=np.float32,
        )
        if raw_boxes.size == 0 or raw_names.size == 0:
            return {"labels": labels, "boxes": boxes, "velocity": velocity}, torch.tensor(False)
        if raw_velocity.size == 0:
            raw_velocity = np.zeros((len(raw_boxes), 3), dtype=np.float32)

        xy_min, xy_max = self.xy_range
        z_min, z_max = self.z_range
        candidates: list[tuple[float, int, torch.Tensor, torch.Tensor]] = []
        for box, raw_name, item_velocity in zip(raw_boxes, raw_names, raw_velocity):
            name = str(raw_name)
            if name not in OPENSCENE_AGENT_CLASS_TO_ID:
                continue
            x, y, z, dx, dy, dz, yaw = (float(value) for value in box[:7])
            if not (
                xy_min <= x <= xy_max
                and xy_min <= y <= xy_max
                and z_min <= z <= z_max
            ):
                continue
            center = torch.tensor(
                [
                    (x - xy_min) / (xy_max - xy_min),
                    (y - xy_min) / (xy_max - xy_min),
                    (z - z_min) / (z_max - z_min),
                ]
            )
            size = (
                torch.tensor([dx, dy, dz], dtype=torch.float32) / self.size_norm
            ).clamp(0.0, 1.0)
            yaw_vector = torch.tensor([np.sin(yaw), np.cos(yaw)], dtype=torch.float32)
            item_box = torch.cat([center.float().clamp(0.0, 1.0), size, yaw_vector])
            item_velocity = (
                torch.as_tensor(item_velocity[:3], dtype=torch.float32)
                / self.velocity_norm
            )
            candidates.append(
                (x * x + y * y, OPENSCENE_AGENT_CLASS_TO_ID[name], item_box, item_velocity)
            )
        candidates.sort(key=lambda item: item[0])
        for index, (_, class_id, box, item_velocity) in enumerate(
            candidates[: self.max_agent_instances]
        ):
            labels[index] = class_id
            boxes[index] = box
            velocity[index] = item_velocity
        valid = torch.tensor(bool(candidates))
        return {"labels": labels, "boxes": boxes, "velocity": velocity}, valid

    def _load_soft_labels(self, token: str) -> dict[str, Any]:
        if self.soft_labels_root is None:
            return {}
        path = self.soft_labels_root / f"{token}.pt"
        if not path.exists():
            return {}
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError(f"soft-label file must contain a mapping: {path}")
        payload_token = str(payload.get("token", token))
        if payload_token != token:
            raise ValueError(f"soft-label token mismatch: {payload_token} != {token}")
        return {
            task: payload[task]
            for task in SOFT_LABEL_TASKS
            if task in payload and payload[task] is not None
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        info = self.infos[index]
        token = str(info["token"])
        agent_gt, agent_valid = self._load_agent_gt(info)
        sample: dict[str, Any] = {
            **self._load_multiview(info),
            "sample_token": token,
            "camera_names": list(self.camera_names),
            "ego_state": torch.as_tensor(info["can_bus"][:9], dtype=torch.float32),
            "ego_pose": torch.as_tensor(info["ego2global"], dtype=torch.float32),
            "agent_gt": agent_gt,
            "agent_valid": agent_valid,
            "seg_valid": torch.tensor(False),
            "depth_valid": torch.tensor(False),
            "map_valid": torch.tensor(False),
        }
        if self.soft_labels_root is not None:
            sample["soft_labels"] = self._load_soft_labels(token)
        return sample
