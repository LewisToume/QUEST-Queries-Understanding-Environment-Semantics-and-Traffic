from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Sequence

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


def openscene_flat_indices_to_xyz(
    flat_indices: torch.Tensor,
    occ_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode OpenScene's x-fastest flat voxel index into QUEST [X, Y, Z]."""
    size_x, size_y, size_z = occ_size
    flat_indices = flat_indices.long()
    if bool(((flat_indices < 0) | (flat_indices >= size_x * size_y * size_z)).any()):
        raise ValueError("OpenScene flat voxel index is outside the configured occupancy grid")
    voxel_x = flat_indices % size_x
    voxel_y = (flat_indices // size_x) % size_y
    voxel_z = flat_indices // (size_x * size_y)
    return voxel_x, voxel_y, voxel_z


def scatter_openscene_sparse_flow(
    sparse_occ: np.ndarray,
    sparse_flow: np.ndarray,
    occ_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter aligned OpenScene [index, class] and [vx, vy] rows to [2, X, Y, Z]."""
    if sparse_occ.ndim != 2 or sparse_occ.shape[1] < 2:
        raise ValueError(f"expected sparse OCC [N, 2], got {sparse_occ.shape}")
    if sparse_flow.ndim != 2 or sparse_flow.shape[1] != 2:
        raise ValueError(f"expected sparse Flow [N, 2], got {sparse_flow.shape}")
    if len(sparse_occ) != len(sparse_flow):
        raise ValueError(f"OCC/Flow row mismatch: {len(sparse_occ)} != {len(sparse_flow)}")

    flat_indices = torch.from_numpy(sparse_occ[:, 0].astype(np.int64))
    class_ids = torch.from_numpy(sparse_occ[:, 1].astype(np.int64))
    flow_values = torch.from_numpy(sparse_flow.astype(np.float32))
    voxel_x, voxel_y, voxel_z = openscene_flat_indices_to_xyz(flat_indices, occ_size)

    dense_flow = torch.zeros((2, *occ_size), dtype=torch.float32)
    dense_flow[:, voxel_x, voxel_y, voxel_z] = flow_values.transpose(0, 1)
    flow_mask = torch.zeros(occ_size, dtype=torch.bool)
    # OpenScene's official loss supervises object classes 0-9 and excludes background class 10.
    foreground = class_ids < 10
    flow_mask[voxel_x[foreground], voxel_y[foreground], voxel_z[foreground]] = True
    return dense_flow, flow_mask


def scatter_openscene_sparse_occ(
    sparse_occ: np.ndarray,
    occ_size: tuple[int, int, int],
    num_classes: int,
) -> torch.Tensor:
    """Scatter OpenScene [flat_voxel_index, class] rows to QUEST [X, Y, Z]."""
    if sparse_occ.ndim != 2 or sparse_occ.shape[1] < 2:
        raise ValueError(f"expected sparse OCC [N, 2], got {sparse_occ.shape}")
    dense_occ = torch.full(occ_size, 255, dtype=torch.long)
    if sparse_occ.size == 0:
        return dense_occ
    flat_indices = torch.from_numpy(sparse_occ[:, 0].astype(np.int64))
    class_ids = torch.from_numpy(sparse_occ[:, 1].astype(np.int64))
    if bool(((class_ids < 0) | (class_ids >= num_classes)).any()):
        raise ValueError(f"OpenScene OCC class is outside [0, {num_classes - 1}]")
    voxel_x, voxel_y, voxel_z = openscene_flat_indices_to_xyz(flat_indices, occ_size)
    dense_occ[voxel_x, voxel_y, voxel_z] = class_ids
    return dense_occ


def resolve_openscene_metadata_path(
    raw_path: str,
    local_root: str | Path,
    insert_sensor_mini: bool = False,
) -> Path:
    """Map OpenScene metadata paths onto the local mini archive roots."""
    parts = list(Path(raw_path.replace("\\", "/")).parts)
    if parts and parts[0].lower() == "dataset":
        parts = parts[1:]
    if insert_sensor_mini:
        try:
            sensor_index = parts.index("sensor_blobs") + 1
        except ValueError as error:
            raise ValueError(f"camera path does not contain sensor_blobs: {raw_path}") from error
        if sensor_index >= len(parts) or parts[sensor_index] != "mini":
            parts.insert(sensor_index, "mini")
    return Path(local_root).joinpath(*parts)


class OpenSceneFirstTestDataset(Dataset):
    """
    Minimal OpenScene adapter for the extracted first-test folder.

    It exposes real 8-camera images, camera geometry, ego state, 3D agent boxes,
    classes, velocity, and semantic occupancy. Map and flow interfaces are kept
    explicit, but their losses must stay masked until real GT files are wired.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: tuple[int, int] = (224, 224),
        occ_size: tuple[int, int, int] = (200, 200, 16),
        C_occ: int = 11,
        C_flow: int = 2,
        C_map: int = 4,
        P: int = 20,
        max_agent_instances: int = 64,
        max_map_instances: int = 32,
        camera_names: Sequence[str] = OPENSCENE_CAMERA_NAMES,
        xy_range: tuple[float, float] = (-50.0, 50.0),
        z_range: tuple[float, float] = (-5.0, 5.0),
        size_norm: tuple[float, float, float] = (20.0, 10.0, 8.0),
        velocity_norm: float = 20.0,
        require_all_cameras: bool = True,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.occ_size = occ_size
        self.C_occ = C_occ
        self.C_flow = C_flow
        self.C_map = C_map
        self.P = P
        self.max_agent_instances = max_agent_instances
        self.max_map_instances = max_map_instances
        self.camera_names = tuple(camera_names)
        self.xy_range = xy_range
        self.z_range = z_range
        self.size_norm = torch.tensor(size_norm, dtype=torch.float32)
        self.velocity_norm = float(velocity_norm)

        manifest_path = self.root / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        if require_all_cameras:
            self.manifest = [
                item
                for item in self.manifest
                if all(self._camera_path(self.root / item["sample_id"], cam_name).exists() for cam_name in self.camera_names)
            ]
        if not self.manifest:
            raise RuntimeError(f"no OpenScene samples with all required cameras under {self.root}")

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_image(self, image_path: Path) -> torch.Tensor:
        if not image_path.exists():
            raise FileNotFoundError(f"missing OpenScene camera image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std

    def _camera_path(self, sample_dir: Path, cam_name: str) -> Path:
        multiview_path = sample_dir / "cams" / f"{cam_name}.jpg"
        if multiview_path.exists():
            return multiview_path
        if cam_name == "CAM_F0":
            return sample_dir / "cam_f0.jpg"
        return multiview_path

    def _load_multiview(self, sample_dir: Path, metadata: dict[str, Any]) -> dict[str, torch.Tensor]:
        images: list[torch.Tensor] = []
        intrinsics: list[torch.Tensor] = []
        extrinsics: list[torch.Tensor] = []

        for cam_name in self.camera_names:
            cam_info = metadata["cams"].get(cam_name)
            if cam_info is None:
                raise KeyError(f"OpenScene sample missing camera metadata for {cam_name}")
            image_path = self._camera_path(sample_dir, cam_name)
            with Image.open(image_path) as source_image:
                source_width, source_height = source_image.size
            images.append(self._load_image(image_path))
            intrinsic = torch.as_tensor(cam_info["cam_intrinsic"], dtype=torch.float32)
            intrinsic = intrinsic.clone()
            intrinsic[0, :] *= float(self.image_size[1]) / float(source_width)
            intrinsic[1, :] *= float(self.image_size[0]) / float(source_height)
            intrinsics.append(intrinsic)

            extrinsic = torch.eye(4, dtype=torch.float32)
            extrinsic[:3, :3] = torch.as_tensor(cam_info["sensor2lidar_rotation"], dtype=torch.float32)
            extrinsic[:3, 3] = torch.as_tensor(cam_info["sensor2lidar_translation"], dtype=torch.float32)
            extrinsics.append(extrinsic)

        return {
            "images": torch.stack(images, dim=0),
            "intrinsics": torch.stack(intrinsics, dim=0),
            "extrinsics": torch.stack(extrinsics, dim=0),
        }

    def _load_agent_gt(self, metadata: dict[str, Any]) -> dict[str, torch.Tensor]:
        labels = torch.full((self.max_agent_instances,), -1, dtype=torch.long)
        boxes = torch.zeros((self.max_agent_instances, 8), dtype=torch.float32)
        velocity = torch.zeros((self.max_agent_instances, 3), dtype=torch.float32)
        anns = metadata.get("anns") or metadata
        gt_boxes = np.asarray(anns.get("gt_boxes", []), dtype=np.float32)
        gt_names = np.asarray(anns.get("gt_names", []))
        gt_velocity = np.asarray(
            anns.get("gt_velocity_3d", anns.get("gt_velocity", [])),
            dtype=np.float32,
        )

        if gt_boxes.size == 0 or gt_names.size == 0:
            return {"labels": labels, "boxes": boxes, "velocity": velocity}
        if gt_velocity.size == 0:
            gt_velocity = np.zeros((len(gt_boxes), 3), dtype=np.float32)

        candidates: list[tuple[float, int, torch.Tensor, torch.Tensor]] = []
        xy_min, xy_max = self.xy_range
        z_min, z_max = self.z_range
        xy_span = xy_max - xy_min
        z_span = z_max - z_min

        for box, raw_name, raw_velocity in zip(gt_boxes, gt_names, gt_velocity):
            name = str(raw_name)
            if name not in OPENSCENE_AGENT_CLASS_TO_ID:
                continue
            x, y, z, dx, dy, dz, yaw = [float(v) for v in box[:7]]
            if not (xy_min <= x <= xy_max and xy_min <= y <= xy_max and z_min <= z <= z_max):
                continue

            center = torch.tensor(
                [
                    (x - xy_min) / xy_span,
                    (y - xy_min) / xy_span,
                    (z - z_min) / z_span,
                ],
                dtype=torch.float32,
            )
            size = torch.tensor([dx, dy, dz], dtype=torch.float32) / self.size_norm
            size = size.clamp(0.0, 1.0)
            yaw_vec = torch.tensor([np.sin(yaw), np.cos(yaw)], dtype=torch.float32)
            quest_box = torch.cat([center.clamp(0.0, 1.0), size, yaw_vec], dim=0)
            quest_velocity = torch.as_tensor(raw_velocity[:3], dtype=torch.float32) / self.velocity_norm
            distance = x * x + y * y
            candidates.append((distance, OPENSCENE_AGENT_CLASS_TO_ID[name], quest_box, quest_velocity))

        candidates.sort(key=lambda item: item[0])
        for idx, (_, class_id, quest_box, quest_velocity) in enumerate(candidates[: self.max_agent_instances]):
            labels[idx] = class_id
            boxes[idx] = quest_box
            velocity[idx] = quest_velocity

        return {"labels": labels, "boxes": boxes, "velocity": velocity}

    def _empty_map_gt(self) -> dict[str, torch.Tensor]:
        return {
            "labels": torch.full((self.max_map_instances,), -1, dtype=torch.long),
            "points": torch.zeros((self.max_map_instances, self.P, 2), dtype=torch.float32),
            "valid": torch.tensor(False),
        }

    def _load_occ_gt(self, occ_path: Path) -> torch.Tensor:
        if not occ_path.exists():
            raise FileNotFoundError(f"missing OpenScene occupancy GT: {occ_path}")
        return scatter_openscene_sparse_occ(np.load(occ_path), self.occ_size, self.C_occ)

    def _load_flow_gt(
        self,
        sample_dir: Path,
        metadata: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidates = [
            sample_dir / "flow_gt_final.npy",
            sample_dir / "flow.npy",
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            flow_array = np.load(candidate)
            if flow_array.ndim == 2:
                occ_path = sample_dir / "occ_gt_final.npy"
                if not occ_path.exists():
                    raise FileNotFoundError(f"sparse Flow requires aligned OCC GT: {occ_path}")
                dense_flow, flow_mask = scatter_openscene_sparse_flow(
                    np.load(occ_path),
                    flow_array,
                    self.occ_size,
                )
                return dense_flow, torch.tensor(True), flow_mask

            flow = torch.as_tensor(flow_array, dtype=torch.float32)
            expected = (self.C_flow, *self.occ_size)
            if tuple(flow.shape) == expected:
                return flow, torch.tensor(True), torch.ones(self.occ_size, dtype=torch.bool)
            if tuple(flow.shape) == (*self.occ_size, self.C_flow):
                return (
                    flow.permute(3, 0, 1, 2).contiguous(),
                    torch.tensor(True),
                    torch.ones(self.occ_size, dtype=torch.bool),
                )
            raise ValueError(f"unsupported flow GT shape {tuple(flow.shape)} at {candidate}")
        del metadata
        return (
            torch.zeros((self.C_flow, *self.occ_size), dtype=torch.float32),
            torch.tensor(False),
            torch.zeros(self.occ_size, dtype=torch.bool),
        )

    def _ego_state(self, metadata: dict[str, Any]) -> torch.Tensor:
        can_bus = torch.as_tensor(metadata.get("can_bus", np.zeros(18)), dtype=torch.float32)
        dynamic = torch.as_tensor(metadata.get("ego_dynamic_state", np.zeros(4)), dtype=torch.float32)
        state = torch.cat([can_bus[:5], dynamic[:4]], dim=0)
        if state.numel() < 9:
            state = torch.cat([state, torch.zeros(9 - state.numel(), dtype=torch.float32)])
        return state[:9]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.manifest[idx]
        sample_dir = self.root / item["sample_id"]

        with (sample_dir / "metadata.pkl").open("rb") as f:
            metadata = pickle.load(f)
        multiview = self._load_multiview(sample_dir, metadata)
        flow_gt, flow_valid, flow_mask = self._load_flow_gt(sample_dir, metadata)

        return {
            "images": multiview["images"],
            "intrinsics": multiview["intrinsics"],
            "extrinsics": multiview["extrinsics"],
            "ego_state": self._ego_state(metadata),
            "ego_pose": torch.as_tensor(metadata.get("ego2global", np.eye(4)), dtype=torch.float32),
            "agent_gt": self._load_agent_gt(metadata),
            "map_gt": self._empty_map_gt(),
            "occ_gt": self._load_occ_gt(sample_dir / "occ_gt_final.npy"),
            "occ_valid": torch.tensor(True),
            "flow_gt": flow_gt,
            "flow_valid": flow_valid,
            "flow_mask": flow_mask,
        }


class OpenSceneMetadataDataset(OpenSceneFirstTestDataset):
    """OpenScene Stage1 dataset backed directly by the official metadata pickle."""

    def __init__(
        self,
        metadata_path: str | Path,
        camera_root: str | Path,
        occupancy_root: str | Path,
        max_samples: int = 0,
        image_size: tuple[int, int] = (224, 224),
        occ_size: tuple[int, int, int] = (200, 200, 16),
        C_occ: int = 11,
        C_flow: int = 2,
        C_map: int = 4,
        P: int = 20,
        max_agent_instances: int = 64,
        max_map_instances: int = 32,
        camera_names: Sequence[str] = OPENSCENE_CAMERA_NAMES,
        xy_range: tuple[float, float] = (-50.0, 50.0),
        z_range: tuple[float, float] = (-5.0, 5.0),
        size_norm: tuple[float, float, float] = (20.0, 10.0, 8.0),
        velocity_norm: float = 20.0,
    ) -> None:
        if C_flow != 2:
            raise ValueError(f"OpenScene Flow GT has two channels [vx, vy], got C_flow={C_flow}")
        self.metadata_path = Path(metadata_path)
        self.camera_root = Path(camera_root)
        self.occupancy_root = Path(occupancy_root)
        self.image_size = tuple(image_size)
        self.occ_size = tuple(occ_size)
        self.C_occ = C_occ
        self.C_flow = C_flow
        self.C_map = C_map
        self.P = P
        self.max_agent_instances = max_agent_instances
        self.max_map_instances = max_map_instances
        self.camera_names = tuple(camera_names)
        self.xy_range = tuple(xy_range)
        self.z_range = tuple(z_range)
        self.size_norm = torch.tensor(size_norm, dtype=torch.float32)
        self.velocity_norm = float(velocity_norm)

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
            raise RuntimeError(f"no complete OpenScene frames found in {self.metadata_path}")
        if max_samples > 0 and len(self.infos) < max_samples:
            raise RuntimeError(
                f"requested {max_samples} complete OpenScene frames, found {len(self.infos)}"
            )

    def __len__(self) -> int:
        return len(self.infos)

    def _camera_metadata_path(self, info: dict[str, Any], camera_name: str) -> Path:
        return resolve_openscene_metadata_path(
            info["cams"][camera_name]["data_path"],
            self.camera_root,
            insert_sensor_mini=True,
        )

    def _occupancy_metadata_path(self, info: dict[str, Any], key: str) -> Path:
        return resolve_openscene_metadata_path(info[key], self.occupancy_root)

    def _is_complete_frame(self, info: dict[str, Any]) -> bool:
        cams = info.get("cams") or {}
        if any(camera_name not in cams for camera_name in self.camera_names):
            return False
        for camera_name in self.camera_names:
            camera = cams[camera_name]
            intrinsic = np.asarray(camera.get("cam_intrinsic"))
            rotation = np.asarray(camera.get("sensor2lidar_rotation"))
            translation = np.asarray(camera.get("sensor2lidar_translation"))
            if intrinsic.shape != (3, 3) or rotation.shape != (3, 3) or translation.shape != (3,):
                return False
            if not all(np.isfinite(value).all() for value in (intrinsic, rotation, translation)):
                return False
        if np.asarray(info.get("ego2global")).shape != (4, 4):
            return False
        if np.asarray(info.get("can_bus")).size < 9:
            return False
        if not np.isfinite(np.asarray(info["ego2global"])).all():
            return False
        if not np.isfinite(np.asarray(info["can_bus"])[:9]).all():
            return False
        try:
            camera_paths = [
                self._camera_metadata_path(info, camera_name) for camera_name in self.camera_names
            ]
            occ_path = self._occupancy_metadata_path(info, "occ_gt_final_path")
            flow_path = self._occupancy_metadata_path(info, "flow_gt_final_path")
        except (KeyError, TypeError, ValueError):
            return False
        return all(path.exists() for path in [*camera_paths, occ_path, flow_path])

    def _load_metadata_multiview(self, info: dict[str, Any]) -> dict[str, torch.Tensor]:
        images: list[torch.Tensor] = []
        intrinsics: list[torch.Tensor] = []
        extrinsics: list[torch.Tensor] = []
        for camera_name in self.camera_names:
            camera = info["cams"][camera_name]
            image_path = self._camera_metadata_path(info, camera_name)
            with Image.open(image_path) as source_image:
                source_width, source_height = source_image.size
            images.append(self._load_image(image_path))

            intrinsic = torch.as_tensor(camera["cam_intrinsic"], dtype=torch.float32).clone()
            intrinsic[0, :] *= self.image_size[1] / source_width
            intrinsic[1, :] *= self.image_size[0] / source_height
            intrinsics.append(intrinsic)

            extrinsic = torch.eye(4, dtype=torch.float32)
            extrinsic[:3, :3] = torch.as_tensor(camera["sensor2lidar_rotation"], dtype=torch.float32)
            extrinsic[:3, 3] = torch.as_tensor(camera["sensor2lidar_translation"], dtype=torch.float32)
            extrinsics.append(extrinsic)
        return {
            "images": torch.stack(images),
            "intrinsics": torch.stack(intrinsics),
            "extrinsics": torch.stack(extrinsics),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        info = self.infos[idx]
        multiview = self._load_metadata_multiview(info)
        occ_path = self._occupancy_metadata_path(info, "occ_gt_final_path")
        flow_path = self._occupancy_metadata_path(info, "flow_gt_final_path")
        sparse_occ = np.load(occ_path)
        sparse_flow = np.load(flow_path)
        occ_gt = scatter_openscene_sparse_occ(sparse_occ, self.occ_size, self.C_occ)
        flow_gt, flow_mask = scatter_openscene_sparse_flow(
            sparse_occ,
            sparse_flow,
            self.occ_size,
        )
        return {
            **multiview,
            "ego_state": self._ego_state(info),
            "ego_pose": torch.as_tensor(info["ego2global"], dtype=torch.float32),
            "agent_gt": self._load_agent_gt(info),
            "map_gt": self._empty_map_gt(),
            "occ_gt": occ_gt,
            "occ_valid": torch.tensor(True),
            "flow_gt": flow_gt,
            "flow_valid": torch.tensor(True),
            "flow_mask": flow_mask,
        }
