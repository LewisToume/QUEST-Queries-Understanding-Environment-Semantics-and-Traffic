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
        C_flow: int = 3,
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
        anns = metadata.get("anns") or {}
        gt_boxes = np.asarray(anns.get("gt_boxes", []), dtype=np.float32)
        gt_names = np.asarray(anns.get("gt_names", []))
        gt_velocity = np.asarray(anns.get("gt_velocity_3d", []), dtype=np.float32)

        if gt_boxes.size == 0 or gt_names.size == 0:
            return {"labels": labels, "boxes": boxes, "velocity": velocity}
        if gt_velocity.size == 0:
            gt_velocity = np.zeros((len(gt_boxes), 3), dtype=np.float32)

        candidates: list[tuple[float, int, torch.Tensor]] = []
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
        sparse_occ = np.load(occ_path)
        dense_occ = torch.zeros(self.occ_size, dtype=torch.long)
        if sparse_occ.size == 0:
            return dense_occ

        flat_indices = torch.from_numpy(sparse_occ[:, 0].astype(np.int64))
        class_ids = torch.from_numpy(sparse_occ[:, 1].astype(np.int64)).clamp(0, self.C_occ - 1)
        valid = (flat_indices >= 0) & (flat_indices < dense_occ.numel())
        dense_occ.view(-1)[flat_indices[valid]] = class_ids[valid]
        return dense_occ

    def _load_flow_gt(self, sample_dir: Path, metadata: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = [
            sample_dir / "flow_gt_final.npy",
            sample_dir / "flow.npy",
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            flow = torch.as_tensor(np.load(candidate), dtype=torch.float32)
            expected = (self.C_flow, *self.occ_size)
            if tuple(flow.shape) == expected:
                return flow, torch.tensor(True)
            if tuple(flow.shape) == (*self.occ_size, self.C_flow):
                return flow.permute(3, 0, 1, 2).contiguous(), torch.tensor(True)
            raise ValueError(f"unsupported flow GT shape {tuple(flow.shape)} at {candidate}")
        del metadata
        return torch.zeros((self.C_flow, *self.occ_size), dtype=torch.float32), torch.tensor(False)

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
        flow_gt, flow_valid = self._load_flow_gt(sample_dir, metadata)

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
        }
