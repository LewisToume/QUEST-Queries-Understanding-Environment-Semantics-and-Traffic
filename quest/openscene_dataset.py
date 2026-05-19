from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


OPENSCENE_AGENT_CLASS_TO_ID = {
    "vehicle": 0,
    "pedestrian": 1,
    "traffic_cone": 2,
    "generic_object": 3,
}


class OpenSceneFirstTestDataset(Dataset):
    """
    Minimal OpenScene adapter for the extracted 10-sample first-test folder.

    It intentionally exposes only the two real supervised branches we can use
    now: agent boxes from metadata anns, and semantic occupancy from occ_final.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: tuple[int, int] = (224, 224),
        occ_size: tuple[int, int, int] = (200, 200, 16),
        C_occ: int = 11,
        max_agent_instances: int = 64,
        xy_range: tuple[float, float] = (-50.0, 50.0),
        z_range: tuple[float, float] = (-5.0, 5.0),
        size_norm: tuple[float, float, float] = (20.0, 10.0, 8.0),
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.occ_size = occ_size
        self.C_occ = C_occ
        self.max_agent_instances = max_agent_instances
        self.xy_range = xy_range
        self.z_range = z_range
        self.size_norm = torch.tensor(size_norm, dtype=torch.float32)

        manifest_path = self.root / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_image(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std

    def _load_agent_gt(self, metadata: dict[str, Any]) -> dict[str, torch.Tensor]:
        labels = torch.full((self.max_agent_instances,), -1, dtype=torch.long)
        boxes = torch.zeros((self.max_agent_instances, 8), dtype=torch.float32)
        anns = metadata.get("anns") or {}
        gt_boxes = np.asarray(anns.get("gt_boxes", []), dtype=np.float32)
        gt_names = np.asarray(anns.get("gt_names", []))

        if gt_boxes.size == 0 or gt_names.size == 0:
            return {"labels": labels, "boxes": boxes}

        candidates: list[tuple[float, int, torch.Tensor]] = []
        xy_min, xy_max = self.xy_range
        z_min, z_max = self.z_range
        xy_span = xy_max - xy_min
        z_span = z_max - z_min

        for box, raw_name in zip(gt_boxes, gt_names):
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
            distance = x * x + y * y
            candidates.append((distance, OPENSCENE_AGENT_CLASS_TO_ID[name], quest_box))

        candidates.sort(key=lambda item: item[0])
        for idx, (_, class_id, quest_box) in enumerate(candidates[: self.max_agent_instances]):
            labels[idx] = class_id
            boxes[idx] = quest_box

        return {"labels": labels, "boxes": boxes}

    def _load_occ_gt(self, occ_path: Path) -> torch.Tensor:
        sparse_occ = np.load(occ_path)
        dense_occ = torch.zeros(self.occ_size, dtype=torch.long)
        if sparse_occ.size == 0:
            return dense_occ

        flat_indices = torch.from_numpy(sparse_occ[:, 0].astype(np.int64))
        class_ids = torch.from_numpy(sparse_occ[:, 1].astype(np.int64)).clamp(0, self.C_occ - 1)
        valid = (flat_indices >= 0) & (flat_indices < dense_occ.numel())
        dense_occ.view(-1)[flat_indices[valid]] = class_ids[valid]
        return dense_occ

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.manifest[idx]
        sample_dir = self.root / item["sample_id"]

        with (sample_dir / "metadata.pkl").open("rb") as f:
            metadata = pickle.load(f)

        return {
            "image": self._load_image(sample_dir / "cam_f0.jpg"),
            "agent_gt": self._load_agent_gt(metadata),
            "occ_gt": self._load_occ_gt(sample_dir / "occ_gt_final.npy"),
        }
