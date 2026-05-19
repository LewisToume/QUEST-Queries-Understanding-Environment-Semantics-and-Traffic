from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class NuPlanDummyDataset(Dataset):
    """
    Dummy single-frame dataset for the QUEST research prototype.

    This is still placeholder data. The structure is intentionally aligned with
    future nuPlan annotations and teacher outputs so the rest of the training
    stack can already follow expert-style interfaces.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        image_size: Tuple[int, int] = (224, 224),
        seg_size: Tuple[int, int] = (64, 64),
        C_seg: int = 6,
        max_agent_instances: int = 16,
        C_agent: int = 10,
        D_box: int = 8,
        max_map_instances: int = 12,
        C_map: int = 4,
        P: int = 20,
        occ_size: Tuple[int, int, int] = (64, 64, 16),
        C_occ: int = 4,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.seg_size = seg_size
        self.C_seg = C_seg
        self.max_agent_instances = max_agent_instances
        self.C_agent = C_agent
        self.D_box = D_box
        self.max_map_instances = max_map_instances
        self.C_map = C_map
        self.P = P
        self.occ_size = occ_size
        self.C_occ = C_occ

        np.random.seed(seed)
        torch.manual_seed(seed)

    def __len__(self) -> int:
        return self.num_samples

    def _build_agent_gt(self) -> Dict[str, torch.Tensor]:
        """
        Structured dummy agent GT.

        This does not try to reproduce real nuPlan annotations. The goal is to
        generate a small, interpretable set of object instances so QUEST can
        test object layout learning and query matching, rather than memorizing a
        random table of boxes.
        """

        labels = torch.full((self.max_agent_instances,), -1, dtype=torch.long)
        boxes = torch.zeros(self.max_agent_instances, self.D_box, dtype=torch.float32)

        num_valid = min(
            self.max_agent_instances,
            int(torch.randint(1, min(6, self.max_agent_instances) + 1, (1,)).item()),
        )

        size_templates = torch.tensor(
            [
                [0.10, 0.05, 0.05],
                [0.12, 0.06, 0.06],
                [0.16, 0.07, 0.07],
                [0.18, 0.08, 0.08],
            ],
            dtype=torch.float32,
        )
        size_jitter = torch.tensor([0.025, 0.018, 0.015], dtype=torch.float32)
        centers: list[torch.Tensor] = []
        min_center_distance = 0.14

        for agent_idx in range(num_valid):
            label = int(torch.randint(0, self.C_agent, (1,)).item())
            labels[agent_idx] = label

            template = size_templates[label % size_templates.shape[0]]

            center_xy = None
            for _ in range(20):
                candidate = torch.empty(2, dtype=torch.float32)
                candidate[0] = torch.empty(1).uniform_(0.08, 0.92).item()
                candidate[1] = torch.empty(1).uniform_(0.08, 0.92).item()
                if not centers:
                    center_xy = candidate
                    break
                distances = [torch.norm(candidate - existing).item() for existing in centers]
                if min(distances) >= min_center_distance:
                    center_xy = candidate
                    break
            if center_xy is None:
                center_xy = torch.empty(2, dtype=torch.float32)
                center_xy[0] = torch.empty(1).uniform_(0.08, 0.92).item()
                center_xy[1] = torch.empty(1).uniform_(0.08, 0.92).item()

            centers.append(center_xy)

            center_z = float(torch.empty(1).uniform_(0.02, 0.12).item())
            size = template + (torch.rand(3, dtype=torch.float32) - 0.5) * 2 * size_jitter
            size = torch.clamp(size, min=torch.tensor([0.05, 0.03, 0.03]), max=torch.tensor([0.28, 0.14, 0.12]))

            yaw = float(torch.empty(1).uniform_(-np.pi, np.pi).item())

            boxes[agent_idx, 0] = center_xy[0]
            boxes[agent_idx, 1] = center_xy[1]
            boxes[agent_idx, 2] = center_z
            boxes[agent_idx, 3:6] = size
            boxes[agent_idx, 6] = np.sin(yaw)
            boxes[agent_idx, 7] = np.cos(yaw)

            if self.D_box > 8:
                boxes[agent_idx, 8:] = torch.empty(self.D_box - 8, dtype=torch.float32).uniform_(0.0, 0.1)

        return {
            "labels": labels,
            "boxes": boxes,
        }

    def _build_map_gt(self) -> Dict[str, torch.Tensor]:
        """
        Structured dummy map elements.

        This is structured dummy map GT. The goal is to generate road-like
        polylines with simple topology templates so QUEST can test whether
        query-based map prediction learns road layout and polyline geometry,
        rather than fitting random point sets. It does not try to reproduce
        real nuPlan maps; it only tries to look more like roads, junctions,
        branches, and lane elements.
        """

        labels = torch.full((self.max_map_instances,), -1, dtype=torch.long)
        points = torch.zeros(
            self.max_map_instances,
            self.P,
            2,
            dtype=torch.float32,
        )

        def class_id(desired: int) -> int:
            return min(max(0, self.C_map - 1), desired)

        def clip_polyline(polyline: torch.Tensor) -> torch.Tensor:
            clipped = polyline.clone()
            clipped[:, 0] = clipped[:, 0].clamp(0.02, 0.98)
            clipped[:, 1] = clipped[:, 1].clamp(0.02, 0.98)
            return clipped

        def maybe_swap(polyline: torch.Tensor, swap_axes: bool) -> torch.Tensor:
            if not swap_axes:
                return polyline
            return polyline[:, [1, 0]]

        def horizontal_curve(
            center_y: float,
            slope: float,
            curvature: float,
            phase: float,
            x_start: float = 0.05,
            x_end: float = 0.95,
            y_offset: float = 0.0,
            swap_axes: bool = False,
        ) -> torch.Tensor:
            x = torch.linspace(x_start, x_end, steps=self.P, dtype=torch.float32)
            normalized_x = (x - x_start) / max(1e-4, (x_end - x_start))
            y = center_y + y_offset + slope * (normalized_x - 0.5)
            y = y + curvature * torch.sin(np.pi * normalized_x + phase)
            polyline = torch.stack([x, y], dim=-1)
            return clip_polyline(maybe_swap(polyline, swap_axes))

        def branch_curve(
            start: tuple[float, float],
            end: tuple[float, float],
            bend: float = 0.0,
        ) -> torch.Tensor:
            t = torch.linspace(0.0, 1.0, steps=self.P, dtype=torch.float32)
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            if abs(end[0] - start[0]) >= abs(end[1] - start[1]):
                y = y + bend * torch.sin(np.pi * t)
            else:
                x = x + bend * torch.sin(np.pi * t)
            polyline = torch.stack([x, y], dim=-1)
            return clip_polyline(polyline)

        def apply_jitter(polyline: torch.Tensor, scale: float = 0.004) -> torch.Tensor:
            noise = torch.randn_like(polyline) * scale
            noise[0] = 0.0
            noise[-1] = 0.0
            return clip_polyline(polyline + noise)

        template_name = [
            "straight",
            "curve",
            "t_junction",
            "cross_junction",
            "y_split",
        ][int(torch.randint(0, 5, (1,)).item())]

        elements: list[tuple[int, torch.Tensor]] = []
        lane_spacing = float(torch.empty(1).uniform_(0.06, 0.10).item())
        swap_axes = bool(torch.randint(0, 2, (1,)).item())
        center = float(torch.empty(1).uniform_(0.38, 0.62).item())
        slope = float(torch.empty(1).uniform_(-0.08, 0.08).item())
        phase = float(torch.empty(1).uniform_(0.0, 2 * np.pi).item())
        curvature = float(torch.empty(1).uniform_(0.015, 0.06).item())

        if template_name == "straight":
            offsets = [-1.5, -0.5, 0.5, 1.5]
            for idx, offset in enumerate(offsets):
                desired_class = 0 if idx in (0, len(offsets) - 1) else 1
                polyline = horizontal_curve(
                    center_y=center,
                    slope=slope,
                    curvature=0.01,
                    phase=phase,
                    y_offset=offset * lane_spacing,
                    swap_axes=swap_axes,
                )
                elements.append((class_id(desired_class), apply_jitter(polyline, scale=0.002)))

        elif template_name == "curve":
            offsets = [-1.5, -0.5, 0.5, 1.5]
            for idx, offset in enumerate(offsets):
                desired_class = 0 if idx in (0, len(offsets) - 1) else 1
                polyline = horizontal_curve(
                    center_y=center,
                    slope=slope,
                    curvature=curvature,
                    phase=phase,
                    y_offset=offset * lane_spacing,
                    swap_axes=swap_axes,
                )
                elements.append((class_id(desired_class), apply_jitter(polyline, scale=0.003)))

        elif template_name == "t_junction":
            trunk_center = center
            branch_y = float(torch.empty(1).uniform_(0.42, 0.62).item())
            for offset, desired_class in [(-lane_spacing, 0), (0.0, 1), (lane_spacing, 0)]:
                polyline = branch_curve(
                    (trunk_center + offset, 0.05),
                    (trunk_center + offset, 0.95),
                    bend=float(torch.empty(1).uniform_(-0.02, 0.02).item()),
                )
                elements.append((class_id(desired_class), apply_jitter(polyline, scale=0.002)))
            branch_side = -1 if bool(torch.randint(0, 2, (1,)).item()) else 1
            branch_end_x = 0.08 if branch_side < 0 else 0.92
            for offset, desired_class in [(-0.5 * lane_spacing, 2), (0.5 * lane_spacing, 0)]:
                polyline = branch_curve(
                    (trunk_center, branch_y + offset),
                    (branch_end_x, branch_y + offset),
                    bend=float(torch.empty(1).uniform_(-0.03, 0.03).item()),
                )
                elements.append((class_id(desired_class), apply_jitter(polyline, scale=0.002)))

        elif template_name == "cross_junction":
            cross_center_x = center
            cross_center_y = float(torch.empty(1).uniform_(0.38, 0.62).item())
            for offset, desired_class in [(-lane_spacing, 0), (0.0, 1), (lane_spacing, 0)]:
                vertical = branch_curve(
                    (cross_center_x + offset, 0.05),
                    (cross_center_x + offset, 0.95),
                    bend=float(torch.empty(1).uniform_(-0.015, 0.015).item()),
                )
                horizontal = branch_curve(
                    (0.05, cross_center_y + offset),
                    (0.95, cross_center_y + offset),
                    bend=float(torch.empty(1).uniform_(-0.015, 0.015).item()),
                )
                elements.append((class_id(desired_class), apply_jitter(vertical, scale=0.002)))
                elements.append((class_id(desired_class if offset != 0.0 else 3), apply_jitter(horizontal, scale=0.002)))

        else:
            split_x = float(torch.empty(1).uniform_(0.42, 0.58).item())
            split_y = float(torch.empty(1).uniform_(0.40, 0.55).item())
            stem = branch_curve(
                (split_x, 0.05),
                (split_x, split_y),
                bend=float(torch.empty(1).uniform_(-0.02, 0.02).item()),
            )
            left_branch = branch_curve(
                (split_x, split_y),
                (0.15, 0.90),
                bend=float(torch.empty(1).uniform_(-0.04, 0.04).item()),
            )
            right_branch = branch_curve(
                (split_x, split_y),
                (0.85, 0.90),
                bend=float(torch.empty(1).uniform_(-0.04, 0.04).item()),
            )
            left_boundary = branch_curve((split_x - lane_spacing, 0.05), (0.08, 0.86), bend=0.02)
            right_boundary = branch_curve((split_x + lane_spacing, 0.05), (0.92, 0.86), bend=-0.02)
            elements.extend(
                [
                    (class_id(1), apply_jitter(stem, scale=0.002)),
                    (class_id(2), apply_jitter(left_branch, scale=0.003)),
                    (class_id(2), apply_jitter(right_branch, scale=0.003)),
                    (class_id(0), apply_jitter(left_boundary, scale=0.002)),
                    (class_id(0), apply_jitter(right_boundary, scale=0.002)),
                ]
            )

        num_valid = min(self.max_map_instances, len(elements))
        for idx, (elem_class, elem_points) in enumerate(elements[:num_valid]):
            labels[idx] = elem_class
            points[idx] = elem_points

        return {
            "labels": labels,
            "points": points,
        }

    def _build_occ_gt(self) -> torch.Tensor:
        """
        Structured dummy occupancy.

        This does not try to simulate real nuPlan occupancy. The goal is to
        generate a learnable spatial signal with contiguous occupied regions so
        QUEST can verify whether the OCC branch learns spatial structure, rather
        than memorizing uniform voxel noise.
        """

        size_x, size_y, size_z = self.occ_size
        if self.C_occ == 1:
            occ_gt = torch.zeros(self.occ_size, dtype=torch.float32)
        else:
            occ_gt = torch.zeros(self.occ_size, dtype=torch.long)

        num_blocks = int(torch.randint(1, 5, (1,)).item())
        min_sizes = (
            max(2, size_x // 10),
            max(2, size_y // 10),
            max(2, size_z // 6),
        )
        max_sizes = (
            max(min_sizes[0], size_x // 3),
            max(min_sizes[1], size_y // 3),
            max(min_sizes[2], max(2, size_z // 2)),
        )

        for _ in range(num_blocks):
            block_size_x = int(torch.randint(min_sizes[0], max_sizes[0] + 1, (1,)).item())
            block_size_y = int(torch.randint(min_sizes[1], max_sizes[1] + 1, (1,)).item())
            block_size_z = int(torch.randint(min_sizes[2], max_sizes[2] + 1, (1,)).item())

            center_x = int(torch.randint(0, size_x, (1,)).item())
            center_y = int(torch.randint(0, size_y, (1,)).item())
            center_z = int(torch.randint(0, size_z, (1,)).item())

            x0 = max(0, center_x - block_size_x // 2)
            y0 = max(0, center_y - block_size_y // 2)
            z0 = max(0, center_z - block_size_z // 2)
            x1 = min(size_x, x0 + block_size_x)
            y1 = min(size_y, y0 + block_size_y)
            z1 = min(size_z, z0 + block_size_z)

            if self.C_occ == 1:
                fill_value = 1.0
            else:
                fill_value = int(torch.randint(1, self.C_occ, (1,)).item())

            occ_gt[x0:x1, y0:y1, z0:z1] = fill_value

        return occ_gt

    def _build_seg_gt(self) -> torch.Tensor:
        """
        Structured dummy segmentation.

        This does not try to mimic real nuPlan semantics. The goal is to create
        spatially coherent segmentation regions so QUEST can test whether the
        segmentation head learns layout and boundaries, instead of fitting pure
        random mosaic noise.
        """

        height, width = self.seg_size
        seg_gt = torch.zeros(self.seg_size, dtype=torch.long)

        if self.C_seg <= 1:
            return seg_gt

        y_coords = torch.arange(height, dtype=torch.long).unsqueeze(1).expand(height, width)
        x_coords = torch.arange(width, dtype=torch.long).unsqueeze(0).expand(height, width)

        road_center = int(torch.randint(height // 3, max(height // 3 + 1, (2 * height) // 3), (1,)).item())
        road_half_width = int(torch.randint(max(2, height // 10), max(max(3, height // 10 + 1), height // 5 + 1), (1,)).item())
        road_mask = (y_coords >= max(0, road_center - road_half_width)) & (
            y_coords < min(height, road_center + road_half_width)
        )
        seg_gt[road_mask] = 1

        if self.C_seg > 2:
            num_regions = int(torch.randint(1, 4, (1,)).item())
            for region_idx in range(num_regions):
                class_id = 2 + (region_idx % max(1, self.C_seg - 2))
                rect_h = int(torch.randint(max(3, height // 10), max(max(4, height // 10 + 1), height // 4 + 1), (1,)).item())
                rect_w = int(torch.randint(max(3, width // 10), max(max(4, width // 10 + 1), width // 4 + 1), (1,)).item())
                top = int(torch.randint(0, max(1, height - rect_h + 1), (1,)).item())
                left = int(torch.randint(0, max(1, width - rect_w + 1), (1,)).item())
                seg_gt[top : top + rect_h, left : left + rect_w] = class_id

        if self.C_seg > 3:
            horizon = int(torch.randint(height // 8, max(height // 8 + 1, height // 3), (1,)).item())
            seg_gt[:horizon] = min(self.C_seg - 1, 2)

        return seg_gt

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        del idx

        image = torch.rand(3, *self.image_size, dtype=torch.float32)

        return {
            "image": image,
            "seg_gt": self._build_seg_gt(),
            "agent_gt": self._build_agent_gt(),
            "map_gt": self._build_map_gt(),
            "occ_gt": self._build_occ_gt(),
        }


def _recursive_stack(items: list[Any]) -> Any:
    first = items[0]
    if isinstance(first, dict):
        return {key: _recursive_stack([item[key] for item in items]) for key in first}
    if torch.is_tensor(first):
        return torch.stack(items, dim=0)
    raise TypeError(f"Unsupported batch item type: {type(first)!r}")


def collate_fn(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    return _recursive_stack(batch)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print("=" * 60)
    print("QUEST dummy dataset smoke test")
    print("=" * 60)

    dataset = NuPlanDummyDataset(num_samples=4, seed=42)
    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key:<8}: shape={tuple(sub_value.shape)} dtype={sub_value.dtype}")
        else:
            print(f"{key:<10}: shape={tuple(value.shape)} dtype={value.dtype}")

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    batch = next(iter(dataloader))
    print("-" * 60)
    print(f"image      : shape={tuple(batch['image'].shape)} dtype={batch['image'].dtype}")
    print(f"seg_gt     : shape={tuple(batch['seg_gt'].shape)} dtype={batch['seg_gt'].dtype}")
    print(f"agent_gt.labels: shape={tuple(batch['agent_gt']['labels'].shape)}")
    print(f"agent_gt.boxes : shape={tuple(batch['agent_gt']['boxes'].shape)}")
    print(f"map_gt.labels  : shape={tuple(batch['map_gt']['labels'].shape)}")
    print(f"map_gt.points  : shape={tuple(batch['map_gt']['points'].shape)}")
    print(f"occ_gt     : shape={tuple(batch['occ_gt'].shape)} dtype={batch['occ_gt'].dtype}")
    print("=" * 60)
