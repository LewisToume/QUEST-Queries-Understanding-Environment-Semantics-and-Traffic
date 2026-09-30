from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import torch


AGENT_CLASS_COUNT = 4
NAVFORMER_CLASS_SUPPORT = torch.tensor([True, True, False, False])
NAVFORMER_TO_QUEST = {0: 0, 2: 1}
CANONICAL_TARGET_FIELDS = {
    "agent": (
        "labels", "boxes_metric", "velocity_mps", "scores",
        "class_support_mask", "valid_mask",
    ),
    "map": ("labels", "points_metric", "scores", "class_support_mask", "valid_mask"),
    "seg": ("labels_or_probabilities", "valid_mask"),
    "depth": ("depth_m", "confidence", "valid_mask"),
}


class TeacherAdapter(ABC):
    """Convert one teacher's native output into a task-level canonical target."""

    task: str

    @abstractmethod
    def adapt(self, raw_output: Mapping[str, Any], **context: Any) -> dict[str, Any]:
        raise NotImplementedError


def empty_canonical_agent(max_instances: int = 64) -> dict[str, torch.Tensor]:
    return {
        "labels": torch.full((max_instances,), -1, dtype=torch.long),
        "boxes_metric": torch.zeros((max_instances, 7), dtype=torch.float32),
        "velocity_mps": torch.zeros((max_instances, 3), dtype=torch.float32),
        "scores": torch.zeros((max_instances,), dtype=torch.float32),
        "class_support_mask": torch.zeros(AGENT_CLASS_COUNT, dtype=torch.bool),
        "valid_mask": torch.zeros((max_instances,), dtype=torch.bool),
    }


def validate_canonical_agent(
    target: Mapping[str, Any], max_instances: int | None = None
) -> None:
    required = {
        "labels",
        "boxes_metric",
        "velocity_mps",
        "scores",
        "class_support_mask",
        "valid_mask",
    }
    if not required.issubset(target):
        raise ValueError(f"canonical Agent target requires {sorted(required)}")
    if not all(torch.is_tensor(target[key]) for key in required):
        raise TypeError("canonical Agent fields must be tensors")
    count = int(target["labels"].shape[0])
    if max_instances is not None and count != max_instances:
        raise ValueError(f"canonical Agent count must be {max_instances}, got {count}")
    expected = {
        "labels": (count,),
        "boxes_metric": (count, 7),
        "velocity_mps": (count, 3),
        "scores": (count,),
        "class_support_mask": (AGENT_CLASS_COUNT,),
        "valid_mask": (count,),
    }
    for key, shape in expected.items():
        if tuple(target[key].shape) != shape:
            raise ValueError(f"canonical Agent {key} must be {shape}")
    valid = target["valid_mask"].bool()
    if valid.any():
        labels = target["labels"][valid]
        if bool(((labels < 0) | (labels >= AGENT_CLASS_COUNT)).any()):
            raise ValueError("canonical Agent labels are outside QUEST taxonomy")
        for key in ("boxes_metric", "velocity_mps", "scores"):
            if not torch.isfinite(target[key][valid]).all():
                raise ValueError(f"canonical Agent {key} contains NaN or Inf")
    if "class_probs" in target and tuple(target["class_probs"].shape) != (
        count,
        AGENT_CLASS_COUNT,
    ):
        raise ValueError("canonical Agent class_probs must be [N,4]")
    if "ignore_boxes" in target:
        ignore_boxes = target["ignore_boxes"]
        if not torch.is_tensor(ignore_boxes) or ignore_boxes.ndim != 2 or ignore_boxes.shape[1] != 7:
            raise ValueError("canonical Agent ignore_boxes must be [M,7]")


def validate_canonical_map(target: Mapping[str, Any]) -> None:
    required = {"labels", "points_metric", "scores", "class_support_mask", "valid_mask"}
    if not required.issubset(target):
        raise ValueError(f"canonical Map target requires {sorted(required)}")


def validate_canonical_seg(target: Mapping[str, Any]) -> None:
    if "valid_mask" not in target or not ({"labels", "probabilities"} & set(target)):
        raise ValueError("canonical Seg target requires labels/probabilities and valid_mask")


def validate_canonical_depth(target: Mapping[str, Any]) -> None:
    required = {"depth_m", "confidence", "valid_mask"}
    if not required.issubset(target):
        raise ValueError(f"canonical Depth target requires {sorted(required)}")


def merge_canonical_agent_targets(
    hard_target: Mapping[str, torch.Tensor],
    teacher_target: Mapping[str, torch.Tensor],
    max_instances: int = 64,
    duplicate_distance_m: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Merge explicit hybrid supervision with hard GT taking precedence."""

    validate_canonical_agent(hard_target)
    validate_canonical_agent(teacher_target)
    output = empty_canonical_agent(max_instances)
    output["class_support_mask"] = (
        hard_target["class_support_mask"].bool()
        | teacher_target["class_support_mask"].bool()
    )
    records: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    hard_valid = hard_target["valid_mask"].bool()
    for index in torch.nonzero(hard_valid, as_tuple=False).flatten().tolist():
        records.append(
            (
                int(hard_target["labels"][index]),
                hard_target["boxes_metric"][index],
                hard_target["velocity_mps"][index],
                hard_target["scores"][index],
            )
        )
    teacher_valid = teacher_target["valid_mask"].bool()
    for index in torch.nonzero(teacher_valid, as_tuple=False).flatten().tolist():
        label = int(teacher_target["labels"][index])
        center = teacher_target["boxes_metric"][index, :3]
        duplicate = any(
            existing_label == label
            and float(torch.linalg.vector_norm(existing_box[:3] - center))
            <= duplicate_distance_m
            for existing_label, existing_box, _, _ in records
        )
        if not duplicate:
            records.append(
                (
                    label,
                    teacher_target["boxes_metric"][index],
                    teacher_target["velocity_mps"][index],
                    teacher_target["scores"][index],
                )
            )
    for output_index, (label, box, velocity, score) in enumerate(records[:max_instances]):
        output["labels"][output_index] = label
        output["boxes_metric"][output_index] = box
        output["velocity_mps"][output_index] = velocity
        output["scores"][output_index] = score
        output["valid_mask"][output_index] = True
    validate_canonical_agent(output, max_instances)
    return output


def box_observability_mask(
    boxes_metric: torch.Tensor,
    lidar2img: torch.Tensor,
    image_shapes: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Return boxes with a center or corner visible in at least one current camera."""

    boxes = torch.as_tensor(boxes_metric, dtype=torch.float32)
    matrices = torch.as_tensor(lidar2img, dtype=torch.float32)
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError("boxes_metric must be [N,>=7]")
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError("lidar2img must be [N_camera,4,4]")
    if len(image_shapes) != matrices.shape[0]:
        raise ValueError("image_shapes must match lidar2img cameras")
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.bool)

    signs = torch.tensor(
        [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ],
        dtype=boxes.dtype,
        device=boxes.device,
    )
    local = signs.unsqueeze(0) * boxes[:, None, 3:6] * 0.5
    cosine = torch.cos(boxes[:, 6])
    sine = torch.sin(boxes[:, 6])
    rotated_x = local[..., 0] * cosine[:, None] - local[..., 1] * sine[:, None]
    rotated_y = local[..., 0] * sine[:, None] + local[..., 1] * cosine[:, None]
    corners = torch.stack((rotated_x, rotated_y, local[..., 2]), dim=-1)
    corners = corners + boxes[:, None, :3]
    points = torch.cat((boxes[:, None, :3], corners), dim=1)
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    projected = torch.einsum("cij,npj->cnpi", matrices.to(boxes.device), homogeneous)
    depth = projected[..., 2]
    safe_depth = depth.clamp_min(torch.finfo(projected.dtype).eps)
    pixel_x = projected[..., 0] / safe_depth
    pixel_y = projected[..., 1] / safe_depth
    visible = depth > 0
    for camera_index, shape in enumerate(image_shapes):
        height, width = int(shape[0]), int(shape[1])
        visible[camera_index] &= (
            (pixel_x[camera_index] >= 0)
            & (pixel_x[camera_index] < width)
            & (pixel_y[camera_index] >= 0)
            & (pixel_y[camera_index] < height)
            & torch.isfinite(pixel_x[camera_index])
            & torch.isfinite(pixel_y[camera_index])
        )
    return visible.any(dim=0).any(dim=-1).cpu()


class NavformerAgentAdapter(TeacherAdapter):
    task = "agent"

    def __init__(
        self,
        max_instances: int = 64,
        score_threshold: float = 0.25,
        max_distance_m: float = 50.0,
        z_range: tuple[float, float] = (-5.0, 5.0),
    ) -> None:
        self.max_instances = int(max_instances)
        self.score_threshold = float(score_threshold)
        self.max_distance_m = float(max_distance_m)
        self.z_range = tuple(float(value) for value in z_range)
        if self.max_instances <= 0:
            raise ValueError("max_instances must be positive")
        if self.score_threshold < 0 or self.max_distance_m <= 0:
            raise ValueError("invalid Navformer Agent filter configuration")

    def adapt(self, raw_output: Mapping[str, Any], **context: Any) -> dict[str, torch.Tensor]:
        boxes_value = raw_output.get("boxes_3d", raw_output.get("track_bbox_results"))
        if hasattr(boxes_value, "tensor"):
            boxes_value = boxes_value.tensor
        scores_value = raw_output.get("scores_3d", raw_output.get("track_scores"))
        labels_value = raw_output.get("labels_3d", raw_output.get("track_labels"))
        if boxes_value is None or scores_value is None or labels_value is None:
            raise ValueError("Navformer output requires boxes_3d, scores_3d, and labels_3d")
        boxes = torch.as_tensor(boxes_value).detach().cpu().float()
        scores = torch.as_tensor(scores_value).detach().cpu().float().reshape(-1)
        labels = torch.as_tensor(labels_value).detach().cpu().long().reshape(-1)
        if boxes.ndim != 2 or boxes.shape[1] < 9:
            raise ValueError(f"Navformer boxes must be [N,>=9], got {tuple(boxes.shape)}")
        if scores.shape != (boxes.shape[0],) or labels.shape != (boxes.shape[0],):
            raise ValueError("Navformer boxes, scores, and labels must have equal length")

        output = empty_canonical_agent(self.max_instances)
        output["class_support_mask"] = NAVFORMER_CLASS_SUPPORT.clone()
        if not boxes.shape[0]:
            return output
        z_min, z_max = self.z_range
        finite = torch.isfinite(boxes[:, :9]).all(dim=1) & torch.isfinite(scores)
        distance = torch.linalg.vector_norm(boxes[:, :2], dim=1)
        allowed = (labels == 0) | (labels == 2)
        keep = (
            finite
            & allowed
            & (scores >= self.score_threshold)
            & (distance <= self.max_distance_m)
            & (boxes[:, 2] >= z_min)
            & (boxes[:, 2] <= z_max)
        )
        lidar2img = context.get("lidar2img")
        image_shapes = context.get("image_shapes")
        if lidar2img is None or image_shapes is None:
            raise ValueError("Navformer Agent adapter requires current-frame lidar2img and image_shapes")
        observable = box_observability_mask(boxes[:, :7], lidar2img, image_shapes)
        keep &= observable
        indices = torch.nonzero(keep, as_tuple=False).flatten()
        indices = indices[torch.argsort(scores[indices], descending=True)][: self.max_instances]
        count = int(indices.numel())
        if count:
            output["labels"][:count] = torch.tensor(
                [NAVFORMER_TO_QUEST[int(label)] for label in labels[indices]],
                dtype=torch.long,
            )
            output["boxes_metric"][:count] = boxes[indices, :7]
            output["velocity_mps"][:count, :2] = boxes[indices, 7:9]
            output["scores"][:count] = scores[indices]
            output["valid_mask"][:count] = True
        validate_canonical_agent(output, self.max_instances)
        return output
