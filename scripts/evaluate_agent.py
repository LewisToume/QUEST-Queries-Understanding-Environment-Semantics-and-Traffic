from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.dataset import collate_fn
from quest.losses import compute_agent_loss
from quest.model import QUESTModel
from quest.openscene_dataset import OpenSceneMetadataDataset
from quest.utils import load_yaml_config


CLASS_NAMES = ("vehicle", "pedestrian", "traffic_cone", "generic_object")
BACKGROUND_CLASS = 4
EXPECTED_AGENT_SHAPES = {
    "agent_cls_logits": (1, 100, 5),
    "agent_boxes": (1, 100, 8),
    "agent_velocity": (1, 100, 3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a QUEST Agent checkpoint on held-out OpenScene frames"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/quest_stage2_agent.pt",
    )
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--distance-threshold", type=float, default=2.0)
    return parser.parse_args()


def _torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model_checkpoint(
    model: torch.nn.Module, checkpoint_path: Path, device: torch.device
) -> int:
    checkpoint = _torch_load(checkpoint_path, device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"checkpoint must contain checkpoint['model_state_dict']: {checkpoint_path}"
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    return int(checkpoint.get("epoch", 0))


def normalized_center_to_metric(center: torch.Tensor) -> torch.Tensor:
    if center.shape[-1] != 3:
        raise ValueError(f"center must end in xyz, got {tuple(center.shape)}")
    metric = center.clone()
    metric[..., 0:2] = metric[..., 0:2] * 100.0 - 50.0
    metric[..., 2] = metric[..., 2] * 10.0 - 5.0
    return metric


def filter_predictions(
    cls_logits: torch.Tensor,
    boxes: torch.Tensor,
    confidence_threshold: float,
) -> dict[str, torch.Tensor]:
    if cls_logits.ndim != 2 or cls_logits.shape[-1] != 5:
        raise ValueError(f"agent_cls_logits must be [N,5], got {tuple(cls_logits.shape)}")
    if boxes.ndim != 2 or boxes.shape != (cls_logits.shape[0], 8):
        raise ValueError(f"agent_boxes must be [N,8], got {tuple(boxes.shape)}")
    probabilities = cls_logits.softmax(dim=-1)
    confidence, labels = probabilities.max(dim=-1)
    keep = (labels != BACKGROUND_CLASS) & (
        confidence >= float(confidence_threshold)
    )
    return {
        "labels": labels[keep],
        "scores": confidence[keep],
        "centers_m": normalized_center_to_metric(boxes[keep, :3]),
    }


def match_agents(
    prediction_centers: torch.Tensor,
    prediction_labels: torch.Tensor,
    gt_centers: torch.Tensor,
    gt_labels: torch.Tensor,
    distance_threshold: float = 2.0,
) -> list[tuple[int, int, float]]:
    if prediction_centers.numel() == 0 or gt_centers.numel() == 0:
        return []
    matches = []
    for class_id in range(len(CLASS_NAMES)):
        class_prediction_indices = torch.nonzero(
            prediction_labels == class_id, as_tuple=False
        ).flatten()
        class_gt_indices = torch.nonzero(gt_labels == class_id, as_tuple=False).flatten()
        if not class_prediction_indices.numel() or not class_gt_indices.numel():
            continue
        distances = torch.cdist(
            prediction_centers[class_prediction_indices].float(),
            gt_centers[class_gt_indices].float(),
            p=2,
        )
        prediction_indices, gt_indices = linear_sum_assignment(
            distances.detach().cpu().numpy()
        )
        for prediction_index, gt_index in zip(prediction_indices, gt_indices):
            global_prediction_index = int(class_prediction_indices[prediction_index])
            global_gt_index = int(class_gt_indices[gt_index])
            distance = float(distances[prediction_index, gt_index])
            if (
                prediction_labels[global_prediction_index]
                == gt_labels[global_gt_index]
                and distance <= float(distance_threshold)
            ):
                matches.append(
                    (global_prediction_index, global_gt_index, distance)
                )
    matches.sort(key=lambda item: item[0])
    return matches


def validation_subset(dataset: Dataset, start: int, count: int) -> Subset:
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}")
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    end = start + count
    if end > len(dataset):
        raise IndexError(
            f"validation complete-frame range [{start}, {end}) exceeds {len(dataset)}"
        )
    return Subset(dataset, range(start, end))


@dataclass
class AgentMetrics:
    evaluated_samples: int = 0
    total_gt: int = 0
    total_predictions: int = 0
    matched: int = 0
    class_correct: int = 0
    center_error_sum: float = 0.0
    agent_loss_sum: float = 0.0
    class_gt: list[int] = field(default_factory=lambda: [0] * len(CLASS_NAMES))
    class_predictions: list[int] = field(
        default_factory=lambda: [0] * len(CLASS_NAMES)
    )
    class_matched: list[int] = field(
        default_factory=lambda: [0] * len(CLASS_NAMES)
    )

    def update(
        self,
        prediction_labels: torch.Tensor,
        gt_labels: torch.Tensor,
        matches: list[tuple[int, int, float]],
        agent_loss: float,
    ) -> None:
        self.evaluated_samples += 1
        self.total_gt += int(gt_labels.numel())
        self.total_predictions += int(prediction_labels.numel())
        self.agent_loss_sum += float(agent_loss)
        for class_id in range(len(CLASS_NAMES)):
            self.class_gt[class_id] += int((gt_labels == class_id).sum())
            self.class_predictions[class_id] += int(
                (prediction_labels == class_id).sum()
            )
        for prediction_index, gt_index, distance in matches:
            prediction_class = int(prediction_labels[prediction_index])
            gt_class = int(gt_labels[gt_index])
            if prediction_class != gt_class:
                continue
            self.matched += 1
            self.class_correct += 1
            self.center_error_sum += distance
            self.class_matched[gt_class] += 1

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    def summary(self) -> dict[str, Any]:
        mean_center_error = (
            self.center_error_sum / self.matched if self.matched else math.nan
        )
        mean_agent_loss = (
            self.agent_loss_sum / self.evaluated_samples
            if self.evaluated_samples
            else math.nan
        )
        per_class = {}
        for class_id, name in enumerate(CLASS_NAMES):
            per_class[name] = {
                "gt": self.class_gt[class_id],
                "predictions": self.class_predictions[class_id],
                "matched": self.class_matched[class_id],
                "precision": self._ratio(
                    self.class_matched[class_id], self.class_predictions[class_id]
                ),
                "recall": self._ratio(
                    self.class_matched[class_id], self.class_gt[class_id]
                ),
            }
        return {
            "evaluated_samples": self.evaluated_samples,
            "total_gt": self.total_gt,
            "total_predictions": self.total_predictions,
            "matched": self.matched,
            "precision": self._ratio(self.matched, self.total_predictions),
            "recall": self._ratio(self.matched, self.total_gt),
            "mean_center_error_m": mean_center_error,
            "class_accuracy_on_matched": self._ratio(
                self.class_correct, self.matched
            ),
            "mean_agent_loss_on_hard_gt": mean_agent_loss,
            "per_class": per_class,
        }


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    stage1 = load_yaml_config(PROJECT_ROOT / "configs/stage1.yaml")
    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    local_backbone = _resolve_path(model_config["local_backbone_dir"])
    if not local_backbone.is_dir():
        raise FileNotFoundError(f"local DINOv2 weights not found: {local_backbone}")
    model_config["local_backbone_dir"] = str(local_backbone)

    dataset_config = dict(stage1["dataset"])
    dataset_config["metadata_path"] = str(_resolve_path(dataset_config["metadata_path"]))
    dataset_config["camera_root"] = str(_resolve_path(dataset_config["camera_root"]))
    dataset_config["max_agent_instances"] = 64
    complete_frame_count = args.start + args.count
    dataset = OpenSceneMetadataDataset(
        max_samples=complete_frame_count,
        **dataset_config,
    )
    subset = validation_subset(dataset, args.start, args.count)
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QUESTModel(**model_config).to(device)
    checkpoint_path = _resolve_path(args.checkpoint)
    epoch = load_model_checkpoint(model, checkpoint_path, device)
    model.eval()
    print(
        f"checkpoint={checkpoint_path} epoch={epoch} device={device} "
        f"complete_frames=[{args.start},{args.start + args.count})"
    )

    metrics = AgentMetrics()
    agent_loss_config = stage1["loss"]["agent"]
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            outputs = model(
                batch["images"].to(device),
                batch["intrinsics"].to(device),
                batch["extrinsics"].to(device),
                batch["ego_state"].to(device),
            )
            for key, expected_shape in EXPECTED_AGENT_SHAPES.items():
                if tuple(outputs[key].shape) != expected_shape:
                    raise ValueError(
                        f"{key} shape {tuple(outputs[key].shape)} != {expected_shape}"
                    )

            predictions = filter_predictions(
                outputs["agent_cls_logits"][0],
                outputs["agent_boxes"][0],
                args.confidence_threshold,
            )
            valid_gt = batch["agent_gt"]["labels"][0] >= 0
            gt_labels = batch["agent_gt"]["labels"][0][valid_gt].to(device)
            gt_centers = normalized_center_to_metric(
                batch["agent_gt"]["boxes"][0][valid_gt, :3].to(device)
            )
            matches = match_agents(
                predictions["centers_m"],
                predictions["labels"],
                gt_centers,
                gt_labels,
                args.distance_threshold,
            )
            hard_gt = {
                key: value.to(device) for key, value in batch["agent_gt"].items()
            }
            agent_loss = compute_agent_loss(
                outputs["agent_cls_logits"],
                outputs["agent_boxes"],
                outputs["agent_velocity"],
                hard_gt,
                agent_loss_config,
            )["agent_loss"]
            metrics.update(
                predictions["labels"].cpu(),
                gt_labels.cpu(),
                matches,
                float(agent_loss),
            )
            print(
                f"\rprogress {index}/{args.count} token={batch['sample_token'][0]}",
                end="",
                flush=True,
            )
    print()

    summary = metrics.summary()
    print("Agent evaluation summary")
    for key in (
        "evaluated_samples",
        "total_gt",
        "total_predictions",
        "matched",
        "precision",
        "recall",
        "mean_center_error_m",
        "class_accuracy_on_matched",
        "mean_agent_loss_on_hard_gt",
    ):
        value = summary[key]
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    print("class_accuracy_on_matched_note: strict class-aware matches only")
    print("per_class:")
    for name in CLASS_NAMES:
        values = summary["per_class"][name]
        print(
            f"  {name}: gt={values['gt']} predictions={values['predictions']} "
            f"matched={values['matched']} precision={values['precision']:.6f} "
            f"recall={values['recall']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
