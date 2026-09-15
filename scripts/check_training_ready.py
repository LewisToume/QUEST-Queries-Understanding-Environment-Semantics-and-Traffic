from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.dataset import collate_fn
from quest.losses import compute_total_loss
from quest.model import QUESTModel
from quest.openscene_dataset import OpenSceneMetadataDataset
from quest.utils import load_yaml_config


EXPECTED_INPUTS = {
    "images": (1, 8, 3, 224, 224),
    "intrinsics": (1, 8, 3, 3),
    "extrinsics": (1, 8, 4, 4),
    "ego_state": (1, 9),
}
EXPECTED_OUTPUTS = {
    "seg_logits": (1, 8, 6, 64, 64),
    "depth": (1, 8, 1, 64, 64),
    "agent_cls_logits": (1, 100, 5),
    "agent_boxes": (1, 100, 8),
    "agent_velocity": (1, 100, 3),
    "map_cls_logits": (1, 50, 5),
    "map_points": (1, 50, 20, 2),
    "bev_features": (1, 256, 32, 32),
}


def resolve_dataset(config: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)
    for key in ("metadata_path", "camera_root"):
        path = Path(config[key])
        if not path.is_absolute():
            config[key] = str(PROJECT_ROOT / path)
    return config


def main() -> int:
    blockers: list[str] = []
    model_config = load_yaml_config(PROJECT_ROOT / "configs" / "model.yaml")["model"]
    stage = load_yaml_config(PROJECT_ROOT / "configs" / "stage1.yaml")
    dataset = OpenSceneMetadataDataset(
        max_samples=1, **resolve_dataset(stage["dataset"])
    )
    batch = next(
        iter(DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_fn))
    )
    for key, expected in EXPECTED_INPUTS.items():
        tensor = batch[key]
        if tuple(tensor.shape) != expected:
            blockers.append(f"{key}: {tuple(tensor.shape)} != {expected}")
        if not torch.isfinite(tensor).all():
            blockers.append(f"{key}: non-finite input")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QUESTModel(**model_config).to(device).train()
    predictions = model(
        batch["images"].to(device),
        batch["intrinsics"].to(device),
        batch["extrinsics"].to(device),
        batch["ego_state"].to(device),
    )
    if set(predictions) != set(EXPECTED_OUTPUTS):
        blockers.append(f"output keys: {sorted(predictions)}")
    for key, expected in EXPECTED_OUTPUTS.items():
        tensor = predictions[key]
        if tuple(tensor.shape) != expected:
            blockers.append(f"{key}: {tuple(tensor.shape)} != {expected}")
        if not torch.isfinite(tensor).all():
            blockers.append(f"{key}: non-finite output")
    if not bool((predictions["depth"] > 0).all()):
        blockers.append("depth is not strictly positive")

    targets = {
        "agent_gt": {
            key: value.to(device) for key, value in batch["agent_gt"].items()
        },
        "agent_valid": batch["agent_valid"].to(device),
        "seg_valid": batch["seg_valid"].to(device),
        "depth_valid": batch["depth_valid"].to(device),
        "map_valid": batch["map_valid"].to(device),
    }
    losses = compute_total_loss(predictions, targets, stage["loss"])
    total = losses["total_loss"]
    if not math.isfinite(total.item()) or total.item() <= 0:
        blockers.append(f"invalid Stage1 loss: {total.item()}")
    total.backward()
    for name, module in (("BEVEncoder", model.bev_encoder), ("AgentHead", model.agent_head)):
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not any(
            torch.isfinite(gradient).all() and bool((gradient != 0).any())
            for gradient in gradients
        ):
            blockers.append(f"{name}: missing finite nonzero gradient")
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        blockers.append("DINOv2 backbone is not frozen")

    print("input shapes", {key: tuple(batch[key].shape) for key in EXPECTED_INPUTS})
    print("output shapes", {key: tuple(value.shape) for key, value in predictions.items()})
    print(f"Stage1 loss {total.item():.6f}")
    print("blockers", blockers if blockers else "NONE")
    print(f"QUEST_8VIEW_TRAINING_READY = {'NO' if blockers else 'YES'}")
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
