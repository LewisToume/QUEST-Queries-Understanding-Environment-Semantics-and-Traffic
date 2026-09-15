from __future__ import annotations

import argparse
import contextlib
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QUEST Stage1 on OpenScene metadata")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def resolve_dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    for key in ("metadata_path", "camera_root"):
        path = Path(result[key])
        if not path.is_absolute():
            result[key] = str(PROJECT_ROOT / path)
    return result


def move_gts(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "agent_gt": {
            key: value.to(device) for key, value in batch["agent_gt"].items()
        },
        "agent_valid": batch["agent_valid"].to(device),
        "seg_valid": batch["seg_valid"].to(device),
        "depth_valid": batch["depth_valid"].to(device),
        "map_valid": batch["map_valid"].to(device),
    }


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def main() -> int:
    args = parse_args()
    torch.manual_seed(42)
    model_config = load_yaml_config(PROJECT_ROOT / "configs" / "model.yaml")["model"]
    stage_config = load_yaml_config(PROJECT_ROOT / "configs" / "stage1.yaml")
    train_config = stage_config["train"]
    num_samples = args.num_samples or int(train_config["num_samples"])
    epochs = args.epochs or int(train_config["num_epochs"])
    dataset = OpenSceneMetadataDataset(
        max_samples=num_samples,
        **resolve_dataset_config(stage_config["dataset"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    configured_device = str(train_config.get("device", "auto"))
    device = torch.device(
        "cuda" if configured_device == "auto" and torch.cuda.is_available()
        else "cpu" if configured_device == "auto"
        else configured_device
    )
    model = QUESTModel(**model_config).to(device).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    first_step_checked = False
    final_losses: dict[str, torch.Tensor] = {}
    for epoch in range(epochs):
        for step, batch in enumerate(loader, start=1):
            images = batch["images"].to(device)
            intrinsics = batch["intrinsics"].to(device)
            extrinsics = batch["extrinsics"].to(device)
            ego_state = batch["ego_state"].to(device)
            gts = move_gts(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                predictions = model(images, intrinsics, extrinsics, ego_state)
            predictions = {key: value.float() for key, value in predictions.items()}
            final_losses = compute_total_loss(predictions, gts, stage_config["loss"])
            total_loss = final_losses["total_loss"]
            if not math.isfinite(float(total_loss.item())) or total_loss.item() <= 0:
                raise RuntimeError(f"Stage1 total loss must be finite and positive: {total_loss.item()}")
            total_loss.backward()
            if not first_step_checked:
                bev_gradients = [
                    parameter.grad
                    for parameter in model.bev_encoder.parameters()
                    if parameter.grad is not None
                ]
                agent_gradients = [
                    parameter.grad
                    for parameter in model.agent_head.parameters()
                    if parameter.grad is not None
                ]
                for name, gradients in (
                    ("BEVEncoder", bev_gradients),
                    ("AgentHead", agent_gradients),
                ):
                    if not gradients or not any(
                        torch.isfinite(gradient).all() and bool((gradient != 0).any())
                        for gradient in gradients
                    ):
                        raise RuntimeError(f"{name} did not receive finite nonzero gradients")
                first_step_checked = True
            optimizer.step()
            print(
                f"epoch={epoch + 1} step={step}/{len(loader)} "
                f"token={batch['sample_token'][0]} "
                f"agent={final_losses['agent_loss'].item():.6f} "
                f"total={total_loss.item():.6f}"
            )
    print(f"Agent loss: {final_losses['agent_loss'].item():.6f}")
    print(f"Total loss: {final_losses['total_loss'].item():.6f}")
    print("Backward: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
