from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

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
    parser = argparse.ArgumentParser(description="Offline soft-label training for QUEST")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    return parser.parse_args()


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def build_targets(
    batch: dict[str, Any], device: torch.device, use_hard_gt: bool = True
) -> dict[str, Any]:
    batch_size = int(batch["images"].shape[0])
    targets: dict[str, Any] = {
        "agent_valid": torch.zeros(batch_size, dtype=torch.bool, device=device),
        "seg_valid": torch.zeros(batch_size, dtype=torch.bool, device=device),
        "depth_valid": torch.zeros(batch_size, dtype=torch.bool, device=device),
        "map_valid": torch.zeros(batch_size, dtype=torch.bool, device=device),
    }
    if use_hard_gt:
        targets["agent_gt"] = _to_device(batch["agent_gt"], device)
        targets["agent_valid"] = batch["agent_valid"].to(device)
    labels = batch.get("soft_labels", {})
    if isinstance(labels, list):
        if len(labels) != 1:
            raise ValueError("offline distillation currently requires batch_size=1")
        labels = labels[0]
    labels = _to_device(labels, device)
    if "seg" in labels:
        seg = labels["seg"]
        targets["seg_gt"] = seg["labels"] if isinstance(seg, Mapping) else seg
        targets["seg_valid"] = torch.tensor([True], device=device)
    if "depth" in labels:
        depth = labels["depth"]
        targets["depth_gt"] = depth["values"] if isinstance(depth, Mapping) else depth
        if isinstance(depth, Mapping) and "valid_mask" in depth:
            targets["depth_mask"] = depth["valid_mask"]
        targets["depth_valid"] = torch.tensor([True], device=device)
    if "agent" in labels:
        agent = labels["agent"]
        required = {"labels", "boxes", "velocity"}
        if not isinstance(agent, Mapping) or not required.issubset(agent):
            raise ValueError("agent soft label requires labels, boxes, and velocity")
        targets["agent_gt"] = dict(agent)
        targets["agent_valid"] = torch.ones(
            batch_size, dtype=torch.bool, device=device
        )
    if "map" in labels:
        vector_map = labels["map"]
        required = {"labels", "points"}
        if not isinstance(vector_map, Mapping) or not required.issubset(vector_map):
            raise ValueError("map soft label requires labels and points")
        targets["map_gt"] = dict(vector_map)
        targets["map_valid"] = torch.tensor([True], device=device)
    return targets


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_config: Mapping[str, Any],
    use_hard_gt: bool,
    epoch: int,
    epochs: int,
) -> tuple[float, float]:
    model.train()
    total_loss_sum = 0.0
    agent_loss_sum = 0.0
    trained_steps = 0
    skipped_steps = 0
    for step, batch in enumerate(loader, start=1):
        targets = build_targets(batch, device, use_hard_gt=use_hard_gt)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(
            batch["images"].to(device),
            batch["intrinsics"].to(device),
            batch["extrinsics"].to(device),
            batch["ego_state"].to(device),
        )
        losses = compute_total_loss(predictions, targets, loss_config)
        total_loss = losses["total_loss"]
        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"non-finite total loss at epoch={epoch} step={step} "
                f"token={batch['sample_token'][0]}"
            )
        if total_loss.item() == 0:
            skipped_steps += 1
            print(
                f"epoch={epoch}/{epochs} step={step}/{len(loader)} "
                f"token={batch['sample_token'][0]} skipped: no labels"
            )
            continue
        total_loss.backward()
        optimizer.step()
        trained_steps += 1
        total_loss_sum += float(total_loss.detach())
        agent_loss_sum += float(losses["agent_loss"].detach())
        print(
            f"epoch={epoch}/{epochs} step={step}/{len(loader)} "
            f"token={batch['sample_token'][0]} "
            f"total={total_loss.item():.6f} agent={losses['agent_loss'].item():.6f}"
        )
    if trained_steps == 0:
        raise RuntimeError(
            f"epoch {epoch} has no trainable labels; "
            "check data/soft_labels and use_hard_gt"
        )
    average_total = total_loss_sum / trained_steps
    average_agent = agent_loss_sum / trained_steps
    print(
        f"epoch={epoch}/{epochs} summary trained={trained_steps} skipped={skipped_steps} "
        f"avg_total_loss={average_total:.6f} avg_agent_loss={average_agent:.6f}"
    )
    return average_total, average_agent


def save_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
        },
        temporary_path,
    )
    temporary_path.replace(path)


def main() -> int:
    args = parse_args()
    model_config = load_yaml_config(PROJECT_ROOT / "configs" / "model.yaml")["model"]
    stage1 = load_yaml_config(PROJECT_ROOT / "configs" / "stage1.yaml")
    stage2 = load_yaml_config(PROJECT_ROOT / "configs" / "stage2_distill.yaml")
    if stage2["interfaces"].get("online_teacher"):
        raise RuntimeError("Stage2 must not instantiate online teachers")
    dataset_config = dict(stage1["dataset"])
    dataset_config.update(stage2.get("dataset", {}))
    for key in ("metadata_path", "camera_root"):
        path = Path(dataset_config[key])
        if not path.is_absolute():
            dataset_config[key] = str(PROJECT_ROOT / path)
    soft_root = Path(stage2["paths"]["soft_labels_dir"])
    if not soft_root.is_absolute():
        soft_root = PROJECT_ROOT / soft_root
    num_samples = (
        args.num_samples
        if args.num_samples is not None
        else int(stage2["distill"]["num_samples"])
    )
    epochs = (
        args.epochs
        if args.epochs is not None
        else int(stage2["distill"]["epochs"])
    )
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    dataset = OpenSceneMetadataDataset(
        max_samples=num_samples,
        soft_labels_root=soft_root,
        **dataset_config,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(stage2["distill"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QUESTModel(**model_config).to(device).train()
    model_path = stage2["distill"].get("model_path")
    if model_path:
        checkpoint_path = Path(model_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(stage2["distill"]["lr"]),
    )
    loss_config = {
        **stage1["loss"],
        "tasks": dict(stage2["tasks"]),
        "task_weights": stage2["loss_weights"],
    }
    use_hard_gt = bool(stage2.get("use_hard_gt", True))
    for epoch in range(1, epochs + 1):
        train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            loss_config,
            use_hard_gt,
            epoch,
            epochs,
        )

    checkpoint_path = args.checkpoint_path or Path(stage2["paths"]["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    save_training_checkpoint(checkpoint_path, model, optimizer, epochs)
    print(f"checkpoint saved: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
