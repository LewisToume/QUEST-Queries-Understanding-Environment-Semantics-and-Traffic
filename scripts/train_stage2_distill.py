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
from quest.model import QUESTModel, QUEST_ARCHITECTURE_VERSION, load_quest_v3_checkpoint
from quest.openscene_dataset import OpenSceneMetadataDataset
from quest.teacher_adapters import (
    NAVFORMER_CLASS_SUPPORT,
    merge_canonical_agent_targets,
    validate_canonical_agent,
)
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


def initial_trained_class_support_mask(supervision_source: str) -> torch.Tensor:
    if supervision_source == "teacher_only":
        return NAVFORMER_CLASS_SUPPORT.clone()
    if supervision_source in {"hard_gt_only", "hybrid"}:
        return torch.ones(4, dtype=torch.bool)
    raise ValueError(f"unknown supervision_source: {supervision_source}")


def build_targets(
    batch: dict[str, Any],
    device: torch.device,
    supervision_source: str = "teacher_only",
) -> dict[str, Any]:
    batch_size = int(batch["images"].shape[0])
    targets: dict[str, Any] = {
        "agent_valid": torch.zeros(batch_size, dtype=torch.bool, device=device),
        "seg_valid": torch.zeros(batch_size, dtype=torch.bool, device=device),
        "depth_valid": torch.zeros(batch_size, dtype=torch.bool, device=device),
        "map_valid": torch.zeros(batch_size, dtype=torch.bool, device=device),
    }
    if supervision_source not in {"teacher_only", "hard_gt_only", "hybrid"}:
        raise ValueError(f"unknown supervision_source: {supervision_source}")
    hard_agent = _to_device(batch["agent_gt"], device)
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
    teacher_agent = labels.get("agent") if isinstance(labels, Mapping) else None
    if teacher_agent is not None and not isinstance(teacher_agent, Mapping):
        raise ValueError("Agent soft label must use the canonical mapping")
    if supervision_source == "hard_gt_only":
        targets["agent_gt"] = hard_agent
        targets["agent_valid"] = batch["agent_valid"].to(device)
    elif supervision_source == "teacher_only" and teacher_agent is not None:
        targets["agent_gt"] = dict(teacher_agent)
        targets["agent_valid"] = teacher_agent["valid_mask"].bool().any(dim=1)
    elif supervision_source == "hybrid" and teacher_agent is not None:
        merged = [
            merge_canonical_agent_targets(
                {key: value[index] for key, value in hard_agent.items()},
                {key: value[index] for key, value in teacher_agent.items()},
                max_instances=int(hard_agent["labels"].shape[1]),
            )
            for index in range(batch_size)
        ]
        targets["agent_gt"] = {
            key: torch.stack([item[key] for item in merged]) for key in merged[0]
        }
        targets["agent_valid"] = targets["agent_gt"]["valid_mask"].any(dim=1)
    elif supervision_source == "hybrid":
        targets["agent_gt"] = hard_agent
        targets["agent_valid"] = batch["agent_valid"].to(device)
    if "agent_gt" in targets:
        for index in range(batch_size):
            validate_canonical_agent(
                {key: value[index] for key, value in targets["agent_gt"].items()}
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
    supervision_source: str,
    proposal_warmup_epochs: int,
    epoch: int,
    epochs: int,
    trained_class_support_mask: torch.Tensor | None = None,
) -> tuple[float, float]:
    model.train()
    total_loss_sum = 0.0
    agent_loss_sum = 0.0
    component_keys = (
        "proposal_objectness_loss",
        "proposal_offset_loss",
        "agent_cls_loss",
        "agent_center_loss",
        "agent_size_loss",
        "agent_yaw_loss",
        "agent_velocity_loss",
    )
    component_sums = {key: 0.0 for key in component_keys}
    valid_agent_sum = 0
    vehicle_sum = 0
    pedestrian_sum = 0
    proposal_spatial_std_sum = 0.0
    bev_visible_ratio_sum = 0.0
    trained_steps = 0
    skipped_steps = 0
    for step, batch in enumerate(loader, start=1):
        targets = build_targets(batch, device, supervision_source=supervision_source)
        if trained_class_support_mask is not None and "agent_gt" in targets:
            available = targets["agent_valid"].bool()
            if available.any():
                observed_support = targets["agent_gt"]["class_support_mask"][available]
                trained_class_support_mask |= observed_support.any(dim=0).detach().cpu()
        optimizer.zero_grad(set_to_none=True)
        predictions = model(
            batch["images"].to(device),
            batch["intrinsics"].to(device),
            batch["extrinsics"].to(device),
            batch["ego_state"].to(device),
        )
        decoder_enabled = epoch > proposal_warmup_epochs
        losses = compute_total_loss(
            predictions,
            targets,
            loss_config,
            agent_decoder_enabled=decoder_enabled,
        )
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
        for key in component_keys:
            component_sums[key] += float(losses[key].detach())
        agent_target = targets.get("agent_gt", {})
        valid_labels = agent_target.get(
            "labels", torch.empty(0, device=device, dtype=torch.long)
        )
        valid_mask = agent_target.get(
            "valid_mask", torch.zeros_like(valid_labels, dtype=torch.bool)
        )
        vehicle_count = int(((valid_labels == 0) & valid_mask).sum())
        pedestrian_count = int(((valid_labels == 1) & valid_mask).sum())
        valid_agent_sum += int(valid_mask.sum())
        vehicle_sum += vehicle_count
        pedestrian_sum += pedestrian_count
        proposal_spatial_std_sum += float(predictions["proposal_spatial_std"].mean())
        bev_visible_ratio_sum += float(predictions["bev_visible_ratio"].mean())
        print(
            f"epoch={epoch}/{epochs} step={step}/{len(loader)} "
            f"token={batch['sample_token'][0]} "
            f"decoder_enabled={decoder_enabled} total={total_loss.item():.6f} "
            f"proposal_obj={losses['proposal_objectness_loss'].item():.6f} "
            f"proposal_offset={losses['proposal_offset_loss'].item():.6f} "
            f"cls={losses['agent_cls_loss'].item():.6f} "
            f"center={losses['agent_center_loss'].item():.6f} "
            f"size={losses['agent_size_loss'].item():.6f} "
            f"yaw={losses['agent_yaw_loss'].item():.6f} "
            f"velocity={losses['agent_velocity_loss'].item():.6f} "
            f"agent={losses['agent_loss'].item():.6f} "
            f"valid_agents={int(valid_mask.sum())} vehicle={vehicle_count} "
            f"pedestrian={pedestrian_count} "
            f"proposal_spatial_std={predictions['proposal_spatial_std'].mean().item():.6f} "
            f"bev_visible_ratio={predictions['bev_visible_ratio'].mean().item():.6f}"
        )
    if trained_steps == 0:
        raise RuntimeError(
            f"epoch {epoch} has no trainable labels; "
            "check data/soft_labels_navformer and supervision_source"
        )
    average_total = total_loss_sum / trained_steps
    average_agent = agent_loss_sum / trained_steps
    component_averages = {
        key: value / trained_steps for key, value in component_sums.items()
    }
    print(
        f"epoch={epoch}/{epochs} summary trained={trained_steps} skipped={skipped_steps} "
        f"avg_total_loss={average_total:.6f} avg_agent_loss={average_agent:.6f} "
        + " ".join(
            f"avg_{key}={value:.6f}" for key, value in component_averages.items()
        )
        + f" avg_valid_agents={valid_agent_sum / trained_steps:.3f}"
        + f" vehicle_labels={vehicle_sum} pedestrian_labels={pedestrian_sum}"
        + f" avg_proposal_spatial_std={proposal_spatial_std_sum / trained_steps:.6f}"
        + f" avg_bev_visible_ratio={bev_visible_ratio_sum / trained_steps:.6f}"
    )
    return average_total, average_agent


def save_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    trained_class_support_mask: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "architecture_version": QUEST_ARCHITECTURE_VERSION,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "trained_class_support_mask": trained_class_support_mask.detach().cpu().bool(),
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
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        load_quest_v3_checkpoint(model, checkpoint)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(stage2["distill"]["lr"]),
    )
    loss_config = {
        **stage1["loss"],
        "tasks": dict(stage2["tasks"]),
        "task_weights": stage2["loss_weights"],
    }
    supervision_source = str(stage2.get("supervision_source", "teacher_only"))
    proposal_warmup_epochs = int(stage2["distill"].get("proposal_warmup_epochs", 1))
    trained_class_support_mask = initial_trained_class_support_mask(
        supervision_source
    )
    for epoch in range(1, epochs + 1):
        train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            loss_config,
            supervision_source,
            proposal_warmup_epochs,
            epoch,
            epochs,
            trained_class_support_mask,
        )

    checkpoint_path = args.checkpoint_path or Path(stage2["paths"]["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    save_training_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epochs,
        trained_class_support_mask,
    )
    print(f"checkpoint saved: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
