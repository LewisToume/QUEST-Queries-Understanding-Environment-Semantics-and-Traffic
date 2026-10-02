from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.agent_training import (
    TRAINABLE_AGENT_MODULES,
    compute_stage2_agent_loss,
    configure_agent_training,
    forward_agent_end_to_end,
    load_checkpoint_cpu,
    prepare_navformer_agent_batch,
    save_agent_stage2_checkpoint,
)
from quest.bev_pretraining import gradient_rms, load_bev_pretrain_checkpoint
from quest.dataset import collate_fn
from quest.model import QUESTModel
from quest.utils import load_yaml_config
from scripts.train_stage1_bev import build_dataset, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 QUEST end-to-end Agent training")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--bev-pretrain-checkpoint", type=Path)
    return parser.parse_args()


def train_one_epoch(
    model: QUESTModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_config: dict,
    agent_loss_config: dict,
    epoch: int,
    epochs: int,
) -> None:
    model.eval()
    for name in TRAINABLE_AGENT_MODULES:
        getattr(model, name).train()
    model.geometry_lift.ego_mlp.eval()
    metrics = (
        "total_agent_loss", "proposal_loss", "objectness_loss",
        "positive_objectness_loss", "negative_objectness_loss", "offset_loss",
        "agent_cls_loss", "agent_center_loss", "agent_size_loss",
        "agent_yaw_loss", "agent_velocity_loss",
        "mean_positive_objectness_probability",
        "mean_negative_objectness_probability",
    )
    sums = {name: 0.0 for name in metrics}
    gradients = {name: 0.0 for name in TRAINABLE_AGENT_MODULES}
    vehicle_count = 0
    pedestrian_count = 0
    positive_cells = 0
    collisions = 0
    spread_sum = 0.0
    steps = 0
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    for batch in loader:
        target = prepare_navformer_agent_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = forward_agent_end_to_end(model, batch, device)
        losses = compute_stage2_agent_loss(
            model, predictions, target, train_config, agent_loss_config
        )
        total = losses["total_agent_loss"]
        if not bool(torch.isfinite(total)):
            raise RuntimeError(f"non-finite Stage 2 Agent loss for {batch['sample_token']}")
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            trainable, float(train_config["gradient_clip_norm"])
        )
        for name in gradients:
            gradients[name] += gradient_rms(getattr(model, name))
        optimizer.step()
        steps += 1
        for name in metrics:
            sums[name] += float(losses[name].detach())
        valid = target["valid_mask"]
        vehicle_count += int(((target["labels"] == 0) & valid).sum())
        pedestrian_count += int(((target["labels"] == 1) & valid).sum())
        positive_cells += losses["positive_cell_count"]
        collisions += losses["collision_count"]
        spread_sum += float(predictions["proposal_spatial_std"].mean().detach())
        if steps == 1:
            print(
                f"tensor_shapes objectness={tuple(predictions['proposal_objectness_logits'].shape)} "
                f"offsets={tuple(predictions['proposal_xy_offsets'].shape)} "
                f"class={tuple(predictions['agent_cls_logits'].shape)} "
                f"boxes={tuple(predictions['agent_boxes'].shape)} "
                f"velocity={tuple(predictions['agent_velocity'].shape)}"
            )
    if not steps:
        raise RuntimeError("Stage 2 Agent training received no samples")
    print(
        f"epoch={epoch}/{epochs} steps={steps} "
        f"background_weight={float(train_config['background_weight']):.3f} "
        + " ".join(f"avg_{name}={sums[name] / steps:.6f}" for name in metrics)
        + f" vehicle_target_count={vehicle_count} pedestrian_target_count={pedestrian_count}"
        + f" positive_cell_count={positive_cells} collision_count={collisions}"
        + f" proposal_top100_spatial_spread={spread_sum / steps:.6f}"
    )
    print(
        "gradient_RMS "
        + " ".join(
            f"{label}={gradients[name] / steps:.9e}"
            for name, label in (
                ("geometry_lift", "GeometryAwareBEVLift"),
                ("bev_encoder", "BEVEncoder"),
                ("agent_proposal_head", "AgentProposalHead"),
                ("agent_decoder", "AgentDecoder"),
                ("agent_head", "AgentHead"),
            )
        )
    )


def main() -> int:
    args = parse_args()
    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    stage1_config = load_yaml_config(PROJECT_ROOT / "configs/stage1.yaml")
    config = load_yaml_config(PROJECT_ROOT / "configs/stage2_agent.yaml")
    train_config = config["train"]
    num_samples = args.num_samples if args.num_samples is not None else int(train_config["num_samples"])
    epochs = args.epochs if args.epochs is not None else int(train_config["epochs"])
    if num_samples <= 0 or epochs <= 0:
        raise ValueError("num_samples and epochs must be positive")
    dataset = build_dataset(
        stage1_config, config, int(train_config["start_index"]), num_samples
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QUESTModel(**model_config).to(device)
    stage1_path = resolve_path(
        args.bev_pretrain_checkpoint or config["paths"]["bev_pretrain_checkpoint"]
    )
    stage1_checkpoint = load_checkpoint_cpu(stage1_path)
    stage1_epoch = load_bev_pretrain_checkpoint(model, stage1_checkpoint)
    del stage1_checkpoint
    print(f"loaded_stage1_bev={stage1_path} epoch={stage1_epoch}")

    bev_parameters, agent_parameters = configure_agent_training(model)
    trainable_names = {
        name.split(".", 1)[0] for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if trainable_names != set(TRAINABLE_AGENT_MODULES):
        raise RuntimeError(f"unexpected trainable modules: {sorted(trainable_names)}")
    print(f"trainable_modules={','.join(TRAINABLE_AGENT_MODULES)}")
    print(f"trainable_bev_parameters={sum(p.numel() for p in bev_parameters)}")
    print(f"trainable_agent_parameters={sum(p.numel() for p in agent_parameters)}")
    optimizer = torch.optim.AdamW(
        [
            {"params": bev_parameters, "lr": float(train_config["bev_lr"])},
            {"params": agent_parameters, "lr": float(train_config["agent_lr"])},
        ],
        weight_decay=float(train_config["weight_decay"]),
    )
    print(
        f"bev_lr={optimizer.param_groups[0]['lr']:.9g} "
        f"agent_lr={optimizer.param_groups[1]['lr']:.9g} "
        f"gradient_clip_norm={float(train_config['gradient_clip_norm']):.3f}"
    )
    checkpoint_path = resolve_path(args.checkpoint_path or config["paths"]["checkpoint_path"])
    for epoch in range(1, epochs + 1):
        train_one_epoch(
            model, loader, optimizer, device, train_config, config["agent_loss"], epoch, epochs
        )
        save_agent_stage2_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch,
            {
                **config,
                "effective_num_samples": num_samples,
                "effective_epochs": epochs,
                "source_bev_pretrain_checkpoint": str(stage1_path),
            },
        )
        print(f"checkpoint_saved={checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
