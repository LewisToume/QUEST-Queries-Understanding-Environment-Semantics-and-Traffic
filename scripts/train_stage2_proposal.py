from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.bev_pretraining import (
    extract_canonical_agent_batch,
    gradient_rms,
    load_bev_pretrain_checkpoint,
)
from quest.dataset import collate_fn
from quest.model import QUESTModel
from quest.proposal_pretraining import (
    compute_proposal_pretraining_loss,
    configure_proposal_pretraining,
    rasterize_proposal_targets,
    save_proposal_pretrain_checkpoint,
)
from quest.utils import load_yaml_config
from scripts.train_stage1_bev import build_dataset, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 Agent Proposal-only pretraining")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--bev-pretrain-checkpoint", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    return parser.parse_args()


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def train_one_epoch(
    model: QUESTModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_config: dict,
    epoch: int,
    epochs: int,
) -> None:
    model.eval()
    model.agent_proposal_head.train()
    metric_names = (
        "proposal_loss",
        "objectness_loss",
        "positive_objectness_loss",
        "negative_objectness_loss",
        "offset_loss",
        "mean_positive_objectness_probability",
        "mean_negative_objectness_probability",
    )
    totals = {name: 0.0 for name in metric_names}
    positive_cell_count = 0
    collision_count = 0
    gradient_sum = 0.0
    steps = 0
    for batch in loader:
        agent_target = extract_canonical_agent_batch(batch, device)
        if agent_target is None:
            raise RuntimeError(f"missing Navformer Agent target for {batch['sample_token']}")
        with torch.no_grad():
            encoded = model.encode_image(
                batch["images"].to(device),
                batch["intrinsics"].to(device),
                batch["extrinsics"].to(device),
                batch["ego_state"].to(device),
                use_ego_state=False,
            )
        objectness, offsets = model.agent_proposal_head(encoded["bev_tokens"])
        targets = rasterize_proposal_targets(
            agent_target,
            model.geometry_lift.x_range,
            model.geometry_lift.y_range,
            model.bev_h,
            model.bev_w,
        )
        losses = compute_proposal_pretraining_loss(
            objectness,
            offsets,
            targets,
            negative_loss_weight=float(train_config["negative_loss_weight"]),
            lambda_objectness=float(train_config["lambda_objectness"]),
            lambda_offset=float(train_config["lambda_offset"]),
        )
        if not bool(torch.isfinite(losses["proposal_loss"])):
            raise RuntimeError(f"non-finite proposal loss for {batch['sample_token']}")
        optimizer.zero_grad(set_to_none=True)
        losses["proposal_loss"].backward()
        gradient_sum += gradient_rms(model.agent_proposal_head)
        optimizer.step()
        steps += 1
        for name in metric_names:
            totals[name] += float(losses[name].detach())
        positive_cell_count += targets["positive_cell_count"]
        collision_count += targets["collision_count"]
        if steps == 1:
            print(
                f"tensor_shapes bev_tokens={tuple(encoded['bev_tokens'].shape)} "
                f"objectness={tuple(objectness.shape)} offsets={tuple(offsets.shape)}"
            )
    if not steps:
        raise RuntimeError("Stage 2 Proposal training received no samples")
    print(
        f"epoch={epoch}/{epochs} steps={steps} "
        + " ".join(f"avg_{name}={totals[name] / steps:.6f}" for name in metric_names)
        + f" positive_cell_count={positive_cell_count}"
        + f" collision_count={collision_count}"
        + f" AgentProposalHead_gradient_RMS={gradient_sum / steps:.9e}"
    )


def main() -> int:
    args = parse_args()
    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    stage1_config = load_yaml_config(PROJECT_ROOT / "configs/stage1.yaml")
    proposal_config = load_yaml_config(PROJECT_ROOT / "configs/stage2_proposal.yaml")
    train_config = proposal_config["train"]
    num_samples = (
        args.num_samples if args.num_samples is not None else int(train_config["num_samples"])
    )
    epochs = args.epochs if args.epochs is not None else int(train_config["epochs"])
    if num_samples <= 0 or epochs <= 0:
        raise ValueError("num_samples and epochs must be positive")
    dataset = build_dataset(
        stage1_config,
        proposal_config,
        int(train_config["start_index"]),
        num_samples,
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
    bev_checkpoint_path = resolve_path(
        args.bev_pretrain_checkpoint or proposal_config["paths"]["bev_pretrain_checkpoint"]
    )
    bev_checkpoint = load_checkpoint(bev_checkpoint_path)
    bev_epoch = load_bev_pretrain_checkpoint(model, bev_checkpoint)
    del bev_checkpoint
    print(f"loaded_stage1_bev={bev_checkpoint_path} epoch={bev_epoch}")

    trainable = configure_proposal_pretraining(model)
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if {name.split(".", 1)[0] for name in trainable_names} != {"agent_proposal_head"}:
        raise RuntimeError(f"unexpected trainable modules: {trainable_names}")
    print("trainable_modules=agent_proposal_head")
    print(f"trainable_parameter_count={sum(parameter.numel() for parameter in trainable)}")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    checkpoint_path = resolve_path(
        args.checkpoint_path or proposal_config["paths"]["checkpoint_path"]
    )
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, loader, optimizer, device, train_config, epoch, epochs)
        save_proposal_pretrain_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch,
            {
                **proposal_config,
                "effective_num_samples": num_samples,
                "effective_epochs": epochs,
                "source_bev_pretrain_checkpoint": str(bev_checkpoint_path),
            },
        )
        print(f"checkpoint_saved={checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
