from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.bev_pretraining import (
    BEVAuxiliaryHead,
    compute_bev_pretraining_loss,
    configure_bev_pretraining,
    extract_canonical_agent_batch,
    gradient_rms,
    rasterize_agent_centers,
    save_bev_pretrain_checkpoint,
)
from quest.dataset import collate_fn
from quest.model import QUESTModel
from quest.openscene_dataset import OpenSceneMetadataDataset
from quest.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 QUEST BEV pretraining")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--checkpoint-path", type=Path)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def build_dataset(
    stage1_config: Mapping[str, Any],
    bev_config: Mapping[str, Any],
    start_index: int,
    num_samples: int,
) -> Subset:
    if start_index < 0 or num_samples <= 0:
        raise ValueError("start_index must be non-negative and num_samples positive")
    dataset_config = dict(stage1_config["dataset"])
    dataset_config["metadata_path"] = str(resolve_path(dataset_config["metadata_path"]))
    dataset_config["camera_root"] = str(resolve_path(dataset_config["camera_root"]))
    dataset_config["max_agent_instances"] = 64
    dataset = OpenSceneMetadataDataset(
        max_samples=start_index + num_samples,
        soft_labels_root=resolve_path(bev_config["paths"]["soft_labels_dir"]),
        **dataset_config,
    )
    end_index = start_index + num_samples
    if len(dataset) < end_index:
        raise IndexError(
            f"requested complete-frame range [{start_index},{end_index}) exceeds {len(dataset)}"
        )
    return Subset(dataset, range(start_index, end_index))


def train_one_epoch(
    model: QUESTModel,
    auxiliary_head: BEVAuxiliaryHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_class: float,
    epoch: int,
    epochs: int,
) -> dict[str, float]:
    model.eval()
    model.geometry_lift.train()
    model.bev_encoder.train()
    auxiliary_head.train()
    metric_names = (
        "total_loss",
        "foreground_loss",
        "positive_loss",
        "negative_loss",
        "class_loss",
        "mean_positive_foreground_probability",
        "mean_negative_foreground_probability",
    )
    totals = {name: 0.0 for name in metric_names}
    totals.update(
        {
            "vehicle_target_count": 0.0,
            "pedestrian_target_count": 0.0,
            "occupied_bev_cells": 0.0,
            "target_collision_count": 0.0,
            "geometry_gradient_rms": 0.0,
            "encoder_gradient_rms": 0.0,
            "auxiliary_gradient_rms": 0.0,
        }
    )
    trained_steps = 0
    skipped_steps = 0
    for step, batch in enumerate(loader, start=1):
        agent_target = extract_canonical_agent_batch(batch, device)
        if agent_target is None:
            skipped_steps += 1
            print(
                f"epoch={epoch}/{epochs} step={step}/{len(loader)} "
                f"token={batch['sample_token'][0]} skipped=no_teacher_target"
            )
            continue
        optimizer.zero_grad(set_to_none=True)
        encoded = model.encode_image(
            batch["images"].to(device),
            batch["intrinsics"].to(device),
            batch["extrinsics"].to(device),
            batch["ego_state"].to(device),
            use_ego_state=False,
        )
        predictions = auxiliary_head(encoded["bev_features"])
        targets = rasterize_agent_centers(
            agent_target,
            model.geometry_lift.x_range,
            model.geometry_lift.y_range,
            model.geometry_lift.bev_h,
            model.geometry_lift.bev_w,
        )
        losses = compute_bev_pretraining_loss(
            predictions, targets, lambda_class=lambda_class
        )
        if not torch.isfinite(losses["total_loss"]):
            raise RuntimeError(
                f"non-finite BEV loss at token={batch['sample_token'][0]}"
            )
        losses["total_loss"].backward()
        geometry_rms = gradient_rms(model.geometry_lift)
        encoder_rms = gradient_rms(model.bev_encoder)
        auxiliary_rms = gradient_rms(auxiliary_head)
        optimizer.step()
        trained_steps += 1
        for name in metric_names:
            totals[name] += float(losses[name].detach())
        for name in (
            "vehicle_target_count",
            "pedestrian_target_count",
            "occupied_bev_cells",
            "target_collision_count",
        ):
            totals[name] += float(targets[name])
        totals["geometry_gradient_rms"] += geometry_rms
        totals["encoder_gradient_rms"] += encoder_rms
        totals["auxiliary_gradient_rms"] += auxiliary_rms
        if trained_steps == 1:
            print(
                "tensor_shapes "
                f"bev_features={tuple(encoded['bev_features'].shape)} "
                f"foreground_logits={tuple(predictions['foreground_logits'].shape)} "
                f"class_logits={tuple(predictions['class_logits'].shape)}"
            )
    if trained_steps == 0:
        raise RuntimeError("no Stage 1 steps had Navformer canonical Agent targets")
    averages = {
        name: totals[name] / trained_steps
        for name in metric_names
    }
    averages.update(
        {
            "geometry_gradient_rms": totals["geometry_gradient_rms"] / trained_steps,
            "encoder_gradient_rms": totals["encoder_gradient_rms"] / trained_steps,
            "auxiliary_gradient_rms": totals["auxiliary_gradient_rms"] / trained_steps,
        }
    )
    print(
        f"epoch={epoch}/{epochs} trained={trained_steps} skipped={skipped_steps} "
        + " ".join(f"avg_{name}={value:.6f}" for name, value in averages.items())
        + f" vehicle_target_count={int(totals['vehicle_target_count'])}"
        + f" pedestrian_target_count={int(totals['pedestrian_target_count'])}"
        + f" occupied_bev_cells={int(totals['occupied_bev_cells'])}"
        + f" target_collision_count={int(totals['target_collision_count'])}"
    )
    print(
        "gradient_RMS "
        f"GeometryAwareBEVLift={averages['geometry_gradient_rms']:.9e} "
        f"BEVEncoder={averages['encoder_gradient_rms']:.9e} "
        f"BEVAuxiliaryHead={averages['auxiliary_gradient_rms']:.9e}"
    )
    return {
        **averages,
        "vehicle_target_count": totals["vehicle_target_count"],
        "pedestrian_target_count": totals["pedestrian_target_count"],
        "occupied_bev_cells": totals["occupied_bev_cells"],
        "target_collision_count": totals["target_collision_count"],
        "trained_steps": float(trained_steps),
    }


def main() -> int:
    args = parse_args()
    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    stage1_config = load_yaml_config(PROJECT_ROOT / "configs/stage1.yaml")
    bev_config = load_yaml_config(PROJECT_ROOT / "configs/stage1_bev.yaml")
    train_config = dict(bev_config["train"])
    num_samples = (
        args.num_samples
        if args.num_samples is not None
        else int(train_config["num_samples"])
    )
    epochs = args.epochs if args.epochs is not None else int(train_config["epochs"])
    if num_samples <= 0 or epochs <= 0:
        raise ValueError("num_samples and epochs must be positive")
    dataset = build_dataset(
        stage1_config,
        bev_config,
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
    auxiliary_head = BEVAuxiliaryHead(
        hidden_dim=model.hidden_dim,
        intermediate_dim=int(bev_config["auxiliary_head"]["intermediate_dim"]),
    ).to(device)
    trainable = configure_bev_pretraining(model, auxiliary_head)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    checkpoint_path = resolve_path(
        args.checkpoint_path or bev_config["paths"]["checkpoint_path"]
    )
    for epoch in range(1, epochs + 1):
        train_one_epoch(
            model,
            auxiliary_head,
            loader,
            optimizer,
            device,
            float(train_config["lambda_class"]),
            epoch,
            epochs,
        )
        save_bev_pretrain_checkpoint(
            checkpoint_path,
            model,
            auxiliary_head,
            optimizer,
            epoch,
            {**bev_config, "effective_num_samples": num_samples, "effective_epochs": epochs},
        )
        print(f"checkpoint_saved={checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
