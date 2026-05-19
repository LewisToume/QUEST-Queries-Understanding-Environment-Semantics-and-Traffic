from __future__ import annotations

import argparse
import contextlib
import csv
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.dataset import NuPlanDummyDataset, collate_fn
from quest.losses import compute_total_loss
from quest.model import QUESTModel
from quest.utils import load_yaml_config
from visualize_dummy_predictions import autocast_context, move_batch_to_device, resolve_device, visualize_batch


def load_configs() -> tuple[dict, dict]:
    model_config = load_yaml_config(PROJECT_ROOT / "configs" / "model.yaml")["model"]
    stage_config = load_yaml_config(PROJECT_ROOT / "configs" / "stage1.yaml")

    dataset_config = stage_config.setdefault("dataset", {})
    dataset_config.setdefault("seg_size", model_config["seg_size"])
    dataset_config.setdefault("C_seg", model_config["C_seg"])
    dataset_config.setdefault("C_agent", model_config["C_agent"])
    dataset_config.setdefault("D_box", model_config["D_box"])
    dataset_config.setdefault("C_map", model_config["C_map"])
    dataset_config.setdefault("P", model_config["P"])
    dataset_config.setdefault("occ_size", (model_config["X"], model_config["Y"], model_config["Z"]))
    dataset_config.setdefault("C_occ", model_config["C_occ"])
    return model_config, stage_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit QUEST on one dummy batch.")
    parser.add_argument("--steps", type=int, default=100, help="Number of optimization steps.")
    parser.add_argument("--vis-interval", type=int, default=10, help="Visualization interval in steps.")
    parser.add_argument("--batch-size", type=int, default=1, help="Fixed batch size used for overfit.")
    parser.add_argument("--device", type=str, default=None, help="Override device.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "debug_vis" / "overfit"),
        help="Directory used to save overfit diagnostics.",
    )
    return parser.parse_args()


def log_losses(step: int, losses: dict[str, torch.Tensor]) -> None:
    print(
        f"step={step:04d} "
        f"seg={losses['seg_loss'].item():.4f} "
        f"agent_cls={losses['agent_cls_loss'].item():.4f} "
        f"agent_box={losses['agent_box_loss'].item():.4f} "
        f"agent_dn={losses['agent_dn_loss'].item():.4f} "
        f"map_cls={losses['map_cls_loss'].item():.4f} "
        f"map_pts={losses['map_pts_loss'].item():.4f} "
        f"map_dir={losses['map_dir_loss'].item():.4f} "
        f"occ_main={losses['occ_main_loss'].item():.4f} "
        f"occ_sem={losses['occ_sem_scal_loss'].item():.4f} "
        f"occ_geo={losses['occ_geo_scal_loss'].item():.4f} "
        f"occ_lovasz={losses['occ_lovasz_loss'].item():.4f} "
        f"total={losses['total_loss'].item():.4f}"
    )


def main() -> None:
    args = parse_args()
    model_config, stage_config = load_configs()
    train_config = stage_config["train"]
    dataset_config = stage_config["dataset"]
    loss_config = stage_config.get("loss", {})
    device = resolve_device(args.device or train_config.get("device", "auto"))
    output_root = Path(args.output_dir)
    run_output_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = NuPlanDummyDataset(
        num_samples=max(args.batch_size, 1),
        **dataset_config,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )
    batch_cpu = next(iter(dataloader))
    batch = move_batch_to_device(batch_cpu, device)

    model = QUESTModel(**model_config).to(device).train()
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr or train_config["lr"]),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
        betas=(0.9, 0.999),
    )

    history: list[dict[str, float]] = []

    with torch.no_grad():
        model.eval()
        with autocast_context(device):
            initial_outputs = model(batch["image"])
        initial_outputs = {key: value.float() for key, value in initial_outputs.items()}
        visualize_batch(batch, initial_outputs, run_output_dir / "step_0000", prefix="overfit")
        model.train()

    for step in range(1, args.steps + 1):
        with autocast_context(device):
            outputs = model(batch["image"])
        outputs = {key: value.float() for key, value in outputs.items()}
        losses = compute_total_loss(outputs, batch, loss_config)

        optimizer.zero_grad()
        losses["total_loss"].backward()
        optimizer.step()

        history.append(
            {
                "step": float(step),
                "seg_loss": float(losses["seg_loss"].item()),
                "agent_cls_loss": float(losses["agent_cls_loss"].item()),
                "agent_box_loss": float(losses["agent_box_loss"].item()),
                "agent_dn_loss": float(losses["agent_dn_loss"].item()),
                "map_cls_loss": float(losses["map_cls_loss"].item()),
                "map_pts_loss": float(losses["map_pts_loss"].item()),
                "map_dir_loss": float(losses["map_dir_loss"].item()),
                "occ_main_loss": float(losses["occ_main_loss"].item()),
                "occ_sem_scal_loss": float(losses["occ_sem_scal_loss"].item()),
                "occ_geo_scal_loss": float(losses["occ_geo_scal_loss"].item()),
                "occ_lovasz_loss": float(losses["occ_lovasz_loss"].item()),
                "total_loss": float(losses["total_loss"].item()),
            }
        )

        if step == 1 or step % args.vis_interval == 0 or step == args.steps:
            log_losses(step, losses)
            model.eval()
            with torch.no_grad():
                with autocast_context(device):
                    vis_outputs = model(batch["image"])
            vis_outputs = {key: value.float() for key, value in vis_outputs.items()}
            visualize_batch(batch, vis_outputs, run_output_dir / f"step_{step:04d}", prefix="overfit")
            model.train()

    csv_path = run_output_dir / "loss_history.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    plt.figure(figsize=(8, 5))
    steps = [int(item["step"]) for item in history]
    plt.plot(steps, [item["total_loss"] for item in history], label="total_loss", linewidth=2.0)
    plt.plot(steps, [item["seg_loss"] for item in history], label="seg_loss", alpha=0.8)
    plt.plot(steps, [item["agent_cls_loss"] for item in history], label="agent_cls_loss", alpha=0.8)
    plt.plot(steps, [item["map_pts_loss"] for item in history], label="map_pts_loss", alpha=0.8)
    plt.plot(steps, [item["occ_main_loss"] for item in history], label="occ_main_loss", alpha=0.8)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("One-batch overfit loss curves")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(run_output_dir / "loss_curve.png", dpi=180, bbox_inches="tight")
    plt.close()

    first = history[0]
    last = history[-1]
    print("=" * 80)
    print("Overfit summary")
    print("=" * 80)
    print(f"total_loss   : {first['total_loss']:.4f} -> {last['total_loss']:.4f}")
    print(f"seg_loss     : {first['seg_loss']:.4f} -> {last['seg_loss']:.4f}")
    print(f"agent_cls    : {first['agent_cls_loss']:.4f} -> {last['agent_cls_loss']:.4f}")
    print(f"agent_box    : {first['agent_box_loss']:.4f} -> {last['agent_box_loss']:.4f}")
    print(f"map_cls      : {first['map_cls_loss']:.4f} -> {last['map_cls_loss']:.4f}")
    print(f"map_pts      : {first['map_pts_loss']:.4f} -> {last['map_pts_loss']:.4f}")
    print(f"map_dir      : {first['map_dir_loss']:.4f} -> {last['map_dir_loss']:.4f}")
    print(f"occ_main     : {first['occ_main_loss']:.4f} -> {last['occ_main_loss']:.4f}")
    print(f"saved logs   : {csv_path}")
    print(f"saved visuals: {run_output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
