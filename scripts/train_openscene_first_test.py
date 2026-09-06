from __future__ import annotations

import argparse
import csv
import contextlib
import sys
from datetime import datetime
from itertools import cycle
from pathlib import Path

import matplotlib
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.dataset import collate_fn
from quest.losses import compute_agent_loss, compute_occ_loss
from quest.model import QUESTModel
from quest.openscene_dataset import OpenSceneFirstTestDataset
from quest.utils import load_yaml_config


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def autocast_context(device: str):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QUEST on the 10-sample OpenScene first test set.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "openscene_first_test"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-agent-instances", type=int, default=64)
    parser.add_argument("--agent-weight", type=float, default=2.0)
    parser.add_argument("--occ-weight", type=float, default=1.0)
    parser.add_argument(
        "--vis-interval",
        type=int,
        default=0,
        help="Save agent/occ visualizations every N steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "debug_vis" / "openscene_first_test"),
        help="Root directory for timestamped visualization runs.",
    )
    parser.add_argument("--top-k-agent", type=int, default=10)
    return parser.parse_args()


def build_model_config() -> dict:
    config = load_yaml_config(PROJECT_ROOT / "configs" / "model.yaml")["model"]
    config.update(
        {
            "C_agent": 4,
            "D_box": 8,
            "C_occ": 11,
            "X": 200,
            "Y": 200,
            "Z": 16,
        }
    )
    return config


def move_openscene_batch_to_device(batch: dict, device: str) -> dict:
    return {
        "images": batch["images"].to(device),
        "intrinsics": batch["intrinsics"].to(device),
        "extrinsics": batch["extrinsics"].to(device),
        "ego_state": batch["ego_state"].to(device),
        "agent_gt": {
            "labels": batch["agent_gt"]["labels"].to(device),
            "boxes": batch["agent_gt"]["boxes"].to(device),
            "velocity": batch["agent_gt"]["velocity"].to(device),
        },
        "occ_gt": batch["occ_gt"].to(device),
    }


def save_loss_history(loss_history: list[dict[str, float]], output_dir: Path) -> None:
    if not loss_history:
        return

    csv_path = output_dir / "loss_history.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(loss_history[0].keys()))
        writer.writeheader()
        writer.writerows(loss_history)

    steps = [row["step"] for row in loss_history]
    plt.figure(figsize=(8, 5))
    for key in ("total", "agent", "agent_cls", "agent_box", "occ"):
        plt.plot(steps, [row[key] for row in loss_history], label=key)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("OpenScene first-test loss history")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=180, bbox_inches="tight")
    plt.close()


def save_agent_vis(
    agent_cls_logits: torch.Tensor,
    agent_boxes: torch.Tensor,
    agent_gt: dict[str, torch.Tensor],
    output_dir: Path,
    prefix: str,
    top_k: int,
) -> None:
    probs = torch.softmax(agent_cls_logits, dim=-1)
    fg_probs = probs[:, :-1]
    bg_probs = probs[:, -1]
    fg_scores, top_classes = fg_probs.max(dim=-1)
    fg_minus_bg = fg_scores - bg_probs
    fg_dominant = fg_minus_bg > 0

    if fg_dominant.any():
        candidates = torch.nonzero(fg_dominant, as_tuple=False).squeeze(1)
        order = torch.argsort(fg_scores[candidates], descending=True)
        top_indices = candidates[order][: min(top_k, candidates.numel())]
        title = "OpenScene Agent Queries (fg-dominant top-k)"
    else:
        top_indices = torch.argsort(fg_minus_bg, descending=True)[: min(top_k, agent_boxes.shape[0])]
        title = "OpenScene Agent Queries (fallback least-bg top-k)"

    csv_path = output_dir / f"{prefix}_agent_topk.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "query_idx",
                "pred_class",
                "fg_prob",
                "bg_prob",
                "fg_minus_bg",
                "fg_dominant",
                "cx",
                "cy",
                "cz",
                "dx",
                "dy",
                "dz",
                "sin_yaw",
                "cos_yaw",
            ]
        )
        for rank, query_idx in enumerate(top_indices.tolist(), start=1):
            writer.writerow(
                [
                    rank,
                    query_idx,
                    int(top_classes[query_idx].item()),
                    float(fg_scores[query_idx].item()),
                    float(bg_probs[query_idx].item()),
                    float(fg_minus_bg[query_idx].item()),
                    bool(fg_dominant[query_idx].item()),
                    *[float(v) for v in agent_boxes[query_idx].tolist()],
                ]
            )

    plt.figure(figsize=(8, 8))
    valid_gt = agent_gt["labels"] >= 0
    gt_boxes = agent_gt["boxes"][valid_gt]
    if gt_boxes.numel() > 0:
        plt.scatter(
            gt_boxes[:, 0].cpu().numpy(),
            gt_boxes[:, 1].cpu().numpy(),
            marker="x",
            s=36,
            c="black",
            label="GT centers",
        )

    for rank, query_idx in enumerate(top_indices.tolist(), start=1):
        box = agent_boxes[query_idx].cpu().numpy()
        is_fg = bool(fg_dominant[query_idx].item())
        marker = "o" if is_fg else "^"
        alpha = 0.95 if is_fg else 0.35
        label = (
            f"q{query_idx}: c{int(top_classes[query_idx].item())} "
            f"fg={float(fg_scores[query_idx].item()):.2f} "
            f"bg={float(bg_probs[query_idx].item()):.2f}"
        )
        plt.scatter(box[0], box[1], s=44, marker=marker, alpha=alpha, label=label)
        plt.text(box[0], box[1], str(rank), fontsize=8)

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("normalized cx")
    plt.ylabel("normalized cy")
    plt.title(title)
    plt.grid(alpha=0.25)
    if len(top_indices) > 0 or gt_boxes.numel() > 0:
        plt.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_agent_topk.png", dpi=180, bbox_inches="tight")
    plt.close()


def save_occ_vis(
    occ_logits: torch.Tensor,
    occ_gt: torch.Tensor,
    output_dir: Path,
    prefix: str,
) -> None:
    occ_pred = occ_logits.argmax(dim=0)
    depth = occ_pred.shape[-1]
    slice_indices = sorted({0, depth // 2, depth - 1})

    pred_bev = occ_pred.max(dim=-1).values
    gt_bev = occ_gt.max(dim=-1).values
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pred_bev.cpu().numpy(), cmap="tab20", interpolation="nearest")
    axes[0].set_title("Pred occ semantic BEV")
    axes[0].axis("off")
    axes[1].imshow(gt_bev.cpu().numpy(), cmap="tab20", interpolation="nearest")
    axes[1].set_title("GT occ semantic BEV")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_occ_bev.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pred_mask = (occ_pred > 0).any(dim=-1)
    gt_mask = (occ_gt > 0).any(dim=-1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pred_mask.cpu().numpy(), cmap="gray", interpolation="nearest")
    axes[0].set_title("Pred occupied BEV mask")
    axes[0].axis("off")
    axes[1].imshow(gt_mask.cpu().numpy(), cmap="gray", interpolation="nearest")
    axes[1].set_title("GT occupied BEV mask")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_occ_occupied_mask.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for slice_idx in slice_indices:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(occ_pred[:, :, slice_idx].cpu().numpy(), cmap="tab20", interpolation="nearest")
        axes[0].set_title(f"Pred occ z={slice_idx}")
        axes[0].axis("off")
        axes[1].imshow(occ_gt[:, :, slice_idx].cpu().numpy(), cmap="tab20", interpolation="nearest")
        axes[1].set_title(f"GT occ z={slice_idx}")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_occ_slice_z{slice_idx:02d}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def save_openscene_visualization(
    model: QUESTModel,
    batch: dict,
    device: str,
    output_dir: Path,
    step: int,
    top_k_agent: int,
) -> None:
    step_dir = output_dir / f"step_{step:04d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        with autocast_context(device):
            outputs = model(
                batch["images"],
                intrinsics=batch["intrinsics"],
                extrinsics=batch["extrinsics"],
                ego_state=batch["ego_state"],
            )
    outputs = {key: value.float().detach().cpu() for key, value in outputs.items()}
    batch_cpu = {
        "agent_gt": {
            "labels": batch["agent_gt"]["labels"].detach().cpu(),
            "boxes": batch["agent_gt"]["boxes"].detach().cpu(),
        },
        "occ_gt": batch["occ_gt"].detach().cpu(),
    }

    batch_size = outputs["agent_cls_logits"].shape[0]
    for sample_idx in range(batch_size):
        prefix = f"openscene_{sample_idx}"
        save_agent_vis(
            outputs["agent_cls_logits"][sample_idx],
            outputs["agent_boxes"][sample_idx],
            {
                "labels": batch_cpu["agent_gt"]["labels"][sample_idx],
                "boxes": batch_cpu["agent_gt"]["boxes"][sample_idx],
            },
            step_dir,
            prefix,
            top_k=top_k_agent,
        )
        save_occ_vis(
            outputs["occ_logits"][sample_idx],
            batch_cpu["occ_gt"][sample_idx],
            step_dir,
            prefix,
        )

    if model_was_training:
        model.train()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model_config = build_model_config()

    dataset = OpenSceneFirstTestDataset(
        root=args.data_root,
        image_size=tuple(model_config.get("image_size", (224, 224))),
        occ_size=(model_config["X"], model_config["Y"], model_config["Z"]),
        C_occ=model_config["C_occ"],
        max_agent_instances=args.max_agent_instances,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )
    vis_dataloader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )
    vis_batch = move_openscene_batch_to_device(next(iter(vis_dataloader)), device)

    model = QUESTModel(**model_config).to(device).train()
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    agent_loss_config = {
        "cls_cost_weight": 2.0,
        "bbox_cost_weight": 0.25,
        "cls_gamma": 2.0,
        "cls_alpha": 0.25,
        "lambda_box": 0.25,
        "lambda_dn": 0.0,
        "box_loss_type": "l1",
    }
    occ_loss_config = {
        "class_weight": None,
        "use_camera_mask": False,
        "lambda_sem": 0.0,
        "lambda_geo": 0.0,
        "lambda_lovasz": 0.0,
    }

    print("=" * 72)
    print("QUEST OpenScene first-test training")
    print(f"device      : {device}")
    print(f"samples     : {len(dataset)}")
    print(f"batch_size  : {args.batch_size}")
    print(f"steps       : {args.steps}")
    print(f"C_agent     : {model_config['C_agent']}")
    print(f"occ shape   : ({model_config['C_occ']}, {model_config['X']}, {model_config['Y']}, {model_config['Z']})")
    print("losses      : agent + occ only")
    output_dir = None
    if args.vis_interval > 0:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"vis_dir     : {output_dir}")
    print("=" * 72)

    running_total = 0.0
    loss_history: list[dict[str, float]] = []
    data_iter = cycle(dataloader)
    if output_dir is not None:
        save_openscene_visualization(
            model=model,
            batch=vis_batch,
            device=device,
            output_dir=output_dir,
            step=0,
            top_k_agent=args.top_k_agent,
        )

    for step in range(1, args.steps + 1):
        batch = next(data_iter)
        images = batch["images"].to(device)
        intrinsics = batch["intrinsics"].to(device)
        extrinsics = batch["extrinsics"].to(device)
        ego_state = batch["ego_state"].to(device)
        gts = {
            "agent_gt": {
                "labels": batch["agent_gt"]["labels"].to(device),
                "boxes": batch["agent_gt"]["boxes"].to(device),
                "velocity": batch["agent_gt"]["velocity"].to(device),
            },
            "occ_gt": batch["occ_gt"].to(device),
        }

        with autocast_context(device):
            preds = model(
                images,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                ego_state=ego_state,
            )
        preds = {key: value.float() for key, value in preds.items()}

        agent_losses = compute_agent_loss(
            preds["agent_cls_logits"],
            preds["agent_boxes"],
            preds["agent_velocity"],
            gts["agent_gt"],
            agent_loss_config,
        )
        occ_losses = compute_occ_loss(
            preds["occ_logits"],
            gts["occ_gt"],
            occ_loss_config,
        )
        total_loss = args.agent_weight * agent_losses["agent_loss"] + args.occ_weight * occ_losses["occ_loss"]

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_total += float(total_loss.item())
        loss_history.append(
            {
                "step": float(step),
                "total": float(total_loss.item()),
                "agent": float(agent_losses["agent_loss"].item()),
                "agent_cls": float(agent_losses["agent_cls_loss"].item()),
                "agent_box": float(agent_losses["agent_box_loss"].item()),
                "occ": float(occ_losses["occ_main_loss"].item()),
            }
        )
        if step == 1 or step % 5 == 0 or step == args.steps:
            print(
                f"step={step:04d} "
                f"agent_cls={agent_losses['agent_cls_loss'].item():.4f} "
                f"agent_box={agent_losses['agent_box_loss'].item():.4f} "
                f"agent={agent_losses['agent_loss'].item():.4f} "
                f"occ={occ_losses['occ_main_loss'].item():.4f} "
                f"total={total_loss.item():.4f}"
            )
        if output_dir is not None and (step % args.vis_interval == 0 or step == args.steps):
            save_openscene_visualization(
                model=model,
                batch=vis_batch,
                device=device,
                output_dir=output_dir,
                step=step,
                top_k_agent=args.top_k_agent,
            )
            save_loss_history(loss_history, output_dir)

    if output_dir is not None:
        save_loss_history(loss_history, output_dir)
    print("=" * 72)
    print(f"avg_total_loss: {running_total / max(1, args.steps):.4f}")
    if output_dir is not None:
        print(f"visualizations: {output_dir}")
    print("OpenScene first-test training finished")
    print("=" * 72)


if __name__ == "__main__":
    main()
