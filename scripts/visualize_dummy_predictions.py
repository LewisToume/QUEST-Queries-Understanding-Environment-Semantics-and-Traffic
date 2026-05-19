from __future__ import annotations

import argparse
import contextlib
import csv
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.dataset import NuPlanDummyDataset, collate_fn
from quest.model import QUESTModel
from quest.utils import load_yaml_config


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def autocast_context(device: str):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


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


def move_batch_to_device(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {
        "image": batch["image"].to(device),
        "seg_gt": batch["seg_gt"].to(device),
        "agent_gt": {
            "labels": batch["agent_gt"]["labels"].to(device),
            "boxes": batch["agent_gt"]["boxes"].to(device),
        },
        "map_gt": {
            "labels": batch["map_gt"]["labels"].to(device),
            "points": batch["map_gt"]["points"].to(device),
        },
        "occ_gt": batch["occ_gt"].to(device),
    }


def tensor_stats(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float().cpu()
    return {
        "name": name,
        "shape": tuple(value.shape),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std().item()),
        "has_nan": bool(torch.isnan(value).any().item()),
        "has_inf": bool(torch.isinf(value).any().item()),
    }


def print_output_stats(outputs: Mapping[str, torch.Tensor]) -> None:
    print("=" * 80)
    print("Prediction tensor stats")
    print("=" * 80)
    for name, tensor in outputs.items():
        stats = tensor_stats(name, tensor)
        print(
            f"{name:<18} shape={stats['shape']} "
            f"min={stats['min']:.4f} max={stats['max']:.4f} "
            f"mean={stats['mean']:.4f} std={stats['std']:.4f} "
            f"nan={stats['has_nan']} inf={stats['has_inf']}"
        )
    print("=" * 80)


def _save_array_image(
    array: torch.Tensor,
    path: Path,
    title: str,
    cmap: str = "tab20",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(array.cpu().numpy(), cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def save_segmentation_vis(seg_logits: torch.Tensor, seg_gt: torch.Tensor, path: Path) -> None:
    pred = seg_logits.argmax(dim=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pred.cpu().numpy(), cmap="tab20", interpolation="nearest")
    axes[0].set_title("Pred seg argmax")
    axes[0].axis("off")
    axes[1].imshow(seg_gt.cpu().numpy(), cmap="tab20", interpolation="nearest")
    axes[1].set_title("GT seg")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_map_vis(
    map_cls_logits: torch.Tensor,
    map_points: torch.Tensor,
    map_gt: Mapping[str, torch.Tensor],
    path: Path,
    top_k: int = 12,
) -> None:
    probs = torch.softmax(map_cls_logits, dim=-1)
    cls_probs = probs[:, :-1]
    bg_probs = probs[:, -1]
    fg_scores, top_classes = cls_probs.max(dim=-1)
    fg_minus_bg = fg_scores - bg_probs
    fg_dominant_mask = fg_minus_bg > 0

    if fg_dominant_mask.any():
        candidate_indices = torch.nonzero(fg_dominant_mask, as_tuple=False).squeeze(1)
        rank_scores = fg_scores[candidate_indices]
        sorted_candidates = candidate_indices[torch.argsort(rank_scores, descending=True)]
        top_indices = sorted_candidates[: min(top_k, sorted_candidates.numel())]
        title = "Map Queries (fg-dominant top-k; pred solid, gt dashed)"
    else:
        top_indices = torch.argsort(fg_minus_bg, descending=True)[: min(top_k, map_points.shape[0])]
        title = "Map Queries (fallback least-bg top-k; pred solid, gt dashed)"

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(8, 8))

    gt_labels = map_gt["labels"]
    gt_points = map_gt["points"]
    valid_gt = gt_labels >= 0
    for gt_idx in torch.nonzero(valid_gt, as_tuple=False).squeeze(1).tolist():
        pts = gt_points[gt_idx].cpu().numpy()
        plt.plot(pts[:, 0], pts[:, 1], linestyle="--", linewidth=1.0, color="black", alpha=0.45)

    for rank, query_idx in enumerate(top_indices.tolist(), start=1):
        pts = map_points[query_idx].cpu().numpy()
        cls_id = int(top_classes[query_idx].item())
        color = cmap(cls_id % 10)
        fg_score = float(fg_scores[query_idx].item())
        bg_score = float(bg_probs[query_idx].item())
        alpha = 0.95 if bool(fg_dominant_mask[query_idx].item()) else 0.4
        plt.plot(
            pts[:, 0],
            pts[:, 1],
            linewidth=2.0,
            color=color,
            alpha=alpha,
            label=f"q{query_idx}: c{cls_id} fg={fg_score:.2f} bg={bg_score:.2f}",
        )
        plt.scatter(pts[0, 0], pts[0, 1], color=color, s=14)
        if rank <= 3:
            plt.text(pts[0, 0], pts[0, 1], f"q{query_idx}", fontsize=8, color=color)

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.xlabel("BEV x")
    plt.ylabel("BEV y")
    if len(top_indices) > 0:
        plt.legend(loc="upper right", fontsize=7)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def save_agent_outputs(
    agent_cls_logits: torch.Tensor,
    agent_boxes: torch.Tensor,
    agent_gt: Mapping[str, torch.Tensor],
    output_dir: Path,
    prefix: str,
    top_k: int = 10,
) -> None:
    probs = torch.softmax(agent_cls_logits, dim=-1)
    cls_probs = probs[:, :-1]
    bg_probs = probs[:, -1]
    fg_scores, top_classes = cls_probs.max(dim=-1)
    fg_minus_bg = fg_scores - bg_probs
    fg_dominant_mask = fg_minus_bg > 0

    if fg_dominant_mask.any():
        candidate_indices = torch.nonzero(fg_dominant_mask, as_tuple=False).squeeze(1)
        rank_scores = fg_scores[candidate_indices]
        sorted_candidates = candidate_indices[torch.argsort(rank_scores, descending=True)]
        top_indices = sorted_candidates[: min(top_k, sorted_candidates.numel())]
    else:
        top_indices = torch.argsort(fg_minus_bg, descending=True)[: min(top_k, agent_boxes.shape[0])]

    csv_path = output_dir / f"{prefix}_agent_topk.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "query_idx",
                "pred_class",
                "fg_score",
                "bg_score",
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
            row = [
                rank,
                query_idx,
                int(top_classes[query_idx].item()),
                float(fg_scores[query_idx].item()),
                float(bg_probs[query_idx].item()),
                float(fg_minus_bg[query_idx].item()),
                bool(fg_dominant_mask[query_idx].item()),
            ] + [float(v) for v in agent_boxes[query_idx].tolist()]
            writer.writerow(row)

    plt.figure(figsize=(8, 8))
    valid_gt = agent_gt["labels"] >= 0
    gt_boxes = agent_gt["boxes"][valid_gt]
    if gt_boxes.numel() > 0:
        plt.scatter(
            gt_boxes[:, 0].cpu().numpy(),
            gt_boxes[:, 1].cpu().numpy(),
            marker="x",
            s=40,
            c="black",
            label="GT centers",
        )

    for rank, query_idx in enumerate(top_indices.tolist(), start=1):
        box = agent_boxes[query_idx].cpu().numpy()
        cls_id = int(top_classes[query_idx].item())
        fg_score = float(fg_scores[query_idx].item())
        bg_score = float(bg_probs[query_idx].item())
        is_fg_dominant = bool(fg_dominant_mask[query_idx].item())
        alpha = 0.95 if is_fg_dominant else 0.35
        marker = "o" if is_fg_dominant else "^"
        plt.scatter(
            box[0],
            box[1],
            s=45,
            alpha=alpha,
            marker=marker,
            label=f"q{query_idx}: c{cls_id} fg={fg_score:.2f} bg={bg_score:.2f}",
        )
        plt.text(box[0], box[1], f"{rank}", fontsize=8)

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("cx")
    plt.ylabel("cy")
    if fg_dominant_mask.any():
        plt.title("Agent Query Centers (fg-dominant top-k)")
    else:
        plt.title("Agent Query Centers (no fg-dominant query; showing least-bg top-k)")
    if len(top_indices) > 0 or gt_boxes.numel() > 0:
        plt.legend(loc="upper right", fontsize=7)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_agent_topk.png", dpi=180, bbox_inches="tight")
    plt.close()


def save_occ_vis(
    occ_logits: torch.Tensor,
    occ_gt: torch.Tensor,
    output_dir: Path,
    prefix: str,
    num_slices: int = 3,
) -> None:
    occ_pred = occ_logits.argmax(dim=0)
    depth = occ_pred.shape[-1]
    slice_indices = torch.linspace(0, depth - 1, steps=min(num_slices, depth)).long().tolist()

    for slice_idx in slice_indices:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(occ_pred[:, :, slice_idx].cpu().numpy(), cmap="tab20", interpolation="nearest")
        axes[0].set_title(f"Pred occ z={slice_idx}")
        axes[0].axis("off")
        axes[1].imshow(occ_gt[:, :, slice_idx].cpu().numpy(), cmap="tab20", interpolation="nearest")
        axes[1].set_title(f"GT occ z={slice_idx}")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_occ_slice_z{slice_idx:02d}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    pred_bev = occ_pred.max(dim=-1).values
    gt_bev = occ_gt.max(dim=-1).values
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pred_bev.cpu().numpy(), cmap="tab20", interpolation="nearest")
    axes[0].set_title("Pred occ BEV projection")
    axes[0].axis("off")
    axes[1].imshow(gt_bev.cpu().numpy(), cmap="tab20", interpolation="nearest")
    axes[1].set_title("GT occ BEV projection")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_occ_bev.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def visualize_batch(
    batch: Mapping[str, Any],
    outputs: Mapping[str, torch.Tensor],
    output_dir: Path,
    prefix: str = "sample",
    top_k_agent: int = 10,
    top_k_map: int = 12,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = outputs["seg_logits"].shape[0]

    for sample_idx in range(batch_size):
        sample_prefix = f"{prefix}_{sample_idx}"
        save_segmentation_vis(
            outputs["seg_logits"][sample_idx].detach().cpu(),
            batch["seg_gt"][sample_idx].detach().cpu(),
            output_dir / f"{sample_prefix}_seg.png",
        )
        save_map_vis(
            outputs["map_cls_logits"][sample_idx].detach().cpu(),
            outputs["map_points"][sample_idx].detach().cpu(),
            {
                "labels": batch["map_gt"]["labels"][sample_idx].detach().cpu(),
                "points": batch["map_gt"]["points"][sample_idx].detach().cpu(),
            },
            output_dir / f"{sample_prefix}_map.png",
            top_k=top_k_map,
        )
        save_agent_outputs(
            outputs["agent_cls_logits"][sample_idx].detach().cpu(),
            outputs["agent_boxes"][sample_idx].detach().cpu(),
            {
                "labels": batch["agent_gt"]["labels"][sample_idx].detach().cpu(),
                "boxes": batch["agent_gt"]["boxes"][sample_idx].detach().cpu(),
            },
            output_dir,
            sample_prefix,
            top_k=top_k_agent,
        )
        save_occ_vis(
            outputs["occ_logits"][sample_idx].detach().cpu(),
            batch["occ_gt"][sample_idx].detach().cpu(),
            output_dir,
            sample_prefix,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize QUEST dummy predictions.")
    parser.add_argument("--num-samples", type=int, default=2, help="Number of dummy samples to visualize.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "debug_vis" / "dummy_predictions"),
        help="Directory used to save visualization outputs.",
    )
    parser.add_argument("--device", type=str, default=None, help="Override device.")
    parser.add_argument("--top-k-agent", type=int, default=10, help="Top-k agent queries to export.")
    parser.add_argument("--top-k-map", type=int, default=12, help="Top-k map queries to draw.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config, stage_config = load_configs()
    train_config = stage_config["train"]
    dataset_config = stage_config["dataset"]
    device = resolve_device(args.device or train_config.get("device", "auto"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = NuPlanDummyDataset(
        num_samples=max(args.num_samples, 1),
        **dataset_config,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=min(args.num_samples, len(dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    batch_cpu = next(iter(dataloader))
    batch = move_batch_to_device(batch_cpu, device)

    model = QUESTModel(**model_config).to(device).eval()
    with torch.no_grad():
        with autocast_context(device):
            outputs = model(batch["image"])
    outputs = {key: value.float() for key, value in outputs.items()}

    print(f"device: {device}")
    print_output_stats(outputs)
    visualize_batch(
        batch,
        outputs,
        output_dir,
        prefix="dummy",
        top_k_agent=args.top_k_agent,
        top_k_map=args.top_k_map,
    )
    print(f"saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
