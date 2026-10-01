from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.bev_pretraining import (
    BEVAuxiliaryHead,
    class_accuracy_counts,
    extract_canonical_agent_batch,
    foreground_counts,
    load_bev_pretrain_checkpoint,
    rasterize_agent_centers,
    topk_center_hits,
)
from quest.dataset import collate_fn
from quest.model import QUESTModel
from quest.utils import load_yaml_config
from scripts.train_stage1_bev import build_dataset, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 QUEST BEV pretraining")
    parser.add_argument("--start", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    args = parse_args()
    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    stage1_config = load_yaml_config(PROJECT_ROOT / "configs/stage1.yaml")
    bev_config = load_yaml_config(PROJECT_ROOT / "configs/stage1_bev.yaml")
    eval_config = bev_config["eval"]
    start = args.start if args.start is not None else int(eval_config["start_index"])
    count = args.count if args.count is not None else int(eval_config["num_samples"])
    dataset = build_dataset(stage1_config, bev_config, start, count)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QUESTModel(**model_config).to(device).eval()
    auxiliary_head = BEVAuxiliaryHead(
        hidden_dim=model.hidden_dim,
        intermediate_dim=int(bev_config["auxiliary_head"]["intermediate_dim"]),
    ).to(device).eval()
    checkpoint_path = resolve_path(
        args.checkpoint or bev_config["paths"]["checkpoint_path"]
    )
    checkpoint = load_checkpoint(checkpoint_path, device)
    epoch = load_bev_pretrain_checkpoint(model, checkpoint, auxiliary_head)
    threshold = float(eval_config["foreground_threshold"])
    tolerance = int(eval_config["tolerance_cells"])
    top_ks = [int(value) for value in eval_config["top_k"]]
    counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    topk_hits = {value: 0 for value in top_ks}
    topk_targets = {value: 0 for value in top_ks}
    class_correct = torch.zeros(2, dtype=torch.long)
    class_total = torch.zeros(2, dtype=torch.long)
    evaluated = 0
    skipped = 0
    with torch.no_grad():
        for batch in loader:
            agent_target = extract_canonical_agent_batch(batch, device)
            if agent_target is None:
                skipped += 1
                continue
            encoded = model.encode_image(
                batch["images"].to(device),
                batch["intrinsics"].to(device),
                batch["extrinsics"].to(device),
                batch["ego_state"].to(device),
            )
            predictions = auxiliary_head(encoded["bev_features"])
            targets = rasterize_agent_centers(
                agent_target,
                model.geometry_lift.x_range,
                model.geometry_lift.y_range,
                model.geometry_lift.bev_h,
                model.geometry_lift.bev_w,
            )
            item_counts = foreground_counts(
                predictions["foreground_logits"],
                targets["positive_mask"],
                threshold,
            )
            for key, value in item_counts.items():
                counts[key] += value
            correct, total = class_accuracy_counts(predictions, targets, threshold)
            class_correct += correct.cpu()
            class_total += total.cpu()
            for top_k in top_ks:
                hits, target_count = topk_center_hits(
                    predictions["foreground_logits"],
                    targets["positive_mask"],
                    top_k,
                    tolerance,
                )
                topk_hits[top_k] += hits
                topk_targets[top_k] += target_count
            evaluated += 1
    precision = ratio(
        counts["true_positive"], counts["true_positive"] + counts["false_positive"]
    )
    recall = ratio(
        counts["true_positive"], counts["true_positive"] + counts["false_negative"]
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"checkpoint={checkpoint_path} epoch={epoch}")
    print(f"evaluated_frames={evaluated} skipped_frames={skipped}")
    print(f"foreground_precision={precision:.6f}")
    print(f"foreground_recall={recall:.6f}")
    print(f"foreground_f1={f1:.6f}")
    for top_k in top_ks:
        print(
            f"top_{top_k}_bev_center_recall="
            f"{ratio(topk_hits[top_k], topk_targets[top_k]):.6f}"
        )
    for class_id, name in enumerate(("vehicle", "pedestrian")):
        print(
            f"{name}_accuracy_on_matched_positive_cells="
            f"{ratio(int(class_correct[class_id]), int(class_total[class_id])):.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
