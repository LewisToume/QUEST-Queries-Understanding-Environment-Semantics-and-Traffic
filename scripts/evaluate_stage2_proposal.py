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
    foreground_counts,
    topk_center_hits,
)
from quest.dataset import collate_fn
from quest.model import QUESTModel
from quest.proposal_pretraining import (
    load_proposal_pretrain_checkpoint,
    proposal_center_hits,
    rasterize_proposal_targets,
)
from quest.utils import load_yaml_config
from scripts.train_stage1_bev import build_dataset, resolve_path
from scripts.train_stage2_proposal import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 Agent proposals")
    parser.add_argument("--start", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    args = parse_args()
    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    stage1_config = load_yaml_config(PROJECT_ROOT / "configs/stage1.yaml")
    proposal_config = load_yaml_config(PROJECT_ROOT / "configs/stage2_proposal.yaml")
    eval_config = proposal_config["eval"]
    start = args.start if args.start is not None else int(eval_config["start_index"])
    count = args.count if args.count is not None else int(eval_config["num_samples"])
    dataset = build_dataset(stage1_config, proposal_config, start, count)
    loader = DataLoader(
        dataset,
        batch_size=int(eval_config["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QUESTModel(**model_config).to(device).eval()
    checkpoint_path = resolve_path(args.checkpoint or proposal_config["paths"]["checkpoint_path"])
    checkpoint = load_checkpoint(checkpoint_path)
    epoch = load_proposal_pretrain_checkpoint(model, checkpoint)
    del checkpoint

    top_ks = tuple(int(value) for value in eval_config["top_k"])
    tolerance = int(eval_config["tolerance_cells"])
    thresholds = tuple(float(value) for value in eval_config["objectness_thresholds"])
    distances = tuple(float(value) for value in eval_config["center_distance_thresholds_m"])
    threshold_counts = {
        value: {"true_positive": 0, "false_positive": 0, "false_negative": 0}
        for value in thresholds
    }
    bev_hits = {value: 0 for value in top_ks}
    bev_totals = {value: 0 for value in top_ks}
    exact_hits = {value: 0 for value in top_ks}
    exact_totals = {value: 0 for value in top_ks}
    metric_hits = {(top_k, distance): 0 for top_k in top_ks for distance in distances}
    metric_totals = {(top_k, distance): 0 for top_k in top_ks for distance in distances}
    evaluated = 0

    with torch.no_grad():
        for batch in loader:
            agent_target = extract_canonical_agent_batch(batch, device)
            if agent_target is None:
                raise RuntimeError(f"missing Navformer Agent target for {batch['sample_token']}")
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
            logits_grid = objectness.reshape(-1, model.bev_h, model.bev_w)
            positive_grid = targets["positive_mask"].reshape_as(logits_grid)
            for threshold in thresholds:
                counts = foreground_counts(logits_grid, positive_grid, threshold)
                for key, value in counts.items():
                    threshold_counts[threshold][key] += value
            for top_k in top_ks:
                hits, total = topk_center_hits(logits_grid, positive_grid, top_k, tolerance)
                bev_hits[top_k] += hits
                bev_totals[top_k] += total
                hits, total = topk_center_hits(logits_grid, positive_grid, top_k, 0)
                exact_hits[top_k] += hits
                exact_totals[top_k] += total
                for distance in distances:
                    hits, total = proposal_center_hits(
                        objectness,
                        offsets,
                        targets["target_centers_metric"],
                        top_k,
                        distance,
                        model.geometry_lift.x_range,
                        model.geometry_lift.y_range,
                        model.bev_h,
                        model.bev_w,
                    )
                    metric_hits[top_k, distance] += hits
                    metric_totals[top_k, distance] += total
            evaluated += int(objectness.shape[0])
    if not evaluated:
        raise RuntimeError("Stage 2 Proposal evaluation received no samples")

    print(f"checkpoint={checkpoint_path} epoch={epoch} evaluated_frames={evaluated}")
    for top_k in top_ks:
        print(f"top_{top_k}_bev_center_recall={ratio(bev_hits[top_k], bev_totals[top_k]):.6f}")
        print(
            f"top_{top_k}_exact_bev_center_recall="
            f"{ratio(exact_hits[top_k], exact_totals[top_k]):.6f}"
        )
        for distance in distances:
            print(
                f"top_{top_k}_center_recall_within_{distance:.1f}m="
                f"{ratio(metric_hits[top_k, distance], metric_totals[top_k, distance]):.6f}"
            )
    for threshold in thresholds:
        counts = threshold_counts[threshold]
        true_positive = counts["true_positive"]
        predicted = true_positive + counts["false_positive"]
        positives = true_positive + counts["false_negative"]
        precision = ratio(true_positive, predicted)
        recall = ratio(true_positive, positives)
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(
            f"threshold={threshold:.1f} precision={precision:.6f} "
            f"recall={recall:.6f} f1={f1:.6f} "
            f"predicted_positive_cells_per_frame={predicted / evaluated:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
