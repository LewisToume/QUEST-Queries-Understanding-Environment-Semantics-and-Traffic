"""Print a compact stored-history review; CPU only, no model construction."""

import argparse
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def f1(row):
    precision, recall = row["precision"], row["recall"]
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.


def macro_f1(metrics, task):
    rows = list(metrics[task].values())
    return sum(f1(row) for row in rows) / len(rows)


def number(value):
    return "N/A" if value is None or not math.isfinite(float(value)) else f"{value:.4f}"


def print_row(label, metrics, passed):
    dense = metrics["dense"]
    fields = [dense["seg_teacher_mIoU"], dense["depth_teacher_absrel"], dense["depth_teacher_delta1"],
              macro_f1(metrics, "agent"), macro_f1(metrics, "map")]
    print(f"{str(label):>8} " + " ".join(f"{number(value):>8}" for value in fields) + " " + passed)


def review(saved):
    if saved.get("stage") not in ("shared_scene_four_task_v1", "shared_scene_feature_distillation_v1",
                                  "shared_scene_gradnorm_v1", "shared_scene_map_refinement_v1"):
        raise ValueError("Expected a full shared-scene training checkpoint, not a deployment export")
    baseline, history = saved["baseline_validation"], saved["validation_history"]
    if not history:
        raise ValueError("No completed validation epochs in checkpoint")
    print("READ_ONLY=True GPU=False model_constructed=False reference=PSEUDO_AGENT_SEG_DEPTH_MAP_VECTOR_GT")
    print("epoch Seg_mIoU Depth_AbsRel Depth_delta1 Agent_macroF1 Map_macroF1 gate")
    print_row("baseline", baseline, "N/A")
    for row in history:
        print_row(row["epoch"], row["validation"], "PASS" if row["baseline_gate"]["all_pass"] else "FAIL")
    selectors = {
        "Seg": lambda m: m["dense"]["seg_teacher_mIoU"],
        "Depth": lambda m: -m["dense"]["depth_teacher_absrel"],
        "Agent": lambda m: macro_f1(m, "agent"),
        "Map": lambda m: macro_f1(m, "map"),
    }
    print("history_best_epochs=" + " ".join(
        name + ":" + str(max(history, key=lambda row: score(row["validation"]))["epoch"])
        for name, score in selectors.items()))
    print("history_best_epochs_do_not_imply_saved_earlier_weights=True")
    current = history[-1]["validation"]
    print("last_vs_baseline task/class precision recall F1 (baseline -> last)")
    for task in ("agent", "map"):
        for name, reference in baseline[task].items():
            row = current[task][name]
            print(task + "/" + name + " " + " ".join(
                f"{reference[key]:.4f}->{row[key]:.4f}" for key in ("precision", "recall"))
                + f" {f1(reference):.4f}->{f1(row):.4f}")
    failed = [name for name, ok in history[-1]["baseline_gate"]["checks"].items() if not ok]
    print("failed_checks=" + ",".join(failed))
    old_classes = {row["class"]: row for row in baseline.get("seg_per_class", [])}
    for row in current.get("seg_per_class", []):
        if "seg_class:" + row["class"] in failed:
            reference = old_classes[row["class"]]
            print(f"Seg/{row['class']} IoU={number(reference['iou'])}->{number(row['iou'])} "
                  f"teacher_pixels={row['teacher_pixels']}")
    scales = saved["task_loss_scales_train_only"]
    weights = saved["training_config"]["loss"]["task_weights"]
    raw = history[-1]["mean_train_losses"]
    print("last_epoch_mean_loss task raw scale weighted_normalized")
    for task in ("agent", "map", "seg", "depth"):
        print(f"{task} {raw[task]:.6f} {scales[task]:.6f} {raw[task]*weights[task]/scales[task]:.6f}")
    if saved.get("stage") != "shared_scene_four_task_v1":
        print("above_base_normalization_excludes_feature_KD_and_GradNorm_weights=True")
    if "mean_weighted_task_losses" in history[-1]:
        print("actual_mean_weighted_task_losses=" + str(history[-1]["mean_weighted_task_losses"]))
    if saved.get("gradnorm_state_dict"):
        print("final_gradnorm_weights=" + str(saved["gradnorm_state_dict"]["weights"].tolist()))
    print("passing_epochs=" + str([row["epoch"] for row in history if row["baseline_gate"]["all_pass"]]))
    print("gradient_cosines_are_in_training_log_not_checkpoint=True; macroF1_is_custom_not_official_mAP=True")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/quest_unified_scene_v1.pt"))
    args = parser.parse_args()
    import torch

    path = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    saved = torch.load(path, map_location="cpu", weights_only=True)
    review(saved)


if __name__ == "__main__":
    main()
