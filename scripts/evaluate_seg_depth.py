"""Evaluate dense Student agreement with held-out offline pseudo labels (not GT)."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quest.agent_training import load_checkpoint_cpu
from quest.cuda_runtime import initialize_cuda_device
from quest.seg_depth_diagnostics import DenseEvaluationDiagnostics, preview_positions, save_dense_previews
from quest.seg_depth_losses import compute_seg_depth_loss
from quest.seg_depth_training import load_dense_model, validate_dense_checkpoint
from quest.stage3_split import file_sha256
from scripts.train_seg_depth import dense_predictions, load_context, loader, make_dataset, resolve


def display(value):
    return "N/A" if value is None else f"{value:.4f}"


def print_summary(report):
    summary = {**report["overall"], "mean_batch_loss": report["mean_batch_loss"],
               "frames": len(report["frames"]), "depth_geometry_policy": report["depth_geometry_policy"]}
    print(json.dumps(summary, indent=2, allow_nan=False))
    print("seg_class teacher_pixels student_pixels IoU precision recall")
    for row in report["seg_per_class"]:
        print(f"{row['class'].replace(' ', '_')} {row['teacher_pixels']} {row['student_pixels']} "
              f"{display(row['iou'])} {display(row['precision'])} {display(row['recall'])}")
    print("depth_teacher_Z_bin pixels AbsRel RMSE_m delta1 signed_bias_m")
    for row in report["depth_distance_bins"]:
        print(f"[{row['min_m']:g},{row['max_m']:g}{']' if row['upper_inclusive'] else ')'} "
              f"{row['pixels']} {display(row['absrel'])} {display(row['rmse_m'])} "
              f"{display(row['delta1'])} {display(row['signed_bias_m'])}")
    for scene, metrics in report["per_scene"].items():
        print(f"scene={scene} frames={metrics['frames']} Seg_mIoU={display(metrics['seg_teacher_mIoU'])} "
              f"Depth_AbsRel={display(metrics['depth_teacher_absrel'])}")
    print("full_per_camera_confusion_and_mask_counts=report.json; reference=TEACHER_PSEUDO_NOT_REAL_GT")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/quest_seg_depth_v1.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/diagnostics/seg_depth_review_v1",
                        help="New/empty report directory; existing reports/previews are never overwritten")
    parser.add_argument("--preview-frames", type=int, default=3,
                        help="Prefer one frame per scene, then deterministic spread; 0 disables previews")
    parser.add_argument("--log-interval", type=int, default=40)
    args = parser.parse_args()
    if args.preview_frames < 0 or args.log_interval <= 0:
        parser.error("preview-frames must be nonnegative and log-interval positive")
    output_dir = resolve(args.output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Choose a new/empty --output-dir; will not overwrite " + str(output_dir))
    checkpoint_path = resolve(args.checkpoint)
    checkpoint_hash = file_sha256(checkpoint_path)
    checkpoint = load_checkpoint_cpu(checkpoint_path)
    validate_dense_checkpoint(checkpoint)
    config = checkpoint["training_config"]
    dataset_config, manifest = load_context(config)
    if (manifest["split_sha256"] != checkpoint["split_sha256"]
            or manifest["metadata_sha256"] != checkpoint["metadata_sha256"]):
        raise ValueError("Evaluation split/metadata differs from checkpoint")
    dataset, audit = make_dataset(config, dataset_config, manifest, "validation")
    if audit != checkpoint["label_audits"]["validation"]:
        raise ValueError("Held-out teacher labels changed since training")
    if not torch.cuda.is_available():
        raise RuntimeError("Server CUDA required")
    device = initialize_cuda_device()
    model = load_dense_model(checkpoint).to(device).eval()
    diagnostics = DenseEvaluationDiagnostics(dataset.images.camera_names, config["loss"], config["tasks"])
    lookup = dict(zip(dataset.source_indices, dataset.images.infos))
    preview_indices = {dataset.source_indices[position] for position in preview_positions(
        dataset.images.infos, args.preview_frames)}
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_sum, batches, files, evaluated = 0., 0, [], 0
    with torch.no_grad():
        for batch in loader(dataset, config):
            output = dense_predictions(model, batch, device)
            losses = compute_seg_depth_loss(output, batch["dense_targets"], config["loss"], config["tasks"])
            if not bool(torch.isfinite(losses["total_loss"])):
                raise RuntimeError("Nonfinite evaluation loss; no completed report will be produced")
            infos = [lookup[index] for index in batch["source_index"]]
            diagnostics.update(output, batch["dense_targets"], infos, batch["source_index"], batch["sample_token"])
            loss_sum += float(losses["total_loss"])
            batches += 1
            for position, index in enumerate(batch["source_index"]):
                if index in preview_indices:
                    files.extend(save_dense_previews(
                        output_dir, index, batch["sample_token"][position], str(infos[position]["scene_token"]),
                        batch["images"][position], {key: output[key][position] for key in ("seg_logits", "depth")},
                        batch["dense_targets"][position], dataset.images.camera_names, config["loss"], config["tasks"]))
            evaluated += len(infos)
            if evaluated % args.log_interval == 0 or evaluated == len(dataset):
                print(f"evaluated={evaluated}/{len(dataset)}", flush=True)
    report = {"report_version": 1, "evaluation_complete": True, **diagnostics.summarize(),
              "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
              "split": "validation", "scope": "FULL_VALIDATION",
              "epoch": int(checkpoint["epoch"]), "trained_dense_tasks": checkpoint["trained_dense_tasks"],
              "split_sha256": checkpoint["split_sha256"], "metadata_sha256": checkpoint["metadata_sha256"],
              "validation_labels_sha256": audit["labels_sha256"],
              "teacher_signatures": audit["teacher_signatures"],
              "frozen_spatial_sha256": checkpoint["frozen_spatial_sha256"],
              "frozen_spatial_checkpoint_verified": True,
              "mean_batch_loss": loss_sum / max(batches, 1),
              "depth_geometry_policy": config["depth_geometry_policy"], "preview_files": files,
              "notes": ["No weights/labels/geometry were changed; no teacher inference or training",
                        "Geometry remains UNVERIFIED for experimental_image_plane checkpoints",
                        "Preview depth panels share the evaluation mask and fixed log scale; black pixels are excluded",
                        f"Teacher agreement is not real GT accuracy; {evaluated} frames cover {len(diagnostics.scenes)} temporally correlated scenes"]}
    if (evaluated != len(dataset)
            or [row["source_index"] for row in report["frames"]] != dataset.source_indices):
        raise RuntimeError("Incomplete validation evaluation")
    if file_sha256(checkpoint_path) != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during evaluation; refusing a mixed-version report")
    # Serialize first so a nonfinite metric cannot leave a misleading completed report.
    payload = json.dumps(report, indent=2, allow_nan=False)
    with (output_dir / "report.json").open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print_summary(report)
    print(f"report={output_dir / 'report.json'} previews={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
