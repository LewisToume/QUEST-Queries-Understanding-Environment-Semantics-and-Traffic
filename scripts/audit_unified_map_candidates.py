"""Run the fixed-reference Unified V1 validation once and explain Map misses."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from quest.agent_training import load_checkpoint_cpu
from quest.cuda_runtime import initialize_cuda_device
from quest.map_diagnostics import map_candidate_audit
from quest.map_training import denormalize_points, load_vector_capacity_audit
from quest.stage3_split import file_sha256
from quest.unified_scene import UNIFIED_SCENE_VERSION, UnifiedQUEST
from quest.utils import load_yaml_config
from scripts.train_seg_depth import load_context, resolve
from scripts.train_unified_scene import build_data, forward, loader


@torch.inference_mode()
def evaluate(model, data, device, distance_m, confidence):
    x0, x1 = model.geometry_lift.x_range
    y0, y1 = model.geometry_lift.y_range
    totals = {}
    for batch in loader(data):
        output = forward(model, batch, device)
        scores, labels = output["map_cls_logits"][0].softmax(-1).max(-1)
        points = denormalize_points(output["map_points"][0].cpu(), (x0, y0, x1, y1))
        rows = map_candidate_audit(labels.cpu(), scores.cpu(), points, batch["vector_targets"][0],
                                   distance_m=distance_m, confidence=confidence)
        for name, row in rows.items():
            total = totals.setdefault(name, dict.fromkeys(row, 0))
            for key, count in row.items():
                total[key] += count
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/quest_unified_scene_v1.pt")
    args = parser.parse_args()

    saved = load_checkpoint_cpu(resolve(args.checkpoint))
    if (saved.get("stage") != UNIFIED_SCENE_VERSION
            or saved["profile"].get("refine_map_references", False)
            or set(saved.get("trained_tasks", ())) != {"agent", "map", "seg", "depth"}):
        raise ValueError("Use the full, fixed-reference four-task Unified V1 training checkpoint")
    config = saved["training_config"]
    source_path = resolve(config["paths"]["baseline"])
    if file_sha256(source_path) != saved["source_sha256"]:
        raise ValueError("Baseline provenance changed")
    metadata = load_checkpoint_cpu(source_path)["base_metadata"]
    data_config, manifest = load_context(config)
    if any(saved[key] != manifest[key] for key in ("metadata_sha256", "split_sha256")):
        raise ValueError("Validation split or metadata changed")
    stage3 = load_yaml_config(resolve(config["paths"]["stage3_config"]))
    queries, provenance = load_vector_capacity_audit(resolve(config["paths"]["capacity_audit"]), stage3, manifest)
    if queries != saved["profile"]["map_queries"]:
        raise ValueError("Certified Map query capacity changed")
    data, audit = build_data(config, metadata, data_config, manifest, "validation", provenance)
    if audit != saved["label_audits"]["validation"]:
        raise ValueError("Validation targets or context changed")
    model = UnifiedQUEST(saved["model_config"], saved["profile"])
    model.load_state_dict(saved["model_state_dict"], strict=True)
    previous = saved["validation_history"][-1]["validation"]["map"]
    distance_m = config["eval"]["map_match_distance_m"]
    confidence = config["eval"]["map_confidence_threshold"]
    del saved
    device = initialize_cuda_device()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Original unified evaluation requires CUDA bf16")
    rows = evaluate(model.to(device).eval(), data, device, distance_m, confidence)
    print(f"map_candidate_audit frames={len(data)} ordered_m={distance_m:g} "
          f"confidence={confidence:g} training=False reference=VECTOR_GT")
    print("class gt pred matched no_geom bg_only wrong_class low_score conflict chamfer_near ordered_2x_near")
    for name, row in rows.items():
        print(name + " " + " ".join(str(value) for value in row.values()))
    matches_stored = all((row["gt"], row["predictions"], row["matched"])
                         == (previous[name]["gt_count"], previous[name]["predictions"], previous[name]["matched"])
                         for name, row in rows.items())
    print(f"stored_validation_counts_match={matches_stored}")
    return 0 if matches_stored else 1


if __name__ == "__main__":
    raise SystemExit(main())
