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
from quest.seg_depth_training import load_dense_model, validate_dense_checkpoint
from scripts.train_seg_depth import evaluate, load_context, make_dataset, resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/quest_seg_depth_v1.pt")
    args = parser.parse_args()
    checkpoint = load_checkpoint_cpu(resolve(args.checkpoint))
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
    print(json.dumps(evaluate(model, dataset, config, device), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
