from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def load_manifest(data_root: Path) -> list[dict]:
    manifest_path = data_root / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def colorize_label(label: np.ndarray) -> Image.Image:
    # Deterministic lightweight palette for quick inspection.
    label_int = label.astype(np.int32)
    rgb = np.zeros((*label_int.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (label_int * 37) % 255
    rgb[..., 1] = (label_int * 67) % 255
    rgb[..., 2] = (label_int * 97) % 255
    rgb[label_int == 0] = np.array([0, 0, 0], dtype=np.uint8)
    return Image.fromarray(rgb)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SegFormer pseudo segmentation labels for OpenScene samples.")
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "data" / "openscene_first_test_100"),
        help="OpenScene first-test folder containing sample_xxx directories and manifest.json.",
    )
    parser.add_argument(
        "--model-id",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="Hugging Face SegFormer checkpoint.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples to process.")
    parser.add_argument("--save-probs", action="store_true", help="Save softmax probabilities as seg_probs.npy.")
    parser.add_argument(
        "--output-subdir",
        default="segformer_pseudo",
        help="Subdirectory created under each sample directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    device = resolve_device(args.device)
    manifest = load_manifest(data_root)
    if args.limit is not None:
        manifest = manifest[: args.limit]

    print("=" * 72)
    print("SegFormer pseudo-label generation")
    print(f"data_root : {data_root}")
    print(f"samples   : {len(manifest)}")
    print(f"model_id  : {args.model_id}")
    print(f"device    : {device}")
    print("=" * 72)

    processor = SegformerImageProcessor.from_pretrained(args.model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_id).to(device)
    model.eval()

    for item in tqdm(manifest, desc="segformer"):
        sample_dir = data_root / item["sample_id"]
        image_path = sample_dir / "cam_f0.jpg"
        out_dir = sample_dir / args.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = F.interpolate(
                outputs.logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )

        pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        np.save(out_dir / "seg_pred.npy", pred)
        Image.fromarray(pred).save(out_dir / "seg_pred.png")
        colorize_label(pred).save(out_dir / "seg_pred_color.png")

        if args.save_probs:
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float16)
            np.save(out_dir / "seg_probs.npy", probs)

    print("=" * 72)
    print(f"saved pseudo labels under each sample's {args.output_subdir}/ directory")
    print("=" * 72)


if __name__ == "__main__":
    main()
