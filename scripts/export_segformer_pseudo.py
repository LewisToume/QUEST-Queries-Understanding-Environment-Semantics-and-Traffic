"""Export real eight-view SegFormer Cityscapes19 predictions; no QUEST model."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import pickle
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quest.frame_selection import select_indices  # noqa: E402
from quest.cuda_runtime import initialize_cuda_device  # noqa: E402
from quest.openscene_dataset import OPENSCENE_CAMERA_NAMES, resolve_openscene_camera_path  # noqa: E402
from quest.seg_teacher import (  # noqa: E402
    CITYSCAPES_CLASSES, CITYSCAPES_PALETTE, SEG_PIXEL_ALIGNMENT, SEG_PREPROCESS_VERSION,
    SEG_SCHEMA_VERSION, logits_to_seg_label, prepare_segformer_input,
    sliding_window_logits, validate_seg_record, validate_teacher_config,
)
from quest.stage3_split import file_sha256, load_stage3_split  # noqa: E402


def local_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def build_teacher(bundle, device):
    # Lazy import: preflight never loads teacher/QUEST/DINO models or reaches CUDA.
    from transformers import SegformerForSemanticSegmentation

    use_safetensors = (bundle / "model.safetensors").is_file()
    model, loading = SegformerForSemanticSegmentation.from_pretrained(
        str(bundle), local_files_only=True, output_loading_info=True, use_safetensors=use_safetensors,
    )
    errors = {key: loading.get(key, []) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
    print("teacher_checkpoint_load=" + json.dumps(errors, default=str))
    if any(errors.values()):
        raise ValueError("Seg teacher checkpoint mismatch; random/missing heads cannot export labels")
    model.requires_grad_(False)
    return model.to(device).eval()


def inspect_cameras(info, camera_root, output_hw):
    if not info.get("token") or not info.get("scene_token") or not np.isfinite(float(info["timestamp"])):
        raise ValueError("Require current sample token, scene and finite timestamp")
    cameras = []
    for name in OPENSCENE_CAMERA_NAMES:
        path = resolve_openscene_camera_path(info["cams"][name]["data_path"], camera_root)
        with Image.open(path) as image:
            width, height = image.size
            if image.format != "JPEG":
                raise ValueError("Expected original OpenScene JPEG: " + str(path))
        if min(height, width) <= 0 or not np.isclose(width / height, output_hw[1] / output_hw[0], rtol=1e-3):
            raise ValueError("Student grid would distort image aspect ratio: " + name)
        cameras.append({"name": name, "path": path, "source_hw": [height, width]})
    return cameras


def infer_frame(model, cameras, output_hw, device, save_probabilities):
    labels, confidence, valid, probabilities, rgbs, resize_meta = [], [], [], [], [], []
    for camera in cameras:
        with Image.open(camera["path"]) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        images, resize = prepare_segformer_input(rgb)
        logits = sliding_window_logits(model, images, device)
        result = logits_to_seg_label(logits, camera["source_hw"], output_hw)
        labels.append(result["labels"])
        confidence.append(result["confidence"])
        valid.append(result["valid_mask"])
        if save_probabilities:
            probabilities.append(result["probabilities"].to(torch.float16))
        rgbs.append(Image.fromarray(rgb).resize((output_hw[1], output_hw[0]), Image.Resampling.BILINEAR))
        resize_meta.append(resize)
    seg = {"labels": torch.stack(labels), "confidence": torch.stack(confidence), "valid_mask": torch.stack(valid)}
    if save_probabilities:
        seg["probabilities"] = torch.stack(probabilities)
    return seg, rgbs, resize_meta


def preview_image(rgbs, labels):
    height, width = labels.shape[-2:]
    tile_w, tile_h = width * 2 + 8, height + 24
    canvas = Image.new("RGB", (tile_w * 2, tile_h * 4 + 40), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), "Original RGB | Cityscapes19 prediction; candidate labels, NOT verified GT", fill="black")
    draw.text((4, 20), "No undistortion. No legacy six-class remapping. Same pixel coordinates.", fill="black")
    palette = np.asarray(CITYSCAPES_PALETTE, dtype=np.uint8)
    for index, name in enumerate(OPENSCENE_CAMERA_NAMES):
        left, top = index % 2 * tile_w, index // 2 * tile_h + 40
        draw.text((left + 4, top + 4), name, fill="black")
        canvas.paste(rgbs[index], (left, top + 24))
        canvas.paste(Image.fromarray(palette[labels[index].numpy()]), (left + width + 4, top + 24))
    return canvas


def save_record_exclusive(record, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(record, temporary)
        os.link(temporary, path)  # Atomic publish; fail rather than overwrite.
    finally:
        temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seg_teacher.yaml")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--split-manifest")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--teacher-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--audit-dir")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true", help="Save only PNG/JSON, no PT labels")
    parser.add_argument("--save-probabilities", action="store_true", default=None)
    args = parser.parse_args(argv)
    with local_path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    teacher, dataset, export = config["teacher"], config["dataset"], config["export"]
    if (teacher["name"] != "SegFormer-B5" or teacher["native_classes"] != 19
            or teacher["model_id"] != "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"):
        raise ValueError("Only SegFormer-B5 Cityscapes19 is implemented")
    bundle = local_path(args.teacher_dir or teacher["local_dir"])
    metadata_path, camera_root = local_path(dataset["metadata_path"]), local_path(dataset["camera_root"])
    output_hw = tuple(dataset["image_size"])
    if len(output_hw) != 2 or min(output_hw) <= 0 or any(not isinstance(value, int) for value in output_hw):
        raise ValueError("image_size must be positive integer [H,W]")
    output_dir, audit_dir = local_path(args.output_dir or export["output_dir"]), local_path(args.audit_dir or export["audit_dir"])
    log_interval = int(export["log_interval"])
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")
    save_probabilities = bool(export["save_probabilities"]) if args.save_probabilities is None else args.save_probabilities
    with metadata_path.open("rb") as stream:
        metadata = pickle.load(stream)  # Trusted local metadata only, not downloaded pickles.
    infos = metadata.get("infos") if isinstance(metadata, Mapping) else None
    if not isinstance(infos, list):
        raise ValueError("Expected OpenScene metadata['infos']")
    manifest = load_stage3_split(local_path(args.split_manifest), infos, metadata_path) if args.split_manifest else None
    indices = select_indices(infos, args.sample_index, args.num_frames, args.indices, manifest, args.split)
    if args.diagnostic_only and len(indices) > 3:
        raise ValueError("Diagnostic-only mode is limited to 3 frames")
    weight = bundle / "model.safetensors"
    if not weight.is_file():
        weight = bundle / "pytorch_model.bin"
    required = (bundle / "config.json", bundle / "preprocessor_config.json", weight)
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        for path in missing:
            print("MISSING " + path)
        return 2
    with (bundle / "config.json").open(encoding="utf-8") as stream:
        validate_teacher_config(json.load(stream))
    for index in indices:
        token = str(infos[index]["token"])
        inspect_cameras(infos[index], camera_root, output_hw)
        paths = [audit_dir / (token + ".png"), audit_dir / (token + ".json")]
        if not args.diagnostic_only:
            paths.append(output_dir / (token + ".pt"))
        if any(path.exists() for path in paths):
            raise FileExistsError("Outputs exist for " + token + "; choose a new output/audit directory")
    print("SEG_PREFLIGHT=PASS frames={} cameras=8 native_classes=19 no_student_model=True".format(len(indices)))
    if args.preflight_only:
        print("runtime=NOT_CHECKED GPU=False outputs_written=False")
        return 0
    device = initialize_cuda_device()
    model = build_teacher(bundle, device)
    provenance = {"teacher": teacher["name"], "model_id": teacher["model_id"],
                  "checkpoint_sha256": file_sha256(weight), "config_sha256": file_sha256(bundle / "config.json"),
                  "preprocessor_config_sha256": file_sha256(bundle / "preprocessor_config.json"),
                  "adapter_sha256": file_sha256(ROOT / "quest/seg_teacher.py"),
                  "export_config_sha256": file_sha256(local_path(args.config)),
                  "hf_processor_used": False,
                  "preprocessing": SEG_PREPROCESS_VERSION, "crop_hw": [1024, 1024], "stride_hw": [768, 768],
                  "scale_wh": [2048, 1024], "checkpoint_load": "strict_loading_report", "confidence_calibrated": False,
                  "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "Pillow")}}
    metadata_hash = file_sha256(metadata_path)
    total_counts = torch.zeros(19, dtype=torch.int64)
    for offset, index in enumerate(indices):
        info = infos[index]
        token = str(info["token"])
        cameras = inspect_cameras(info, camera_root, output_hw)
        seg, rgbs, resize = infer_frame(model, cameras, output_hw, device, save_probabilities)
        record = {"schema_version": SEG_SCHEMA_VERSION, "token": token, "source_index": index,
                  "scene_token": str(info["scene_token"]), "timestamp": float(info["timestamp"]),
                  "camera_names": list(OPENSCENE_CAMERA_NAMES), "class_names": list(CITYSCAPES_CLASSES),
                  "label_space": "cityscapes19", "pixel_alignment": SEG_PIXEL_ALIGNMENT,
                  "source_image_hw": [camera["source_hw"] for camera in cameras], "output_hw": list(output_hw),
                  "metadata_sha256": metadata_hash, "split_sha256": manifest["split_sha256"] if manifest else None,
                  "adaptation_verified": False, "provenance": provenance, "seg": seg}
        validate_seg_record(record, token, index, OPENSCENE_CAMERA_NAMES, output_hw)
        counts = torch.bincount(seg["labels"].flatten().long(), minlength=19)
        total_counts += counts
        audit_dir.mkdir(parents=True, exist_ok=True)
        with (audit_dir / (token + ".png")).open("xb") as stream:
            preview_image(rgbs, seg["labels"]).save(stream, format="PNG")
        summary = {key: value for key, value in record.items() if key != "seg"}
        summary.update({"labels_shape": list(seg["labels"].shape), "resize": resize,
                        "class_pixel_counts": dict(zip(CITYSCAPES_CLASSES, counts.tolist())),
                        "mean_confidence_uncalibrated": float(seg["confidence"].float().mean())})
        with (audit_dir / (token + ".json")).open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
        if not args.diagnostic_only:
            save_record_exclusive(record, output_dir / (token + ".pt"))
        if offset == 0 or (offset + 1) % log_interval == 0 or offset + 1 == len(indices):
            print("frame={}/{} source_index={} token={} labels={} saved_pt={}".format(
                offset + 1, len(indices), index, token, tuple(seg["labels"].shape), not args.diagnostic_only))
    print("class_pixel_counts=" + json.dumps(dict(zip(CITYSCAPES_CLASSES, total_counts.tolist()))))
    print("SEG_EXPORT=PASS frames={} adaptation_verified=False peak_allocated_MiB={:.1f}".format(
        len(indices), torch.cuda.max_memory_allocated(device) / 1024 ** 2))
    print("preview_dir=" + str(audit_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
