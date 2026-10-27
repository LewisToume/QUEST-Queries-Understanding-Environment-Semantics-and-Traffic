"""Native Cityscapes19 teacher adaptation, without importing QUEST models."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F


SEG_SCHEMA_VERSION = 1
SEG_PIXEL_ALIGNMENT = "original_openscene_jpeg_resized_v1"
SEG_PREPROCESS_VERSION = "segformer_b5_keep_ratio_slide_v1"
CITYSCAPES_CLASSES = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
)
CITYSCAPES_PALETTE = (
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
)


def validate_teacher_config(config):
    names = tuple(config.get("id2label", {}).get(str(index)) for index in range(19))
    if (config.get("model_type") != "segformer" or names != CITYSCAPES_CLASSES
            or config.get("depths") != [3, 6, 40, 3]
            or config.get("hidden_sizes") != [64, 128, 320, 512]
            or config.get("decoder_hidden_size") != 768
            or len(config.get("id2label", {})) != 19):
        raise ValueError("Require the official SegFormer-B5 Cityscapes19 config/weight pair")


def prepare_segformer_input(rgb, scale_wh=(2048, 1024)):
    import cv2

    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) <= 0:
        raise ValueError("Expected nonempty original uint8 RGB image")
    height, width = rgb.shape[:2]
    # MMCV imrescale uses long/short-edge limits and round(value + 0.5).
    ratio = min(max(scale_wh) / max(height, width), min(scale_wh) / min(height, width))
    target_h, target_w = int(height * ratio + 0.5), int(width * ratio + 0.5)
    resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float()
    tensor = (tensor - tensor.new_tensor([123.675, 116.28, 103.53])[:, None, None]) / tensor.new_tensor(
        [58.395, 57.12, 57.375])[:, None, None]
    return tensor.unsqueeze(0), {"source_hw": [height, width], "resized_hw": [target_h, target_w],
                                 "resize_ratio": ratio, "keep_ratio": True}


def sliding_windows(height, width, crop_hw=(1024, 1024), stride_hw=(768, 768)):
    crop_h, crop_w = crop_hw
    stride_h, stride_w = stride_hw
    if min(height, width, crop_h, crop_w, stride_h, stride_w) <= 0:
        raise ValueError("Image, crop and stride dimensions must be positive")
    if stride_h > crop_h or stride_w > crop_w:
        raise ValueError("Sliding windows must not leave uncovered gaps")
    rows = max(height - crop_h + stride_h - 1, 0) // stride_h + 1
    cols = max(width - crop_w + stride_w - 1, 0) // stride_w + 1
    for row in range(rows):
        for col in range(cols):
            bottom, right = min(row * stride_h + crop_h, height), min(col * stride_w + crop_w, width)
            yield max(bottom - crop_h, 0), bottom, max(right - crop_w, 0), right


@torch.no_grad()
def sliding_window_logits(model, images, device, crop_hw=(1024, 1024), stride_hw=(768, 768)):
    if images.ndim != 4 or tuple(images.shape[:2]) != (1, 3):
        raise ValueError("One camera [1,3,H,W] is processed at a time")
    height, width = images.shape[-2:]
    # CPU accumulation bounds GPU memory independently of the original JPEG width.
    total = torch.zeros(1, 19, height, width, dtype=torch.float32)
    count = torch.zeros(1, 1, height, width, dtype=torch.float32)
    for top, bottom, left, right in sliding_windows(height, width, crop_hw, stride_hw):
        output = model(pixel_values=images[..., top:bottom, left:right].to(device))
        logits = output.logits
        if logits.ndim != 4 or tuple(logits.shape[:2]) != (1, 19) or not bool(torch.isfinite(logits).all()):
            raise ValueError("Native SegFormer logits must be finite [1,19,H,W]")
        resized = F.interpolate(logits.float(), (bottom - top, right - left), mode="bilinear",
                                align_corners=False).cpu()
        total[..., top:bottom, left:right] += resized
        count[..., top:bottom, left:right] += 1
    if not bool((count > 0).all()):
        raise RuntimeError("Sliding-window inference left uncovered pixels")
    return total / count


def logits_to_seg_label(logits, source_hw, output_hw):
    if logits.ndim != 4 or tuple(logits.shape[:2]) != (1, 19):
        raise ValueError("Seg teacher uses 19 native classes, not legacy QUEST C_seg=6")
    if (len(source_hw) != 2 or len(output_hw) != 2
            or not bool(torch.isfinite(logits).all()) or min(*source_hw, *output_hw) <= 0):
        raise ValueError("Invalid teacher logits or image dimensions")
    # Follow native rescale to original pixels before mapping onto the student grid.
    original = F.interpolate(logits.float(), tuple(source_hw), mode="bilinear", align_corners=False)
    resized = F.interpolate(original, tuple(output_hw), mode="bilinear", align_corners=False)
    probabilities = resized.softmax(1)[0]
    confidence, labels = probabilities.max(0)
    return {"labels": labels.to(torch.uint8), "confidence": confidence.to(torch.float16),
            "valid_mask": torch.ones_like(labels, dtype=torch.bool), "probabilities": probabilities}


def validate_seg_record(record, token, source_index, camera_names, output_hw):
    if not isinstance(record, Mapping) or record.get("schema_version") != SEG_SCHEMA_VERSION:
        raise ValueError("Unsupported Seg label schema")
    if record.get("token") != token or record.get("source_index") != source_index:
        raise ValueError("Seg token/original source index mismatch")
    if tuple(record.get("camera_names", ())) != tuple(camera_names):
        raise ValueError("Seg camera order mismatch")
    if (record.get("label_space") != "cityscapes19" or record.get("pixel_alignment") != SEG_PIXEL_ALIGNMENT
            or tuple(record.get("class_names", ())) != CITYSCAPES_CLASSES):
        raise ValueError("Seg taxonomy/pixel alignment mismatch")
    shape = (len(camera_names), *output_hw)
    seg = record.get("seg", {})
    if not isinstance(seg, Mapping):
        raise ValueError("Seg payload must be a mapping")
    for key in ("labels", "confidence", "valid_mask"):
        if not torch.is_tensor(seg.get(key)) or tuple(seg[key].shape) != shape:
            raise ValueError("Seg field shape mismatch: " + key)
    if seg["labels"].dtype != torch.uint8 or seg["valid_mask"].dtype != torch.bool:
        raise ValueError("Seg labels must be uint8 and validity bool")
    if not bool(((seg["labels"] < 19) | ~seg["valid_mask"]).all()):
        raise ValueError("Native Seg class ID out of range")
    confidence = seg["confidence"]
    if not confidence.is_floating_point() or not bool((torch.isfinite(confidence) & (confidence >= 0)
                                                      & (confidence <= 1)).all()):
        raise ValueError("Invalid Seg confidence; no calibrated-probability claim is made")
    if "probabilities" in seg:
        probability = seg["probabilities"]
        if (not torch.is_tensor(probability) or tuple(probability.shape) != (len(camera_names), 19, *output_hw)
                or not probability.is_floating_point() or not bool(torch.isfinite(probability).all())
                or not bool(((probability >= 0) & (probability <= 1)).all())
                or not torch.allclose(probability.float().sum(1),
                                      torch.ones_like(probability[:, 0], dtype=torch.float32), atol=2e-3, rtol=0)):
            raise ValueError("Invalid optional Seg class probabilities")
