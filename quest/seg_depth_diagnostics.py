"""Read-only dense teacher-agreement breakdowns and fixed-scale previews."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .seg_depth_losses import (depth_valid_mask, empty_dense_metrics,
                               summarize_dense_metrics, update_dense_metrics)
from .seg_teacher import CITYSCAPES_CLASSES, CITYSCAPES_PALETTE


def class_agreement(confusion):
    """Rows=teacher, columns=student. Empty classes are N/A, not perfect."""
    rows = []
    for index, name in enumerate(CITYSCAPES_CLASSES):
        truth, predicted, matched = int(confusion[index].sum()), int(confusion[:, index].sum()), int(confusion[index, index])
        union = truth + predicted - matched
        rows.append({"id": index, "class": name, "teacher_pixels": truth,
                     "student_pixels": predicted, "matched_pixels": matched,
                     "precision": matched / predicted if predicted else None,
                     "recall": matched / truth if truth else None,
                     "iou": matched / union if union else None})
    return rows


def depth_ranges(settings):
    minimum, maximum = float(settings["min_depth_m"]), float(settings["max_depth_m"])
    if not (math.isfinite(minimum) and math.isfinite(maximum) and 0 < minimum < maximum):
        raise ValueError("Invalid depth diagnostic range")
    edges = [minimum, *(value for value in (10., 20., 30., 40., 50.) if minimum < value < maximum), maximum]
    return [(left, right, index == len(edges) - 2)
            for index, (left, right) in enumerate(zip(edges, edges[1:]))]


def empty_depth_statistics():
    return {"pixels": 0, "absrel_sum": 0., "squared_sum": 0., "absolute_sum": 0.,
            "signed_sum": 0., "delta1_sum": 0, "teacher_sum": 0., "student_sum": 0., "log_ratio_sum": 0.}


def add_depth_statistics(statistics, truth, predicted):
    truth, predicted = truth.double(), predicted.double()
    if truth.numel() != predicted.numel() or not bool(
            torch.isfinite(truth).all() & torch.isfinite(predicted).all()
            & (truth > 0).all() & (predicted > 0).all()):
        raise ValueError("Depth diagnostic values must be finite, positive and aligned")
    difference = predicted - truth
    statistics["pixels"] += truth.numel()
    statistics["absrel_sum"] += float((difference.abs() / truth).sum())
    statistics["absolute_sum"] += float(difference.abs().sum())
    statistics["squared_sum"] += float(difference.square().sum())
    statistics["signed_sum"] += float(difference.sum())
    statistics["delta1_sum"] += int((torch.maximum(predicted / truth, truth / predicted) < 1.25).sum())
    statistics["teacher_sum"] += float(truth.sum())
    statistics["student_sum"] += float(predicted.sum())
    statistics["log_ratio_sum"] += float((predicted.log() - truth.log()).sum())


def summarize_depth(statistics):
    count = statistics["pixels"]
    return {"pixels": count,
            "absrel": statistics["absrel_sum"] / count if count else None,
            "rmse_m": math.sqrt(statistics["squared_sum"] / count) if count else None,
            "mae_m": statistics["absolute_sum"] / count if count else None,
            "signed_bias_m": statistics["signed_sum"] / count if count else None,
            "delta1": statistics["delta1_sum"] / count if count else None,
            "teacher_mean_m": statistics["teacher_sum"] / count if count else None,
            "student_mean_m": statistics["student_sum"] / count if count else None,
            "geometric_mean_student_teacher_ratio": math.exp(statistics["log_ratio_sum"] / count) if count else None}


def slice_target(target, camera):
    return {task: ({key: value[camera:camera + 1] if torch.is_tensor(value) else value
                    for key, value in payload.items()} if payload is not None else None)
            for task, payload in target.items()}


def detailed_summary(metrics):
    return {**summarize_dense_metrics(metrics), "seg_pixels": int(metrics["confusion"].sum()),
            "depth_pixels": metrics["depth_pixels"], "seg_per_class": class_agreement(metrics["confusion"])}


class DenseEvaluationDiagnostics:
    def __init__(self, camera_names, settings, tasks):
        self.camera_names, self.settings, self.tasks = tuple(camera_names), settings, tuple(tasks)
        self.overall = empty_dense_metrics()
        self.cameras = {name: empty_dense_metrics() for name in self.camera_names}
        self.scenes, self.scene_frames, self.frames, self.seen_tokens = {}, {}, [], set()
        self.ranges = depth_ranges(settings["depth"])
        self.bin_statistics = [empty_depth_statistics() for _ in self.ranges]
        self.depth_statistics = empty_depth_statistics()
        self.depth_mask_counts = {"total_pixels": 0, "export_valid_pixels": 0,
                                  "excluded_by_range_or_nonfinite": 0, "excluded_by_sky_or_seg_validity": 0,
                                  "evaluated_pixels": 0}

    @torch.no_grad()
    def update(self, predictions, targets, infos, source_indices, tokens):
        if not (len(targets) == len(infos) == len(source_indices) == len(tokens) == predictions["depth"].shape[0]):
            raise ValueError("Diagnostic batch metadata mismatch")
        if predictions["depth"].shape[1] != len(self.camera_names):
            raise ValueError("Diagnostic camera count mismatch")
        for info, index, token in zip(infos, source_indices, tokens):
            if str(info["token"]) != token or token in self.seen_tokens:
                raise ValueError("Diagnostic token join mismatch or duplicate frame")
            self.seen_tokens.add(token)
            scene = str(info["scene_token"])
            self.frames.append({"source_index": index, "token": token, "scene_token": scene})
        update_dense_metrics(self.overall, predictions, targets, self.settings, self.tasks)
        for index, (target, info) in enumerate(zip(targets, infos)):
            scene = str(info["scene_token"])
            scene_metrics = self.scenes.setdefault(scene, empty_dense_metrics())
            self.scene_frames[scene] = self.scene_frames.get(scene, 0) + 1
            sample = {key: predictions[key][index:index + 1] for key in ("seg_logits", "depth") if key in predictions}
            update_dense_metrics(scene_metrics, sample, [target], self.settings, self.tasks)
            for camera, name in enumerate(self.camera_names):
                camera_prediction = {key: value[:, camera:camera + 1] for key, value in sample.items()}
                update_dense_metrics(self.cameras[name], camera_prediction, [slice_target(target, camera)],
                                     self.settings, self.tasks)
            depth = target.get("depth")
            if "depth" not in self.tasks or depth is None:
                continue
            valid = depth_valid_mask(depth, target.get("seg"), self.settings["depth"])
            truth = depth["depth_m"][valid]
            predicted = predictions["depth"][index].detach().cpu()[valid]
            add_depth_statistics(self.depth_statistics, truth, predicted)
            for (left, right, inclusive), statistics in zip(self.ranges, self.bin_statistics):
                selected = (truth >= left) & ((truth <= right) if inclusive else (truth < right))
                add_depth_statistics(statistics, truth[selected], predicted[selected])
            exported = depth["valid_mask"]
            range_valid = (exported & torch.isfinite(depth["depth_m"])
                           & (depth["depth_m"] >= self.settings["depth"]["min_depth_m"])
                           & (depth["depth_m"] <= self.settings["depth"]["max_depth_m"]))
            self.depth_mask_counts["total_pixels"] += exported.numel()
            self.depth_mask_counts["export_valid_pixels"] += int(exported.sum())
            self.depth_mask_counts["excluded_by_range_or_nonfinite"] += int((exported & ~range_valid).sum())
            self.depth_mask_counts["excluded_by_sky_or_seg_validity"] += int((range_valid & ~valid).sum())
            self.depth_mask_counts["evaluated_pixels"] += int(valid.sum())

    def summarize(self):
        if sum(item["pixels"] for item in self.bin_statistics) != self.overall["depth_pixels"]:
            raise RuntimeError("Depth distance bins do not cover the exact evaluated pixel set")
        return {"overall": summarize_dense_metrics(self.overall),
                "seg_per_class": class_agreement(self.overall["confusion"]),
                "seg_confusion_matrix": self.overall["confusion"].tolist(),
                "seg_confusion_axes": {"rows": "teacher", "columns": "student"},
                "depth_overall": summarize_depth(self.depth_statistics),
                "depth_distance_bins": [{"min_m": left, "max_m": right, "upper_inclusive": inclusive,
                                         **summarize_depth(statistics)}
                                        for (left, right, inclusive), statistics in zip(self.ranges, self.bin_statistics)],
                "depth_mask_counts": dict(self.depth_mask_counts),
                "per_camera": {name: detailed_summary(value) for name, value in self.cameras.items()},
                "per_scene": {name: {"frames": self.scene_frames[name], **detailed_summary(value)}
                              for name, value in self.scenes.items()},
                "frames": list(self.frames),
                "metric_conventions": {"seg_mIoU": "Mean over classes with nonzero teacher/student union; not frequency-weighted",
                                       "depth_distance": "Teacher camera Z, not student distance or LiDAR radial range",
                                       "depth_bins": "Lower inclusive; upper exclusive except final bin",
                                       "depth_bias": "Student minus teacher; no scale or median alignment",
                                       "aggregation": "Pixel-weighted, not a claim of independent-frame sampling"}}


def preview_positions(infos, count):
    if count < 0:
        raise ValueError("Preview count must be nonnegative")
    if count == 0 or not infos:
        return []
    selected, scenes = [], set()
    for index, info in enumerate(infos):
        scene = str(info["scene_token"])
        if scene not in scenes:
            selected.append(index)
            scenes.add(scene)
            if len(selected) == count:
                return selected
    for index in np.linspace(0, len(infos) - 1, min(len(infos), count), dtype=int).tolist() + list(range(len(infos))):
        if index not in selected:
            selected.append(index)
        if len(selected) >= count:
            break
    return sorted(selected)


def rgb_from_normalized(image):
    mean = image.new_tensor([.485, .456, .406])[:, None, None]
    std = image.new_tensor([.229, .224, .225])[:, None, None]
    return ((image.detach().cpu() * std.cpu() + mean.cpu()).clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def color_depth(depth, valid, minimum, maximum):
    values = np.clip((np.log(np.maximum(depth, minimum)) - math.log(minimum))
                     / (math.log(maximum) - math.log(minimum)), 0, 1)
    color = np.stack((values, values ** .6, 1 - values), -1)
    color[~valid] = 0
    return (color * 255).astype(np.uint8)


def save_dense_previews(output_dir, source_index, token, scene, images, predictions, target,
                        camera_names, settings, tasks):
    """Common teacher/student mask and fixed scales; never per-image normalization."""
    output_dir = Path(output_dir)
    tile_w, tile_h, row_h, header_h = 336, 189, 215, 72
    canvases = {}
    for task in tasks:
        if target.get(task) is None:
            continue
        canvas = Image.new("RGB", (tile_w * 8, header_h + row_h * math.ceil(len(camera_names) / 2)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 4), f"index={source_index} token={token} scene={scene}", fill="black")
        draw.text((8, 20), "OFFLINE TEACHER PSEUDO, NOT REAL GT; same pixel grid; black=excluded", fill="black")
        description = ("RGB | teacher Cityscapes19 | QUEST Cityscapes19 | disagreement red / agreement gray"
                       if task == "seg" else
                       f"RGB | teacher camera-Z | QUEST camera-Z | absolute relative error (blue=0, red>=0.5); depth log scale {settings['depth']['min_depth_m']:g}-{settings['depth']['max_depth_m']:g}m")
        draw.text((8, 36), description, fill="black")
        canvases[task] = canvas
    predicted_seg = predictions["seg_logits"].detach().argmax(1).cpu().numpy() if "seg" in canvases else None
    predicted_depth = predictions["depth"].detach().cpu().numpy()[:, 0]
    palette = np.asarray(CITYSCAPES_PALETTE, dtype=np.uint8)
    for camera, name in enumerate(camera_names):
        rgb = rgb_from_normalized(images[camera])
        for task, canvas in canvases.items():
            draw = ImageDraw.Draw(canvas)
            x, y = (camera % 2) * tile_w * 4, header_h + (camera // 2) * row_h
            draw.text((x + 4, y + 4), name, fill="black")
            if task == "seg":
                labels = target["seg"]["labels"][camera].cpu().numpy()
                valid = target["seg"]["valid_mask"][camera].cpu().numpy()
                teacher = palette[np.where(valid, labels, 0)].copy()
                student = palette[predicted_seg[camera]].copy()
                teacher[~valid], student[~valid] = 0, 0
                disagreement = np.repeat(rgb.mean(-1, keepdims=True).astype(np.uint8), 3, -1)
                disagreement[valid & (labels != predicted_seg[camera])] = (255, 30, 30)
                disagreement[~valid] = 0
            else:
                valid = depth_valid_mask(target["depth"], target.get("seg"), settings["depth"])[camera, 0].numpy()
                truth = target["depth"]["depth_m"][camera, 0].numpy()
                minimum, maximum = settings["depth"]["min_depth_m"], settings["depth"]["max_depth_m"]
                teacher = color_depth(truth, valid, minimum, maximum)
                student = color_depth(predicted_depth[camera], valid, minimum, maximum)
                relative = np.clip(np.abs(predicted_depth[camera] - truth) / np.maximum(truth, minimum) / .5, 0, 1)
                disagreement = (np.stack((relative, np.zeros_like(relative), 1 - relative), -1) * 255).astype(np.uint8)
                disagreement[~valid] = 0
            for panel, value in enumerate((rgb, teacher, student, disagreement)):
                method = Image.Resampling.NEAREST if task == "seg" and panel > 0 else Image.Resampling.BILINEAR
                canvas.paste(Image.fromarray(value).resize((tile_w, tile_h), method), (x + panel * tile_w, y + 24))
    paths = []
    for task, canvas in canvases.items():
        path = output_dir / f"validation_{source_index}_{task}.png"
        with path.open("xb") as stream:
            canvas.save(stream, format="PNG")
        paths.append(path.name)
    return paths
