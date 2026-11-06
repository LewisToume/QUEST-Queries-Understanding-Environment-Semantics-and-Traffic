"""Training-only frozen pre-integration QUEST guides; not original specialists."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .seg_depth_losses import depth_valid_mask


FEATURE_KD_STAGE = "shared_scene_feature_distillation_v1"
GUIDE_ROLE = "FROZEN_PREINTEGRATION_QUEST_PROXY_NOT_ORIGINAL_TEACHERS"


class FrozenDenseGuide(nn.Module):
    def __init__(self, combined_baseline):
        super().__init__()
        self.seg_branch = combined_baseline.seg_branch
        self.depth_frontend = combined_baseline.base.dense_frontend
        if hasattr(self.depth_frontend, "agent_temporal"):
            self.depth_frontend.agent_temporal = None
        self.seg_channels = self.seg_branch.decode_head.classifier.in_channels
        self.depth_channels = self.depth_frontend.options["dpt_features"]
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, levels, previous_levels, inputs, grid_size):
        captured = {}

        def capture(name):
            def hook(module, args):
                value = args[0]
                if value.ndim != 4 or not bool(torch.isfinite(value).all()):
                    raise RuntimeError("Invalid frozen-guide feature: " + name)
                captured[name] = F.adaptive_avg_pool2d(value.float(), grid_size).detach()
            return hook

        handles = [self.seg_branch.decode_head.classifier.register_forward_pre_hook(capture("seg")),
                   self.depth_frontend.depth_decoder.output.register_forward_pre_hook(capture("depth"))]
        try:
            with torch.autocast(device_type=inputs["images"].device.type, dtype=torch.bfloat16,
                                enabled=inputs["images"].device.type == "cuda"):
                self.seg_branch(levels, inputs["images"])
                previous = {**inputs["history"], "feature_levels": previous_levels}
                self.depth_frontend(levels, inputs["intrinsics"], inputs["extrinsics"], inputs["distortion"],
                    tuple(inputs["images"].shape[-2:]), history=previous)
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != {"seg", "depth"}:
            raise RuntimeError("Frozen guide feature hooks did not run")
        return captured


class SharedFeatureDistiller(nn.Module):
    """Two training adapters constrain ONE common dense tensor, never inference."""

    def __init__(self, student_channels, seg_channels, depth_channels):
        super().__init__()
        if student_channels != depth_channels:
            raise ValueError("Depth retention requires aligned shared/guide channels, no learned escape adapter")
        self.semantic_adapter = nn.Conv2d(student_channels, seg_channels, 1, bias=False)

    @staticmethod
    def distance(student, teacher, mask):
        if student.shape != teacher.shape or mask.shape != (student.shape[0], 1, *student.shape[-2:]):
            raise ValueError("Feature distillation channel/grid/mask mismatch")
        if not all(bool(torch.isfinite(x).all()) for x in (student, teacher)) or not bool(mask.any()):
            raise ValueError("Invalid or empty masked feature supervision")
        # Channel RMS normalization prevents arbitrary feature magnitude from setting the loss scale.
        a, b = student.float(), teacher.detach().float()
        a = a / a.square().mean(1, keepdim=True).add(1e-6).sqrt()
        b = b / b.square().mean(1, keepdim=True).add(1e-6).sqrt()
        cosine = 1 - F.cosine_similarity(a, b, dim=1, eps=1e-6)
        l1 = (a - b).abs().mean(1)
        return (cosine + l1)[mask[:, 0]].mean()

    def forward(self, shared, guide_features, targets, depth_settings, options):
        grid = tuple(options["feature_grid"])
        if (len(grid) != 2 or any(not isinstance(x, int) or x <= 0 for x in grid)
                or not math.isfinite(options["min_valid_fraction"]) or not 0 < options["min_valid_fraction"] <= 1):
            raise ValueError("Invalid feature grid/validity fraction")
        if shared.ndim != 4 or shared.shape[0] != sum(row["seg"]["labels"].shape[0] for row in targets):
            raise ValueError("Shared feature batch/camera order differs from targets")
        semantic = torch.cat([row["seg"]["valid_mask"].unsqueeze(1) for row in targets]).to(shared.device)
        geometry = torch.cat([depth_valid_mask(row["depth"], row.get("seg"), depth_settings) for row in targets]).to(shared.device)
        masks = {name: F.adaptive_avg_pool2d(value.float(), grid) >= options["min_valid_fraction"]
                 for name, value in (("seg", semantic), ("depth", geometry))}
        pooled = F.adaptive_avg_pool2d(shared.float(), grid)
        projected = self.semantic_adapter(pooled)
        return {"seg_feature_kd": self.distance(projected, guide_features["seg"], masks["seg"]),
                "depth_feature_kd": self.distance(pooled, guide_features["depth"], masks["depth"])}


def restore_student(parent, model):
    if parent.get("stage") not in (FEATURE_KD_STAGE, "shared_scene_four_task_v1"):
        raise ValueError("Feature KD requires a trained shared-scene student")
    if set(parent.get("trained_tasks", [])) != {"agent", "map", "seg", "depth"}:
        raise ValueError("Feature KD cannot silently start from a single-task ablation")
    if parent["model_config"] != model.model_config or parent["profile"] != model.profile:
        raise ValueError("Feature KD must preserve the exact trained student architecture")
    model.load_state_dict(parent["model_state_dict"], strict=True)


def deployment_payload(model, parent, epoch, validation, gate):
    """Only the original student architecture/state; guides/adapters cannot be loaded."""
    return {"stage": "shared_scene_four_task_v1", "model_config": model.model_config, "profile": model.profile,
            "model_state_dict": model.state_dict(), "trained_tasks": list(parent["trained_tasks"]),
            "epoch": epoch, "experiment": FEATURE_KD_STAGE, "validation": validation, "baseline_gate": gate,
            "source_epoch": parent.get("epoch", 0), "total_student_epochs": parent.get("epoch", 0) + epoch,
            "depth_geometry_policy": "experimental_image_plane", "inference_guides": 0, "inference_adapters": 0}


def load_deployment(payload, backbone=None):
    from .unified_scene import UNIFIED_SCENE_VERSION, UnifiedQUEST

    if (payload.get("stage") != UNIFIED_SCENE_VERSION or payload.get("experiment") != FEATURE_KD_STAGE
            or payload.get("inference_guides") != 0 or payload.get("inference_adapters") != 0):
        raise ValueError("Not a standalone feature-KD deployment checkpoint")
    model = UnifiedQUEST(payload["model_config"], payload["profile"], backbone=backbone)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model
