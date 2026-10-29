"""Explicit dense-head warm-start and versioned four-task checkpoints."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import torch

from .dense_perception import SHARED_IMAGE_FRONTEND_VERSION
from .model import QUESTModel, load_quest_v3_checkpoint
from .seg_teacher import CITYSCAPES_CLASSES


DENSE_STAGE = "seg_depth_offline"
DENSE_CHECKPOINT_VERSION = 1


def resolve_seg_class_balance(loss_settings, train_audit, *, saved_weights_required=False):
    """Bounded inverse-square-root weights from actual training pixels only."""
    semantic = loss_settings.get("seg", {})
    balance = semantic.get("class_balance")
    if balance is None:
        return None
    if balance.get("mode") != "inverse_sqrt_train_pixels":
        raise ValueError("Unknown Seg class balance recipe")
    lower, upper = float(balance["min_weight"]), float(balance["max_weight"])
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower <= 1 <= upper):
        raise ValueError("Invalid Seg class balance limits")
    counts = torch.as_tensor(train_audit.get("seg_class_pixel_counts", []))
    if (counts.shape != (19,) or counts.dtype not in (torch.int32, torch.int64)
            or bool((counts < 0).any()) or int(counts.sum()) <= 0
            or int(counts.sum()) != train_audit["tasks"]["seg"]["valid_pixels"]):
        raise ValueError("Missing/inconsistent full training Seg pixel frequencies")
    positive = counts[counts > 0].double()
    median = positive.median()
    values = (median / counts.double().clamp_min(1)).sqrt().clamp(lower, upper).tolist()
    if saved_weights_required and (semantic.get("class_weights") != values
                                   or semantic.get("class_names") != list(CITYSCAPES_CLASSES)):
        raise ValueError("Saved Seg class weights differ from audited training frequencies")
    semantic["class_weights"] = values
    semantic["class_names"] = list(CITYSCAPES_CLASSES)
    return {"counts": counts.tolist(), "class_weights": values,
            "labels_sha256": train_audit["labels_sha256"], "source": "TRAIN_ONLY"}


def validate_dense_data_join(checkpoint, manifest, audits, model_config, initialization, *, smoke=False):
    """Fine-tuning may add derived counts, never change source labels or joins."""
    validate_dense_checkpoint(checkpoint)
    if (checkpoint["split_sha256"] != manifest["split_sha256"]
            or checkpoint["metadata_sha256"] != manifest["metadata_sha256"]
            or checkpoint["initialization"] != initialization or checkpoint["model_config"] != model_config):
        raise ValueError("Fine-tune source/config/split changed")
    if set(audits) != ({"train"} if smoke else {"train", "validation"}):
        raise ValueError("Fine-tune audits do not cover the required frame lists")
    for split, audit in audits.items():
        previous = dict(checkpoint["label_audits"][split])
        current = dict(audit)
        previous.pop("seg_class_pixel_counts", None)
        current.pop("seg_class_pixel_counts", None)
        if previous != current:
            raise ValueError("Fine-tuning cannot change label hashes/tokens/teacher: " + split)


def validate_dense_config(config, tasks):
    if config.get("stage") != DENSE_STAGE or not tasks or set(tasks) - {"seg", "depth"} or len(set(tasks)) != len(tasks):
        raise ValueError("Invalid dense training stage/tasks")
    frontend, train = config["frontend"], config["train"]
    if (frontend["version"] != SHARED_IMAGE_FRONTEND_VERSION
            or frontend["preserve_spatial_features"] is not True
            or tuple(frontend["output_size"]) != (252, 448)):
        raise ValueError("Dense profile must preserve the existing spatial branch and 252x448 pixels")
    for name in ("epochs", "num_samples", "batch_size", "log_interval"):
        if not isinstance(train[name], int) or train[name] <= 0:
            raise ValueError("Expected positive integer train." + name)
    if train["num_workers"] != 0:
        raise ValueError("Initial audited dense pipeline uses num_workers=0")
    for name in ("lr", "gradient_clip_norm", "weight_decay"):
        if not math.isfinite(float(train[name])) or float(train[name]) < (0 if name == "weight_decay" else 1e-12):
            raise ValueError("Invalid train." + name)
    weights = config["loss"]["weights"]
    required = {"seg_ce", "depth_log", "depth_relative", "depth_gradient"}
    if set(weights) != required or any(not math.isfinite(float(v)) or float(v) < 0 for v in weights.values()):
        raise ValueError("Invalid dense loss weights")
    if "seg" in tasks and weights["seg_ce"] <= 0:
        raise ValueError("Enabled Seg needs a positive loss weight")
    if "depth" in tasks and sum(weights[key] for key in required - {"seg_ce"}) <= 0:
        raise ValueError("Enabled Depth needs a positive loss weight")
    if config["loss"].get("seg", {}).get("class_balance") is not None and "seg" not in tasks:
        raise ValueError("Seg class balance requires the Seg task")


def configure_dense_training(model, tasks):
    if model.dense_frontend is None:
        raise ValueError("Dense training requires a versioned frontend")
    model.requires_grad_(False)
    model.eval()
    dense = model.dense_frontend
    for name in ("lateral", "refine"):
        getattr(dense, name).requires_grad_(True)
    if "seg" in tasks:
        dense.seg_decoder.requires_grad_(True)
    if "depth" in tasks:
        dense.camera_condition.requires_grad_(True)
        dense.depth_decoder.requires_grad_(True)
    # Neither the unused BEV adapter nor any existing spatial module is trained.
    dense.train()
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def frozen_spatial_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith("dense_frontend."):
            continue
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_dense_checkpoint(checkpoint):
    if (checkpoint.get("stage") != DENSE_STAGE
            or checkpoint.get("dense_checkpoint_version") != DENSE_CHECKPOINT_VERSION
            or checkpoint.get("architecture_version") != 3
            or checkpoint.get("spatial_preservation_verified") is not True
            or checkpoint.get("bev_use_ego_state") is not False):
        raise ValueError("Not a compatible/preserved Seg/Depth QUEST checkpoint")
    if checkpoint.get("dense_frontend") != checkpoint.get("model_config", {}).get("dense_frontend"):
        raise ValueError("Checkpoint frontend/model profile mismatch")
    support = checkpoint.get("trained_class_support_mask")
    if not torch.is_tensor(support) or not torch.equal(support.cpu().bool(), torch.tensor([True, True, False, False])):
        raise ValueError("Dense checkpoint Agent support mask mismatch")
    if checkpoint.get("depth_geometry_policy") not in ("verified", "experimental_image_plane"):
        raise ValueError("Checkpoint depth geometry policy missing")
    if (checkpoint["depth_geometry_policy"] != checkpoint.get("training_config", {}).get("depth_geometry_policy")
            or checkpoint.get("trained_dense_tasks") != checkpoint.get("training_config", {}).get("tasks")):
        raise ValueError("Checkpoint trained dense tasks/geometry metadata inconsistent")


def load_dense_model(checkpoint, model_factory=QUESTModel):
    validate_dense_checkpoint(checkpoint)
    model = model_factory(**checkpoint["model_config"])
    load_quest_v3_checkpoint(model, checkpoint)
    if frozen_spatial_fingerprint(model) != checkpoint["frozen_spatial_sha256"]:
        raise ValueError("Checkpoint frozen Agent/Map checksum mismatch")
    return model


def save_dense_checkpoint(path, model, optimizer, scheduler, epoch, config, model_config,
                          manifest, audits, baseline_hash, initialization, loader_generator=None):
    if frozen_spatial_fingerprint(model) != baseline_hash:
        raise RuntimeError("Agent/Map/backbone changed during dense training; checkpoint refused")
    payload = {
        "stage": DENSE_STAGE, "dense_checkpoint_version": DENSE_CHECKPOINT_VERSION,
        "architecture_version": 3, "dense_frontend": model.dense_frontend_config,
        "model_config": model_config, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "loader_rng_state": loader_generator.get_state() if loader_generator is not None else None,
        "epoch": epoch, "training_config": config, "initialization": initialization,
        "split_sha256": manifest["split_sha256"], "metadata_sha256": manifest["metadata_sha256"],
        "label_audits": audits, "frozen_spatial_sha256": baseline_hash,
        "spatial_preservation_verified": True, "bev_use_ego_state": False,
        "depth_geometry_policy": config["depth_geometry_policy"],
        "trained_dense_tasks": list(config["tasks"]),
        "trained_class_support_mask": torch.tensor([True, True, False, False]),
        "reference": "OFFLINE_TEACHER_PSEUDO_NOT_REAL_GT",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    created = False
    try:
        with temporary.open("xb") as stream:
            created = True
            torch.save(payload, stream)
        os.replace(temporary, path)
    finally:
        if created and temporary.exists():
            temporary.unlink()
