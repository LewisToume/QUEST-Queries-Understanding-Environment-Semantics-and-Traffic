"""Explicit dense-head warm-start and versioned four-task checkpoints."""

from __future__ import annotations

import hashlib
import math
import os
import random
from pathlib import Path

import torch

from .dense_perception import SHARED_IMAGE_FRONTEND_VERSION
from .geometry_depth import (GEOMETRY_DEPTH_VERSION, GEOMETRY_DEPTH_VERSIONS,
                             TEMPORAL_DEPTH_VERSION, validate_geometry_depth_options)
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
    if (frontend["version"] not in (SHARED_IMAGE_FRONTEND_VERSION, *GEOMETRY_DEPTH_VERSIONS)
            or frontend["preserve_spatial_features"] is not True
            or tuple(frontend["output_size"]) != (252, 448)):
        raise ValueError("Dense profile must preserve the existing spatial branch and 252x448 pixels")
    if frontend["version"] in GEOMETRY_DEPTH_VERSIONS:
        validate_geometry_depth_options(frontend)
        if list(tasks) != ["depth"] or config["loss"]["weights"]["seg_ce"] != 0:
            raise ValueError("Geometry Depth V2 is Depth-only; no untrained Seg output is exported")
        if config["loss"]["depth"]["exclude_sky"] or train["missing_labels"] != "error":
            raise ValueError("Depth V2 requires actual Depth targets and no unqualified Seg dependency")
        if config["loss"]["depth"]["max_depth_m"] != frontend["ray_max_depth_m"]:
            raise ValueError("Depth V2 target and geometry ranges differ")
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
    agent = config.get("temporal_agent", {})
    enabled = agent.get("enabled", False)
    profile_agent = frontend.get("temporal", {}).get("agent_bev")
    if bool(enabled) != (profile_agent is not None):
        raise ValueError("Temporal Agent training and model profile must be explicitly enabled together")
    if enabled and (frontend["version"] != TEMPORAL_DEPTH_VERSION or tasks != ["depth"]
            or not math.isfinite(agent["loss_weight"]) or agent["loss_weight"] <= 0):
        raise ValueError("Joint temporal Agent/Depth requires V3 and a positive Agent objective")


def configure_dense_training(model, tasks, *, train_temporal_agent=False):
    if model.dense_frontend is None:
        raise ValueError("Dense training requires a versioned frontend")
    model.requires_grad_(False)
    model.eval()
    dense = model.dense_frontend
    if model.dense_frontend_config["version"] in GEOMETRY_DEPTH_VERSIONS:
        if list(tasks) != ["depth"]:
            raise ValueError("Geometry Depth V2 only trains Depth")
        dense.requires_grad_(True)
        dense.train()
        agent = getattr(dense, "agent_temporal", None)
        if agent is not None and not train_temporal_agent:
            agent.requires_grad_(False)
            agent.eval()
        return [parameter for parameter in dense.parameters() if parameter.requires_grad]
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
    profile = checkpoint.get("dense_frontend") or {}
    expected_version = {GEOMETRY_DEPTH_VERSION: 2, TEMPORAL_DEPTH_VERSION: 3}.get(
        profile.get("version"), DENSE_CHECKPOINT_VERSION)
    if (checkpoint.get("stage") != DENSE_STAGE
            or checkpoint.get("dense_checkpoint_version") != expected_version
            or checkpoint.get("architecture_version") != 3
            or checkpoint.get("spatial_preservation_verified") is not True
            or checkpoint.get("bev_use_ego_state") is not False):
        raise ValueError("Not a compatible/preserved Seg/Depth QUEST checkpoint")
    if checkpoint.get("dense_frontend") != checkpoint.get("model_config", {}).get("dense_frontend"):
        raise ValueError("Checkpoint frontend/model profile mismatch")
    if profile.get("version") in GEOMETRY_DEPTH_VERSIONS:
        if profile != checkpoint.get("training_config", {}).get("frontend"):
            raise ValueError("Checkpoint frontend/training profile mismatch")
        validate_dense_config(checkpoint["training_config"], checkpoint["trained_dense_tasks"])
    if bool(checkpoint.get("trained_temporal_agent", False)) != bool(
            checkpoint.get("training_config", {}).get("temporal_agent", {}).get("enabled", False)):
        raise ValueError("Checkpoint temporal Agent training declaration mismatch")
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


def validate_depth_warm_start(source, config, model_config, manifest, audits, initialization, *, smoke=False):
    """Only the explicit V2 -> V2/control or V2 -> temporal-V3 migration is allowed."""
    validate_dense_checkpoint(source)
    if (source["dense_frontend"]["version"] != GEOMETRY_DEPTH_VERSION
            or config["frontend"]["version"] not in GEOMETRY_DEPTH_VERSIONS
            or source["trained_dense_tasks"] != ["depth"] or config["tasks"] != ["depth"]):
        raise ValueError("Depth warm-start requires the trained, Depth-only V2 baseline")
    base_profile = dict(config["frontend"])
    base_profile["version"] = GEOMETRY_DEPTH_VERSION
    base_profile.pop("temporal", None)
    expected_model = dict(model_config, dense_frontend=base_profile)
    if (expected_model != source["model_config"] or config["loss"] != source["training_config"]["loss"]
            or source["initialization"] != initialization
            or source["depth_geometry_policy"] != config["depth_geometry_policy"]
            or source["split_sha256"] != manifest["split_sha256"]
            or source["metadata_sha256"] != manifest["metadata_sha256"]):
        raise ValueError("Depth warm-start cannot change spatial model, baseline loss, split or geometry policy")
    for split, audit in audits.items():
        current = {key: value for key, value in audit.items() if key not in ("temporal_context", "agent_temporal_labels")}
        old = source["label_audits"][split]
        if smoke and split == "train":
            # Partial smoke never saves; the full training run must match every label hash.
            if (current["frames"] > old["frames"] or current["teacher_signatures"] != old["teacher_signatures"]):
                raise ValueError("Smoke teacher differs from Depth baseline")
        elif current != old:
            raise ValueError("Depth warm-start label hashes/frame list differ: " + split)


def load_depth_warm_start(model, source):
    validate_dense_checkpoint(source)
    if source["dense_frontend"]["version"] != GEOMETRY_DEPTH_VERSION:
        raise ValueError("Only the single-frame V2 Depth source can initialize this migration")
    if hasattr(model.dense_frontend, "temporal_blocks") and any(
            bool((block.gate.detach() != 0).any()) for block in model.dense_frontend.temporal_blocks):
        raise ValueError("Temporal migration must start with zero residual gates")
    agent = getattr(model.dense_frontend, "agent_temporal", None)
    if agent is not None and bool((agent.gates.detach() != 0).any()):
        raise ValueError("Agent temporal migration must start with zero residual gates")
    expected, state = model.state_dict(), source["model_state_dict"]
    additions = set(expected) - set(state)
    if set(state) - set(expected) or any(not name.startswith(("dense_frontend.temporal_blocks.",
                                                             "dense_frontend.agent_temporal.")) for name in additions):
        raise ValueError("Unexpected Depth warm-start state keys; no automatic partial load")
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != additions or result.unexpected_keys:
        raise ValueError("Unexpected Depth migration keys")
    if frozen_spatial_fingerprint(model) != source["frozen_spatial_sha256"]:
        raise ValueError("Depth warm-start changed frozen Agent/Map/backbone")
    return len(additions)


def save_dense_checkpoint(path, model, optimizer, scheduler, epoch, config, model_config,
                          manifest, audits, baseline_hash, initialization, loader_generator=None,
                          validation_history=None):
    if frozen_spatial_fingerprint(model) != baseline_hash:
        raise RuntimeError("Agent/Map/backbone changed during dense training; checkpoint refused")
    payload = {
        "stage": DENSE_STAGE, "dense_checkpoint_version": {
            GEOMETRY_DEPTH_VERSION: 2, TEMPORAL_DEPTH_VERSION: 3}.get(
                model.dense_frontend_config["version"], DENSE_CHECKPOINT_VERSION),
        "architecture_version": 3, "dense_frontend": model.dense_frontend_config,
        "model_config": model_config, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state_all": (torch.cuda.get_rng_state_all()
                               if next(model.parameters()).device.type == "cuda" else None),
        "loader_rng_state": loader_generator.get_state() if loader_generator is not None else None,
        "epoch": epoch, "training_config": config, "initialization": initialization,
        "split_sha256": manifest["split_sha256"], "metadata_sha256": manifest["metadata_sha256"],
        "label_audits": audits, "frozen_spatial_sha256": baseline_hash,
        "spatial_preservation_verified": True, "bev_use_ego_state": False,
        "depth_geometry_policy": config["depth_geometry_policy"],
        "trained_dense_tasks": list(config["tasks"]),
        "trained_temporal_agent": bool(config.get("temporal_agent", {}).get("enabled", False)),
        "trained_class_support_mask": torch.tensor([True, True, False, False]),
        "reference": "OFFLINE_TEACHER_PSEUDO_NOT_REAL_GT",
        "validation_history": list(validation_history or []),
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
