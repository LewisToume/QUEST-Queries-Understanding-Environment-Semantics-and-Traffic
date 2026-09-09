from __future__ import annotations

import contextlib
import copy
import importlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.dataset import collate_fn
from quest.losses import compute_total_loss
from quest.model import QUESTModel
from quest.openscene_dataset import OpenSceneMetadataDataset
from quest.utils import load_yaml_config


EXPECTED_INPUT_SHAPES = {
    "images": (1, 8, 3, 224, 224),
    "intrinsics": (1, 8, 3, 3),
    "extrinsics": (1, 8, 4, 4),
    "ego_state": (1, 9),
}
EXPECTED_OUTPUT_SHAPES = {
    "agent_cls_logits": (1, 100, 5),
    "agent_boxes": (1, 100, 8),
    "agent_velocity": (1, 100, 3),
    "map_cls_logits": (1, 50, 5),
    "map_points": (1, 50, 20, 2),
    "occ_logits": (1, 11, 200, 200, 16),
    "flow_logits": (1, 2, 200, 200, 16),
}


def module_version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "imported (version unavailable)"))


def tensor_stats(tensor: torch.Tensor) -> dict[str, float | bool]:
    value = tensor.detach().float()
    return {
        "finite": bool(torch.isfinite(value).all().item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
    }


def move_gt_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "agent_gt": {
            "labels": batch["agent_gt"]["labels"].to(device),
            "boxes": batch["agent_gt"]["boxes"].to(device),
            "velocity": batch["agent_gt"]["velocity"].to(device),
        },
        "map_gt": {
            "labels": batch["map_gt"]["labels"].to(device),
            "points": batch["map_gt"]["points"].to(device),
        },
        "map_valid": batch["map_gt"]["valid"].to(device),
        "occ_gt": batch["occ_gt"].to(device),
        "occ_valid": batch["occ_valid"].to(device),
        "flow_gt": batch["flow_gt"].to(device),
        "flow_valid": batch["flow_valid"].to(device),
        "flow_mask": batch["flow_mask"].to(device),
    }


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def forward_model(model: QUESTModel, tensors: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    with autocast_context(device):
        outputs = model(
            tensors["images"],
            intrinsics=tensors["intrinsics"],
            extrinsics=tensors["extrinsics"],
            ego_state=tensors["ego_state"],
        )
    return {name: value.float() for name, value in outputs.items()}


def check_intrinsics_against_source(
    dataset: OpenSceneMetadataDataset,
    batch: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    metadata = dataset.infos[0]

    details: list[dict[str, Any]] = []
    all_match = True
    for camera_index, camera_name in enumerate(dataset.camera_names):
        image_path = dataset._camera_metadata_path(metadata, camera_name)
        with Image.open(image_path) as image:
            source_width, source_height = image.size
        source = torch.as_tensor(metadata["cams"][camera_name]["cam_intrinsic"], dtype=torch.float32).clone()
        source[0, :] *= dataset.image_size[1] / source_width
        source[1, :] *= dataset.image_size[0] / source_height
        actual = batch["intrinsics"][0, camera_index]
        matches = bool(torch.allclose(actual, source, rtol=1e-5, atol=1e-4))
        all_match = all_match and matches
        details.append(
            {
                "camera": camera_name,
                "source_size": [source_width, source_height],
                "fx": float(actual[0, 0].item()),
                "fy": float(actual[1, 1].item()),
                "cx": float(actual[0, 2].item()),
                "cy": float(actual[1, 2].item()),
                "matches_actual_resize": matches,
            }
        )
    return all_match, details


def print_step(step: int, losses: dict[str, torch.Tensor]) -> None:
    print(
        f"step={step:02d} total={losses['total_loss'].item():.6f} "
        f"agent={losses['agent_loss'].item():.6f} "
        f"occ={losses['occ_loss'].item():.6f} "
        f"map={losses['map_loss'].item():.6f} "
        f"flow={losses['flow_loss'].item():.6f}"
    )


def run() -> tuple[dict[str, Any], list[str]]:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    blockers: list[str] = []
    report: dict[str, Any] = {}

    dependency_versions = {
        "python": sys.version.split()[0],
        "torch": module_version("torch"),
        "torchvision": module_version("torchvision"),
        "transformers": module_version("transformers"),
        "scipy": module_version("scipy"),
        "numpy": module_version("numpy"),
        "pillow": module_version("PIL"),
        "yaml": module_version("yaml"),
    }
    report["dependencies"] = dependency_versions
    print("DEPENDENCIES", json.dumps(dependency_versions, ensure_ascii=True))

    model_config = load_yaml_config(PROJECT_ROOT / "configs" / "model.yaml")["model"]
    stage_config = load_yaml_config(PROJECT_ROOT / "configs" / "stage1.yaml")
    dataset_config = dict(stage_config["dataset"])
    dataset_config.pop("C_agent", None)
    dataset_config.setdefault("P", model_config["P"])
    for path_key in ("metadata_path", "camera_root", "occupancy_root"):
        path = Path(dataset_config[path_key])
        if not path.is_absolute():
            dataset_config[path_key] = str(PROJECT_ROOT / path)
    dataset = OpenSceneMetadataDataset(max_samples=1, **dataset_config)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)))

    input_stats: dict[str, Any] = {}
    for name, expected_shape in EXPECTED_INPUT_SHAPES.items():
        value = batch[name]
        stats = tensor_stats(value)
        stats["shape"] = list(value.shape)
        input_stats[name] = stats
        if tuple(value.shape) != expected_shape:
            blockers.append(f"{name} shape {tuple(value.shape)} != {expected_shape}")
        if not stats["finite"]:
            blockers.append(f"{name} contains NaN or Inf")
    extrinsics_nonempty = bool((batch["extrinsics"].abs().sum(dim=(-1, -2)) > 0).all().item())
    if not extrinsics_nonempty:
        blockers.append("one or more extrinsics are empty")
    rotation_determinants = torch.linalg.det(batch["extrinsics"][..., :3, :3])
    extrinsics_nonsingular = bool((rotation_determinants.abs() > 1e-5).all().item())
    if not extrinsics_nonsingular:
        blockers.append("one or more extrinsic rotations are singular")
    intrinsics_match, intrinsic_details = check_intrinsics_against_source(dataset, batch)
    if not intrinsics_match:
        blockers.append("resized intrinsics do not match source image dimensions")
    intrinsics_reasonable = all(
        detail["fx"] > 0.0
        and detail["fy"] > 0.0
        and 0.0 <= detail["cx"] <= dataset.image_size[1]
        and 0.0 <= detail["cy"] <= dataset.image_size[0]
        for detail in intrinsic_details
    )
    if not intrinsics_reasonable:
        blockers.append("resized intrinsic focal lengths or principal points are unreasonable")
    report["inputs"] = input_stats
    report["intrinsics"] = intrinsic_details
    report["extrinsics_nonempty"] = extrinsics_nonempty
    report["extrinsics_nonsingular"] = extrinsics_nonsingular
    print("INPUT_SHAPES", {name: tuple(batch[name].shape) for name in EXPECTED_INPUT_SHAPES})
    print("INTRINSICS", intrinsic_details)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report["device"] = str(device)
    tensors = {name: batch[name].to(device) for name in EXPECTED_INPUT_SHAPES}
    gts = move_gt_to_device(batch, device)
    model = QUESTModel(**model_config).to(device).train()

    backbone_frozen = all(not parameter.requires_grad for parameter in model.backbone.parameters())
    backbone_eval = not model.backbone.model.training
    non_backbone_trainable = all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    )
    if not backbone_frozen:
        blockers.append("DINOv2 has trainable parameters")
    if not backbone_eval:
        blockers.append("DINOv2 is not in eval mode during QUEST training")
    if not non_backbone_trainable:
        blockers.append("one or more non-backbone QUEST parameters are frozen")
    report["requires_grad"] = {
        "backbone_frozen": backbone_frozen,
        "backbone_eval": backbone_eval,
        "other_modules_trainable": non_backbone_trainable,
    }

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen_params = total_params - trainable_params
    report["parameters"] = {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
    }
    print("PARAMETERS", report["parameters"])

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(stage_config["train"]["lr"]),
        weight_decay=float(stage_config["train"].get("weight_decay", 1e-4)),
    )
    tracked_parameter = model.fusion.token_proj[0].weight
    tracked_before = tracked_parameter.detach().clone()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        memory_baseline = torch.cuda.memory_allocated(device)
    else:
        memory_baseline = 0

    outputs = forward_model(model, tensors, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        forward_peak = torch.cuda.max_memory_allocated(device)
    else:
        forward_peak = 0

    output_stats: dict[str, Any] = {}
    for name, expected_shape in EXPECTED_OUTPUT_SHAPES.items():
        value = outputs[name]
        stats = tensor_stats(value)
        stats["shape"] = list(value.shape)
        output_stats[name] = stats
        if tuple(value.shape) != expected_shape:
            blockers.append(f"{name} shape {tuple(value.shape)} != {expected_shape}")
        if not stats["finite"]:
            blockers.append(f"{name} contains NaN or Inf")
        if max(abs(float(stats["min"])), abs(float(stats["max"]))) >= 1e4:
            blockers.append(f"{name} magnitude is unreasonable")
    agent_center_size = outputs["agent_boxes"][..., :6]
    if float(agent_center_size.min().item()) < -1e-5 or float(agent_center_size.max().item()) > 1.00001:
        blockers.append("agent box center/size values are outside [0, 1]")
    yaw_norm = torch.linalg.vector_norm(outputs["agent_boxes"][..., 6:8], dim=-1)
    if not bool(torch.allclose(yaw_norm, torch.ones_like(yaw_norm), rtol=0.02, atol=0.02)):
        blockers.append("agent box yaw vectors are not normalized")
    map_points = outputs["map_points"]
    if float(map_points.min().item()) < -1e-5 or float(map_points.max().item()) > 1.00001:
        blockers.append("map points are outside [0, 1]")
    report["outputs"] = output_stats
    print("OUTPUT_SHAPES", {name: tuple(outputs[name].shape) for name in EXPECTED_OUTPUT_SHAPES})
    print("OUTPUT_STATS", output_stats)

    loss_config = stage_config["loss"]
    losses = compute_total_loss(outputs, gts, loss_config)
    initial_loss = float(losses["total_loss"].item())
    losses_finite = all(bool(torch.isfinite(value).all().item()) for value in losses.values())
    mask_probe_config = copy.deepcopy(loss_config)
    mask_probe_config["tasks"]["map"] = True
    mask_probe_config["tasks"]["flow"] = True
    mask_probe_config["task_weights"]["map"] = 1.0
    mask_probe_config["task_weights"]["flow"] = 1.0
    mask_probe_losses = compute_total_loss(outputs, gts, mask_probe_config)
    map_masked = float(mask_probe_losses["map_loss"].item()) == 0.0
    flow_supervised = bool(gts["flow_valid"].any()) and float(mask_probe_losses["flow_loss"].item()) > 0.0
    if not losses_finite or not math.isfinite(initial_loss) or initial_loss <= 0.0:
        blockers.append("initial total loss is not finite and positive")
    if not map_masked:
        blockers.append("Map loss is not masked without GT")
    if not flow_supervised:
        blockers.append("Flow GT is available but Flow loss is not positive")
    if float(losses["agent_loss"].item()) <= 0.0:
        blockers.append("Agent GT is available but Agent loss is not positive")
    if float(losses["occ_loss"].item()) <= 0.0:
        blockers.append("OCC GT is available but OCC loss is not positive")

    optimizer.zero_grad(set_to_none=True)
    losses["total_loss"].backward()
    gradients = [(name, parameter.grad) for name, parameter in model.named_parameters() if parameter.grad is not None]
    gradients_finite = bool(gradients) and all(bool(torch.isfinite(grad).all().item()) for _, grad in gradients)
    gradient_abs_sum = sum(float(grad.detach().abs().sum().item()) for _, grad in gradients)
    nonzero_gradient_names = [name for name, grad in gradients if bool((grad != 0).any().item())]
    active_prefixes = ("fusion.", "decoder.", "agent_head.", "occ_head.", "flow_head.")
    active_groups_nonzero = {
        prefix.rstrip("."): any(name.startswith(prefix) for name in nonzero_gradient_names)
        for prefix in active_prefixes
    }
    if not gradients_finite:
        blockers.append("gradients contain NaN/Inf or no gradients were produced")
    if gradient_abs_sum <= 0.0:
        blockers.append("gradient sum is zero")
    for group_name, has_nonzero in active_groups_nonzero.items():
        if not has_nonzero:
            blockers.append(f"{group_name} has no nonzero gradient")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        backward_peak = torch.cuda.max_memory_allocated(device)
    else:
        backward_peak = 0
    optimizer.step()
    parameter_delta = float((tracked_parameter.detach() - tracked_before).abs().max().item())
    parameter_updated = parameter_delta > 0.0
    if not parameter_updated:
        blockers.append("optimizer.step did not change the tracked fusion parameter")

    report["initial_loss"] = initial_loss
    report["loss_checks"] = {
        "all_finite": losses_finite,
        "map_masked": map_masked,
        "flow_supervised": flow_supervised,
    }
    report["gradient"] = {
        "finite": gradients_finite,
        "absolute_sum": gradient_abs_sum,
        "nonzero_tensor_count": len(nonzero_gradient_names),
        "active_groups_nonzero": active_groups_nonzero,
    }
    report["parameter_update"] = {
        "updated": parameter_updated,
        "max_absolute_delta": parameter_delta,
        "parameter": "fusion.token_proj.0.weight",
    }
    report["vram_mib"] = {
        "baseline": memory_baseline / (1024**2),
        "forward_peak": forward_peak / (1024**2),
        "forward_increment": (forward_peak - memory_baseline) / (1024**2),
        "forward_backward_peak": backward_peak / (1024**2),
        "forward_backward_increment": (backward_peak - memory_baseline) / (1024**2),
    }
    print("INITIAL_LOSS", initial_loss)
    print("LOSS_MASKS", report["loss_checks"])
    print("GRADIENT_CHECK", report["gradient"])
    print("PARAMETER_UPDATE_CHECK", report["parameter_update"])
    print("VRAM_MIB", report["vram_mib"])

    history = [initial_loss]
    print("TINY_OVERFIT")
    print_step(1, losses)
    for step in range(2, 21):
        optimizer.zero_grad(set_to_none=True)
        step_outputs = forward_model(model, tensors, device)
        step_losses = compute_total_loss(step_outputs, gts, loss_config)
        step_losses["total_loss"].backward()
        optimizer.step()
        history.append(float(step_losses["total_loss"].item()))
        print_step(step, step_losses)

    model.eval()
    with torch.no_grad():
        final_outputs = forward_model(model, tensors, device)
        final_losses = compute_total_loss(final_outputs, gts, loss_config)
    final_loss = float(final_losses["total_loss"].item())
    print(
        f"FINAL_AFTER_20 total={final_loss:.6f} "
        f"agent={final_losses['agent_loss'].item():.6f} "
        f"occ={final_losses['occ_loss'].item():.6f}"
    )
    first_five_mean = sum(history[:5]) / 5
    last_five_mean = sum([*history[-4:], final_loss]) / 5
    overfit_decreased = final_loss < initial_loss * 0.95 and last_five_mean < first_five_mean
    if not overfit_decreased:
        blockers.append(
            f"20-step overfit did not decrease clearly: {initial_loss:.6f} -> {final_loss:.6f}"
        )
    report["overfit"] = {
        "steps": 20,
        "loss_history": history,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "first_five_mean": first_five_mean,
        "last_five_mean": last_five_mean,
        "clear_decrease": overfit_decreased,
    }
    return report, blockers


def main() -> int:
    report: dict[str, Any] = {}
    blockers: list[str] = []
    try:
        report, blockers = run()
    except Exception as error:
        blockers.append(f"{type(error).__name__}: {error}")
        traceback.print_exc()

    ready = not blockers
    report["training_ready"] = ready
    report["blockers"] = blockers
    print("BLOCKERS", blockers if blockers else "NONE")
    print(f"QUEST_8VIEW_TRAINING_READY = {'YES' if ready else 'NO'}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
