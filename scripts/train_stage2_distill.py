from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.dataset import collate_fn
from quest.losses import compute_total_loss
from quest.model import QUESTModel
from quest.openscene_dataset import OpenSceneFirstTestDataset
from quest.teachers import TeacherUnavailableError, build_enabled_teachers
from quest.utils import load_yaml_config


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def autocast_context(device: str):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def move_batch_to_device(batch: dict[str, Any], device: str) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
    model_inputs = {
        "intrinsics": batch["intrinsics"].to(device),
        "extrinsics": batch["extrinsics"].to(device),
        "ego_state": batch["ego_state"].to(device),
    }
    gts = {
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
    }
    return batch["images"].to(device), gts, model_inputs


def zero_like_loss(preds: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(value.sum() * 0.0 for value in preds.values() if torch.is_tensor(value))


def distill_from_teacher_outputs(
    student_preds: dict[str, torch.Tensor],
    teacher_outputs: dict[str, dict[str, Any]],
    loss_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    kd_loss = zero_like_loss(student_preds)
    logs: dict[str, float] = {}

    agent = teacher_outputs.get("agent")
    if agent and "cls_logits" in agent:
        target = agent["cls_logits"].to(student_preds["agent_cls_logits"].device)
        if target.shape == student_preds["agent_cls_logits"].shape:
            loss = F.kl_div(
                F.log_softmax(student_preds["agent_cls_logits"], dim=-1),
                F.softmax(target, dim=-1),
                reduction="batchmean",
            )
            kd_loss = kd_loss + float(loss_weights.get("agent", 0.0)) * loss
            logs["kd_agent"] = float(loss.item())

    occ = teacher_outputs.get("occ")
    if occ and "occ_logits" in occ:
        target = occ["occ_logits"].to(student_preds["occ_logits"].device)
        if target.shape == student_preds["occ_logits"].shape:
            loss = F.kl_div(
                F.log_softmax(student_preds["occ_logits"], dim=1),
                F.softmax(target, dim=1),
                reduction="batchmean",
            )
            kd_loss = kd_loss + float(loss_weights.get("occ", 0.0)) * loss
            logs["kd_occ"] = float(loss.item())

    flow = teacher_outputs.get("flow")
    if flow and "flow" in flow:
        target = flow["flow"].to(student_preds["flow_logits"].device)
        if target.shape == student_preds["flow_logits"].shape:
            loss = F.smooth_l1_loss(student_preds["flow_logits"], target)
            kd_loss = kd_loss + float(loss_weights.get("flow", 0.0)) * loss
            logs["kd_flow"] = float(loss.item())

    mapping = teacher_outputs.get("map")
    if mapping and "points" in mapping:
        target = mapping["points"].to(student_preds["map_points"].device)
        if target.shape == student_preds["map_points"].shape:
            loss = F.smooth_l1_loss(student_preds["map_points"], target)
            kd_loss = kd_loss + float(loss_weights.get("map", 0.0)) * loss
            logs["kd_map"] = float(loss.item())

    return kd_loss, logs


def load_configs() -> tuple[dict, dict, dict]:
    model_config = load_yaml_config(PROJECT_ROOT / "configs" / "model.yaml")["model"]
    stage1_config = load_yaml_config(PROJECT_ROOT / "configs" / "stage1.yaml")
    stage2_config = load_yaml_config(PROJECT_ROOT / "configs" / "stage2_distill.yaml")
    return model_config, stage1_config, stage2_config


def build_dataset(model_config: dict, stage1_config: dict, num_samples: int) -> OpenSceneFirstTestDataset:
    dataset_kwargs = dict(stage1_config.get("dataset", {}))
    data_root = dataset_kwargs.pop("root", "data/openscene_first_test_100")
    dataset_kwargs.pop("C_agent", None)
    dataset_kwargs.pop("D_box", None)
    dataset_kwargs.setdefault("camera_names", model_config["camera_names"])
    dataset_kwargs.setdefault("C_map", model_config["C_map"])
    dataset_kwargs.setdefault("P", model_config["P"])
    dataset_kwargs.setdefault("C_occ", model_config["C_occ"])
    dataset_kwargs.setdefault("C_flow", model_config["C_flow"])
    dataset_kwargs.setdefault("occ_size", (model_config["X"], model_config["Y"], model_config["Z"]))
    dataset = OpenSceneFirstTestDataset(root=data_root, **dataset_kwargs)
    if num_samples > 0 and num_samples < len(dataset):
        dataset.manifest = dataset.manifest[:num_samples]
    return dataset


def main() -> None:
    model_config, stage1_config, stage2_config = load_configs()
    distill_config = stage2_config["distill"]
    device = resolve_device(distill_config.get("device", "auto"))
    dataset = build_dataset(model_config, stage1_config, int(distill_config.get("num_samples", 1)))
    dataloader = DataLoader(
        dataset,
        batch_size=int(distill_config.get("batch_size", 1)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )

    student = QUESTModel(**model_config).to(device).train()
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=float(distill_config.get("lr", 5e-5)),
        weight_decay=1e-4,
    )
    teachers = build_enabled_teachers(stage2_config.get("teachers", {}))
    for teacher in teachers.values():
        teacher.eval()
        teacher.requires_grad_(False)

    batch = next(iter(dataloader))
    images, gts, model_inputs = move_batch_to_device(batch, device)
    teacher_outputs: dict[str, dict[str, Any]] = {}
    teacher_status: dict[str, str] = {}
    for task, teacher in teachers.items():
        try:
            with torch.no_grad():
                teacher_outputs[task] = teacher(batch)
            teacher_status[task] = "available"
        except TeacherUnavailableError as exc:
            teacher_status[task] = f"skipped: {exc}"

    with autocast_context(device):
        preds = student(images, **model_inputs)
    preds = {key: value.float() for key, value in preds.items()}
    hard_losses = compute_total_loss(preds, gts, stage1_config.get("loss", {}))
    kd_loss, kd_logs = distill_from_teacher_outputs(preds, teacher_outputs, stage2_config.get("loss_weights", {}))
    total_loss = hard_losses["total_loss"] + kd_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print("=" * 72)
    print("QUEST stage2 distillation skeleton")
    print(f"device        : {device}")
    print(f"samples       : {len(dataset)}")
    print("teacher status:")
    for task, status in teacher_status.items():
        print(f"  {task:<6}: {status}")
    print("loss:")
    print(f"  hard_total  : {hard_losses['total_loss'].item():.4f}")
    print(f"  kd_total    : {kd_loss.item():.4f}")
    for key, value in kd_logs.items():
        print(f"  {key:<11}: {value:.4f}")
    print(f"  total       : {total_loss.item():.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
