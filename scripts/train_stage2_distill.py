from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.model import QUESTModel
from quest.utils import load_yaml_config


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def autocast_context(device: str):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class SoftLabelLoader:
    """
    Placeholder soft-label loader.

    This keeps the distillation interface aligned with the new four-task structure
    while real teacher exports are still unavailable.
    """

    def __init__(self, model_config: dict, soft_label_dir: str, num_samples: int = 1000):
        self.model_config = model_config
        self.soft_label_dir = Path(soft_label_dir)
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def load_sample(self, idx: int) -> dict:
        del idx

        seg_channels = int(self.model_config["seg_channels"])
        seg_h, seg_w = self.model_config["seg_size"]
        num_agent_queries = int(self.model_config["num_agent_queries"])
        map_channels = int(self.model_config["map_channels"])
        map_h, map_w = self.model_config["map_size"]
        occ_channels = int(self.model_config["occ_channels"])
        occ_x, occ_y, occ_z = self.model_config["occ_size"]

        seg_soft = F.softmax(torch.randn(seg_channels, seg_h, seg_w), dim=0)
        agent_reg_soft = torch.randn(num_agent_queries, 8)
        agent_cls_soft = F.softmax(torch.randn(num_agent_queries, 2), dim=-1)
        map_soft = F.softmax(torch.randn(map_channels, map_h, map_w), dim=0)
        occ_soft = F.softmax(torch.randn(occ_channels, occ_x, occ_y, occ_z), dim=0)

        return {
            "seg_soft": seg_soft,
            "agent_reg_soft": agent_reg_soft,
            "agent_cls_soft": agent_cls_soft,
            "map_soft": map_soft,
            "occ_soft": occ_soft,
        }


class DistillationLoss:
    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def kl_div_loss(self, student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = teacher_probs.clamp_min(1e-6)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    def agent_cls_loss(self, student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = teacher_probs.clamp_min(1e-6)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    def mse_loss(self, student_pred: torch.Tensor, teacher_target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(student_pred, teacher_target, reduction="mean")


class DistillationTrainer:
    def __init__(
        self,
        model_config: dict,
        distill_config: dict,
        loss_weights: dict,
        soft_label_dir: str,
    ) -> None:
        self.device = resolve_device(distill_config.get("device", "auto"))
        self.student = QUESTModel(**model_config).to(self.device)
        self.loss_weights = {
            "seg": 1.0,
            "agent": 1.0,
            "map": 1.0,
            "occ": 1.0,
        }
        self.loss_weights.update(loss_weights)
        self.distillation_loss = DistillationLoss(
            temperature=float(distill_config.get("temperature", 1.0))
        )
        self.soft_label_loader = SoftLabelLoader(
            model_config=model_config,
            soft_label_dir=soft_label_dir,
            num_samples=int(distill_config.get("num_samples", 100)),
        )
        self.optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=float(distill_config.get("lr", 5e-5)),
            weight_decay=1e-4,
        )

        model_path = distill_config.get("model_path")
        if model_path:
            candidate = Path(model_path)
            if candidate.exists():
                state_dict = torch.load(candidate, map_location="cpu")
                self.student.load_state_dict(state_dict, strict=False)

    def distillation_step(self, images: torch.Tensor, teacher_soft: dict) -> dict:
        self.student.train()
        with autocast_context(self.device):
            student_pred = self.student(images)

            seg_loss = self.distillation_loss.kl_div_loss(
                student_pred["seg_pred"],
                teacher_soft["seg_soft"].to(self.device),
            )
            agent_reg_loss = self.distillation_loss.mse_loss(
                student_pred["agent_pred"][:, :, :8],
                teacher_soft["agent_reg_soft"].to(self.device),
            )
            agent_cls_loss = self.distillation_loss.agent_cls_loss(
                student_pred["agent_pred"][:, :, 8:10],
                teacher_soft["agent_cls_soft"].to(self.device),
            )
            agent_loss = agent_reg_loss + agent_cls_loss
            map_loss = self.distillation_loss.kl_div_loss(
                student_pred["map_pred"],
                teacher_soft["map_soft"].to(self.device),
            )
            occ_loss = self.distillation_loss.kl_div_loss(
                student_pred["occ_pred"],
                teacher_soft["occ_soft"].to(self.device),
            )

            total_loss = (
                self.loss_weights["seg"] * seg_loss
                + self.loss_weights["agent"] * agent_loss
                + self.loss_weights["map"] * map_loss
                + self.loss_weights["occ"] * occ_loss
            )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "seg_loss": seg_loss.item(),
            "agent_reg_loss": agent_reg_loss.item(),
            "agent_cls_loss": agent_cls_loss.item(),
            "agent_loss": agent_loss.item(),
            "map_loss": map_loss.item(),
            "occ_loss": occ_loss.item(),
            "total_loss": total_loss.item(),
        }


def load_configs() -> tuple[dict, dict]:
    model_config = load_yaml_config(PROJECT_ROOT / "configs" / "model.yaml")["model"]
    stage_config = load_yaml_config(PROJECT_ROOT / "configs" / "stage2_distill.yaml")
    return model_config, stage_config


if __name__ == "__main__":
    model_config, stage_config = load_configs()
    distill_config = stage_config["distill"]
    soft_labels_dir = stage_config.get("paths", {}).get("soft_labels_dir", "data/soft_labels")
    trainer = DistillationTrainer(
        model_config=model_config,
        distill_config=distill_config,
        loss_weights=stage_config.get("loss_weights", {}),
        soft_label_dir=soft_labels_dir,
    )

    batch_size = int(distill_config["batch_size"])
    device = trainer.device
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    sample_soft = trainer.soft_label_loader.load_sample(0)

    teacher_soft_batch = {
        "seg_soft": sample_soft["seg_soft"].unsqueeze(0).repeat(batch_size, 1, 1, 1),
        "agent_reg_soft": sample_soft["agent_reg_soft"].unsqueeze(0).repeat(batch_size, 1, 1),
        "agent_cls_soft": sample_soft["agent_cls_soft"].unsqueeze(0).repeat(batch_size, 1, 1),
        "map_soft": sample_soft["map_soft"].unsqueeze(0).repeat(batch_size, 1, 1, 1),
        "occ_soft": sample_soft["occ_soft"].unsqueeze(0).repeat(batch_size, 1, 1, 1, 1),
    }

    print("=" * 60)
    print("QUEST stage2 distillation demo")
    print(f"device       : {device}")
    print(f"batch_size   : {batch_size}")
    print(f"temperature  : {distill_config['temperature']}")
    print(f"soft_labels  : {soft_labels_dir}")
    print("=" * 60)

    losses = trainer.distillation_step(dummy_images, teacher_soft_batch)
    for key, value in losses.items():
        print(f"{key:<16}: {value:.4f}")

    print("=" * 60)
