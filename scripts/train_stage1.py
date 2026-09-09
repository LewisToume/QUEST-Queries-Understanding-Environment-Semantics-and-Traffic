from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.dataset import collate_fn
from quest.losses import compute_total_loss
from quest.model import QUESTModel
from quest.openscene_dataset import OpenSceneMetadataDataset
from quest.utils import load_yaml_config


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def autocast_context(device: str):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class QUESTTrainer:
    def __init__(
        self,
        model_config: dict,
        train_config: dict,
        dataset_config: dict,
        loss_config: dict,
    ) -> None:
        self.device = resolve_device(train_config.get("device", "auto"))
        self.loss_config = loss_config
        self.train_config = train_config
        self.dataset_config = dataset_config

        self.model = QUESTModel(**model_config).to(self.device)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=float(train_config["lr"]),
            weight_decay=float(train_config.get("weight_decay", 1e-4)),
            betas=(0.9, 0.999),
        )
        self.scheduler = None
        self.backward_passed = False

    def build_dataloader(self) -> DataLoader:
        dataset_kwargs = dict(self.dataset_config)
        dataset_kwargs.pop("C_agent", None)
        dataset_kwargs.pop("D_box", None)
        max_samples = int(self.train_config["num_samples"])
        dataset = OpenSceneMetadataDataset(max_samples=max_samples, **dataset_kwargs)
        return DataLoader(
            dataset,
            batch_size=int(self.train_config["batch_size"]),
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def build_scheduler(self, dataloader_len: int) -> None:
        total_steps = max(1, int(self.train_config["num_epochs"]) * dataloader_len)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6,
        )

    def train_step(self, batch: dict) -> dict:
        images = batch["images"].to(self.device)
        intrinsics = batch["intrinsics"].to(self.device)
        extrinsics = batch["extrinsics"].to(self.device)
        ego_state = batch["ego_state"].to(self.device)
        gts = {
            "agent_gt": {
                "labels": batch["agent_gt"]["labels"].to(self.device),
                "boxes": batch["agent_gt"]["boxes"].to(self.device),
                "velocity": batch["agent_gt"]["velocity"].to(self.device),
            },
            "map_gt": {
                "labels": batch["map_gt"]["labels"].to(self.device),
                "points": batch["map_gt"]["points"].to(self.device),
            },
            "map_valid": batch["map_gt"]["valid"].to(self.device),
            "occ_gt": batch["occ_gt"].to(self.device),
            "occ_valid": batch["occ_valid"].to(self.device),
            "flow_gt": batch["flow_gt"].to(self.device),
            "flow_valid": batch["flow_valid"].to(self.device),
            "flow_mask": batch["flow_mask"].to(self.device),
        }

        with autocast_context(self.device):
            preds = self.model(
                images,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                ego_state=ego_state,
            )
        preds = {key: value.float() for key, value in preds.items()}
        losses = compute_total_loss(preds, gts, self.loss_config)
        if not torch.isfinite(losses["total_loss"]):
            raise RuntimeError("Stage1 total loss is not finite")

        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        gradients = [parameter.grad for parameter in self.model.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("Stage1 backward produced missing or non-finite gradients")
        if bool(gts["flow_valid"].any()) and bool(self.loss_config["tasks"].get("flow", False)):
            flow_gradients = [
                parameter.grad
                for name, parameter in self.model.named_parameters()
                if name.startswith("flow_head.") and parameter.grad is not None
            ]
            if not flow_gradients or not any(bool((gradient != 0).any()) for gradient in flow_gradients):
                raise RuntimeError("Flow loss did not produce nonzero Flow Head gradients")
        self.backward_passed = True
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return losses

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> None:
        self.model.train()
        tracked_keys = [
            "agent_loss",
            "agent_cls_loss",
            "agent_box_loss",
            "agent_velocity_loss",
            "agent_dn_loss",
            "map_cls_loss",
            "map_pts_loss",
            "map_dir_loss",
            "occ_main_loss",
            "occ_sem_scal_loss",
            "occ_geo_scal_loss",
            "occ_lovasz_loss",
            "occ_loss",
            "flow_loss",
            "total_loss",
        ]
        running = {key: 0.0 for key in tracked_keys}

        epoch_start_time = time.time()
        for batch_idx, batch in enumerate(dataloader, start=1):
            losses = self.train_step(batch)
            for key in tracked_keys:
                running[key] += float(losses[key].item())

            if batch_idx % 5 == 0 or batch_idx == len(dataloader):
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch}] Step [{batch_idx}/{len(dataloader)}] "
                    f"agent_cls={losses['agent_cls_loss'].item():.4f} "
                    f"agent_vel={losses['agent_velocity_loss'].item():.4f} "
                    f"map_pts={losses['map_pts_loss'].item():.4f} "
                    f"occ={losses['occ_main_loss'].item():.4f} "
                    f"flow={losses['flow_loss'].item():.4f} "
                    f"total={losses['total_loss'].item():.4f} "
                    f"lr={current_lr:.6f}"
                )

        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch [{epoch}] finished in {epoch_time:.2f}s")
        for key in tracked_keys:
            print(f"  avg {key:<18}: {running[key] / len(dataloader):.4f}")
        print("-" * 60)


def load_configs() -> tuple[dict, dict]:
    model_config = load_yaml_config(PROJECT_ROOT / "configs" / "model.yaml")["model"]
    stage_config = load_yaml_config(PROJECT_ROOT / "configs" / "stage1.yaml")

    dataset_config = stage_config.setdefault("dataset", {})
    dataset_config.setdefault(
        "metadata_path",
        str(PROJECT_ROOT / "data" / "openscene" / "meta_datas" / "openscene-v1.0" / "meta_datas" / "meta_data_mini.pkl"),
    )
    dataset_config.setdefault(
        "camera_root",
        str(PROJECT_ROOT / "data" / "openscene" / "sensor_blobs_mini"),
    )
    dataset_config.setdefault(
        "occupancy_root",
        str(PROJECT_ROOT / "data" / "openscene" / "occ_mini"),
    )
    dataset_config.setdefault("camera_names", model_config["camera_names"])
    dataset_config.setdefault("C_agent", model_config["C_agent"])
    dataset_config.setdefault("D_box", model_config["D_box"])
    dataset_config.setdefault("C_map", model_config["C_map"])
    dataset_config.setdefault("P", model_config["P"])
    dataset_config.setdefault("occ_size", (model_config["X"], model_config["Y"], model_config["Z"]))
    dataset_config.setdefault("C_occ", model_config["C_occ"])
    dataset_config.setdefault("C_flow", model_config["C_flow"])
    for path_key in ("metadata_path", "camera_root", "occupancy_root"):
        path = Path(dataset_config[path_key])
        if not path.is_absolute():
            dataset_config[path_key] = str(PROJECT_ROOT / path)
    return model_config, stage_config


if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    model_config, stage_config = load_configs()
    train_config = stage_config["train"]
    loss_config = stage_config.get("loss", {})

    print("=" * 60)
    print("QUEST stage1 training")
    print(f"device       : {resolve_device(train_config.get('device', 'auto'))}")
    print(f"batch_size   : {train_config['batch_size']}")
    print(f"num_epochs   : {train_config['num_epochs']}")
    print(f"num_samples  : {train_config['num_samples']}")
    print(f"task_weights : {loss_config.get('task_weights', {})}")
    print("=" * 60)

    trainer = QUESTTrainer(
        model_config=model_config,
        train_config=train_config,
        dataset_config=stage_config["dataset"],
        loss_config=loss_config,
    )
    dataloader = trainer.build_dataloader()
    if len(dataloader) == 0:
        raise RuntimeError("dataloader is empty, increase num_samples or lower batch_size")
    trainer.build_scheduler(len(dataloader))

    print("\nStarting training loop...")
    print("-" * 60)
    for epoch in range(1, int(train_config["num_epochs"]) + 1):
        trainer.train_epoch(dataloader, epoch)

    print("\nStage1 training finished")
    print(f"backward     : {'PASS' if trainer.backward_passed else 'FAIL'}")
    print("=" * 60)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated_mb = torch.cuda.memory_allocated() / 1e6
        print(f"CUDA allocated after cleanup: {allocated_mb:.2f} MB")
