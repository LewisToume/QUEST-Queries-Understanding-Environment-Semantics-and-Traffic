from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.agent_training import AGENT_END_TO_END_STAGE, load_checkpoint_cpu, prepare_navformer_agent_batch
from quest.bev_pretraining import gradient_rms
from quest.map_teacher import MapRasterDistillHead
from quest.map_training import (
    STAGE3_MODULES, VECTOR_PROVENANCE_KEYS, configure_stage3, load_stage2_for_stage3,
    load_vector_capacity_audit,
    mixed_stage3_loss, save_stage3_checkpoint, stage3_forward, stage3_kd_enabled,
    stage3_trainable_modules, frozen_agent_fingerprint,
    stage3_model_config,
    validate_vector_record,
)
from quest.model import QUESTModel, QUEST_ARCHITECTURE_VERSION
from quest.stage3_dataset import (
    Stage3JoinedDataset, collate_stage3, load_record, load_teacher_audit,
)
from quest.stage3_split import load_stage3_split
from quest.teacher_adapters import validate_canonical_agent
from quest.utils import load_yaml_config
from quest.vector_map_labels import MAP_CLASS_NAMES


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def build_dataset(model_config: dict, config: dict, stage1: dict, audit: dict | None,
                  source_indices: list[int], vector_provenance: dict) -> Stage3JoinedDataset:
    dataset_config = dict(stage1["dataset"])
    dataset_config["metadata_path"] = resolve(dataset_config["metadata_path"])
    dataset_config["camera_root"] = resolve(dataset_config["camera_root"])
    dataset_config["max_agent_instances"] = 64
    paths = config["paths"]
    kd_enabled = stage3_kd_enabled(config)
    return Stage3JoinedDataset(
        dataset_config, source_indices,
        resolve(paths["agent_soft_labels_dir"]), resolve(paths["vector_gt_dir"]),
        resolve(paths["map_teacher_dir"]) if kd_enabled else None, audit,
        (model_config["x_range"][0], model_config["y_range"][0],
         model_config["x_range"][1], model_config["y_range"][1]),
        int(model_config["bev_h"]), int(model_config["bev_w"]),
        skip_missing_teacher=kd_enabled and config["train"]["missing_teacher"] == "skip",
        vector_provenance=vector_provenance,
    )


def preflight_vectors(dataset: Stage3JoinedDataset, query_count: int) -> dict:
    provenance = None
    scene_locations = {}
    counts = [0] * len(MAP_CLASS_NAMES)
    nonempty = 0
    for index, info in zip(dataset.source_indices, dataset.images.infos):
        token = str(info["token"])
        record = load_record(dataset.vector_dir / f"{token}.pt")
        validate_vector_record(record, token, index, dataset.quest_range, expected_info=info)
        scene = str(info["scene_token"])
        previous_location = scene_locations.setdefault(scene, record["map_location"])
        if previous_location != record["map_location"]:
            raise ValueError(f"mixed vector GT cities within scene={scene}")
        current = {key: record[key] for key in VECTOR_PROVENANCE_KEYS}
        if provenance is None:
            provenance = current
        elif current != provenance:
            raise ValueError(f"mixed vector GT export provenance at index={index} token={token}")
        if len(record["class_ids"]) > query_count:
            raise ValueError(
                f"map GT exceeds {query_count} queries at index={index} token={token}: "
                f"instances={len(record['class_ids'])}; no GT truncation is allowed"
            )
        if len(record["class_ids"]):
            nonempty += 1
        for class_id in range(len(MAP_CLASS_NAMES)):
            counts[class_id] += int((record["class_ids"] == class_id).sum())
    if provenance is None:
        raise ValueError("Stage 3 has no vector GT records")
    print(f"vector_gt_preflight_frames={len(dataset)} nonempty_frames={nonempty} "
          f"class_counts={dict(zip(MAP_CLASS_NAMES, counts))}")
    return provenance


def preflight_agents(dataset: Stage3JoinedDataset) -> None:
    counts = [0, 0]
    expected_support = torch.tensor([True, True, False, False])
    for index, info in zip(dataset.source_indices, dataset.images.infos):
        token = str(info["token"])
        record = load_record(dataset.images.soft_labels_root / f"{token}.pt")
        if str(record.get("token")) != token or ("sample_index" in record
                and record["sample_index"] != index):
            raise ValueError(f"Agent pseudo label index/token mismatch at {index} {token}")
        agent = record.get("agent")
        if not isinstance(agent, Mapping):
            raise ValueError(f"Agent pseudo label missing at {index} {token}")
        validate_canonical_agent(agent, dataset.images.max_agent_instances)
        if not torch.equal(agent["class_support_mask"].bool(), expected_support):
            raise ValueError(f"Agent class support differs from Stage 2 at {index} {token}")
        labels = agent["labels"][agent["valid_mask"].bool()]
        if bool(((labels != 0) & (labels != 1)).any()):
            raise ValueError(f"unsupported Agent pseudo class at {index} {token}")
        for class_id in range(2):
            counts[class_id] += int((labels == class_id).sum())
    print(f"agent_preflight_frames={len(dataset)} vehicle={counts[0]} pedestrian={counts[1]}")


def preflight_stage2_checkpoint(path: Path) -> None:
    checkpoint = load_checkpoint_cpu(path)
    required = ("geometry_aware_bev_lift_state_dict", "bev_encoder_state_dict",
                "agent_proposal_head_state_dict", "agent_decoder_state_dict",
                "agent_head_state_dict")
    if (checkpoint.get("stage") != AGENT_END_TO_END_STAGE
            or checkpoint.get("architecture_version") != QUEST_ARCHITECTURE_VERSION
            or checkpoint.get("bev_use_ego_state") is not False
            or any(key not in checkpoint for key in required)):
        raise ValueError("Stage 2 initialization checkpoint is incompatible with QUEST V3")
    support = checkpoint.get("trained_class_support_mask")
    if not torch.is_tensor(support) or not torch.equal(
        support.cpu().bool(), torch.tensor([True, True, False, False])
    ):
        raise ValueError("Stage 2 checkpoint Agent class support is incompatible")
    print(f"stage2_checkpoint_preflight=PASS epoch={checkpoint.get('epoch')}")


def train_one_epoch(model: QUESTModel, raster_head: MapRasterDistillHead | None,
                    loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, config: dict, audit: dict | None, epoch: int,
                    scheduler=None) -> None:
    model.eval()
    trainable_modules = stage3_trainable_modules(config["train"])
    for name in trainable_modules:
        getattr(model, name).train()
    model.geometry_lift.ego_mlp.eval()
    if model.map_spatial is not None:
        model.map_spatial.geometry_lift.ego_mlp.eval()
    if raster_head is not None:
        raster_head.train()
    keys = (
        "total_loss", "raw_map_gt_loss", "raw_map_kd_loss", "raw_agent_loss",
        "weighted_map_gt_loss", "weighted_map_kd_loss", "weighted_agent_loss",
        "map_cls_loss", "map_point_loss", "map_direction_loss",
        "map_matched_point_error_m",
    )
    if model.map_spatial is not None:
        keys += ("map_dense_focal_loss", "map_dense_dice_loss", "map_dense_gt_loss")
    class_errors, class_matches = [0.] * len(MAP_CLASS_NAMES), [0] * len(MAP_CLASS_NAMES)
    sums = {key: 0.0 for key in keys}
    module_names = (*trainable_modules, *(("map_raster_distill_head",) if raster_head is not None else ()))
    gradient_modules = {name: (raster_head if name == "map_raster_distill_head" else getattr(model, name))
                        for name in module_names}
    if model.map_spatial is not None:
        gradient_modules.update(map_visual_lift=model.map_spatial.geometry_lift,
                                 map_bev_encoder=model.map_spatial.layers,
                                 map_dense_head=model.map_spatial.dense_head)
    gradients = {name: 0.0 for name in gradient_modules}
    counts = [0] * len(MAP_CLASS_NAMES)
    raster_probability = 0.0
    raster_probability_min = 1.0
    raster_probability_max = 0.0
    steps = 0
    window_map = window_error = 0.0
    window_steps = 0
    support = (torch.tensor(audit["teacher_channel_support_mask"], device=device, dtype=torch.bool)
               if audit is not None else None)
    weights = (torch.tensor(audit["teacher_channel_weights"], device=device, dtype=torch.float32)
               if audit is not None else None)
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        target = prepare_navformer_agent_batch(batch, device)
        vectors = [{key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in record.items()} for record in batch["vector_targets"]]
        for record in vectors:
            for class_id in range(len(MAP_CLASS_NAMES)):
                counts[class_id] += int((record["class_ids"] == class_id).sum())
        predictions = stage3_forward(model, raster_head, batch, device)
        losses = mixed_stage3_loss(
            model, predictions, vectors,
            batch["teacher_map_aligned"].to(device) if raster_head is not None else None,
            batch["teacher_map_valid"].to(device) if raster_head is not None else None,
            support, weights, target, config,
        )
        if not bool(torch.isfinite(losses["total_loss"])):
            raise RuntimeError(f"non-finite Stage 3 loss for {batch['sample_token']}")
        losses["total_loss"].backward()
        if any(not bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters() if parameter.grad is not None):
            raise RuntimeError(f"non-finite Stage 3 model gradient for {batch['sample_token']}")
        if raster_head is not None and any(not bool(torch.isfinite(parameter.grad).all())
                                           for parameter in raster_head.parameters()
                                           if parameter.grad is not None):
            raise RuntimeError(f"non-finite Stage 3 raster gradient for {batch['sample_token']}")
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad]
            + ([p for p in raster_head.parameters() if p.requires_grad] if raster_head is not None else []),
            float(config["train"]["gradient_clip_norm"]),
        )
        for name, module in gradient_modules.items():
            gradients[name] += gradient_rms(module)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        for key in keys:
            sums[key] += float(losses[key].detach())
        for class_id, name in enumerate(MAP_CLASS_NAMES):
            class_errors[class_id] += float(losses[f"map_{name}_point_error_sum_m"])
            class_matches[class_id] += int(losses[f"map_{name}_matched_count"])
        if raster_head is not None:
            raster_values = predictions["student_map_raster_logits"].sigmoid().detach()
            raster_probability += float(raster_values.mean())
            raster_probability_min = min(raster_probability_min, float(raster_values.min()))
            raster_probability_max = max(raster_probability_max, float(raster_values.max()))
        steps += 1
        window_map += float(losses["raw_map_gt_loss"].detach())
        window_error += float(losses["map_matched_point_error_m"].detach())
        window_steps += 1
        interval = int(config["train"].get("log_interval", 500))
        if interval > 0 and steps % interval == 0:
            print(f"epoch={epoch} step={steps}/{len(loader)} "
                  f"window_map={window_map / window_steps:.4f} "
                  f"window_point_error_m={window_error / window_steps:.3f} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
            window_map = window_error = 0.0
            window_steps = 0
    if not steps:
        raise RuntimeError("Stage 3 dataset produced no batches")
    print(f"epoch={epoch} steps={steps} map_kd={'ON' if raster_head is not None else 'OFF'} "
          f"effective_frames={len(loader.dataset)} "
          f"mean_map_instances_per_frame={sum(counts) / len(loader.dataset):.3f} "
          f"supported_teacher_channels={int(support.sum()) if support is not None else 'N/A'}")
    if raster_head is not None:
        print(f"student_raster_mean_probability={raster_probability / steps:.6f} "
              f"student_raster_min_probability={raster_probability_min:.6f} "
              f"student_raster_max_probability={raster_probability_max:.6f}")
    print(" ".join(f"{key}={sums[key] / steps:.6f}" for key in keys))
    print("map_gt_class_counts=" + str(dict(zip(MAP_CLASS_NAMES, counts))))
    print("matched_training_point_error_m " + " ".join(
        f"{name}={class_errors[i] / class_matches[i]:.3f}" if class_matches[i] else f"{name}=N/A"
        for i, name in enumerate(MAP_CLASS_NAMES)))
    print("gradient_RMS " + " ".join(f"{name}={gradients[name] / steps:.3e}" for name in gradients))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 vector GT + Agent training with optional map KD")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = load_yaml_config(PROJECT_ROOT / "configs/stage3_map.yaml")
    stage1 = load_yaml_config(PROJECT_ROOT / "configs/stage1.yaml")
    stage2 = load_yaml_config(PROJECT_ROOT / "configs/stage2_agent.yaml")
    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    kd_enabled = stage3_kd_enabled(config)
    model_config = stage3_model_config(model_config, config)
    if (tuple(stage1["dataset"]["camera_names"]) != tuple(model_config["camera_names"])
            or len(model_config["camera_names"]) != 8
            or int(model_config["architecture_version"]) != QUEST_ARCHITECTURE_VERSION):
        raise ValueError("Stage 3 requires the QUEST V3 eight-camera model/dataset contract")
    metadata_path = resolve(stage1["dataset"]["metadata_path"])
    from scripts.run_navformer_openscene_teacher import load_infos
    manifest = load_stage3_split(resolve(config["paths"]["split_manifest_path"]),
                                 load_infos(metadata_path), metadata_path)
    if kd_enabled and config["train"]["missing_teacher"] not in ("error", "skip"):
        raise ValueError("missing_teacher must be error or explicit skip")
    if list(config["map"]["class_names"]) != list(MAP_CLASS_NAMES) or int(config["map"]["num_points"]) != 20:
        raise ValueError("Stage 3 map taxonomy or point count is incompatible")
    auxiliary = config["map"].get("aux_layer_weights")
    if auxiliary is not None and len(auxiliary) != int(model_config["map_decoder_layers"]):
        raise ValueError("map auxiliary weights must match configured decoder layer count")
    query_count, audited_provenance = load_vector_capacity_audit(
        resolve(config["paths"]["vector_capacity_audit_path"]), config, manifest
    )
    model_config.update(C_map=len(MAP_CLASS_NAMES), N_map=query_count, P=20)
    audit = (load_teacher_audit(resolve(config["paths"]["teacher_audit_path"]),
                                expected_split_sha256=manifest["split_sha256"])
             if kd_enabled else None)
    indices = [row["index"] for row in manifest["train"]["frames"]]
    if args.num_samples is not None and args.num_samples != len(indices):
        raise ValueError("formal Stage 3 training must use every manifest training frame")
    epochs = args.epochs if args.epochs is not None else int(config["train"]["epochs"])
    if not indices or epochs <= 0:
        raise ValueError("num-samples and epochs must be positive")
    dataset = build_dataset(model_config, config, stage1, audit, indices, audited_provenance)
    if dataset.source_indices != indices:
        raise ValueError("formal Stage 3 training cannot silently skip manifest frames")
    vector_provenance = preflight_vectors(dataset, query_count)
    if vector_provenance != audited_provenance:
        raise ValueError("training vector GT provenance differs from certified capacity audit")
    preflight_agents(dataset)
    stage2_path = resolve(config["paths"]["stage2_checkpoint"])
    if not stage2_path.is_file():
        raise FileNotFoundError(f"Stage 2 initialization checkpoint missing: {stage2_path}")
    preflight_stage2_checkpoint(stage2_path)
    backbone_path = resolve(model_config["local_backbone_dir"])
    if not backbone_path.is_dir():
        raise FileNotFoundError(f"local DINOv2 backbone missing: {backbone_path}")
    print(f"MAP_KD_DISABLED={not kd_enabled} stage3_split_sha256={manifest['split_sha256']} "
          f"train_frames={len(indices)} map_queries={query_count} "
          f"stage2_checkpoint={stage2_path} dinov2={backbone_path}")
    print(f"trainable_modules={stage3_trainable_modules(config['train'])} "
          f"map_objective={config['map'].get('objective_version', 'legacy_v1')} "
          f"map_architecture={(config.get('map_architecture') or {}).get('version', 'legacy_instance_v1')}")
    if args.preflight_only:
        eval_indices = [row["index"] for row in manifest["validation"]["frames"]]
        eval_dataset = build_dataset(model_config, config, stage1, audit, eval_indices, audited_provenance)
        if eval_dataset.source_indices != eval_indices:
            raise ValueError("Stage 3 validation preflight cannot skip manifest frames")
        if preflight_vectors(eval_dataset, query_count) != audited_provenance:
            raise ValueError("validation Vector GT provenance differs from capacity audit")
        preflight_agents(eval_dataset)
        print("STAGE3_PREFLIGHT = PASS; no model instantiated, no forward or training")
        return
    output_path = resolve(args.checkpoint or config["paths"]["checkpoint_path"])
    if output_path.exists():
        raise FileExistsError(f"Stage 3 checkpoint already exists; choose a new --checkpoint path: {output_path}")
    loader = DataLoader(dataset, batch_size=int(config["train"]["batch_size"]),
                        shuffle=True, num_workers=0, collate_fn=collate_stage3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(config["train"].get("seed", 42)))
    model = QUESTModel(**model_config)
    stage2_epoch = load_stage2_for_stage3(model, load_checkpoint_cpu(stage2_path))
    print(f"loaded_stage2_agent={stage2_path} epoch={stage2_epoch}")
    model = model.to(device)
    raster_head = (MapRasterDistillHead(model.hidden_dim, len(audit["teacher_channel_names_or_ids"])).to(device)
                   if kd_enabled else None)
    optimizer = configure_stage3(model, raster_head, config["train"])
    total_steps = epochs * len(loader)
    warmup = int(config["train"].get("warmup_steps", 0))
    minimum = float(config["train"].get("min_lr_ratio", 1.0))
    if warmup < 0 or not 0 < minimum <= 1:
        raise ValueError("invalid learning-rate warmup/min_lr_ratio")

    def lr_factor(step):
        if step < warmup:
            return max(1e-3, (step + 1) / max(warmup, 1))
        progress = min(1.0, (step - warmup) / max(total_steps - warmup, 1))
        return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    effective = {**config, "agent_train": stage2["train"], "agent_loss": stage2["agent_loss"],
                 "map_loss": config["map"], "effective_num_samples": len(dataset),
                 "effective_epochs": epochs, "effective_map_query_count": query_count,
                 "stage3_split_sha256": manifest["split_sha256"], "map_kd_enabled": kd_enabled}
    if config["train"].get("freeze_agent_and_bev", False):
        effective["agent_baseline_sha256"] = frozen_agent_fingerprint(model)
    print("optimizer_groups=" + str([(group["lr"], len(group["params"])) for group in optimizer.param_groups]))
    print("parameters=" + str(model.parameter_summary()))
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, raster_head, loader, optimizer, device, effective, audit, epoch, scheduler)
        save_stage3_checkpoint(
            output_path, model, raster_head, optimizer, epoch, effective, audit, vector_provenance,
            scheduler=scheduler,
        )
        if effective.get("agent_baseline_sha256"):
            print("frozen_shared_BEV_and_Agent=UNCHANGED")
        print(f"checkpoint_saved={output_path}")


if __name__ == "__main__":
    main()
