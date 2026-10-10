from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.agent_training import load_checkpoint_cpu, prepare_navformer_agent_batch
from quest.bev_pretraining import gradient_rms
from quest.map_teacher import MapRasterDistillHead
from quest.map_training import (
    STAGE3_MODULES, VECTOR_PROVENANCE_KEYS, configure_stage3, load_stage2_for_stage3,
    load_vector_capacity_audit,
    mixed_stage3_loss, save_stage3_checkpoint, stage3_forward,
    validate_vector_record,
)
from quest.model import QUESTModel
from quest.stage3_dataset import (
    Stage3JoinedDataset, collate_stage3, load_record, load_teacher_audit,
)
from quest.stage3_split import load_stage3_split
from quest.utils import load_yaml_config
from quest.vector_map_labels import MAP_CLASS_NAMES


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def build_dataset(model: QUESTModel, config: dict, stage1: dict, audit: dict,
                  source_indices: list[int]) -> Stage3JoinedDataset:
    dataset_config = dict(stage1["dataset"])
    dataset_config["metadata_path"] = resolve(dataset_config["metadata_path"])
    dataset_config["camera_root"] = resolve(dataset_config["camera_root"])
    dataset_config["max_agent_instances"] = 64
    paths = config["paths"]
    return Stage3JoinedDataset(
        dataset_config, source_indices,
        resolve(paths["agent_soft_labels_dir"]), resolve(paths["vector_gt_dir"]),
        resolve(paths["map_teacher_dir"]), audit,
        (model.geometry_lift.x_range[0], model.geometry_lift.y_range[0],
         model.geometry_lift.x_range[1], model.geometry_lift.y_range[1]),
        model.bev_h, model.bev_w,
        skip_missing_teacher=config["train"]["missing_teacher"] == "skip",
    )


def preflight_vectors(dataset: Stage3JoinedDataset, query_count: int) -> dict:
    provenance = None
    scene_locations = {}
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
    if provenance is None:
        raise ValueError("Stage 3 has no vector GT records")
    return provenance


def train_one_epoch(model: QUESTModel, raster_head: MapRasterDistillHead,
                    loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, config: dict, audit: dict, epoch: int) -> None:
    model.eval()
    for name in STAGE3_MODULES:
        getattr(model, name).train()
    model.geometry_lift.ego_mlp.eval()
    raster_head.train()
    keys = (
        "total_loss", "raw_map_gt_loss", "raw_map_kd_loss", "raw_agent_loss",
        "weighted_map_gt_loss", "weighted_map_kd_loss", "weighted_agent_loss",
        "map_cls_loss", "map_point_loss", "map_direction_loss",
    )
    sums = {key: 0.0 for key in keys}
    module_names = (*STAGE3_MODULES, "map_raster_distill_head")
    gradients = {name: 0.0 for name in module_names}
    counts = [0] * len(MAP_CLASS_NAMES)
    raster_probability = 0.0
    raster_probability_min = 1.0
    raster_probability_max = 0.0
    steps = 0
    support = torch.tensor(audit["teacher_channel_support_mask"], device=device, dtype=torch.bool)
    weights = torch.tensor(audit["teacher_channel_weights"], device=device, dtype=torch.float32)
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
            model, predictions, vectors, batch["teacher_map_aligned"].to(device),
            batch["teacher_map_valid"].to(device),
            support, weights, target, config,
        )
        if not bool(torch.isfinite(losses["total_loss"])):
            raise RuntimeError(f"non-finite Stage 3 loss for {batch['sample_token']}")
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in (*model.parameters(), *raster_head.parameters()) if p.requires_grad],
            float(config["train"]["gradient_clip_norm"]),
        )
        for name in STAGE3_MODULES:
            gradients[name] += gradient_rms(getattr(model, name))
        gradients["map_raster_distill_head"] += gradient_rms(raster_head)
        optimizer.step()
        for key in keys:
            sums[key] += float(losses[key].detach())
        raster_values = predictions["student_map_raster_logits"].sigmoid().detach()
        raster_probability += float(raster_values.mean())
        raster_probability_min = min(raster_probability_min, float(raster_values.min()))
        raster_probability_max = max(raster_probability_max, float(raster_values.max()))
        steps += 1
    if not steps:
        raise RuntimeError("Stage 3 dataset produced no batches")
    print(f"epoch={epoch} steps={steps} supported_teacher_channels={int(support.sum())} "
          f"mean_map_instances_per_frame={sum(counts) / len(loader.dataset):.3f} "
          f"student_raster_mean_probability={raster_probability / steps:.6f} "
          f"student_raster_min_probability={raster_probability_min:.6f} "
          f"student_raster_max_probability={raster_probability_max:.6f}")
    print(" ".join(f"{key}={sums[key] / steps:.6f}" for key in keys))
    print("map_gt_class_counts=" + str(dict(zip(MAP_CLASS_NAMES, counts))))
    print("gradient_RMS " + " ".join(f"{name}={gradients[name] / steps:.9e}" for name in module_names))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 hybrid vector-map training")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    config = load_yaml_config(PROJECT_ROOT / "configs/stage3_map.yaml")
    stage1 = load_yaml_config(PROJECT_ROOT / "configs/stage1.yaml")
    stage2 = load_yaml_config(PROJECT_ROOT / "configs/stage2_agent.yaml")
    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    metadata_path = resolve(stage1["dataset"]["metadata_path"])
    from scripts.run_navformer_openscene_teacher import load_infos
    manifest = load_stage3_split(resolve(config["paths"]["split_manifest_path"]),
                                 load_infos(metadata_path), metadata_path)
    if config["train"]["missing_teacher"] not in ("error", "skip"):
        raise ValueError("missing_teacher must be error or explicit skip")
    if list(config["map"]["class_names"]) != list(MAP_CLASS_NAMES) or int(config["map"]["num_points"]) != 20:
        raise ValueError("Stage 3 map taxonomy or point count is incompatible")
    query_count, audited_provenance = load_vector_capacity_audit(
        resolve(config["paths"]["vector_capacity_audit_path"]), config, manifest
    )
    model_config.update(C_map=len(MAP_CLASS_NAMES), N_map=query_count, P=20)
    model = QUESTModel(**model_config)
    audit = load_teacher_audit(resolve(config["paths"]["teacher_audit_path"]),
                               expected_split_sha256=manifest["split_sha256"])
    indices = [row["index"] for row in manifest["train"]["frames"]]
    if args.num_samples is not None and args.num_samples != len(indices):
        raise ValueError("formal Stage 3 training must use every manifest training frame")
    epochs = args.epochs if args.epochs is not None else int(config["train"]["epochs"])
    if not indices or epochs <= 0:
        raise ValueError("num-samples and epochs must be positive")
    dataset = build_dataset(model, config, stage1, audit, indices)
    if dataset.source_indices != indices:
        raise ValueError("formal Stage 3 training cannot silently skip manifest frames")
    vector_provenance = preflight_vectors(dataset, query_count)
    if vector_provenance != audited_provenance:
        raise ValueError("training vector GT provenance differs from certified capacity audit")
    loader = DataLoader(dataset, batch_size=int(config["train"]["batch_size"]),
                        shuffle=True, num_workers=0, collate_fn=collate_stage3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage2_path = resolve(config["paths"]["stage2_checkpoint"])
    stage2_epoch = load_stage2_for_stage3(model, load_checkpoint_cpu(stage2_path))
    print(f"loaded_stage2_agent={stage2_path} epoch={stage2_epoch}")
    model = model.to(device)
    raster_head = MapRasterDistillHead(model.hidden_dim, len(audit["teacher_channel_names_or_ids"])).to(device)
    optimizer = configure_stage3(model, raster_head, config["train"])
    effective = {**config, "agent_train": stage2["train"], "agent_loss": stage2["agent_loss"],
                 "map_loss": config["map"], "effective_num_samples": len(dataset),
                 "effective_epochs": epochs, "effective_map_query_count": query_count,
                 "stage3_split_sha256": manifest["split_sha256"]}
    print("optimizer_groups=" + str([(group["lr"], len(group["params"])) for group in optimizer.param_groups]))
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, raster_head, loader, optimizer, device, effective, audit, epoch)
        path = resolve(args.checkpoint or config["paths"]["checkpoint_path"])
        save_stage3_checkpoint(
            path, model, raster_head, optimizer, epoch, effective, audit, vector_provenance
        )
        print(f"checkpoint_saved={path}")


if __name__ == "__main__":
    main()
