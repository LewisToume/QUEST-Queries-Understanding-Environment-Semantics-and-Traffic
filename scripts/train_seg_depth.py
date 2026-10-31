"""Train offline Seg/Depth heads while preserving the existing QUEST Agent/Map."""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quest.agent_training import load_checkpoint_cpu
from quest.cuda_runtime import initialize_cuda_device
from quest.map_training import load_stage3_checkpoint, load_vector_capacity_audit, stage3_model_config
from quest.model import QUESTModel
from quest.geometry_depth import GEOMETRY_DEPTH_VERSION
from quest.seg_depth_dataset import SegDepthDataset, collate_seg_depth
from quest.seg_depth_losses import (compute_seg_depth_loss, empty_dense_metrics,
                                    summarize_dense_metrics, update_dense_metrics)
from quest.seg_depth_training import (configure_dense_training, frozen_spatial_fingerprint,
                                      load_dense_model, save_dense_checkpoint, validate_dense_config,
                                      resolve_seg_class_balance, validate_dense_checkpoint, validate_dense_data_join)
from quest.seg_teacher import CITYSCAPES_CLASSES
from quest.stage3_split import file_sha256, load_stage3_split, selected_frames
from quest.utils import load_yaml_config


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_context(config):
    paths = config["paths"]
    dataset = dict(load_yaml_config(resolve(paths["dataset_config"]))["dataset"])
    for key in ("metadata_path", "camera_root"):
        dataset[key] = resolve(dataset[key])
    dataset["max_agent_instances"] = 64
    dataset.pop("soft_labels_root", None)
    with dataset["metadata_path"].open("rb") as stream:
        infos = pickle.load(stream)["infos"]  # Trusted local OpenScene metadata only.
    manifest = load_stage3_split(resolve(paths["split_manifest"]), infos, dataset["metadata_path"])
    return dataset, manifest


def make_dataset(config, dataset, manifest, split, count=None):
    frames = selected_frames(manifest, split)
    if count is not None:
        if count <= 0 or count > len(frames):
            raise ValueError(f"Requested {count} {split} frames but manifest has {len(frames)}")
        frames = frames[:count]
    paths = config["paths"]
    experimental = config["depth_geometry_policy"] == "experimental_image_plane"
    result = SegDepthDataset(
        dataset, [row["index"] for row in frames], resolve(paths["seg_labels"]),
        resolve(paths["candidate_depth_labels"] if experimental else paths["depth_labels"]),
        manifest["metadata_sha256"], manifest["split_sha256"], config["tasks"],
        missing_policy=config["train"]["missing_labels"], experimental_depth=experimental,
        require_seg_for_depth=config["loss"]["depth"]["exclude_sky"],
        required_depth_geometry=(config["frontend"]["image_geometry"]
                                 if config["frontend"]["version"] == GEOMETRY_DEPTH_VERSION else None),
    )
    count_classes = split == "train" and config["loss"].get("seg", {}).get("class_balance") is not None
    audit = result.audit(config["loss"]["depth"], include_seg_class_counts=count_classes)
    print(f"dense_preflight split={split} frames={len(result)} tasks={json.dumps(audit['tasks'])}")
    return result, audit


def prepare_spatial_initialization(config, manifest):
    paths = config["paths"]
    stage3 = load_yaml_config(resolve(paths["stage3_config"]))
    query_count, _ = load_vector_capacity_audit(resolve(paths["capacity_audit"]), stage3, manifest)
    source_path = resolve(paths["init_checkpoint"])
    source = load_checkpoint_cpu(source_path)
    saved = source.get("training_config", {})
    if (source.get("stage") != "map_gt_agent" or source.get("architecture_version") != 3
            or saved.get("stage3_split_sha256") != manifest["split_sha256"]
            or saved.get("effective_map_query_count") != query_count):
        raise ValueError("Initialization must be the matching GT-only Stage 3 Agent/Map checkpoint")
    model_config = stage3_model_config(load_yaml_config(resolve(paths["model_config"]))["model"], stage3, source)
    model_config.update(N_map=query_count, C_map=3, P=20, C_seg=19, seg_size=[252, 448], depth_size=[252, 448],
                        dense_frontend=copy.deepcopy(config["frontend"]))
    backbone = resolve(model_config["local_backbone_dir"])
    if not (backbone / "config.json").is_file() or not any((backbone / name).is_file()
            for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")):
        raise FileNotFoundError("Frozen local DINO assets missing: " + str(backbone))
    model_config["local_backbone_dir"] = str(backbone)
    identity = {"path": str(source_path), "sha256": file_sha256(source_path), "epoch": int(source["epoch"])}
    identity["spatial_metadata"] = {key: copy.deepcopy(source.get(key)) for key in (
        "map_architecture", "map_objective_version", "vector_gt_provenance", "map_class_names")}
    return model_config, source, identity


def loader(dataset, config, shuffle=False):
    generator = torch.Generator().manual_seed(config["train"]["seed"])
    return DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=shuffle,
                      num_workers=0, collate_fn=collate_seg_depth, generator=generator)


def dense_predictions(model, batch, device):
    geometry = model.dense_frontend_config["version"] == GEOMETRY_DEPTH_VERSION
    amp = geometry and model.dense_frontend_config.get("amp_bf16", False) and device.type == "cuda"
    if amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This Depth profile requests bf16; no automatic precision/model fallback")
    extra = ({key: batch[key].to(device) for key in ("extrinsics", "distortion")} if geometry else {})
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        return model.forward_dense(batch["images"].to(device), batch["intrinsics"].to(device), **extra)


@torch.no_grad()
def evaluate(model, dataset, config, device):
    model.eval()
    metrics, loss_sum, batches = empty_dense_metrics(), 0.0, 0
    for batch in loader(dataset, config):
        output = dense_predictions(model, batch, device)
        losses = compute_seg_depth_loss(output, batch["dense_targets"], config["loss"], config["tasks"])
        if not bool(torch.isfinite(losses["total_loss"])):
            raise RuntimeError("Nonfinite evaluation loss")
        loss_sum += float(losses["total_loss"])
        batches += 1
        update_dense_metrics(metrics, output, batch["dense_targets"], config["loss"], config["tasks"])
    return {**summarize_dense_metrics(metrics), "mean_batch_loss": loss_sum / max(batches, 1),
            "frames": len(dataset), "depth_geometry_policy": config["depth_geometry_policy"]}


def gradient_health(model):
    dense = model.dense_frontend
    if model.dense_frontend_config["version"] == GEOMETRY_DEPTH_VERSION:
        groups = {"neck": (dense.level_projections, dense.ray_embedding),
                  "geometry": (dense.geometry_blocks,), "depth": (dense.depth_decoder,), "seg": ()}
    else:
        groups = {"neck": (dense.lateral, dense.refine), "seg": (dense.seg_decoder,),
                  "depth": (dense.camera_condition, dense.depth_decoder)}
    result = {}
    for name, modules in groups.items():
        square, count = 0.0, 0
        for module in modules:
            for parameter in module.parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError("Nonfinite dense gradient in " + name)
                    square += float(parameter.grad.detach().double().square().sum())
                    count += parameter.numel()
        result[name] = math.sqrt(square / count) if count else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/seg_depth.yaml")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--tasks", nargs="+", choices=("seg", "depth"))
    parser.add_argument("--checkpoint", type=Path, help="New output; will not overwrite an existing run")
    parser.add_argument("--resume", type=Path, help="Resume exactly the saved config and label set")
    parser.add_argument("--finetune-from", type=Path,
                        help="Load existing dense model, reset optimizer/schedule, require a NEW --checkpoint")
    parser.add_argument("--seg-class-balance", action="store_true",
                        help="Use bounded training-frequency weighted CE; taxonomy and evaluation stay unchanged")
    parser.add_argument("--balance-config", type=Path, default=ROOT / "configs/seg_balance.yaml")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0, help="Repeat selected data; NEVER save a checkpoint")
    parser.add_argument("--experimental-unverified-depth", action="store_true",
                        help="Use separate image-plane candidate depth labels; no geometry/GT accuracy claim")
    args = parser.parse_args()
    if args.smoke_steps < 0 or (args.resume and args.smoke_steps):
        parser.error("smoke steps must be nonnegative and cannot resume")
    if args.resume and args.finetune_from:
        parser.error("Choose resume OR finetune-from, not both")
    if args.finetune_from and (not args.checkpoint or not args.seg_class_balance):
        parser.error("Balanced fine-tuning requires --seg-class-balance and a NEW --checkpoint")
    resumed = load_checkpoint_cpu(resolve(args.resume)) if args.resume else None
    fine_source = load_checkpoint_cpu(resolve(args.finetune_from)) if args.finetune_from else None
    if fine_source is not None:
        validate_dense_checkpoint(fine_source)
        if args.num_samples is not None or args.tasks is not None or args.experimental_unverified_depth:
            parser.error("Fine-tuning preserves source tasks, complete frame lists and depth geometry policy")
    if resumed is not None:
        if any((args.num_samples is not None, args.epochs is not None, args.tasks is not None,
                args.experimental_unverified_depth, args.checkpoint is not None, args.lr is not None,
                args.seg_class_balance)):
            parser.error("Resume does not accept configuration/output overrides")
        config = copy.deepcopy(resumed["training_config"])
    elif fine_source is not None:
        config = copy.deepcopy(fine_source["training_config"])
    else:
        config = load_yaml_config(resolve(args.config))
        config["depth_geometry_policy"] = "experimental_image_plane" if args.experimental_unverified_depth else "verified"
        if args.tasks is not None:
            config["tasks"] = args.tasks
        if args.num_samples is not None:
            config["train"]["num_samples"] = args.num_samples
    if args.seg_class_balance:
        recipe = load_yaml_config(resolve(args.balance_config))
        config["loss"].setdefault("seg", {})["class_balance"] = {
            key: recipe[key] for key in ("mode", "min_weight", "max_weight")}
        if fine_source is not None:
            config["train"]["epochs"] = int(recipe["finetune_epochs"])
            config["train"]["lr"] = float(recipe["finetune_lr"])
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.lr is not None:
        config["train"]["lr"] = args.lr
    if fine_source is not None:
        config["dense_finetune_source"] = {"path": str(resolve(args.finetune_from)),
                                            "sha256": file_sha256(resolve(args.finetune_from)),
                                            "epoch": int(fine_source["epoch"]),
                                            "previous_class_balance": copy.deepcopy(
                                                fine_source["training_config"]["loss"].get("seg"))}
    validate_dense_config(config, config["tasks"])
    random.seed(config["train"]["seed"])
    torch.manual_seed(config["train"]["seed"])
    dataset_config, manifest = load_context(config)
    train_data, train_audit = make_dataset(config, dataset_config, manifest, "train", config["train"]["num_samples"])
    balance = resolve_seg_class_balance(config["loss"], train_audit, saved_weights_required=resumed is not None)
    if balance is not None:
        print("SEG_CLASS_BALANCE=TRAIN_ONLY formula=clip(sqrt(median_positive_count/count),min,max) "
              "normalization=SUM_TARGET_WEIGHTS")
        print("seg_class_weights=" + json.dumps(dict(zip(CITYSCAPES_CLASSES, balance["class_weights"]))))
    audits = {"train": train_audit}
    validation = None
    if not args.smoke_steps:
        validation, audits["validation"] = make_dataset(config, dataset_config, manifest, "validation")
        if train_audit["teacher_signatures"] != audits["validation"]["teacher_signatures"]:
            raise ValueError("Training/validation teachers or preprocessing differ")
    model_config, source, initialization = prepare_spatial_initialization(config, manifest)
    if fine_source is not None:
        validate_dense_data_join(fine_source, manifest, audits, model_config, initialization,
                                 smoke=bool(args.smoke_steps))
    if resumed is not None:
        if (resumed["split_sha256"] != manifest["split_sha256"]
                or resumed["metadata_sha256"] != manifest["metadata_sha256"]
                or resumed["label_audits"] != audits or resumed["initialization"] != initialization
                or resumed["model_config"] != model_config):
            raise ValueError("Resume source/config/split/label hash changed")
    experimental = config["depth_geometry_policy"] == "experimental_image_plane"
    output_path = resolve(args.resume or args.checkpoint or config["paths"][
        "experimental_checkpoint" if experimental else "checkpoint"])
    if output_path == resolve(config["paths"]["init_checkpoint"]):
        raise ValueError("Refusing to overwrite the Agent/Map initialization checkpoint")
    if args.finetune_from and output_path.resolve() == resolve(args.finetune_from).resolve():
        raise ValueError("Fine-tuning must not overwrite the original dense checkpoint")
    if not args.resume and not args.smoke_steps and output_path.exists():
        raise FileExistsError("Checkpoint already exists; use --resume or a new --checkpoint: " + str(output_path))
    print(f"tasks={config['tasks']} depth_geometry_policy={config['depth_geometry_policy']} "
          f"Agent_Map_DINO=FROZEN split_sha256={manifest['split_sha256']} "
          f"checkpoint_save={'OFF' if args.smoke_steps else output_path}")
    if args.preflight_only:
        print("SEG_DEPTH_PREFLIGHT=PASS no_student_model=True no_forward=True; runtime_NOT_VERIFIED")
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for server training; no automatic model/resolution fallback")
    device = initialize_cuda_device()
    if fine_source is not None:
        model = load_dense_model(fine_source)
        print(f"dense_finetune_loaded_epoch={fine_source['epoch']} optimizer=NEW "
              f"epochs={config['train']['epochs']} lr={config['train']['lr']}; original checkpoint preserved")
        del fine_source
    elif resumed is None:
        model = QUESTModel(**model_config)
        source_epoch = load_stage3_checkpoint(model, None, source, expected_map_kd_enabled=False)
        print(f"spatial_warm_start_epoch={source_epoch}; new_dense_tasks={config['tasks']}")
    else:
        model = load_dense_model(resumed)
    del source
    model.to(device)
    baseline_hash = frozen_spatial_fingerprint(model)
    parameters = configure_dense_training(model, config["tasks"])
    optimizer = torch.optim.AdamW(parameters, lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"])
    train_loader = loader(train_data, config, shuffle=not args.smoke_steps)
    total_steps = config["train"]["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(total_steps, 1), eta_min=config["train"]["lr"] * 0.05)
    start_epoch = 0
    if resumed is not None:
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scheduler.load_state_dict(resumed["scheduler_state_dict"])
        start_epoch = int(resumed["epoch"])
        torch.set_rng_state(resumed["torch_rng_state"])
        train_loader.generator.set_state(resumed["loader_rng_state"])
    print(f"parameters={model.parameter_summary()} frontend={model.dense_frontend_config['version']} "
          f"input=[B,8,3,252,448] tasks={config['tasks']} depth=[B,8,1,252,448]")
    if model.dense_frontend_config["version"] == GEOMETRY_DEPTH_VERSION:
        dense = model.dense_frontend
        print("depth_component_parameters=" + str({
            "image_projections": sum(p.numel() for p in dense.level_projections.parameters()),
            "ray_embedding": sum(p.numel() for p in dense.ray_embedding.parameters()),
            "geometry_encoder": sum(p.numel() for p in dense.geometry_blocks.parameters()),
            "dpt_decoder": sum(p.numel() for p in dense.depth_decoder.parameters())}))
    epochs = range(1) if args.smoke_steps else range(start_epoch, config["train"]["epochs"])
    last_gradient = {}
    for epoch in epochs:
        configure_dense_training(model, config["tasks"])
        sums, active_steps, skipped, iterator = {}, 0, 0, iter(train_loader)
        steps = args.smoke_steps or len(train_loader)
        before = [parameter.detach().clone() for parameter in parameters] if args.smoke_steps else None
        for step in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            output = dense_predictions(model, batch, device)
            losses = compute_seg_depth_loss(output, batch["dense_targets"], config["loss"], config["tasks"])
            if not bool(torch.isfinite(losses["total_loss"])):
                raise RuntimeError("Nonfinite dense training loss")
            if losses["seg_pixels"] + losses["depth_pixels"] == 0:
                skipped += 1
                continue
            losses["total_loss"].backward()
            last_gradient = gradient_health(model)
            if last_gradient["neck"] is None or (args.smoke_steps and last_gradient["neck"] <= 0):
                raise RuntimeError("Dense neck received no training signal")
            if "geometry" in last_gradient and (last_gradient["geometry"] is None
                    or (args.smoke_steps and last_gradient["geometry"] <= 0)):
                raise RuntimeError("Cross-view geometry encoder received no training signal")
            for task in config["tasks"]:
                if losses[task + "_pixels"] > 0 and (last_gradient[task] is None
                        or (args.smoke_steps and last_gradient[task] <= 0)):
                    raise RuntimeError("Enabled supervised head received no gradient: " + task)
            torch.nn.utils.clip_grad_norm_(parameters, config["train"]["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            if not args.smoke_steps:
                scheduler.step()
            active_steps += 1
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            if step == 0 or (step + 1) % config["train"]["log_interval"] == 0 or step + 1 == steps:
                print(f"epoch={epoch+1} step={step+1}/{steps} total={float(losses['total_loss']):.4f} "
                      f"seg={float(losses['seg_ce']):.4f} depth_log={float(losses['depth_log']):.4f} "
                      f"depth_absrel={float(losses['depth_absrel']):.4f}" +
                      (f" cross_view_coverage={float(output['depth_cross_view_coverage']):.3f}"
                       if "depth_cross_view_coverage" in output else ""), flush=True)
        if not active_steps or frozen_spatial_fingerprint(model) != baseline_hash:
            raise RuntimeError("No optimizer steps or frozen Agent/Map/backbone changed")
        print(f"epoch={epoch+1} means={json.dumps({key: value/active_steps for key, value in sums.items()})} "
              f"skipped_batches={skipped} gradient_RMS={last_gradient} frozen_Agent_Map=UNCHANGED")
        if args.smoke_steps:
            if not any(not torch.equal(old, new) for old, new in zip(before, parameters)):
                raise RuntimeError("Optimizer did not change dense parameters")
            model.eval()
            with torch.no_grad():
                full = model(**{key: batch[key].to(device) for key in
                                ("images", "intrinsics", "extrinsics", "ego_state", "distortion") if key in batch})
            print("available_task_shapes=" + str({key: tuple(full[key].shape) for key in (
                "agent_cls_logits", "agent_boxes", "map_cls_logits", "map_points", "seg_logits", "depth") if key in full}))
            print("SEG_DEPTH_SMOKE=PASS parameter_update=PASS gradients=FINITE checkpoint_saved=False")
        else:
            print("held_out_teacher_agreement=" + json.dumps(evaluate(model, validation, config, device), allow_nan=False))
            save_dense_checkpoint(output_path, model, optimizer, scheduler, epoch + 1, config,
                                  model_config, manifest, audits, baseline_hash, initialization, train_loader.generator)
            print("checkpoint_saved=" + str(output_path), flush=True)
    print(f"peak_allocated_MiB={torch.cuda.max_memory_allocated(device)/2**20:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
