from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.agent_training import load_checkpoint_cpu, prepare_navformer_agent_batch
from quest.bev_pretraining import gradient_rms
from quest.map_training import (
    STAGE3_MODULES, configure_stage3, load_stage2_for_stage3,
    load_vector_capacity_audit, mixed_stage3_loss, stage3_forward, stage3_kd_enabled,
    stage3_trainable_modules, frozen_agent_fingerprint, stage3_model_config,
)
from quest.model import QUESTModel, QUEST_ARCHITECTURE_VERSION
from quest.stage3_dataset import collate_stage3
from quest.stage3_split import load_stage3_split
from quest.utils import load_yaml_config
from quest.vector_map_labels import MAP_CLASS_NAMES
from scripts.run_navformer_openscene_teacher import load_infos
from scripts.train_stage3_map import build_dataset, preflight_agents, preflight_vectors, resolve


def snapshot_trainable(model: torch.nn.Module) -> dict:
    # Keep comparison copies on CPU so the check does not double GPU parameter memory.
    return {
        module_name: {name: parameter.detach().cpu().clone()
                      for name, parameter in getattr(model, module_name).named_parameters()
                      if parameter.requires_grad}
        for module_name in (*STAGE3_MODULES, *(("map_spatial",) if getattr(model, "map_spatial", None) is not None else ()))
    }


def check_parameter_updates(model: torch.nn.Module, before: dict, *, verbose: bool = True) -> None:
    for module_name in before:
        if not before[module_name]:
            continue
        changed = False
        for name, parameter in getattr(model, module_name).named_parameters():
            if name not in before[module_name]:
                continue
            current = parameter.detach().cpu()
            if not bool(torch.isfinite(current).all()):
                raise RuntimeError(f"non-finite parameter after step: {module_name}.{name}")
            changed |= not torch.equal(current, before[module_name][name])
        if verbose:
            print(f"parameter_update {module_name}={'PASS' if changed else 'FAIL'}", flush=True)
        if not changed:
            raise RuntimeError(f"optimizer did not change any parameter in {module_name}")


def select_training_indices(indices: list[int], sample_index: int | None, steps: int) -> list[int]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if sample_index is not None and sample_index not in indices:
        raise ValueError("sample-index must be an original index in the training manifest")
    start = indices.index(sample_index) if sample_index is not None else 0
    selected = indices[start:start + steps]
    if len(selected) != steps:
        raise ValueError("not enough training frames for requested steps; no wrap or validation frames allowed")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Short real CUDA Stage 3 GT-only check; no files saved")
    parser.add_argument("--sample-index", type=int, help="Original training metadata index; default first manifest frame")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--overfit-samples", type=int,
                        help="Repeat this many real training frames for --steps; no checkpoint is saved")
    parser.add_argument("--verbose", action="store_true", help="Also print shapes and individual losses")
    args = parser.parse_args()
    config = load_yaml_config(PROJECT_ROOT / "configs/stage3_map.yaml")
    stage1 = load_yaml_config(PROJECT_ROOT / "configs/stage1.yaml")
    stage2 = load_yaml_config(PROJECT_ROOT / "configs/stage2_agent.yaml")
    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    if stage3_kd_enabled(config):
        raise ValueError("this check requires GT-only Stage 3 (map KD disabled)")
    model_config = stage3_model_config(model_config, config)
    if (tuple(stage1["dataset"]["camera_names"]) != tuple(model_config["camera_names"])
            or len(model_config["camera_names"]) != 8
            or int(model_config["architecture_version"]) != QUEST_ARCHITECTURE_VERSION
            or tuple(stage1["dataset"]["image_size"]) != (252, 448)):
        raise ValueError("expected QUEST V3 eight-camera 252x448 inputs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this check will not fall back to CPU")
    if not resolve(model_config["local_backbone_dir"]).is_dir():
        raise FileNotFoundError("local DINOv2 weights are required; no download will be attempted")
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    print(f"device={torch.cuda.get_device_name(device)} MAP_KD_DISABLED=True checkpoint_save=OFF", flush=True)
    metadata_path = resolve(stage1["dataset"]["metadata_path"])
    manifest = load_stage3_split(resolve(config["paths"]["split_manifest_path"]),
                                 load_infos(metadata_path), metadata_path)
    query_count, provenance = load_vector_capacity_audit(
        resolve(config["paths"]["vector_capacity_audit_path"]), config, manifest)
    indices = [row["index"] for row in manifest["train"]["frames"]]
    if args.overfit_samples is not None and not 1 <= args.overfit_samples <= args.steps:
        raise ValueError("overfit-samples must be between 1 and steps")
    selected = select_training_indices(indices, args.sample_index, args.overfit_samples or args.steps)
    model_config.update(C_map=len(MAP_CLASS_NAMES), N_map=query_count, P=20)
    dataset = build_dataset(model_config, config, stage1, None, selected, provenance)
    if preflight_vectors(dataset, query_count) != provenance:
        raise ValueError("sample Vector GT differs from certified capacity audit")
    preflight_agents(dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_stage3)
    print(f"steps={args.steps} batch_size=1 map_queries={query_count} "
          f"split_sha256={manifest['split_sha256']}", flush=True)
    model = QUESTModel(**model_config)
    epoch = load_stage2_for_stage3(model, load_checkpoint_cpu(resolve(config["paths"]["stage2_checkpoint"])))
    print(f"stage2_loaded_epoch={epoch}", flush=True)
    model.to(device).eval()
    optimizer = configure_stage3(model, None, config["train"])
    print(f"map_architecture={(model.map_architecture or {}).get('version', 'legacy_instance_v1')} "
          f"parameters={model.parameter_summary()}", flush=True)
    trainable_modules = stage3_trainable_modules(config["train"])
    for name in trainable_modules:
        getattr(model, name).train()
    model.geometry_lift.ego_mlp.eval()
    if model.map_spatial is not None:
        model.map_spatial.geometry_lift.ego_mlp.eval()
    for name, module in (("DINOv2", model.backbone), ("ego_mlp", model.geometry_lift.ego_mlp)):
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError(f"{name} must remain frozen")
    effective = {**config, "agent_train": stage2["train"], "agent_loss": stage2["agent_loss"],
                 "map_loss": config["map"]}
    before = snapshot_trainable(model)
    frozen_before = (frozen_agent_fingerprint(model)
                     if config["train"].get("freeze_agent_and_bev", False) else None)
    sums = {"map": 0.0, "agent": 0.0, "total": 0.0}
    batches = itertools.islice(itertools.cycle(loader), args.steps) if args.overfit_samples else loader
    initial_error = final_error = None
    for step, batch in enumerate(batches, 1):
        sample_index = selected[(step - 1) % len(selected)]
        context = f"step={step} index={sample_index} token={batch['sample_token'][0]}"
        if tuple(batch["images"].shape) != (1, 8, 3, 252, 448):
            raise ValueError(f"{context}: unexpected image shape: {tuple(batch['images'].shape)}")
        for key in ("images", "intrinsics", "extrinsics", "ego_state"):
            if not bool(torch.isfinite(batch[key]).all()):
                raise ValueError(f"{context}: non-finite input: {key}")
        optimizer.zero_grad(set_to_none=True)
        target = prepare_navformer_agent_batch(batch, device)
        vectors = [{key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in record.items()} for record in batch["vector_targets"]]
        predictions = stage3_forward(model, None, batch, device)
        for key, value in predictions.items():
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"{context}: non-finite output: {key}")
        losses = mixed_stage3_loss(model, predictions, vectors, None, None, None, None, target, effective)
        for key, value in losses.items():
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"{context}: non-finite loss: {key}")
        if float(losses["total_loss"].detach()) <= 0:
            raise RuntimeError(f"{context}: expected positive total loss")
        if args.verbose:
            for key in ("agent_cls_logits", "agent_boxes", "agent_velocity", "map_cls_logits", "map_points"):
                print(f"output {key}={tuple(predictions[key].shape)}", flush=True)
            print(" ".join(f"{key}={float(value.detach()):.6f}" for key, value in losses.items()), flush=True)
        losses["total_loss"].backward()
        for name in trainable_modules:
            gradients = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
            if any(not bool(torch.isfinite(grad).all()) for grad in gradients):
                raise RuntimeError(f"{context}: non-finite gradient: {name}")
            if not any(bool(torch.count_nonzero(grad)) for grad in gradients):
                raise RuntimeError(f"{context}: missing or zero gradient: {name}")
        frozen_modules = [model.backbone, model.geometry_lift.ego_mlp]
        if model.map_spatial is not None:
            frozen_modules.append(model.map_spatial.geometry_lift.ego_mlp)
        for module in frozen_modules:
            if any(p.grad is not None for p in module.parameters()):
                raise RuntimeError(f"{context}: frozen DINOv2/ego_mlp unexpectedly received gradients")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            float(config["train"]["gradient_clip_norm"]))
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"{context}: non-finite total gradient norm during clipping")
        optimizer.step()
        values = {"map": float(losses["raw_map_gt_loss"].detach()),
                  "agent": float(losses["raw_agent_loss"].detach()),
                  "total": float(losses["total_loss"].detach())}
        for key, value in values.items():
            sums[key] += value
        final_error = float(losses["map_matched_point_error_m"])
        if initial_error is None:
            initial_error = final_error
        if step == 1 or step == args.steps or step % (20 if args.overfit_samples else 1) == 0:
            print(f"step={step}/{args.steps} index={sample_index} "
                  + " ".join(f"{key}={value:.5f}" for key, value in values.items())
                  + f" point_error_m={final_error:.3f}", flush=True)
        # Release each step's graph before starting the next forward pass.
        del predictions, losses, vectors, target
    check_parameter_updates(model, before, verbose=args.verbose)
    if frozen_before is not None and frozen_agent_fingerprint(model) != frozen_before:
        raise RuntimeError("frozen shared BEV/Agent changed during short check")
    torch.cuda.synchronize(device)
    print("mean_loss " + " ".join(f"{key}={value / args.steps:.5f}" for key, value in sums.items()), flush=True)
    print("final_gradient_RMS " + " ".join(
        f"{name}={gradient_rms(getattr(model, name)):.3e}" for name in trainable_modules), flush=True)
    if model.map_spatial is not None:
        print(f"map_visual_lift_gradient_RMS={gradient_rms(model.map_spatial.geometry_lift):.3e} "
              f"map_bev_encoder_gradient_RMS={gradient_rms(model.map_spatial.layers):.3e} "
              f"map_dense_head_gradient_RMS={gradient_rms(model.map_spatial.dense_head):.3e}", flush=True)
    status = "STAGE3_ONE_BATCH" if args.steps == 1 else "STAGE3_SHORT_RUN"
    if args.overfit_samples:
        print(f"overfit_frames={len(selected)} initial_point_error_m={initial_error:.3f} "
              f"final_point_error_m={final_error:.3f} "
              "fit_quality_requires_manual_review=True", flush=True)
    print(f"{status} = PASS; optimizer_steps={args.steps} gradients=finite_nonzero_all_steps "
          f"parameter_updates={len(trainable_modules)}/{len(trainable_modules)} "
          f"frozen_agent_unchanged={frozen_before is not None} checkpoint_saved=False", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("STAGE3_CHECK = FAIL", flush=True)
        raise
    finally:
        if torch.cuda.is_initialized():
            print(f"peak_allocated_MiB={torch.cuda.max_memory_allocated() / 2**20:.2f} "
                  f"peak_reserved_MiB={torch.cuda.max_memory_reserved() / 2**20:.2f}", flush=True)
