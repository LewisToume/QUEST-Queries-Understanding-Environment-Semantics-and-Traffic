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
from quest.map_training import (
    STAGE3_MODULES, configure_stage3, load_stage2_for_stage3,
    load_vector_capacity_audit, mixed_stage3_loss, stage3_forward, stage3_kd_enabled,
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
        for module_name in STAGE3_MODULES
    }


def check_parameter_updates(model: torch.nn.Module, before: dict) -> None:
    for module_name in STAGE3_MODULES:
        changed = False
        for name, parameter in getattr(model, module_name).named_parameters():
            if name not in before[module_name]:
                continue
            current = parameter.detach().cpu()
            if not bool(torch.isfinite(current).all()):
                raise RuntimeError(f"non-finite parameter after step: {module_name}.{name}")
            changed |= not torch.equal(current, before[module_name][name])
        print(f"parameter_update {module_name}={'PASS' if changed else 'FAIL'}", flush=True)
        if not changed:
            raise RuntimeError(f"optimizer did not change any parameter in {module_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="One real CUDA Stage 3 GT-only batch; no files saved")
    parser.add_argument("--sample-index", type=int, help="Original training metadata index; default first manifest frame")
    args = parser.parse_args()
    config = load_yaml_config(PROJECT_ROOT / "configs/stage3_map.yaml")
    stage1 = load_yaml_config(PROJECT_ROOT / "configs/stage1.yaml")
    stage2 = load_yaml_config(PROJECT_ROOT / "configs/stage2_agent.yaml")
    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    if stage3_kd_enabled(config):
        raise ValueError("this check requires GT-only Stage 3 (map KD disabled)")
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
    sample_index = indices[0] if args.sample_index is None else args.sample_index
    if sample_index not in indices:
        raise ValueError("sample-index must be an original index in the training manifest")
    model_config.update(C_map=len(MAP_CLASS_NAMES), N_map=query_count, P=20)
    dataset = build_dataset(model_config, config, stage1, None, [sample_index], provenance)
    if preflight_vectors(dataset, query_count) != provenance:
        raise ValueError("sample Vector GT differs from certified capacity audit")
    preflight_agents(dataset)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False,
                                  num_workers=0, collate_fn=collate_stage3)))
    if tuple(batch["images"].shape) != (1, 8, 3, 252, 448):
        raise ValueError(f"unexpected image shape: {tuple(batch['images'].shape)}")
    for key in ("images", "intrinsics", "extrinsics", "ego_state"):
        if not bool(torch.isfinite(batch[key]).all()):
            raise ValueError(f"non-finite input: {key}")
        print(f"input {key}={tuple(batch[key].shape)}", flush=True)
    print(f"sample_index={sample_index} token={batch['sample_token'][0]} "
          f"split_sha256={manifest['split_sha256']} map_queries={query_count}", flush=True)
    model = QUESTModel(**model_config)
    epoch = load_stage2_for_stage3(model, load_checkpoint_cpu(resolve(config["paths"]["stage2_checkpoint"])))
    print(f"stage2_loaded_epoch={epoch}", flush=True)
    model.to(device).eval()
    optimizer = configure_stage3(model, None, config["train"])
    for name in STAGE3_MODULES:
        getattr(model, name).train()
    model.geometry_lift.ego_mlp.eval()
    for name, module in (("DINOv2", model.backbone), ("ego_mlp", model.geometry_lift.ego_mlp)):
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError(f"{name} must remain frozen")
    target = prepare_navformer_agent_batch(batch, device)
    vectors = [{key: value.to(device) if torch.is_tensor(value) else value
                for key, value in record.items()} for record in batch["vector_targets"]]
    effective = {**config, "agent_train": stage2["train"], "agent_loss": stage2["agent_loss"],
                 "map_loss": config["map"]}
    optimizer.zero_grad(set_to_none=True)
    print("forward_start", flush=True)
    predictions = stage3_forward(model, None, batch, device)
    for key, value in predictions.items():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"non-finite output: {key}")
    for key in ("agent_cls_logits", "agent_boxes", "agent_velocity", "map_cls_logits", "map_points"):
        print(f"output {key}={tuple(predictions[key].shape)}", flush=True)
    losses = mixed_stage3_loss(model, predictions, vectors, None, None, None, None, target, effective)
    for key, value in losses.items():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"non-finite loss: {key}")
        print(f"{key}={float(value.detach()):.8f}", flush=True)
    if float(losses["total_loss"].detach()) <= 0:
        raise RuntimeError("expected positive total loss")
    print("backward_start", flush=True)
    losses["total_loss"].backward()
    for name in STAGE3_MODULES:
        module = getattr(model, name)
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        if any(not bool(torch.isfinite(grad).all()) for grad in gradients):
            raise RuntimeError(f"non-finite gradient: {name}")
        if not any(bool(torch.count_nonzero(grad)) for grad in gradients):
            raise RuntimeError(f"missing or zero gradient: {name}")
        print(f"gradient_RMS {name}={gradient_rms(module):.9e}", flush=True)
    for module in (model.backbone, model.geometry_lift.ego_mlp):
        if any(p.grad is not None for p in module.parameters()):
            raise RuntimeError("frozen DINOv2/ego_mlp unexpectedly received gradients")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        float(config["train"]["gradient_clip_norm"]))
    if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("non-finite total gradient norm during clipping")
    before = snapshot_trainable(model)
    optimizer.step()
    check_parameter_updates(model, before)
    torch.cuda.synchronize(device)
    print("STAGE3_ONE_BATCH = PASS; optimizer_steps=1 checkpoint_saved=False", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("STAGE3_ONE_BATCH = FAIL", flush=True)
        raise
    finally:
        if torch.cuda.is_initialized():
            print(f"peak_allocated_MiB={torch.cuda.max_memory_allocated() / 2**20:.2f} "
                  f"peak_reserved_MiB={torch.cuda.max_memory_reserved() / 2**20:.2f}", flush=True)
