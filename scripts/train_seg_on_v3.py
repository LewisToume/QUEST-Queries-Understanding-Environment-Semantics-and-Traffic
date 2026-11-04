"""One-command SegFormer-head adaptation; preserve the complete trained Depth V3."""

import argparse
import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from quest.agent_training import load_checkpoint_cpu
from quest.cuda_runtime import initialize_cuda_device
from quest.geometry_depth import TEMPORAL_DEPTH_VERSION
from quest.model import QUESTModel
from quest.seg_on_v3 import (SEG_ON_V3_STAGE, DINOSegFormerBranch, SegOnFrozenV3,
    full_base_fingerprint, initialize_segformer_decoder, segmentation_loss, load_seg_on_v3)
from quest.seg_depth_dataset import collate_seg_depth
from quest.seg_depth_losses import empty_dense_metrics, update_dense_metrics, summarize_dense_metrics
from quest.seg_depth_training import load_dense_model, validate_dense_checkpoint, resolve_seg_class_balance
from quest.seg_teacher import validate_teacher_config
from quest.stage3_split import file_sha256, selected_frames
from quest.temporal_depth_dataset import TemporalSegDepthDataset
from quest.utils import load_yaml_config
from scripts.train_seg_depth import load_context, resolve


def inputs(batch, device):
    return {**{key: batch[key].to(device) for key in ("images", "intrinsics", "extrinsics", "ego_state", "distortion")},
            "history": {key: value.to(device) for key, value in batch["history"].items()}}


def dataset(config, base_metadata, data_config, manifest, split, count=None):
    rows = selected_frames(manifest, split)
    if count is not None and not 0 < count <= len(rows):
        raise ValueError("Invalid requested Seg frame count")
    result = TemporalSegDepthDataset(data_config, [row["index"] for row in (rows[:count] if count else rows)],
        resolve(config["seg_labels"]), ROOT / "data/unused_depth_labels", manifest["metadata_sha256"],
        manifest["split_sha256"], tasks=["seg"], require_seg_for_depth=False, missing_policy="error",
        required_depth_geometry="raw_opencv", experimental_depth=False,
        temporal_options=base_metadata["dense_frontend"]["temporal"], history_source_indices=[row["index"] for row in rows])
    audit = result.audit(include_seg_class_counts=split == "train")
    print(f"SEG_PREFLIGHT split={split} frames={len(result)} missing={audit['tasks']['seg']['missing']} "
          f"valid_pixels={audit['tasks']['seg']['valid_pixels']}", flush=True)
    return result, audit


def make_loader(data, *, shuffle=False, generator=None):
    return DataLoader(data, batch_size=1, shuffle=shuffle, num_workers=0,
                      collate_fn=collate_seg_depth, generator=generator)


@torch.no_grad()
def evaluate(model, data, device, *, per_class=False):
    model.eval()
    metrics = empty_dense_metrics()
    for batch in make_loader(data):
        prediction = model.forward_seg(batch["images"].to(device))
        update_dense_metrics(metrics, prediction, batch["dense_targets"], {}, ["seg"])
    result = summarize_dense_metrics(metrics)
    result = {key: value for key, value in result.items() if not key.startswith("depth_")}
    if per_class:
        from quest.seg_teacher import CITYSCAPES_CLASSES

        cm = metrics["confusion"].double()
        rows = []
        for index, name in enumerate(CITYSCAPES_CLASSES):
            truth, predicted, hit = float(cm[index].sum()), float(cm[:, index].sum()), float(cm[index, index])
            union = truth + predicted - hit
            rows.append({"class": name, "teacher_pixels": int(truth), "student_pixels": int(predicted),
                         "iou": hit / union if union else None, "recall": hit / truth if truth else None})
        result["seg_per_class"] = rows
    return result


def save(path, model, optimizer, scheduler, epoch, config, base_metadata, teacher_config, audits,
         base_hash, source_identity, generator, history):
    if full_base_fingerprint(model.base) != base_hash:
        raise RuntimeError("Frozen V3 changed; refusing checkpoint")
    payload = dict(stage=SEG_ON_V3_STAGE, epoch=epoch, training_config=config, base_metadata=base_metadata,
        teacher_config=teacher_config, model_state_dict=model.state_dict(), base_sha256=base_hash,
        optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
        label_audits=audits, source_identity=source_identity, loader_rng_state=generator.get_state(),
        torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state(),
        python_rng_state=random.getstate(), validation_history=history,
        trained_tasks=["seg"], available_tasks=["agent", "map", "depth", "seg"],
        depth_policy="unchanged_temporal_v3", agent_policy="original_current_only",
        reference="OFFLINE_SEGFORMER_PSEUDO_NOT_REAL_GT")
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seg_on_depth_v3_12ep.yaml")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate", type=Path)
    args = parser.parse_args(argv)
    if args.run_all:
        if any((args.preflight_only, args.smoke_steps, args.num_samples, args.resume, args.evaluate)):
            parser.error("--run-all does not accept partial/resume/evaluation overrides")
        config = load_yaml_config(resolve(args.config))
        destination = resolve(config["checkpoint"])
        if destination.exists() or destination.with_stem(destination.stem + "_best").exists():
            raise FileExistsError("Output already exists; use explicit --resume or --evaluate")
        commands = [
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_seg_on_v3.py", "-q"],
            [sys.executable, "-u", __file__, "--config", args.config, "--num-samples", "2", "--smoke-steps", "4"],
            [sys.executable, "-u", __file__, "--config", args.config]]
        for command in commands:
            subprocess.run(command, cwd=ROOT, check=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
        return 0
    if args.smoke_steps < 0 or (args.resume and (args.smoke_steps or args.num_samples)) or (args.resume and args.evaluate):
        parser.error("Invalid smoke/resume/evaluation combination")
    saved = load_checkpoint_cpu(resolve(args.resume or args.evaluate)) if args.resume or args.evaluate else None
    config = copy.deepcopy(saved["training_config"]) if saved else load_yaml_config(resolve(args.config))
    if config["stage"] != SEG_ON_V3_STAGE or config["train"]["batch_size"] != 1:
        raise ValueError("Unsupported Seg-on-V3 stage/batch")
    output = resolve(config["checkpoint"])
    if (not saved and not args.smoke_steps and not args.preflight_only
            and (output.exists() or output.with_stem(output.stem + "_best").exists())):
        raise FileExistsError("Refusing to overwrite Seg run; use --resume")
    source_path, bundle = resolve(config["base_checkpoint"]), resolve(config["teacher_bundle"])
    if output.resolve() == source_path.resolve():
        raise ValueError("Refusing to overwrite V3")
    source = load_checkpoint_cpu(source_path)
    validate_dense_checkpoint(source)
    if source["dense_frontend"]["version"] != TEMPORAL_DEPTH_VERSION or source["trained_dense_tasks"] != ["depth"]:
        raise ValueError("Use the selected trained Depth V3 source")
    teacher_config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    validate_teacher_config(teacher_config)
    weight = bundle / ("model.safetensors" if (bundle / "model.safetensors").is_file() else "pytorch_model.bin")
    identity = {"base_sha256": file_sha256(source_path), "teacher_sha256": file_sha256(weight),
                "teacher_config_sha256": file_sha256(bundle / "config.json")}
    if saved and saved["source_identity"] != identity:
        raise ValueError("V3/teacher source changed since training")
    # Existing context helper reads dataset/split paths without constructing models.
    context = {"paths": {"dataset_config": config["dataset_config"], "split_manifest": config["split_manifest"]}}
    data_config, manifest = load_context(context)
    if (source["split_sha256"] != manifest["split_sha256"]
            or source["metadata_sha256"] != manifest["metadata_sha256"]):
        raise ValueError("Seg data must use the same V3 metadata/split")
    seed = config["train"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    count = args.num_samples if args.num_samples is not None else config["train"]["num_samples"]
    train, train_audit = dataset(config, source, data_config, manifest, "train", count)
    validation, validation_audit = dataset(config, source, data_config, manifest, "validation")
    audits = {"train": train_audit, "validation": validation_audit}
    resolve_seg_class_balance(config["loss"], train_audit, saved_weights_required=bool(saved))
    if saved and saved["label_audits"] != audits:
        raise ValueError("Seg labels/context changed since training")
    if train_audit["teacher_signatures"] != validation_audit["teacher_signatures"]:
        raise ValueError("Mixed training/validation Seg teacher versions")
    signature = train_audit["teacher_signatures"]["seg"]
    if signature[3] != identity["teacher_sha256"] or signature[4] != identity["teacher_config_sha256"]:
        raise ValueError("Exported Seg labels and decoder initialization must use the exact same teacher weights/config")
    print("Seg=TRAINABLE Depth_Agent_Map_DINO=FROZEN epochs=" + str(config["train"]["epochs"]) +
          " Agent_route=original_current_only teacher=SegFormer_B5_Cityscapes19", flush=True)
    if args.preflight_only:
        print("SEG_ON_V3_PREFLIGHT=PASS no_model_load=True no_GPU=True")
        return 0
    device = initialize_cuda_device()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Seg profile requires CUDA bf16; no silent fallback")
    metadata = {key: value for key, value in source.items() if key not in (
        "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "cuda_rng_state_all")}
    if saved:
        model = load_seg_on_v3(saved).to(device)
    else:
        base = load_dense_model(source)
        branch = DINOSegFormerBranch(base.hidden_dim, teacher_config)
        keys = initialize_segformer_decoder(branch, bundle)
        print(f"teacher_decode_head=STRICT_PASS loaded_state_keys={keys}; teacher_encoder_not_loaded=True")
        model = SegOnFrozenV3(base, branch).to(device)
    del source
    base_hash = full_base_fingerprint(model.base)
    parameters = list(model.seg_branch.parameters())
    print(f"parameters_total={sum(p.numel() for p in model.parameters())} trainable_seg={sum(p.numel() for p in parameters)}")
    if args.evaluate:
        print("held_out_teacher_agreement=" + json.dumps(evaluate(model, validation, device, per_class=True)))
        return 0
    optimizer = torch.optim.AdamW(parameters, lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    loader = make_loader(train, shuffle=not args.smoke_steps, generator=generator)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["train"]["epochs"] * len(loader),
                                                          eta_min=config["train"]["lr"] * .05)
    history, start = [], 0
    if args.resume:
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        generator.set_state(saved["loader_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        torch.cuda.set_rng_state(saved["cuda_rng_state"])
        random.setstate(saved["python_rng_state"])
        start, history = saved["epoch"], saved["validation_history"]
    weights = torch.tensor(config["loss"]["seg"]["class_weights"], dtype=torch.float32, device=device)
    probe_position = next((i for i, plan in enumerate(train.history_plans) if plan["history_index"] is not None), 0)
    probe = inputs(collate_seg_depth([train[probe_position]]), device)
    model.eval()
    with torch.no_grad():
        initial = {key: value.cpu() for key, value in model(**probe).items()
                   if key in ("depth", "agent_boxes", "agent_cls_logits", "agent_velocity", "map_cls_logits", "map_points")}
    for epoch in range(start, start + 1 if args.smoke_steps else config["train"]["epochs"]):
        model.train()
        total, active = 0., 0
        iterator = iter(loader)
        steps = args.smoke_steps or len(loader)
        for step in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_seg(batch["images"].to(device))["seg_logits"]
            loss = segmentation_loss(logits, batch["dense_targets"], weights)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite Seg loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            if not float(gradient) > 0:
                raise RuntimeError("No Seg gradient")
            optimizer.step()
            if not args.smoke_steps:
                scheduler.step()
            total += float(loss)
            active += 1
            if step == 0 or (step + 1) % config["train"]["log_interval"] == 0 or step + 1 == steps:
                print(f"epoch={epoch+1} step={step+1}/{steps} seg_ce={float(loss):.4f}", flush=True)
        model.eval()
        if full_base_fingerprint(model.base) != base_hash:
            raise RuntimeError("Frozen V3 weights or buffers changed")
        with torch.no_grad():
            after = model(**probe)
            for key, value in initial.items():
                torch.testing.assert_close(after[key].cpu(), value, rtol=0, atol=0)
        print(f"epoch={epoch+1} mean_seg_ce={total/active:.5f} frozen_V3_weights=UNCHANGED "
              "probe_Depth_Agent_Map=EXACT", flush=True)
        if args.smoke_steps:
            print("SEG_ON_V3_SMOKE=PASS gradients=FINITE_NONZERO checkpoint_saved=False")
            break
        metrics = evaluate(model, validation, device)
        print("held_out_teacher_agreement=" + json.dumps(metrics), flush=True)
        previous_best = max((row["validation"]["seg_teacher_mIoU"] for row in history), default=-1.)
        history.append({"epoch": epoch + 1, "mean_seg_ce": total/active, "validation": metrics})
        save(output, model, optimizer, scheduler, epoch + 1, config, metadata, teacher_config,
             audits, base_hash, identity, generator, history)
        if metrics["seg_teacher_mIoU"] > previous_best:
            save(output.with_stem(output.stem + "_best"), model, optimizer, scheduler, epoch + 1, config, metadata,
                 teacher_config, audits, base_hash, identity, generator, history)
        print("checkpoint_saved=" + str(output), flush=True)
    print(f"peak_allocated_MiB={torch.cuda.max_memory_allocated(device)/2**20:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
