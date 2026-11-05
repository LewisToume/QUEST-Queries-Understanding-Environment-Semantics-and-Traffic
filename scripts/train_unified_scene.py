"""Train one shared trainable scene trunk on Agent/Map/Seg/Depth jointly."""

import argparse
import copy
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from quest.agent_training import load_checkpoint_cpu
from quest.cuda_runtime import initialize_cuda_device
from quest.map_training import class_aware_vector_matches, denormalize_points, load_vector_capacity_audit
from quest.seg_depth_diagnostics import DenseEvaluationDiagnostics
from quest.seg_depth_training import resolve_seg_class_balance
from quest.seg_on_v3 import SEG_ON_V3_STAGE, full_base_fingerprint, load_seg_on_v3
from quest.stage3_split import file_sha256, selected_frames
from quest.temporal_agent_training import empty_agent_agreement, summarize_agent_agreement, update_agent_agreement
from quest.unified_dataset import FourTaskDataset, collate_four_tasks
from quest.unified_scene import UNIFIED_SCENE_VERSION, UnifiedQUEST
from quest.unified_training import (TASKS, four_task_losses, joint_backward, quality_gate,
                                    scaled_task_losses, shared_gradient_audit)
from quest.utils import load_yaml_config
from quest.vector_map_labels import MAP_CLASS_NAMES
from scripts.train_seg_depth import load_context, resolve
from scripts.train_seg_on_v3 import inputs


def loader(data, *, shuffle=False, generator=None):
    return DataLoader(data, batch_size=1, shuffle=shuffle, generator=generator,
                      num_workers=0, collate_fn=collate_four_tasks)


def build_data(config, metadata, data_config, manifest, split, provenance, count=None):
    frames = selected_frames(manifest, split)
    if count is not None and not 0 < count <= len(frames):
        raise ValueError("Invalid joint sample count")
    current = frames[:count] if count is not None else frames
    old = metadata["model_config"]
    data = FourTaskDataset(data_config, [row["index"] for row in current],
        resolve(config["paths"]["seg_labels"]), resolve(config["paths"]["depth_labels"]),
        manifest["metadata_sha256"], manifest["split_sha256"], tasks=("seg", "depth"),
        missing_policy="error", experimental_depth=True, require_seg_for_depth=False,
        required_depth_geometry="raw_opencv", temporal_options=metadata["dense_frontend"]["temporal"],
        history_source_indices=[row["index"] for row in frames], agent_labels=resolve(config["paths"]["agent_labels"]),
        vector_dir=resolve(config["paths"]["vector_labels"]), vector_provenance=provenance,
        xy_range=(old["x_range"][0], old["y_range"][0], old["x_range"][1], old["y_range"][1]),
        map_queries=config["model"]["map_queries"])
    audit = data.audit(config["loss"]["dense"]["depth"], include_seg_class_counts=split == "train")
    print(f"FOUR_TASK_PREFLIGHT split={split} frames={len(data)} tasks=4 history={data.temporal_audit['with_history']} missing=0", flush=True)
    return data, audit


def forward(model, batch, device):
    # Baseline wrapper owns its original mixed-precision policy.
    if isinstance(model, UnifiedQUEST):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            result = model(**inputs(batch, device))
    else:
        result = model(**inputs(batch, device))
    required = ("agent_cls_logits", "agent_boxes", "map_cls_logits", "map_points", "seg_logits", "depth")
    if not all(key in result and bool(torch.isfinite(result[key]).all()) for key in required):
        raise RuntimeError("Invalid/missing joint prediction")
    return {key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
            for key, value in result.items()}


@torch.no_grad()
def evaluate(model, data, device, settings, eval_config):
    model.eval()
    base = model if isinstance(model, UnifiedQUEST) else model.base
    dense = DenseEvaluationDiagnostics(data.images.camera_names, settings["dense"], ("seg", "depth"))
    agent = empty_agent_agreement()
    infos = dict(zip(data.source_indices, data.images.infos))
    maps = {name: {"predictions": 0, "gt_count": 0, "matched": 0} for name in MAP_CLASS_NAMES}
    xy = (base.geometry_lift.x_range[0], base.geometry_lift.y_range[0], base.geometry_lift.x_range[1], base.geometry_lift.y_range[1])
    agent_config = {"temporal_agent": {"recipe": settings["agent"]}}
    for batch in loader(data):
        output = forward(model, batch, device)
        dense.update(output, batch["dense_targets"], [infos[index] for index in batch["source_index"]],
                     batch["source_index"], batch["sample_token"])
        update_agent_agreement(agent, base, output, batch, agent_config, device)
        scores, labels = output["map_cls_logits"][0].softmax(-1).max(-1)
        keep = (labels != 3) & (scores >= eval_config["map_confidence_threshold"])
        labels = labels[keep].cpu()
        points = denormalize_points(output["map_points"][0, keep].cpu(), xy)
        target = batch["vector_targets"][0]
        matches = class_aware_vector_matches(labels, points, target, eval_config["map_match_distance_m"], "ordered")
        for i, name in enumerate(MAP_CLASS_NAMES):
            maps[name]["predictions"] += int((labels == i).sum())
            maps[name]["gt_count"] += int((target["class_ids"] == i).sum())
            maps[name]["matched"] += sum(int(labels[pred]) == i for pred, _, _, _ in matches)
    for row in maps.values():
        row["precision"] = row["matched"] / row["predictions"] if row["predictions"] else 0.
        row["recall"] = row["matched"] / row["gt_count"] if row["gt_count"] else 0.
    details = dense.summarize()
    return {"frames": len(data), "dense": details["overall"],
        **{key: details[key] for key in ("seg_per_class", "depth_distance_bins", "per_camera", "per_scene")},
        "agent": summarize_agent_agreement(agent, agent_config)["classes"], "map": maps,
        "reference": "AGENT_SEG_DEPTH_TEACHER_PSEUDO_MAP_NUPLAN_VECTOR_GT", "official_mAP": False,
        "map_distance_metric": "ordered", "map_threshold_m": eval_config["map_match_distance_m"],
        "agent_eval_protocol": settings["agent"]["eval"], "map_confidence_threshold": eval_config["map_confidence_threshold"]}


def brief(metrics):
    return {"dense": metrics["dense"],
            **{task: {name: {key: value[key] for key in ("precision", "recall")}
                       for name, value in metrics[task].items()} for task in ("agent", "map")}}


def save(path, model, config, settings, audits, manifest, source_sha, initialization,
         optimizer, scheduler, generator, pc_generator, epoch, scales, baseline, history):
    payload = {"stage": UNIFIED_SCENE_VERSION, "model_config": model.model_config, "profile": model.profile,
        "model_state_dict": model.state_dict(), "training_config": config, "loss_settings": settings,
        "label_audits": audits, "split_sha256": manifest["split_sha256"], "metadata_sha256": manifest["metadata_sha256"],
        "source_sha256": source_sha, "initialization": initialization, "epoch": epoch,
        "trained_tasks": config["train"]["active_tasks"], "depth_geometry_policy": "experimental_image_plane",
        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
        "loader_rng": generator.get_state(), "pcgrad_rng": pc_generator.get_state(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "python_rng": random.getstate(),
        "task_loss_scales_train_only": scales, "baseline_validation": baseline, "validation_history": history}
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


def optimizer_for(model, config):
    new_names = ("camera_projection.", "bev_to_camera.", "query_to_camera.", "context_gate", "query_type")
    new, shared = [], []
    for name, parameter in model.trunk.named_parameters():
        if parameter.requires_grad:
            (new if name.startswith(new_names) else shared).append(parameter)
    heads = [p for name, p in model.heads.named_parameters() if not name.startswith(("seg.", "map_dense."))]
    fresh_heads = list(model.heads.seg.parameters()) + list(model.heads.map_dense.parameters())
    groups = [{"params": shared, "lr": config["shared_lr"]}, {"params": new + fresh_heads, "lr": config["new_lr"]},
              {"params": heads, "lr": config["head_lr"]}]
    return torch.optim.AdamW(groups, weight_decay=config["weight_decay"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/unified_scene_v1.yaml")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--experimental-unverified-depth", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--only-task", choices=TASKS, help="Explicit single-task control; same trunk/data/init, separate checkpoint")
    args = parser.parse_args(argv)
    if not args.experimental_unverified_depth:
        parser.error("Explicit --experimental-unverified-depth required; pseudo geometry is not approved GT")
    if args.run_all:
        if any((args.preflight_only, args.smoke_steps, args.num_samples, args.resume, args.evaluate, args.only_task)):
            parser.error("--run-all cannot be combined with partial/resume/evaluation flags")
        for pattern in ("test_unified_scene.py", "test_geometry_depth_v2.py", "test_temporal_depth_v3.py", "test_seg_on_v3.py"):
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-q"],
                           cwd=ROOT, check=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
        for overrides in (["--num-samples", "2", "--smoke-steps", "4"], []):
            subprocess.run([sys.executable, "-u", __file__, "--config", args.config,
                "--experimental-unverified-depth", *overrides], cwd=ROOT, check=True)
        return 0
    if (args.smoke_steps < 0 or (args.num_samples is not None and args.num_samples <= 0)
            or (args.resume and (args.smoke_steps or args.num_samples or args.evaluate))
            or (args.only_task and (args.resume or args.evaluate))):
        parser.error("Invalid smoke/resume/evaluation combination")
    resumed = load_checkpoint_cpu(resolve(args.resume or args.evaluate)) if args.resume or args.evaluate else None
    if resumed and resumed.get("stage") != UNIFIED_SCENE_VERSION:
        raise ValueError("Not a shared-scene checkpoint")
    config = copy.deepcopy(resumed["training_config"]) if resumed else load_yaml_config(resolve(args.config))
    if config["stage"] != UNIFIED_SCENE_VERSION:
        raise ValueError("Wrong joint architecture stage")
    config["train"].setdefault("active_tasks", list(TASKS))
    if args.only_task:
        config["train"]["active_tasks"] = [args.only_task]
        original = Path(config["paths"]["checkpoint"])
        config["paths"]["checkpoint"] = str(original.with_stem(original.stem + "_only_" + args.only_task))
    active_tasks = config["train"]["active_tasks"]
    if not active_tasks or len(set(active_tasks)) != len(active_tasks) or set(active_tasks) - set(TASKS):
        raise ValueError("Invalid active-task selection")
    destination = resolve(config["paths"]["checkpoint"])
    qualified = destination.with_stem(destination.stem + "_qualified")
    if not resumed and not args.smoke_steps and not args.preflight_only and (destination.exists() or qualified.exists()):
        raise FileExistsError("Existing joint run; choose another config or use --resume")
    source_path = resolve(config["paths"]["baseline"])
    if source_path in (destination, qualified):
        raise ValueError("Refusing to overwrite selected baseline")
    source_sha = file_sha256(source_path)
    if resumed and source_sha != resumed["source_sha256"]:
        raise ValueError("Selected baseline changed since joint training")
    source = load_checkpoint_cpu(source_path)
    if source.get("stage") != SEG_ON_V3_STAGE:
        raise ValueError("Use the selected combined Seg-on-V3 baseline, not a single-task checkpoint")
    metadata = source["base_metadata"]
    data_config, manifest = load_context(config)
    if any(manifest[key] != metadata[key] for key in ("metadata_sha256", "split_sha256")):
        raise ValueError("Joint data differs from selected baseline")
    stage3 = load_yaml_config(resolve(config["paths"]["stage3_config"]))
    queries, provenance = load_vector_capacity_audit(resolve(config["paths"]["capacity_audit"]), stage3, manifest)
    if queries != config["model"]["map_queries"] or metadata["model_config"]["N_map"] != queries:
        raise ValueError("Map query capacity must match certified labels and selected checkpoint")
    settings = {"agent": copy.deepcopy(metadata["training_config"]["temporal_agent"]["recipe"]),
                "map": copy.deepcopy(stage3["map"]), "dense": copy.deepcopy(config["loss"]["dense"])}
    layers = config["model"]["query_layers"]
    settings["map"]["aux_layer_weights"] = [.5] * (layers - 1) + [1.]
    settings["agent"]["agent_loss"]["aux_layer_weights"] = [(i + 1) / layers for i in range(layers)]
    train, train_audit = build_data(config, metadata, data_config, manifest, "train", provenance,
                                  args.num_samples or config["train"]["num_samples"])
    validation, validation_audit = build_data(config, metadata, data_config, manifest, "validation", provenance)
    audits = {"train": train_audit, "validation": validation_audit}
    if train_audit["teacher_signatures"] != validation_audit["teacher_signatures"]:
        raise ValueError("Train/validation teacher provenance differs")
    for task, previous in (("seg", source["label_audits"]["validation"]),
                           ("depth", metadata["label_audits"]["validation"])):
        if validation_audit["teacher_signatures"][task] != previous["teacher_signatures"][task]:
            raise ValueError("Joint teacher labels differ from selected " + task + " baseline")
    resolve_seg_class_balance(settings["dense"], train_audit)
    if resumed and (resumed["label_audits"] != audits or resumed["loss_settings"] != settings):
        raise ValueError("Joint labels/objectives/context changed since training")
    print("shared_trunk=TRAINABLE active_losses=" + ",".join(active_tasks) + " DINO=FROZEN old_baseline=READ_ONLY", flush=True)
    if args.preflight_only:
        print("UNIFIED_PREFLIGHT=PASS no_model_load=True runtime_NOT_VERIFIED")
        return 0
    seed = config["train"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    device = initialize_cuda_device()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Joint profile requires CUDA bf16")
    baseline = None
    if resumed:
        model = UnifiedQUEST(resumed["model_config"], resumed["profile"])
        model.load_state_dict(resumed["model_state_dict"], strict=True)
        initialization = resumed["initialization"]
        baseline_metrics = resumed["baseline_validation"]
    else:
        baseline = load_seg_on_v3(source).eval()
        model = UnifiedQUEST(metadata["model_config"], config["model"])
        initialization = model.warm_start(baseline)
        baseline_metrics = None
    del source
    model.to(device)
    backbone_hash = full_base_fingerprint(model.backbone)
    print("parameters=" + json.dumps({"total": sum(p.numel() for p in model.parameters()),
        "shared_trainable": sum(p.numel() for p in model.trunk.parameters() if p.requires_grad),
        "readouts": sum(p.numel() for p in model.heads.parameters())}), flush=True)
    if args.evaluate:
        current = evaluate(model, validation, device, settings, config["eval"])
        print("four_task_validation=" + json.dumps(brief(current), allow_nan=False))
        print("baseline_gate=" + json.dumps(quality_gate(current, baseline_metrics, config["eval"]["gate"])))
        print("full_per_class_per_camera_per_scene_metrics_in_checkpoint_validation_history=True")
        return 0
    if not args.smoke_steps and baseline_metrics is None:
        baseline.to(device)
        baseline_metrics = evaluate(baseline, validation, device, settings, config["eval"])
        print("selected_baseline_validation=" + json.dumps(brief(baseline_metrics), allow_nan=False), flush=True)
        baseline.cpu()
    del baseline
    torch.cuda.empty_cache()
    optimizer = optimizer_for(model, config["train"])
    generator, pc_generator = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed + 1)
    training = loader(train, shuffle=not args.smoke_steps, generator=generator)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["train"]["epochs"] * len(training), eta_min=1e-6)
    start, history = 0, []
    if resumed:
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scheduler.load_state_dict(resumed["scheduler_state_dict"])
        generator.set_state(resumed["loader_rng"])
        pc_generator.set_state(resumed["pcgrad_rng"])
        torch.set_rng_state(resumed["torch_rng"])
        torch.cuda.set_rng_state(resumed["cuda_rng"])
        random.setstate(resumed["python_rng"])
        start, history = resumed["epoch"], resumed["validation_history"]
        scales = resumed["task_loss_scales_train_only"]
    else:
        model.eval()
        scales = dict.fromkeys(TASKS, 0.)
        positions = torch.linspace(0, len(train) - 1, min(len(train), config["train"]["calibration_frames"])).round().long().tolist()
        with torch.no_grad():
            for position in positions:
                batch = collate_four_tasks([train[position]])
                raw, _ = four_task_losses(model, forward(model, batch, device), batch, settings, device)
                for name in TASKS:
                    scales[name] += float(raw[name]) / len(positions)
        scales = {key: max(value, 1e-6) for key, value in scales.items()}
        print("task_loss_scales_TRAIN_ONLY=" + json.dumps(scales), flush=True)
    for epoch in range(start, start + 1 if args.smoke_steps else config["train"]["epochs"]):
        model.train()
        total = dict.fromkeys(TASKS, 0.)
        iterator = iter(training)
        steps = args.smoke_steps or len(training)
        head_probes = {"agent": model.heads.agent.cls_head[-1].weight, "map": model.heads.map_class[-1].weight,
                       "seg": model.heads.seg.weight, "depth": model.heads.depth[-1].weight}
        tracked = [model.trunk.geometry.level_projections[-1].weight] + [head_probes[k] for k in active_tasks]
        before = [value.detach().clone() for value in tracked]
        for step in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(training)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            raw, diagnostics = four_task_losses(model, forward(model, batch, device), batch, settings, device)
            weighted = scaled_task_losses(raw, scales, config["loss"]["task_weights"])
            weighted = {key: weighted[key] for key in active_tasks}
            if step == 0:
                print("shared_gradient_audit=" + json.dumps(shared_gradient_audit(model, weighted)), flush=True)
            joint_backward(model, weighted, mode=config["train"]["gradient_mode"], generator=pc_generator)
            if step == 0:
                for parameter in tracked:
                    if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) or not float(parameter.grad.norm()) > 0:
                        raise RuntimeError("Missing/nonfinite joint trunk or output-head gradient")
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["train"]["clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            if not args.smoke_steps:
                scheduler.step()
            for name in TASKS:
                total[name] += float(raw[name])
            if step == 0 or (step + 1) % config["train"]["log_interval"] == 0 or step + 1 == steps:
                values = " ".join(f"{key}={float(raw[key]):.4f}" for key in TASKS)
                print(f"epoch={epoch+1} step={step+1}/{steps} {values}", flush=True)
        if full_base_fingerprint(model.backbone) != backbone_hash:
            raise RuntimeError("Frozen DINO weights/buffers changed")
        if any(torch.equal(old, new.detach()) for old, new in zip(before, tracked)):
            raise RuntimeError("Missing actual parameter update in trunk or one of the four readouts")
        if args.smoke_steps:
            print("UNIFIED_SMOKE=PASS active_tasks=" + ",".join(active_tasks) +
                  " gradients_reach_shared_geometry_pyramid_BEV_query=True parameter_update=PASS checkpoint_saved=False")
            break
        current = evaluate(model, validation, device, settings, config["eval"])
        gate = quality_gate(current, baseline_metrics, config["eval"]["gate"])
        history.append({"epoch": epoch + 1, "mean_train_losses": {k: v / steps for k, v in total.items()},
                        "validation": current, "baseline_gate": gate})
        print("four_task_validation=" + json.dumps(brief(current), allow_nan=False), flush=True)
        print("baseline_gate=" + json.dumps(gate), flush=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_path = resolve(config["paths"]["reports"]) / f"{destination.stem}_epoch_{epoch+1:02d}_{stamp}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump({"epoch": epoch + 1, "validation": current, "baseline": baseline_metrics,
                       "baseline_gate": gate, "source_sha256": source_sha, "active_tasks": active_tasks,
                       "task_loss_scales_train_only": scales}, stream, indent=2, allow_nan=False)
        save(destination, model, config, settings, audits, manifest, source_sha, initialization,
             optimizer, scheduler, generator, pc_generator, epoch + 1, scales, baseline_metrics, history)
        if gate["all_pass"] and set(active_tasks) == set(TASKS):
            save(qualified, model, config, settings, audits, manifest, source_sha, initialization,
                 optimizer, scheduler, generator, pc_generator, epoch + 1, scales, baseline_metrics, history)
        print("checkpoint_saved=" + str(destination) + " baseline_gate_pass=" + str(gate["all_pass"]), flush=True)
        print("detailed_report=" + str(report_path), flush=True)
    print(f"peak_allocated_MiB={torch.cuda.max_memory_allocated(device)/2**20:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
