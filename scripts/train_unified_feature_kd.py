"""Controlled feature-KD continuation of the trained shared-scene student."""

import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import Subset

from quest.agent_training import load_checkpoint_cpu
from quest.cuda_runtime import initialize_cuda_device
from quest.map_training import load_vector_capacity_audit
from quest.seg_on_v3 import full_base_fingerprint, load_seg_on_v3
from quest.shared_feature_distillation import (FEATURE_KD_STAGE, GUIDE_ROLE, FrozenDenseGuide,
    SharedFeatureDistiller, deployment_payload, restore_student)
from quest.stage3_split import file_sha256
from quest.unified_scene import UNIFIED_SCENE_VERSION, UnifiedQUEST
from quest.unified_gradnorm import GRADNORM_STAGE, TaskGradNorm
from quest.unified_training import TASKS, four_task_losses, quality_gate, scaled_task_losses, shared_gradient_audit
from quest.utils import load_yaml_config
from scripts.train_seg_depth import load_context, resolve
from scripts.train_seg_on_v3 import inputs
from scripts.train_unified_scene import brief, build_data, evaluate, loader, optimizer_for


def atomic_save(path, payload):
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


def predict(student, guide, distiller, batch, device, settings, options):
    data = inputs(batch, device)
    levels = student.extract_levels(data["images"])
    previous = student.extract_levels(data["history"]["images"])
    target = guide(levels, previous, data, tuple(options["feature_grid"])) if guide is not None else None
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = student.forward_from_levels(levels, previous, **data, return_shared_features=target is not None)
        kd = distiller(output["shared_dense_features"], target, batch["dense_targets"], settings["dense"]["depth"], options) if target is not None else None
    output = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in output.items()}
    if kd is None:
        kd = {key: output["depth"].new_zeros(()) for key in ("seg_feature_kd", "depth_feature_kd")}
    return output, kd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/unified_feature_kd_v1.yaml")
    parser.add_argument("--control", action="store_true", help="Matched continuation with KD disabled; separate output")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--experimental-unverified-depth", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if not args.experimental_unverified_depth:
        parser.error("Explicit --experimental-unverified-depth required")
    if args.run_all:
        if any((args.smoke_steps, args.num_samples, args.preflight_only, args.resume)):
            parser.error("--run-all cannot be mixed with partial/resume flags")
        for pattern in ("test_shared_feature_distillation.py", "test_unified_gradnorm.py", "test_unified_scene.py"):
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-q"],
                cwd=ROOT, check=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
        for extra in (["--num-samples", "2", "--smoke-steps", "4"], []):
            subprocess.run([sys.executable, "-u", __file__, "--config", args.config,
                "--experimental-unverified-depth", *(["--control"] if args.control else []), *extra], cwd=ROOT, check=True)
        return 0
    if (args.smoke_steps < 0 or (args.num_samples is not None and (args.num_samples <= 0 or not args.smoke_steps))
            or (args.resume and (args.smoke_steps or args.num_samples or args.control))):
        parser.error("Sample overrides are smoke-only; resume uses saved experiment settings")
    resumed = load_checkpoint_cpu(resolve(args.resume)) if args.resume else None
    if resumed and resumed.get("stage") not in (FEATURE_KD_STAGE, GRADNORM_STAGE):
        raise ValueError("Not a supported controlled-continuation training checkpoint")
    config = copy.deepcopy(resumed["training_config"]) if resumed else load_yaml_config(resolve(args.config))
    balance_only = config["stage"] == GRADNORM_STAGE
    if config["stage"] not in (FEATURE_KD_STAGE, GRADNORM_STAGE):
        raise ValueError("Unexpected experiment or guide role")
    if not balance_only and config["distillation"]["guide_role"] != GUIDE_ROLE:
        raise ValueError("Unexpected feature guide role")
    if not resumed:
        config["control"] = bool(args.control)
        if args.control:
            path = Path(config["paths"]["checkpoint"])
            config["paths"]["checkpoint"] = str(path.with_stem(path.stem + "_control"))
    options = config.get("distillation", {})
    if not balance_only and (config["train"]["epochs"] <= 0 or any(not math.isfinite(options[k]) or options[k] < 0
            for k in ("seg_weight", "depth_weight")) or sum(options[k] for k in ("seg_weight", "depth_weight")) <= 0):
        raise ValueError("Require a positive training budget and nonzero feature supervision")
    if balance_only:
        gn = config["gradnorm"]
        if (config["train"]["epochs"] <= 0 or gn["reference_losses"] != "parent_last_epoch_train_mean"
                or not isinstance(gn["update_every"], int) or gn["update_every"] <= 0
                or not math.isfinite(gn["weight_lr"]) or gn["weight_lr"] <= 0
                or not math.isfinite(gn["alpha"]) or gn["alpha"] < 0):
            raise ValueError("Invalid GradNorm schedule/reference policy")
    destination = resolve(config["paths"]["checkpoint"])
    deploy = destination.with_stem(destination.stem + "_deploy")
    qualified = destination.with_stem(destination.stem + "_qualified_deploy")
    initial_path = resolve(config["paths"]["student_init"])
    if initial_path in (destination, deploy, qualified):
        raise ValueError("Refusing to overwrite initial student")
    if not resumed and not args.smoke_steps and not args.preflight_only and any(p.exists() for p in (destination, deploy, qualified)):
        raise FileExistsError("Existing feature-KD run; use explicit --resume or a new config")
    initial_sha = file_sha256(initial_path)
    if resumed and initial_sha != resumed["student_initialization_sha256"]:
        raise ValueError("Initial V1 student changed since feature KD")
    initial = load_checkpoint_cpu(initial_path)
    if initial.get("stage") != UNIFIED_SCENE_VERSION or set(initial.get("trained_tasks", [])) != set(TASKS):
        raise ValueError("Initialize from the trained four-task unified V1 checkpoint")
    recipe, settings = copy.deepcopy(initial["training_config"]), copy.deepcopy(initial["loss_settings"])
    if recipe["train"]["gradient_mode"] != "sum" or set(recipe["train"]["active_tasks"]) != set(TASKS):
        raise ValueError("First feature-KD experiment requires original four-task summed-gradient control")
    guide_path = resolve(recipe["paths"]["baseline"])
    guide_sha = file_sha256(guide_path)
    if guide_sha != initial["source_sha256"] or (resumed and guide_sha != resumed["source_sha256"]):
        raise ValueError("Frozen pre-integration guide changed")
    if guide_path in (destination, deploy, qualified):
        raise ValueError("Refusing to overwrite frozen guide")
    guide_source = load_checkpoint_cpu(guide_path)
    metadata = guide_source["base_metadata"]
    data_config, manifest = load_context(recipe)
    if any(initial[key] != manifest[key] for key in ("split_sha256", "metadata_sha256")):
        raise ValueError("Feature-KD split/metadata changed")
    stage3 = load_yaml_config(resolve(recipe["paths"]["stage3_config"]))
    queries, provenance = load_vector_capacity_audit(resolve(recipe["paths"]["capacity_audit"]), stage3, manifest)
    if queries != initial["profile"]["map_queries"]:
        raise ValueError("Certified Map query capacity changed")
    train, train_audit = build_data(recipe, metadata, data_config, manifest, "train", provenance, recipe["train"]["num_samples"])
    validation, validation_audit = build_data(recipe, metadata, data_config, manifest, "validation", provenance)
    audits = {"train": train_audit, "validation": validation_audit}
    if audits != initial["label_audits"] or (resumed and audits != resumed["label_audits"]):
        raise ValueError("Feature KD may not change labels, class frequencies or causal context")
    if resumed and settings != resumed["loss_settings"]:
        raise ValueError("Original four-task loss objectives changed")
    role = "NONE_OFFLINE_LABELS_ONLY" if balance_only else GUIDE_ROLE
    prefix = "GRADNORM" if balance_only else "FEATURE_KD"
    print(f"{prefix}_PREFLIGHT=PASS full_train_frames={len(train)} validation_frames={len(validation)} "
          f"init_epoch={initial['epoch']} control={config['control']} guide_role={role}", flush=True)
    if args.preflight_only:
        print("no_model_constructed=True GPU=False")
        return 0
    if args.num_samples is not None:
        if args.num_samples > len(train):
            raise ValueError("Smoke sample count exceeds source training frames")
        train = Subset(train, range(args.num_samples))
        print(f"smoke_subset_frames={len(train)} full_source_label_audit_preserved=True")
    random.seed(config["train"]["seed"])
    torch.manual_seed(config["train"]["seed"])
    device = initialize_cuda_device()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Feature KD requires CUDA bf16")
    student = UnifiedQUEST(initial["model_config"], initial["profile"])
    restore_student(resumed or initial, student)
    guide, distiller = None, None
    if not balance_only:
        old = load_seg_on_v3(guide_source)
        if full_base_fingerprint(student.backbone) != full_base_fingerprint(old.base.backbone):
            raise ValueError("Reusing frozen DINO features requires identical student/guide DINO weights")
        if (student.camera_names != old.base.camera_names
                or student.layer_indices != tuple(old.base.dense_frontend.options["layer_indices"])):
            raise ValueError("Shared frozen features require identical camera and DINO layer ordering")
        guide = FrozenDenseGuide(old)
        distiller = SharedFeatureDistiller(metadata["dense_frontend"]["dpt_features"], guide.seg_channels, guide.depth_channels)
        del old
        if resumed:
            distiller.load_state_dict(resumed["training_distiller_state_dict"], strict=True)
    del guide_source
    student.to(device)
    if guide is not None:
        guide.to(device).eval()
        distiller.to(device).train()
    guide_hash = full_base_fingerprint(guide) if guide is not None else None
    dino_hash = full_base_fingerprint(student.backbone)
    optimizer = optimizer_for(student, config["train"])
    if distiller is not None:
        optimizer.add_param_group({"params": list(distiller.parameters()), "lr": config["train"]["adapter_lr"]})
    scales = initial["task_loss_scales_train_only"]
    weights = recipe["loss"]["task_weights"]
    balancer, balance_optimizer = None, None
    if balance_only:
        reference = {k: initial["validation_history"][-1]["mean_train_losses"][k] * weights[k] / scales[k] for k in TASKS}
        balancer = TaskGradNorm(reference, config["gradnorm"]["alpha"]).to(device)
        balance_optimizer = torch.optim.Adam(balancer.parameters(), lr=config["gradnorm"]["weight_lr"])
        if resumed:
            balancer.load_state_dict(resumed["gradnorm_state_dict"], strict=True)
            balance_optimizer.load_state_dict(resumed["gradnorm_optimizer_state_dict"])
    generator = torch.Generator().manual_seed(config["train"]["seed"])
    batches = loader(train, shuffle=not args.smoke_steps, generator=generator)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["train"]["epochs"] * len(batches), eta_min=1e-6)
    start, history = 0, []
    if resumed:
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scheduler.load_state_dict(resumed["scheduler_state_dict"])
        generator.set_state(resumed["loader_rng"])
        torch.set_rng_state(resumed["torch_rng"])
        torch.cuda.set_rng_state(resumed["cuda_rng"])
        random.setstate(resumed["python_rng"])
        start, history = resumed["epoch"], resumed["validation_history"]
    print("student_architecture=UNCHANGED original_task_loss_scales=UNCHANGED optimizer_schedule=NEW_MATCHED_CONTROL "
          f"GradNorm={balance_only and not config['control']} feature_KD={not balance_only and not config['control']} "
          "PCGrad=OFF Map_reference_update=UNCHANGED", flush=True)
    if not args.smoke_steps and not resumed:
        print("initial_student_validation=" + json.dumps(brief(evaluate(student, validation, device, settings, recipe["eval"])), allow_nan=False), flush=True)
    for epoch in range(start, start + 1 if args.smoke_steps else config["train"]["epochs"]):
        student.train()
        steps, iterator = args.smoke_steps or len(batches), iter(batches)
        totals = dict.fromkeys((*TASKS, "seg_feature_kd", "depth_feature_kd"), 0.)
        weighted_totals = dict.fromkeys(TASKS, 0.)
        probes = [student.trunk.geometry.level_projections[-1].weight, student.heads.seg.weight]
        before = [p.detach().clone() for p in probes]
        for step in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(batches)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            output, kd = predict(student, None if config["control"] else guide, distiller, batch, device, settings, options)
            raw, _ = four_task_losses(student, output, batch, settings, device)
            normalized = scaled_task_losses(raw, scales, weights)
            weighted = balancer.weighted_losses(normalized) if balancer is not None else normalized
            feature_loss = (options["seg_weight"] * kd["seg_feature_kd"] + options["depth_weight"] * kd["depth_feature_kd"]
                            if not balance_only else output["depth"].new_zeros(()))
            update_balance = (balance_only and not config["control"]
                              and (epoch * len(batches) + step) % config["gradnorm"]["update_every"] == 0)
            if update_balance:
                balance_optimizer.zero_grad(set_to_none=True)
                balance_stats = balancer.prepare_update(normalized, student.trunk.pyramid_fusion.fuse_refine[0].layers[-1].weight)
                if step == 0:
                    print("gradnorm_shared_probe=" + json.dumps(balance_stats), flush=True)
            if step == 0:
                print("shared_gradient_audit=" + json.dumps(shared_gradient_audit(student, weighted)), flush=True)
                if not balance_only and not config["control"]:
                    gradient = torch.autograd.grad(feature_loss, probes[0], retain_graph=True)[0]
                    if not bool(torch.isfinite(gradient).all()) or not float(gradient.norm()) > 0:
                        raise RuntimeError("Feature loss does not reach common geometry")
                    print(f"feature_KD_to_shared_geometry_gradient_norm={float(gradient.norm()):.6e}", flush=True)
            total = sum(weighted.values()) + feature_loss
            if not bool(torch.isfinite(total)):
                raise RuntimeError("Nonfinite joint feature-KD loss")
            total.backward()
            if step == 0 and not balance_only and not config["control"]:
                gradient = distiller.semantic_adapter.weight.grad
                if gradient is None or not bool(torch.isfinite(gradient).all()) or not float(gradient.norm()) > 0:
                    raise RuntimeError("Semantic feature adapter has no finite/nonzero update signal")
            torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]], config["train"]["clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            if update_balance:
                balance_optimizer.step()
                balancer.renormalize()
            if not args.smoke_steps:
                scheduler.step()
            for name, value in {**raw, **kd}.items():
                totals[name] += float(value.detach())
            for name, value in weighted.items():
                weighted_totals[name] += float(value.detach())
            if step == 0 or (step + 1) % config["train"]["log_interval"] == 0 or step + 1 == steps:
                if balance_only:
                    summary = " ".join(f"{k}={float(raw[k]):.4f}" for k in TASKS)
                    task_weights = ",".join(f"{k}:{v:.3f}" for k,v in balancer.values().items())
                    print(f"epoch={epoch+1} step={step+1}/{steps} {summary} weights={task_weights}", flush=True)
                else:
                    print(f"epoch={epoch+1} step={step+1}/{steps} seg={float(raw['seg']):.4f} "
                          f"seg_KD={float(kd['seg_feature_kd']):.4f} depth_KD={float(kd['depth_feature_kd']):.4f}", flush=True)
        if ((guide is not None and full_base_fingerprint(guide) != guide_hash)
                or full_base_fingerprint(student.backbone) != dino_hash):
            raise RuntimeError("Frozen guide or DINO changed")
        if any(torch.equal(a, b.detach()) for a, b in zip(before, probes)):
            raise RuntimeError("Shared geometry/Seg parameters did not update")
        if args.smoke_steps:
            print(f"{prefix}_SMOKE=PASS shared_trunk_updated=True inference_teachers=0 checkpoint_saved=False")
            break
        current = evaluate(student, validation, device, settings, recipe["eval"])
        gate = quality_gate(current, initial["baseline_validation"], recipe["eval"]["gate"])
        history.append({"epoch": epoch+1, "mean_train_losses": {k: v/steps for k,v in totals.items()},
                        "mean_weighted_task_losses": {k: v/steps for k,v in weighted_totals.items()},
                        "validation": current, "baseline_gate": gate})
        print("four_task_validation=" + json.dumps(brief(current), allow_nan=False), flush=True)
        print("baseline_gate=" + json.dumps(gate), flush=True)
        if balancer is not None:
            print("gradnorm_weights=" + json.dumps(balancer.values()), flush=True)
        payload = {"stage": config["stage"], "training_config": config, "model_config": student.model_config,
            "profile": student.profile, "model_state_dict": student.state_dict(), "trained_tasks": list(TASKS),
            "training_distiller_state_dict": distiller.state_dict() if distiller is not None else {}, "guide_role": role,
            "gradnorm_state_dict": balancer.state_dict() if balancer is not None else {},
            "gradnorm_optimizer_state_dict": balance_optimizer.state_dict() if balance_optimizer is not None else {},
            "source_sha256": guide_sha, "student_initialization_sha256": initial_sha, "source_epoch": initial["epoch"],
            "loss_settings": settings, "label_audits": audits, "baseline_validation": initial["baseline_validation"],
            "initial_student_validation": initial["validation_history"][-1]["validation"],
            "split_sha256": manifest["split_sha256"], "metadata_sha256": manifest["metadata_sha256"],
            "epoch": epoch+1, "validation_history": history, "task_loss_scales_train_only": scales,
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "loader_rng": generator.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
            "python_rng": random.getstate()}
        atomic_save(destination, payload)
        exported = deployment_payload(student, initial, epoch+1, current, gate, experiment=config["stage"])
        atomic_save(deploy, exported)
        if gate["all_pass"]:
            atomic_save(qualified, exported)
        folder = resolve(config["paths"]["reports"])
        folder.mkdir(parents=True, exist_ok=True)
        report = folder / (destination.stem + f"_epoch_{epoch+1:02d}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
        with report.open("x", encoding="utf-8") as stream:
            json.dump({"epoch": epoch+1, "validation": current, "baseline_gate": gate, "guide_role": role,
                       "gradnorm_weights": balancer.values() if balancer is not None else None,
                       "control": config["control"], "losses": history[-1]["mean_train_losses"]}, stream, indent=2, allow_nan=False)
        print("checkpoint_saved=" + str(destination) + " deploy=" + str(deploy) + " report=" + str(report), flush=True)
    print(f"peak_allocated_MiB={torch.cuda.max_memory_allocated(device)/2**20:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
