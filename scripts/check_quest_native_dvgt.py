"""Verify a full pretrained DVGT transplant inside one QUEST; no training/export."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from quest.cuda_runtime import initialize_cuda_device
from quest.dvgt_depth import (DVGT_CHECKPOINT_SHA256, DVGT_REVISION, DINOV3_REVISION,
                              build_teacher, load_preprocessors, restore_depth)
from quest.native_dvgt import NATIVE_QUEST_VERSION, QUESTNativeDVGT, publish_native_checkpoint
from quest.native_dvgt_inputs import causal_native_window, prepare_native_packet, packet_to_device
from quest.openscene_dataset import OPENSCENE_CAMERA_NAMES
from quest.stage3_split import file_sha256, load_stage3_split, selected_frames
from quest.temporal_depth_dataset import causal_history_plans
from quest.utils import load_yaml_config
from scripts.export_dvgt_depth import inspect_assets
from scripts.export_metric3d_depth import inspect_frame


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/quest_dvgt_native_v4.yaml")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--sample-position", type=int, default=1,
                        help="Manifest position; default 1 has real past context in the current split")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--save-initialized", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_yaml_config(resolve(args.config))
    if (config["profile"] != NATIVE_QUEST_VERSION or config["image_geometry"] != "raw_opencv"
            or config["context"]["policy"] != "same_split_same_scene_strict_past_v1"
            or config["context"]["cold_start"] != "native_single_frame"):
        raise ValueError("Unsupported native QUEST input/geometry policy")
    teacher_config = load_yaml_config(resolve(config["native_config"]))
    teacher = teacher_config["teacher"]
    repo, native_path = resolve(teacher["repo_path"]), resolve(teacher["checkpoint_path"])
    print("NATIVE_QUEST mode=INITIALIZATION_ONLY no_training=True no_label_export=True", flush=True)
    print("native_assets_check=START includes_checkpoint_SHA256; existing labels are not rescanned", flush=True)
    blockers = inspect_assets(repo, native_path, teacher)
    if (teacher["repo_revision"] != DVGT_REVISION or teacher["dinov3_revision"] != DINOV3_REVISION
            or teacher["checkpoint_sha256"] != DVGT_CHECKPOINT_SHA256):
        blockers.append("PINNED_NATIVE_CONFIG_MISMATCH")
    for package in ("transformers", "yaml"):
        if importlib.util.find_spec(package) is None:
            blockers.append("DEPENDENCY_MISSING: " + package)
    source_path = resolve(config["spatial_checkpoint"])
    if not source_path.is_file():
        blockers.append("SPATIAL_CHECKPOINT_MISSING: " + str(source_path))
    for message in blockers:
        print("BLOCKER " + message)
    if blockers:
        return 2
    dataset = load_yaml_config(resolve(config["dataset_config"]))["dataset"]
    metadata_path, camera_root = resolve(dataset["metadata_path"]), resolve(dataset["camera_root"])
    with metadata_path.open("rb") as stream:
        infos = pickle.load(stream)["infos"]  # Trusted local OpenScene metadata.
    manifest = load_stage3_split(resolve(config["split_manifest"]), infos, metadata_path)
    allowed = [row["index"] for row in selected_frames(manifest, args.split)]
    if not 0 <= args.sample_position < len(allowed):
        raise ValueError("sample-position outside manifest split")
    target = allowed[args.sample_position]
    plan = causal_history_plans(infos, [target], allowed, config["context"])[0]
    indices = causal_native_window(plan)
    cameras = {index: inspect_frame(infos[index], camera_root, (252, 448), "raw_opencv") for index in indices}
    # Lazy import: missing native/student dependencies are reported before model construction.
    from quest.agent_training import load_checkpoint_cpu
    from quest.seg_depth_training import load_dense_model, validate_dense_checkpoint

    source = load_checkpoint_cpu(source_path)
    validate_dense_checkpoint(source)
    if (source["dense_frontend"]["version"] != "geometry_dpt_depth_v2"
            or source["split_sha256"] != manifest["split_sha256"]
            or source["metadata_sha256"] != manifest["metadata_sha256"]):
        raise ValueError("Spatial source must be the matching preserved V2 baseline")
    backbone_dir = resolve(source["model_config"]["local_backbone_dir"])
    if (not (backbone_dir / "config.json").is_file()
            or not any((backbone_dir / name).is_file() for name in (
                "model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"))):
        raise FileNotFoundError("Local spatial DINOv2 assets missing: " + str(backbone_dir))
    print(f"context_indices={indices} cold_start={len(indices) == 1} future_context=False")
    print("native=DINOv3_ViT_L16 width=1024 geometry_rounds=24 attention_blocks=72 XYZ+DPT+ego_pose")
    print("spatial=original_current_frame_Agent_Map shared_backbone=False Seg=NOT_AVAILABLE")
    if args.preflight_only:
        print("NATIVE_QUEST_PREFLIGHT=PASS no_model_load=True runtime_NOT_VERIFIED")
        return 0
    report_path, output_path = resolve(config["report"]), resolve(config["checkpoint"])
    if report_path.exists() or (args.save_initialized and (output_path.exists() or output_path.with_suffix(".pt.tmp").exists())):
        raise FileExistsError("Use a new report/checkpoint path; existing experiments are never overwritten")
    if args.save_initialized and shutil.disk_usage(ROOT).free < teacher["checkpoint_bytes"] + 2 * 2**30:
        raise RuntimeError("Need native checkpoint size plus 2 GiB free workspace disk space")
    device = initialize_cuda_device()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native profile requires CUDA bf16; no silent precision fallback")
    packet = packet_to_device(prepare_native_packet(infos, plan, cameras, load_preprocessors(repo)), device)
    spatial = load_dense_model(source).to(device).eval()
    del source["model_state_dict"]
    # Reference only the old spatial route, not its unused Depth decoder.
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        encoded = spatial.encode_image(packet["images"], packet["intrinsics"], packet["extrinsics"],
                                       packet["ego_state"], use_ego_state=False)
        reference = spatial.forward_agent(encoded["bev_tokens"], encoded["bev_features"])
        reference.update(spatial.forward_map(encoded, packet["intrinsics"], packet["extrinsics"],
                                            packet["ego_state"], (252, 448), include_dense=False))
    reference = {key: reference[key].cpu() for key in ("agent_cls_logits", "agent_boxes", "agent_velocity",
                                                      "map_cls_logits", "map_points")}
    del encoded
    native, audit = build_teacher(repo, native_path, device, decoder_chunk_frames=1)
    model = QUESTNativeDVGT(spatial, native).eval()
    print("parameters=" + json.dumps(model.parameter_summary()), flush=True)
    with torch.inference_mode():
        output = model(**packet)
        for key, expected in reference.items():
            torch.testing.assert_close(output[key].cpu(), expected, rtol=0, atol=0)
        # Same resident native component, no second 1.7B teacher copy on the GPU.
        # This is an integration equivalence check, NOT independent teacher accuracy.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            direct = native(packet["native_images"])
        for actual, key in (("native_points", "points"), ("native_points_conf", "points_conf"),
                            ("native_ego_pose_enc", "absolute_ego_pose_enc")):
            torch.testing.assert_close(output[actual], direct[key], rtol=0, atol=0)
    # Cross-check the torch camera-Z adapter against the existing audited CPU adapter.
    from quest.dvgt_depth import DVGT_TEACHER_CAMERAS
    rows = {row["name"]: row for row in cameras[target]}
    maximum_error = 0.
    common_pixels = 0
    for position, name in enumerate(OPENSCENE_CAMERA_NAMES):
        row, native_view = rows[name], DVGT_TEACHER_CAMERAS.index(name)
        restored, valid, _ = restore_depth(
            direct["points"][0, -1, native_view].float().cpu().numpy(),
            direct["points_conf"][0, -1, native_view].float().cpu().numpy(),
            packet["restoration"]["camera_from_first_ego"][0, position].cpu().numpy(),
            row["intrinsic"], row["distortion"],
            packet["restoration"]["native_intrinsics"][0, position].cpu().numpy(),
            row["source_hw"], (252, 448), "raw_opencv")
        torch_depth = output["depth"][0, position].cpu()
        common = valid & output["depth_valid_mask"][0, position].cpu()
        if not bool(common.any()):
            raise RuntimeError("No common restored depth for camera=" + name)
        common_pixels += int(common.sum())
        # OpenCV remap quantizes interpolation coordinates; report, do not demand bit identity.
        maximum_error = max(maximum_error, float((torch_depth - restored).abs()[common].max()))
    provenance = {"native_strict_load": audit["strict"], "runtime_checks_passed": True,
                  "native_checkpoint_sha256": DVGT_CHECKPOINT_SHA256,
                  "repo_revision": DVGT_REVISION, "dinov3_revision": DINOV3_REVISION,
                  "spatial_checkpoint": str(source_path), "spatial_checkpoint_sha256": file_sha256(source_path),
                  "spatial_model_config": source["model_config"], "config": config,
                  "split_sha256": manifest["split_sha256"], "metadata_sha256": manifest["metadata_sha256"]}
    report = {"profile": NATIVE_QUEST_VERSION, "native_architecture": model.native_architecture,
              "parameters": model.parameter_summary(), "spatial_predictions_exactly_preserved": True,
              "native_forward_exactly_preserved": True, "context": plan,
              "depth_valid_fraction": float(output["depth_valid_mask"].float().mean()),
              "restoration_common_pixels": common_pixels,
              "torch_opencv_common_max_difference_m": maximum_error,
              "geometry_approval": "UNVERIFIED", "real_GT_depth_accuracy": "NOT_MEASURED",
              "training_started": False, "reference": "PRETRAINED_COMPONENT_TRANSPLANT_NOT_DISTILLATION",
              "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 2**20,
              "provenance": provenance}
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.save_initialized:
        publish_native_checkpoint(output_path, model, provenance)
        print("initialized_checkpoint_saved=" + str(output_path))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print("NATIVE_QUEST_INTEGRATION=PASS Agent_Map=EXACT native_geometry=EXACT training_started=False")
    print(f"valid_fraction={report['depth_valid_fraction']:.4f} peak_allocated_MiB={report['peak_allocated_MiB']:.1f}")
    print("report=" + str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
