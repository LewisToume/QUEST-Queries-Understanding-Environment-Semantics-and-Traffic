"""Native OpenScene DVGT-1 depth diagnostic/export, without loading the QUEST Student."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quest.cuda_runtime import initialize_cuda_device
from quest.depth_teacher import validate_depth_record
from quest.dvgt_depth import (
    DINOV3_REVISION, DVGT_ALIGNMENT_VERSION, DVGT_CHECKPOINT_SHA256, DVGT_REVISION,
    DVGT_SCHEMA_VERSION, DVGT_SCALE, DVGT_TEACHER_CAMERAS, DVGT_WEIGHT_REVISION,
    build_sequence_plans, build_teacher, camera_from_first_ego, load_preprocessors,
    prepare_image, repository_revision, restore_depth,
)
from quest.frame_selection import select_indices
from quest.openscene_dataset import OPENSCENE_CAMERA_NAMES
from quest.stage3_split import file_sha256, load_stage3_split, selected_frames
# Only existing JPEG/calibration inspection and PNG utilities are reused here.
# This module has no module-level Metric3D/DINO/QUEST model construction.
from scripts.export_metric3d_depth import dependency_versions, inspect_frame, local_path, preview_image


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/depth_teacher_dvgt.yaml")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Original metadata position, or manifest position if --split-manifest is set")
    parser.add_argument("--num-frames", type=int, default=1, help="Target frames to save, NOT context-frame count")
    parser.add_argument("--indices", type=int, nargs="+", help="Explicit original metadata target indices")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--image-geometry", choices=("raw_opencv", "rectified_pinhole"))
    parser.add_argument("--allow-future-context", action="store_true",
                        help="Permit real same-scene future context at scene starts for OFFLINE teacher only")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true", help="Read-only CPU; no model construction")
    modes.add_argument("--diagnostic-only", action="store_true", help="Default: PNG/JSON only, never training .pt")
    modes.add_argument("--export-candidate-labels", action="store_true",
                       help="Explicit experimental image-plane .pt labels; NOT geometry/GT approval")
    args = parser.parse_args(argv)
    if args.indices is not None and (args.sample_index != 0 or args.num_frames != 1):
        parser.error("--indices cannot be combined with --sample-index/--num-frames")
    return args


def validate_config(config):
    teacher, dataset, export = config["teacher"], config["dataset"], config["export"]
    if (teacher["name"] != "DVGT-1" or teacher["repo_revision"] != DVGT_REVISION
            or teacher["dinov3_revision"] != DINOV3_REVISION
            or teacher["checkpoint_sha256"] != DVGT_CHECKPOINT_SHA256
            or teacher["gt_scale_factor"] != DVGT_SCALE):
        raise ValueError("Use the pinned official DVGT-1 task checkpoint/source/metric scale")
    if tuple(dataset["image_size"]) != (252, 448):
        raise ValueError("Do not silently change QUEST label pixel geometry")
    numbers = [export[key] for key in ("min_depth_m", "max_depth_m", "confidence_min", "max_reprojection_px")]
    if (not np.isfinite(numbers).all() or not 0 < numbers[0] < numbers[1]
            or numbers[2] < 0 or numbers[3] <= 0):
        raise ValueError("Invalid DVGT valid-depth/score/reprojection range")
    if any(not isinstance(export[key], int) or export[key] < 1 for key in ("log_interval", "preview_frames")):
        raise ValueError("log_interval/preview_frames must be positive integers")
    if not isinstance(teacher["decoder_chunk_frames"], int) or teacher["decoder_chunk_frames"] < 1:
        raise ValueError("decoder_chunk_frames must be positive")


def inspect_assets(repo, checkpoint, teacher):
    blockers = []
    files = ("dvgt/models/architectures/dvgt1.py", "dvgt/models/backbones/dvgt1_aggregator.py",
             "tools/data_process/utils/image.py", "dvgt/datasets/transforms/transforms.py",
             "configs/_base_/default_dataset.yaml", "third_party/dinov3/hubconf.py")
    for relative in files:
        if not (repo / relative).is_file():
            blockers.append("ASSET_MISSING: " + str(repo / relative))
    if not checkpoint.is_file():
        blockers.append("CHECKPOINT_MISSING: " + str(checkpoint))
    elif (checkpoint.stat().st_size != teacher["checkpoint_bytes"]
          or file_sha256(checkpoint) != DVGT_CHECKPOINT_SHA256):
        blockers.append("CHECKPOINT_HASH_OR_SIZE_MISMATCH: require official RainyNight/DVGT-1 dvgt1.pt")
    if not blockers:
        for directory, revision in ((repo, DVGT_REVISION), (repo / "third_party/dinov3", DINOV3_REVISION)):
            try:
                repository_revision(directory, revision)
            except (ValueError, OSError, subprocess.CalledProcessError) as error:
                blockers.append("SOURCE_REVISION: " + str(error))
    for package in ("cv2", "pandas", "iopath", "safetensors", "huggingface_hub", "einops", "fvcore", "torchvision"):
        if importlib.util.find_spec(package) is None:
            blockers.append("DEPENDENCY_MISSING: " + package)
    return blockers


def check_native_predictions(output, length, image_hw):
    required = ("points", "points_conf", "absolute_ego_pose_enc")
    if not isinstance(output, dict) or any(key not in output for key in required):
        raise ValueError("Native DVGT output keys changed")
    expected = {"points": (1, length, 8, *image_hw, 3),
                "points_conf": (1, length, 8, *image_hw), "absolute_ego_pose_enc": (1, length, 7)}
    for key, shape in expected.items():
        value = output[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid/nonfinite native {key}; expected={shape}")
    return {key: list(output[key].shape) for key in required}


def publish_record(record, output_dir):
    token = record["token"]
    temporary, final = output_dir / (token + ".pt.tmp"), output_dir / (token + ".pt")
    if final.exists():
        raise FileExistsError("Refusing to overwrite " + str(final))
    created = False
    try:
        with temporary.open("xb") as stream:
            created = True
            torch.save(record, stream)
        os.link(temporary, final)
    finally:
        if created:
            temporary.unlink()


def main(argv=None):
    args = parse_args(argv)
    with local_path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    teacher, dataset, settings = config["teacher"], config["dataset"], config["export"]
    geometry = args.image_geometry or dataset["image_geometry"]
    diagnostic = not args.export_candidate_labels
    if geometry == "unverified" and not args.preflight_only:
        raise ValueError("Declare --image-geometry raw_opencv or rectified_pinhole as an explicit hypothesis")
    metadata_path, camera_root = local_path(dataset["metadata_path"]), local_path(dataset["camera_root"])
    with metadata_path.open("rb") as stream:
        metadata = pickle.load(stream)  # Trusted local OpenScene metadata only.
    infos = metadata.get("infos") if isinstance(metadata, dict) else None
    if not isinstance(infos, list):
        raise ValueError("Expected OpenScene metadata['infos']")
    manifest = load_stage3_split(local_path(args.split_manifest), infos, metadata_path) if args.split_manifest else None
    targets = select_indices(infos, args.sample_index, args.num_frames, args.indices, manifest, args.split)
    allowed = {str(row["scene_token"]) for row in selected_frames(manifest, args.split)} if manifest else None
    plans = build_sequence_plans(infos, targets, teacher["context_frames"], allow_future=args.allow_future_context,
                                 timestamp_scale=teacher["timestamp_scale"], max_gap_s=teacher["max_gap_s"],
                                 allowed_scenes=allowed)
    if diagnostic and not args.preflight_only and len(targets) > 3:
        raise ValueError("Diagnostic inference is limited to 3 target frames; bulk export requires explicit candidate mode")
    repo = local_path(args.repo or teacher["repo_path"]).resolve()
    checkpoint = local_path(args.checkpoint or teacher["checkpoint_path"]).resolve()
    output_dir = local_path(args.output_dir or settings["output_dir"])
    audit_dir = local_path(args.audit_dir or settings["audit_dir"]) / ("diagnostic" if diagnostic else "candidate")
    # Never allow an accidentally supplied old Metric3D label directory to be reused.
    if output_dir.resolve() == audit_dir.resolve() or "metric3d" in str(output_dir).lower():
        raise ValueError("Use a distinct DVGT output directory; never overwrite Metric3D experiments")
    blockers = inspect_assets(repo, checkpoint, teacher)
    cameras_by_index = {}
    for index in dict.fromkeys(index for plan in plans for index in plan["indices"]):
        cameras = inspect_frame(infos[index], camera_root, tuple(dataset["image_size"]), geometry)
        # Validate all explicit/derived transforms before allocating the teacher.
        for camera in cameras:
            camera_from_first_ego(infos[index], infos[index], camera["sensor2lidar"])
        cameras_by_index[index] = cameras
    for index in targets:
        token = str(infos[index]["token"])
        paths = [audit_dir / (token + ".json"), audit_dir / (token + ".png")]
        if args.export_candidate_labels:
            paths += [output_dir / (token + ".pt"), output_dir / (token + ".pt.tmp")]
        blockers.extend("OUTPUT_EXISTS: " + str(path) for path in paths if path.exists())
    metadata_hash = file_sha256(metadata_path)
    print(f"teacher=DVGT-1 domain=OpenScene preflight_only={args.preflight_only} no_student_model=True no_training=True")
    print(f"targets={len(targets)} sequences={len(plans)} context={teacher['context_frames']} "
          f"allow_future_context={args.allow_future_context} geometry={geometry}")
    print("student_camera_order=" + ",".join(OPENSCENE_CAMERA_NAMES))
    print("teacher_camera_order=" + ",".join(DVGT_TEACHER_CAMERAS))
    print("versions=" + json.dumps(dependency_versions(), sort_keys=True))
    if blockers:
        for blocker in blockers:
            print("BLOCKER " + blocker)
        print("DVGT_PREFLIGHT=BLOCKED")
        return 2
    print("DVGT_PREFLIGHT=PASS checkpoint_hash=PASS; model_load_and_geometry_quality=NOT_CHECKED")
    if args.preflight_only:
        return 0
    device = initialize_cuda_device()
    helpers = load_preprocessors(repo)
    model, load_audit = build_teacher(repo, checkpoint, device, teacher["decoder_chunk_frames"])
    provenance = {"teacher": "DVGT-1", "teacher_domain": "trained_and_evaluated_on_openscene",
                  "repo_revision": DVGT_REVISION, "dinov3_revision": DINOV3_REVISION,
                  "checkpoint_sha256": DVGT_CHECKPOINT_SHA256, "checkpoint_revision": DVGT_WEIGHT_REVISION,
                  "checkpoint_load": load_audit,
                  "gt_scale_factor": DVGT_SCALE, "native_coordinate_frame": "first_ego_rdf",
                  "pose_source": "openscene_metadata", "adapter_sha256": file_sha256(Path(__file__)),
                  "geometry_adapter_sha256": file_sha256(ROOT / "quest/dvgt_depth.py"),
                  "config_sha256": file_sha256(local_path(args.config)),
                  "geometry_evidence": f"UNVERIFIED_{geometry}_HYPOTHESIS_NOT_CALIBRATION_APPROVAL",
                  "native_saturation_policy": "unbounded_points_finite_positive_range_reprojection",
                  "confidence_min": settings["confidence_min"],
                  "max_reprojection_px": settings["max_reprojection_px"],
                  "preprocess": "native_openscene_annotation_and_deterministic_288x512_eval",
                  "teacher_training_split_overlap": "UNKNOWN", "allow_future_context": args.allow_future_context}
    audit_dir.mkdir(parents=True, exist_ok=True)
    if args.export_candidate_labels:
        output_dir.mkdir(parents=True, exist_ok=True)
    saved_tokens, processed = set(), 0
    for plan in plans:
        tensors, native_ks, rgbs = [], {}, {}
        for index in plan["indices"]:
            cameras = {item["name"]: item for item in cameras_by_index[index]}
            views, native_ks[index], rgbs[index] = [], {}, {}
            for name in DVGT_TEACHER_CAMERAS:
                camera = cameras[name]
                with Image.open(camera["path"]) as stream:
                    rgb = np.asarray(stream.convert("RGB")).copy()
                image, native_k = prepare_image(rgb, camera["intrinsic"], camera["distortion"], geometry, helpers)
                views.append(image)
                native_ks[index][name], rgbs[index][name] = native_k, rgb
            tensors.append(torch.stack(views))
        images = torch.stack(tensors)[None].to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            native = model(images)
        shapes = check_native_predictions(native, len(plan["indices"]), tuple(images.shape[-2:]))
        if processed == 0:
            print("native_outputs=" + json.dumps(shapes))
        points = native["points"][0].float().cpu().numpy()
        confidence = native["points_conf"][0].float().cpu().numpy()
        predicted_pose = native["absolute_ego_pose_enc"][0].float().cpu().tolist()
        del native, images, tensors
        for index in plan["targets"]:
            info, position = infos[index], plan["indices"].index(index)
            depths, masks, stats, camera_transforms = [], [], [], []
            cameras = cameras_by_index[index]
            for camera in cameras:
                name = camera["name"]
                transform = camera_from_first_ego(infos[plan["indices"][0]], info, camera["sensor2lidar"])
                native_view = DVGT_TEACHER_CAMERAS.index(name)
                depth, mask, stat = restore_depth(
                    points[position, native_view], confidence[position, native_view], transform,
                    camera["intrinsic"], camera["distortion"], native_ks[index][name],
                    camera["source_hw"], tuple(dataset["image_size"]), geometry,
                    **{key: settings[key] for key in ("confidence_min", "max_reprojection_px")},
                    min_depth=settings["min_depth_m"], max_depth=settings["max_depth_m"])
                values = depth[mask]
                stat.update(camera=name, median_m=float(values.median()) if values.numel() else None)
                depths.append(depth)
                masks.append(mask)
                stats.append(stat)
                camera_transforms.append(transform.tolist())
            depth, mask = torch.stack(depths), torch.stack(masks)
            token = str(info["token"])
            if not mask.flatten(1).any(1).all():
                raise ValueError(f"NO_VALID_DEPTH_CAMERA token={token} stats={stats}; no training label written")
            record = {"schema_version": DVGT_SCHEMA_VERSION, "alignment_version": DVGT_ALIGNMENT_VERSION,
                      "token": token, "source_index": index, "scene_token": str(info["scene_token"]),
                      "timestamp": float(info["timestamp"]), "metadata_sha256": metadata_hash,
                      "split_sha256": manifest["split_sha256"] if manifest else None,
                      "camera_names": list(OPENSCENE_CAMERA_NAMES), "provenance": provenance,
                      "adaptation_verified": False, "geometry_verified": False,
                      "training_label_scope": "image_plane_candidate" if args.export_candidate_labels else "diagnostic_only",
                      "source_hw": [camera["source_hw"] for camera in cameras],
                      "source_paths": [str(camera["path"]) for camera in cameras],
                      "teacher_intrinsics": [native_ks[index][name].tolist() for name in OPENSCENE_CAMERA_NAMES],
                      "camera_from_first_ego_rdf": camera_transforms,
                      "sequence": {"source_indices": plan["indices"],
                                   "tokens": [str(infos[i]["token"]) for i in plan["indices"]],
                                   "timestamps": [float(infos[i]["timestamp"]) for i in plan["indices"]],
                                   "scene_token": plan["scene_token"], "target_position": position,
                                   "uses_future": position < len(plan["indices"]) - 1,
                                   "predicted_ego_pose_scaled_rdf": predicted_pose},
                      "depth": {"depth_m": depth, "valid_mask": mask, "confidence": None,
                                "definition": "camera_z", "units": "m", "image_geometry": geometry,
                                "min_depth_m": settings["min_depth_m"], "max_depth_m": settings["max_depth_m"],
                                "intrinsics": torch.from_numpy(np.stack([camera["scaled_intrinsic"] for camera in cameras])).float(),
                                "sensor2lidar": torch.from_numpy(np.stack([camera["sensor2lidar"] for camera in cameras])).float(),
                                "distortion": [torch.from_numpy(camera["distortion"]).float() for camera in cameras]}}
            validate_depth_record(record, token, index, OPENSCENE_CAMERA_NAMES)
            report = {"token": token, "source_index": index, "stats": stats, "provenance": provenance,
                      "sequence": record["sequence"], "ground_truth_depth_accuracy": "UNVERIFIED",
                      "teacher_intrinsics": record["teacher_intrinsics"],
                      "camera_from_first_ego_rdf": camera_transforms,
                      "depth_shape": list(depth.shape), "depth_definition": "camera_z_meters",
                      "training_label_export_requested": args.export_candidate_labels,
                      "notes": ["Metadata camera timestamp offsets are not corrected",
                                "Confidence and reprojection masks do not identify sky/reflections",
                                "Teacher training overlap with QUEST validation is unknown"]}
            with (audit_dir / (token + ".json")).open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
            if processed < settings["preview_frames"]:
                with (audit_dir / (token + ".png")).open("xb") as stream:
                    preview_image([rgbs[index][name] for name in OPENSCENE_CAMERA_NAMES], depth.numpy(), mask.numpy(),
                                  OPENSCENE_CAMERA_NAMES, settings["min_depth_m"], settings["max_depth_m"],
                                  note="DVGT-1 candidate camera Z; UNVERIFIED geometry, NOT GT").save(stream, format="PNG")
            if args.export_candidate_labels:
                publish_record(record, output_dir)
                saved_tokens.add(token)
            processed += 1
            if processed == 1:
                for stat in stats:
                    print("  {camera} valid={valid_fraction:.3f} median_m={median_m:.2f} "
                          "reprojection_median_px={native_reprojection_median_px:.2f}".format(**stat))
            if processed == 1 or processed % settings["log_interval"] == 0 or processed == len(targets):
                print(f"frame={processed}/{len(targets)} index={index} token={token} depth={tuple(depth.shape)} "
                      f"valid={mask.float().mean().item():.4f} saved={args.export_candidate_labels}", flush=True)
        del points, confidence
    expected_tokens = {str(infos[index]["token"]) for index in targets}
    if args.export_candidate_labels and saved_tokens != expected_tokens:
        raise RuntimeError("Target/exported token mismatch")
    print(f"DVGT_DEPTH=PASS target_frames={processed} saved={len(saved_tokens)} "
          f"peak_allocated_MiB={torch.cuda.max_memory_allocated(device) / 2**20:.1f} audit_dir={audit_dir}")
    print("GEOMETRY_APPROVAL=UNVERIFIED student_training=NOT_STARTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
