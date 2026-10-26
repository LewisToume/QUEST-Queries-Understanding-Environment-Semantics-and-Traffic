"""Inspect/export real, token-aligned Metric3Dv2 depth. No student model is loaded."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import pickle
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quest.depth_teacher import (  # noqa: E402
    DEPTH_ALIGNMENT_VERSION, DEPTH_SCHEMA_VERSION, IMAGE_GEOMETRIES,
    cross_view_audit, depth_to_original_pixels, prepare_metric3d_input,
    rectified_teacher_image, resize_depth_label, restore_metric_depth,
    scale_intrinsic, sensor_to_lidar, validate_depth_record,
    validate_image_geometry, validate_intrinsic,
)
from quest.openscene_dataset import OPENSCENE_CAMERA_NAMES, resolve_openscene_camera_path  # noqa: E402
from quest.stage3_split import file_sha256, load_stage3_split, selected_frames  # noqa: E402


def local_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def select_indices(infos, sample_index, num_frames, explicit_indices=None, manifest=None, split="train"):
    if explicit_indices is not None:
        if manifest is not None:
            raise ValueError("--indices cannot be combined with --split-manifest")
        indices = list(explicit_indices)
    else:
        if sample_index < 0 or num_frames <= 0:
            raise ValueError("sample-index must be nonnegative and num-frames positive")
        source = [row["index"] for row in selected_frames(manifest, split)] if manifest else range(len(infos))
        if sample_index + num_frames > len(source):
            raise ValueError("Requested range exceeds metadata/split length")
        indices = list(source[sample_index:sample_index + num_frames])
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("Require nonempty, unique original metadata indices")
    if any(index < 0 or index >= len(infos) for index in indices):
        raise ValueError("Original metadata index out of range")
    tokens = [str(infos[index]["token"]) for index in indices]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Selected metadata tokens are not unique")
    for token in tokens:
        if not token or token in (".", "..") or any(char in token for char in ("/", "\\", ":")):
            raise ValueError("Unsafe metadata token filename")
    return indices


def inspect_frame(info, camera_root, output_hw, geometry):
    if not info.get("token") or not info.get("scene_token") or not np.isfinite(float(info["timestamp"])):
        raise ValueError("Frame needs a real token, scene_token and finite timestamp")
    cameras = []
    for name in OPENSCENE_CAMERA_NAMES:
        camera = info["cams"][name]
        path = resolve_openscene_camera_path(str(camera["data_path"]), camera_root)
        if not path.is_file():
            raise FileNotFoundError(f"{name}: JPEG missing: {path}")
        intrinsic = validate_intrinsic(camera["cam_intrinsic"])
        transform = sensor_to_lidar(camera)
        with Image.open(path) as image:
            width, height = image.size
            if image.format != "JPEG":
                raise ValueError(f"{name}: expected original JPEG, got {image.format}")
        if min(height, width) <= 0 or not np.isclose(width / height, output_hw[1] / output_hw[0], rtol=1e-3):
            raise ValueError(f"{name}: aspect ratio would be distorted by student resize")
        if "distortion" not in camera:
            raise ValueError(f"{name}: MISSING distortion")
        distortion = np.asarray(camera["distortion"], dtype=np.float64).reshape(-1)
        if not np.isfinite(distortion).all():
            raise ValueError(f"{name}: non-finite distortion")
        if geometry != "unverified":
            validate_image_geometry(camera, geometry)
        cameras.append({"name": name, "path": path, "source_hw": (height, width),
                        "intrinsic": intrinsic, "scaled_intrinsic": scale_intrinsic(intrinsic, (height, width), output_hw),
                        "sensor2lidar": transform, "distortion": distortion})
    return cameras


def teacher_assets(teacher):
    repo = local_path(teacher["repo_path"]).resolve()
    checkpoint = local_path(teacher["checkpoint_path"]).resolve()
    if teacher["name"] != "Metric3Dv2" or teacher["model"] != "metric3d_vit_large":
        raise ValueError("This adapter implements Metric3Dv2 ViT-L only; do not silently substitute weights")
    config = repo / "mono/configs/HourglassDecoder/vit.raft5.large.py"
    return repo, checkpoint, config


def dependency_versions():
    result = {"python": sys.version.split()[0]}
    for package in ("torch", "torchvision", "numpy", "Pillow", "PyYAML", "mmcv", "mmcv-full",
                    "mmengine", "timm", "matplotlib", "xformers"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "NOT_INSTALLED"
    result["cv2_importable_spec"] = importlib.util.find_spec("cv2") is not None
    return result


def teacher_dependency_blockers(versions):
    blockers = []
    if not versions["cv2_importable_spec"]:
        blockers.append("OPENCV_MISSING")
    # hubconf has an mmengine fallback, but mono/utils/comm.py unconditionally imports mmcv.utils.
    if not any(versions.get(name, "").startswith("1.") for name in ("mmcv", "mmcv-full")):
        blockers.append("MMCV1_MISSING: official mono.utils.comm requires mmcv.utils; mmengine alone is insufficient")
    if versions.get("numpy", "").startswith("2."):
        blockers.append("LEGACY_NUMPY_PROFILE: use numpy<2 in an isolated teacher environment, not in quest_env")
    for package in ("timm", "matplotlib"):
        if versions.get(package, "NOT_INSTALLED") == "NOT_INSTALLED":
            blockers.append(f"TEACHER_DEPENDENCY_MISSING: {package}")
    return blockers


def inference_policy(geometry, evidence, diagnostic_only):
    """An unverified geometry hypothesis is permitted only for non-training diagnostics."""
    if diagnostic_only:
        if geometry not in IMAGE_GEOMETRIES[1:]:
            raise ValueError("Diagnostic inference requires an explicit --image-geometry hypothesis")
        return {"write_labels": False, "blocker": None,
                "evidence": f"UNVERIFIED_GEOMETRY_HYPOTHESIS_ONLY:{geometry}; NOT_CALIBRATION_APPROVAL"}
    blocker = None
    if geometry == "unverified" or not evidence.strip():
        blocker = "IMAGE_GEOMETRY_UNVERIFIED: provide verified image geometry and its evidence"
    return {"write_labels": True, "blocker": blocker, "evidence": evidence}


def save_geometry_preview(cameras, output_path):
    """CPU-only comparison of hypotheses, not automatic calibration verification."""
    width, height = 448, 252
    tile_w, tile_h = width * 2 + 12, height + 28
    canvas = Image.new("RGB", (tile_w * 2, tile_h * 4 + 48), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), "LEFT: original JPEG | RIGHT: metadata OpenCV undistortion HYPOTHESIS", fill="black")
    draw.text((4, 20), "Compare building/pole straightness. This does NOT prove calibration or image provenance.", fill="black")
    for index, camera in enumerate(cameras):
        validate_image_geometry({"distortion": camera["distortion"]}, "raw_opencv")
        with Image.open(camera["path"]) as image:
            original = image.convert("RGB")
            rectified, _ = rectified_teacher_image(np.asarray(original), camera["intrinsic"],
                                                    camera["distortion"], "raw_opencv")
            left, top = (index % 2) * tile_w, (index // 2) * tile_h + 48
            canvas.paste(original.resize((width, height), Image.Resampling.BILINEAR), (left, top + 24))
            canvas.paste(Image.fromarray(rectified).resize((width, height), Image.Resampling.BILINEAR),
                         (left + width + 4, top + 24))
        draw.text((left + 4, top + 4), camera["name"], fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        canvas.save(handle, format="PNG")


def build_teacher(repo, checkpoint, device):
    existing = sys.modules.get("mono")
    if existing is not None:
        module_path = getattr(existing, "__file__", None)
        if not module_path or repo not in Path(module_path).resolve().parents:
            raise RuntimeError("A different mono package is already imported; use an isolated teacher process")
    # Local official hub entry, explicitly without its network checkpoint downloader.
    model = torch.hub.load(str(repo), "metric3d_vit_large", source="local", pretrain=False)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state_dict"), Mapping):
        raise ValueError("Expected official Metric3D checkpoint['model_state_dict']")
    state = payload["model_state_dict"]
    missing = sorted(set(model.state_dict()) - set(state))
    unexpected = sorted(set(state) - set(model.state_dict()))
    print(f"checkpoint_missing_keys={missing} unexpected_keys={unexpected}")
    if missing or unexpected:
        raise ValueError("Incomplete/mismatched teacher weights; no random-head label export allowed")
    model.load_state_dict(state, strict=True)
    del payload, state
    model.requires_grad_(False)
    return model.to(device).eval()


def preview_image(rgbs, depth, valid, names, minimum, maximum, note=""):
    height, width = depth.shape[-2:]
    tile_w, tile_h = width * 2 + 12, height + 28
    header_h = 44 if note else 28
    canvas = Image.new("RGB", (tile_w * 2, tile_h * 4 + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), f"RGB | camera Z (m), log scale {minimum:g} to {maximum:g}; black=invalid", fill="black")
    if note:
        draw.text((4, 20), note, fill="black")
    for index, name in enumerate(names):
        left, top = (index % 2) * tile_w, (index // 2) * tile_h + header_h
        rgb = Image.fromarray(rgbs[index]).resize((width, height), Image.Resampling.BILINEAR)
        values = depth[index, 0]
        normalized = np.clip((np.log(np.maximum(values, minimum)) - np.log(minimum))
                             / (np.log(maximum) - np.log(minimum)), 0, 1)
        # Fixed, shared color scale across all cameras/frames; never normalize each image independently.
        color = np.stack((normalized, normalized ** 0.6, 1 - normalized), axis=-1)
        color[~valid[index, 0]] = 0
        canvas.paste(rgb, (left, top + 24))
        canvas.paste(Image.fromarray((color * 255).astype(np.uint8)), (left + width + 4, top + 24))
        draw.text((left + 4, top + 4), name, fill="black")
    return canvas


def infer_frame(model, cameras, teacher, settings, geometry, device):
    output_hw = tuple(settings["image_size"])
    depths, masks, rgbs, resize_metadata, stats = [], [], [], [], []
    for camera in cameras:
        with Image.open(camera["path"]) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        rectified, image_valid = rectified_teacher_image(
            rgb, camera["intrinsic"], camera["distortion"], geometry)
        image_tensor, resize = prepare_metric3d_input(
            rectified, camera["intrinsic"], tuple(teacher["input_size"]), teacher["canonical_focal_px"])
        with torch.no_grad():
            native = model.inference({"input": image_tensor.to(device)})
        if not isinstance(native, (tuple, list)) or len(native) != 3:
            raise ValueError("Metric3D inference must return depth, confidence, output_dict")
        metric = restore_metric_depth(native[0], resize)
        image_valid &= (np.isfinite(metric) & (metric >= settings["min_depth_m"])
                        & (metric <= settings["max_depth_m"]))
        metric, mask = depth_to_original_pixels(
            metric, image_valid, camera["intrinsic"], camera["distortion"], geometry)
        depth, mask = resize_depth_label(metric, mask, output_hw,
                                         settings["min_depth_m"], settings["max_depth_m"])
        if not mask.any():
            raise ValueError(f"{camera['name']}: no valid depth pixels")
        values = depth[mask].numpy()
        stats.append({"camera": camera["name"], "native_depth_shape": list(native[0].shape),
                      "valid_fraction": float(mask.float().mean()), "min_m": float(values.min()),
                      "median_m": float(np.median(values)), "p95_m": float(np.percentile(values, 95)),
                      "max_m": float(values.max())})
        depths.append(depth)
        masks.append(mask)
        rgbs.append(rgb)
        resize_metadata.append(resize)
    return torch.stack(depths), torch.stack(masks), rgbs, resize_metadata, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/depth_teacher.yaml")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Raw metadata index, or position within selected split when using a manifest")
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--indices", type=int, nargs="+", help="Explicit ORIGINAL metadata indices")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--image-geometry", choices=IMAGE_GEOMETRIES)
    parser.add_argument("--geometry-evidence", help="Source/pipeline evidence for the image geometry declaration")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--geometry-preview-only", action="store_true",
                        help="CPU original/undistortion comparison; no teacher, labels, or geometry approval")
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="Real teacher inference with an unverified geometry hypothesis; PNG/JSON only, never training .pt")
    args = parser.parse_args()
    if sum((args.preflight_only, args.geometry_preview_only, args.diagnostic_only)) > 1:
        parser.error("Choose only one of preflight, geometry preview or diagnostic inference")
    if args.indices is not None and (args.sample_index != 0 or args.num_frames != 1):
        parser.error("--indices cannot be combined with --sample-index/--num-frames")
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    teacher, dataset, settings = config["teacher"], config["dataset"], config["export"]
    for key, value in (("repo_path", args.repo), ("checkpoint_path", args.checkpoint)):
        if value is not None:
            teacher[key] = str(value)
    geometry = args.image_geometry or dataset["image_geometry"]
    if args.geometry_preview_only:
        geometry = "unverified"
    evidence = args.geometry_evidence or dataset.get("geometry_evidence", "")
    policy = inference_policy(geometry, evidence, args.diagnostic_only)
    evidence = policy["evidence"]
    output_dir = local_path(args.output_dir or settings["output_dir"])
    audit_dir = local_path(args.audit_dir or settings["audit_dir"])
    if args.diagnostic_only:
        audit_dir = audit_dir / f"diagnostic_{geometry}"
    metadata_path, camera_root = local_path(dataset["metadata_path"]), local_path(dataset["camera_root"])
    if tuple(dataset["image_size"]) != (252, 448):
        raise ValueError("Current QUEST input is [H,W]=[252,448]; do not silently change label geometry")
    if not (0 < float(settings["min_depth_m"]) < float(settings["max_depth_m"]) <= 200):
        raise ValueError("Invalid depth validity range for official ViT-L export")
    if not np.isfinite([settings["min_depth_m"], settings["max_depth_m"]]).all():
        raise ValueError("Depth bounds must be finite")
    if not isinstance(settings["overlap_stride"], int) or settings["overlap_stride"] <= 0:
        raise ValueError("overlap_stride must be a positive integer")
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)  # Trusted local OpenScene metadata only.
    infos = metadata.get("infos") if isinstance(metadata, Mapping) else None
    if not isinstance(infos, list):
        raise ValueError("Expected OpenScene metadata['infos']")
    manifest = load_stage3_split(args.split_manifest, infos, metadata_path) if args.split_manifest else None
    indices = select_indices(infos, args.sample_index, args.num_frames, args.indices, manifest, args.split)
    if args.diagnostic_only and len(indices) > 3:
        raise ValueError("Unverified diagnostic inference is limited to at most 3 frames; no bulk unaudited export")
    repo, checkpoint, teacher_config = teacher_assets(teacher)
    if (tuple(teacher["input_size"]) != (616, 1064)
            or teacher["canonical_focal_px"] != 1000.0):
        raise ValueError("ViT-L adapter requires official [616,1064] input and canonical focal 1000")
    blockers = []
    if policy["blocker"]:
        blockers.append(policy["blocker"])
    for label, path in (("TEACHER_REPO", repo / "hubconf.py"), ("TEACHER_CONFIG", teacher_config),
                        ("TEACHER_CHECKPOINT", checkpoint)):
        if not path.is_file():
            blockers.append(f"{label}_MISSING: {path}")
    versions = dependency_versions()
    blockers.extend(teacher_dependency_blockers(versions))
    print(f"preflight_only={args.preflight_only} geometry_preview_only={args.geometry_preview_only} "
          f"diagnostic_only={args.diagnostic_only} no_student_model=True no_training=True")
    print(f"versions={json.dumps(versions, sort_keys=True)}")
    print(f"camera_order={','.join(OPENSCENE_CAMERA_NAMES)} image_geometry={geometry}")
    metadata_hash = file_sha256(metadata_path)
    print(f"metadata_sha256={metadata_hash} split_sha256={manifest['split_sha256'] if manifest else 'NOT_SELECTED'}")
    for offset, index in enumerate(indices):
        cameras = inspect_frame(infos[index], camera_root, tuple(dataset["image_size"]), geometry)
        print(f"index={index} token={infos[index]['token']} scene={infos[index]['scene_token']} cameras=8 JPEG_and_calibration=PASS")
        if offset == 0:
            for camera in cameras:
                matrix = camera["intrinsic"]
                print(f"  {camera['name']} HW={camera['source_hw']} "
                      f"fx_fy_cx_cy={matrix[[0,1,0,1],[0,1,2,2]].round(3).tolist()} "
                      f"distortion={camera['distortion'].round(7).tolist()}")
        token = str(infos[index]["token"])
        if args.geometry_preview_only:
            path = audit_dir / "geometry" / f"{token}.png"
            save_geometry_preview(cameras, path)
            print(f"geometry_preview={path}")
            continue
        paths = [audit_dir / f"{token}.png", audit_dir / f"{token}.json"]
        if policy["write_labels"]:
            paths.extend((output_dir / f"{token}.pt", output_dir / f"{token}.pt.tmp"))
        for path in paths:
            if path.exists():
                blockers.append(f"OUTPUT_EXISTS: will not overwrite {path}; choose new output/audit directories")
    print("student_geometry_note=existing_BEV_uses_pinhole_projection_without_distortion; not modified")
    print("depth_confidence_note=teacher_confidence_is_not_used_as_calibrated_probability; valid_mask_is_not_sky_detection")
    if args.geometry_preview_only:
        print("GEOMETRY_STATUS=UNVERIFIED; CPU previews only, teacher_not_loaded=True depth_labels_saved=False")
        return 0
    if blockers:
        for blocker in blockers:
            print(f"BLOCKER {blocker}")
        print("DEPTH_TEACHER_PREFLIGHT=BLOCKED")
        return 2
    print("DEPTH_TEACHER_PREFLIGHT=PASS; checkpoint_contents_and_GPU_inference_not_checked")
    if args.preflight_only:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback or fake depth labels")
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    checkpoint_hash = file_sha256(checkpoint)
    revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=False)
    provenance = {"teacher": teacher["name"], "model": teacher["model"],
                  "repo_revision": revision.stdout.strip() if revision.returncode == 0 else None,
                  "hubconf_sha256": file_sha256(repo / "hubconf.py"),
                  "teacher_config_sha256": file_sha256(teacher_config),
                  "checkpoint_sha256": checkpoint_hash, "input_size": teacher["input_size"],
                  "canonical_focal_px": teacher["canonical_focal_px"], "geometry_evidence": evidence,
                  "versions": versions, "checkpoint_load": "strict", "diagnostic_only": args.diagnostic_only}
    model = build_teacher(repo, checkpoint, device)
    print(f"teacher_model_type={type(model).__name__} device={torch.cuda.get_device_name(device)} "
          "geometry_approval=UNVERIFIED")
    if policy["write_labels"]:
        output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    for position, index in enumerate(indices, start=1):
        info = infos[index]
        cameras = inspect_frame(info, camera_root, tuple(dataset["image_size"]), geometry)
        depth, mask, rgbs, resize, stats = infer_frame(
            model, cameras, teacher, {**settings, "image_size": dataset["image_size"]}, geometry, device)
        names = list(OPENSCENE_CAMERA_NAMES)
        intrinsics = np.stack([camera["scaled_intrinsic"] for camera in cameras])
        transforms = np.stack([camera["sensor2lidar"] for camera in cameras])
        distortions = [camera["distortion"] for camera in cameras]
        record = {"schema_version": DEPTH_SCHEMA_VERSION, "alignment_version": DEPTH_ALIGNMENT_VERSION,
                  "adaptation_verified": False,
                  "token": str(info["token"]), "source_index": index, "scene_token": str(info["scene_token"]),
                  "timestamp": float(info["timestamp"]), "metadata_sha256": metadata_hash,
                  "split_sha256": manifest["split_sha256"] if manifest else None,
                  "camera_names": names, "provenance": provenance,
                  "source_paths": [str(camera["path"]) for camera in cameras],
                  "source_hw": [camera["source_hw"] for camera in cameras], "teacher_resize": resize,
                  "depth": {"depth_m": depth, "valid_mask": mask, "confidence": None,
                            "definition": "camera_z", "units": "m", "image_geometry": geometry,
                            "intrinsics": torch.from_numpy(intrinsics).float(),
                            "sensor2lidar": torch.from_numpy(transforms).float(),
                            "distortion": [torch.from_numpy(value).float() for value in distortions],
                            "min_depth_m": settings["min_depth_m"], "max_depth_m": settings["max_depth_m"]}}
        validate_depth_record(record, str(info["token"]), index, OPENSCENE_CAMERA_NAMES)
        overlaps = cross_view_audit(depth[:, 0].numpy(), mask[:, 0].numpy(), intrinsics,
                                    transforms, distortions, names, geometry, settings["overlap_stride"])
        audit = {"token": record["token"], "source_index": index, "stats": stats, "cross_view": overlaps,
                 "provenance": provenance, "ground_truth_depth_accuracy": "UNVERIFIED",
                 "image_geometry": geometry, "geometry_verified": False,
                 "training_label_export_enabled": policy["write_labels"],
                 "notes": ["Cross-view disagreement includes occlusion, moving objects, sampling and timestamp offsets",
                           "Per-camera timestamp offsets are not audited; exact sensor synchronization unverified",
                           "Geometric validity does not identify sky, reflections or teacher hallucinations",
                           "No Agent box center or virtual map centerline was used as a depth target"]}
        token = record["token"]
        # Previews/reports first; a .pt appears only after the entire frame audit succeeds.
        with (audit_dir / f"{token}.png").open("xb") as handle:
            preview_image(rgbs, depth.numpy(), mask.numpy(), names,
                          settings["min_depth_m"], settings["max_depth_m"],
                          note=f"UNVERIFIED image geometry: {geometry}; diagnostic_only={args.diagnostic_only}").save(handle, format="PNG")
        with (audit_dir / f"{token}.json").open("x", encoding="utf-8") as handle:
            json.dump(audit, handle, indent=2, allow_nan=False)
        if policy["write_labels"]:
            temporary = output_dir / f"{token}.pt.tmp"
            temporary_created = False
            try:
                with temporary.open("xb") as handle:
                    temporary_created = True
                    torch.save(record, handle)
                # Exclusive hard link publishes a complete file without a replace/overwrite race.
                os.link(temporary, output_dir / f"{token}.pt")
            finally:
                if temporary_created and temporary.exists():
                    temporary.unlink()
        print(f"frame={position}/{len(indices)} index={index} token={token} "
              f"depth={tuple(depth.shape)} valid_fraction={mask.float().mean().item():.4f} "
              f"overlap_pairs={len(overlaps)} training_labels_saved={policy['write_labels']}")
        for stat in stats:
            print("  {camera} valid={valid_fraction:.3f} median_m={median_m:.2f} p95_m={p95_m:.2f}".format(**stat))
    print(f"frames_inferred={len(indices)} training_labels_saved={len(indices) if policy['write_labels'] else 0} "
          f"previews_and_reports={audit_dir}")
    print(f"process_peak_allocated_MiB={torch.cuda.max_memory_allocated(device) / 2 ** 20:.2f}")
    print("DEPTH_ADAPTATION_REQUIRES_HUMAN_REVIEW=True; student_training=NOT_STARTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
