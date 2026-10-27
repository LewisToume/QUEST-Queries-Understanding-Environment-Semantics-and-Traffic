"""Token/index-joined offline dense targets; never imports a teacher runtime."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .depth_teacher import scale_intrinsic, sensor_to_lidar, validate_depth_record
from .openscene_dataset import OPENSCENE_CAMERA_NAMES, OpenSceneMetadataDataset
from .seg_teacher import validate_seg_record
from .seg_depth_losses import depth_valid_mask


def read_hashed_record(path):
    content = Path(path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    try:
        record = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    except TypeError:
        record = torch.load(io.BytesIO(content), map_location="cpu")
    if not isinstance(record, Mapping):
        raise ValueError("Dense label must be a mapping: " + str(path))
    return record, digest


def validate_depth_training_scope(record, experimental_depth):
    verified = record.get("geometry_verified") is True and record.get("training_label_scope") == "verified_geometry"
    candidate = (record.get("geometry_verified") is False
                 and record.get("training_label_scope") == "image_plane_candidate")
    if not verified and not (experimental_depth and candidate):
        raise ValueError("DEPTH_GEOMETRY_UNVERIFIED: use verified labels, or explicitly select "
                         "experimental image-plane candidate supervision; no automatic approval")


class SegDepthDataset(Dataset):
    def __init__(self, dataset_config, source_indices, seg_dir, depth_dir,
                 metadata_sha256, split_sha256, tasks=("seg", "depth"),
                 missing_policy="error", experimental_depth=False, require_seg_for_depth=True):
        self.tasks = tuple(tasks)
        if not self.tasks or len(set(tasks)) != len(tasks) or set(tasks) - {"seg", "depth"}:
            raise ValueError("Dense tasks must be a nonempty subset of seg/depth")
        if missing_policy not in ("error", "mask"):
            raise ValueError("missing_policy must be error or mask")
        self.missing_policy, self.experimental_depth = missing_policy, bool(experimental_depth)
        load_tasks = list(self.tasks)
        if "depth" in self.tasks and require_seg_for_depth and "seg" not in load_tasks:
            load_tasks.append("seg")
        self.load_tasks = tuple(load_tasks)
        self.directories = {"seg": Path(seg_dir), "depth": Path(depth_dir)}
        self.metadata_sha256, self.split_sha256 = metadata_sha256, split_sha256
        self.source_indices = list(source_indices)
        self.images = OpenSceneMetadataDataset(**dataset_config, source_indices=self.source_indices)
        if tuple(self.images.camera_names) != OPENSCENE_CAMERA_NAMES or self.images.image_size != (252, 448):
            raise ValueError("Dense supervision requires canonical 8 cameras and [252,448]")
        self.digests, self.teacher_signatures = {}, {}
        tokens = [str(info["token"]) for info in self.images.infos]
        if len(set(tokens)) != len(tokens):
            raise ValueError("Duplicate token in dense dataset")

    def __len__(self):
        return len(self.source_indices)

    def _load_targets(self, position, establish=False):
        info, index = self.images.infos[position], self.source_indices[position]
        token = str(info["token"])
        targets = {"seg": None, "depth": None}
        digests = {}
        for task in self.load_tasks:
            path = self.directories[task] / (token + ".pt")
            if not path.is_file():
                if self.missing_policy == "error":
                    raise FileNotFoundError(f"{task} label missing: source_index={index} token={token} path={path}")
                digests[task] = None
                continue
            record, digest = read_hashed_record(path)
            if (record.get("metadata_sha256") != self.metadata_sha256
                    or record.get("split_sha256") != self.split_sha256
                    or str(record.get("scene_token")) != str(info["scene_token"])):
                raise ValueError(f"{task} metadata/split/scene mismatch: source_index={index} token={token}")
            provenance = record.get("provenance")
            if not isinstance(provenance, Mapping) or not isinstance(provenance.get("checkpoint_sha256"), str):
                raise ValueError(f"{task} teacher provenance missing for {token}")
            if task == "seg":
                validate_seg_record(record, token, index, OPENSCENE_CAMERA_NAMES, self.images.image_size)
                signature = (record["schema_version"], record["label_space"], record["pixel_alignment"],
                             provenance["checkpoint_sha256"], provenance.get("config_sha256"), provenance.get("adapter_sha256"))
                targets[task] = record["seg"]
            else:
                validate_depth_record(record, token, index, OPENSCENE_CAMERA_NAMES)
                validate_depth_training_scope(record, self.experimental_depth)
                depth = record["depth"]
                if provenance.get("native_saturation_excluded") is not True:
                    raise ValueError("Regenerate depth labels with native saturation exclusion")
                if len(record.get("source_hw", [])) != len(OPENSCENE_CAMERA_NAMES):
                    raise ValueError("Depth source image dimensions are missing")
                for camera_index, name in enumerate(OPENSCENE_CAMERA_NAMES):
                    camera = info["cams"][name]
                    intrinsic = scale_intrinsic(camera["cam_intrinsic"], record["source_hw"][camera_index], self.images.image_size)
                    if (not torch.allclose(depth["intrinsics"][camera_index].double(), torch.as_tensor(intrinsic).double(), atol=1e-4, rtol=1e-5)
                            or not torch.allclose(depth["sensor2lidar"][camera_index].double(),
                                                  torch.as_tensor(sensor_to_lidar(camera)).double(), atol=1e-4, rtol=1e-5)
                            or not torch.allclose(depth["distortion"][camera_index].double(),
                                                  torch.as_tensor(camera["distortion"], dtype=torch.float64), atol=1e-7, rtol=1e-5)):
                        raise ValueError(f"Depth calibration differs from metadata for {token} {name}")
                signature = (record["schema_version"], record["alignment_version"], depth["image_geometry"],
                             provenance["checkpoint_sha256"], provenance.get("adapter_sha256"),
                             provenance.get("geometry_adapter_sha256"), record["training_label_scope"])
                targets[task] = depth
            previous = self.teacher_signatures.setdefault(task, signature)
            if previous != signature:
                raise ValueError("Mixed teacher/preprocessing provenance for task=" + task)
            digests[task] = digest
        if establish:
            self.digests[index] = digests
        elif self.digests.get(index) != digests:
            raise ValueError(f"Dense labels changed after preflight: source_index={index} token={token}")
        if (self.missing_policy == "mask" and "depth" in self.tasks
                and "seg" in self.load_tasks and targets["seg"] is None):
            targets["depth"] = None  # No reliable sky exclusion without the aligned Seg target.
        return targets

    def audit(self, depth_settings=None):
        counts = {task: {"frames": 0, "valid_pixels": 0, "missing": 0} for task in self.load_tasks}
        for position in range(len(self)):
            targets = self._load_targets(position, establish=True)
            for task in self.load_tasks:
                target = targets[task]
                if target is None:
                    counts[task]["missing"] += 1
                else:
                    counts[task]["frames"] += 1
                    valid = (depth_valid_mask(target, targets["seg"], depth_settings)
                             if task == "depth" and depth_settings is not None else target["valid_mask"])
                    counts[task]["valid_pixels"] += int(valid.sum())
        if any(counts[task]["valid_pixels"] == 0 for task in self.load_tasks):
            raise ValueError("No valid supervision for an enabled dense task: " + str(counts))
        identity = [(index, str(info["token"]), self.digests[index])
                    for index, info in zip(self.source_indices, self.images.infos)]
        label_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return {"frames": len(self), "tasks": counts, "labels_sha256": label_hash,
                "teacher_signatures": dict(self.teacher_signatures)}

    def __getitem__(self, position):
        if not self.digests:
            raise RuntimeError("Call dataset.audit() before dense training/evaluation")
        targets = self._load_targets(position)
        info = self.images.infos[position]
        # Reuse the existing image/calibration path without loading unrelated Agent GT.
        sample = {**self.images._load_multiview(info), "sample_token": str(info["token"]),
                  "ego_state": torch.as_tensor(info["can_bus"][:9], dtype=torch.float32)}
        return {key: sample[key] for key in ("images", "intrinsics", "extrinsics", "ego_state", "sample_token")} | {
            "source_index": self.source_indices[position], "dense_targets": targets,
        }


def collate_seg_depth(samples):
    # Keep missing targets as None, never fabricate dummy GT for stacking.
    return {key: torch.stack([sample[key] for sample in samples])
            for key in ("images", "intrinsics", "extrinsics", "ego_state")} | {
        "sample_token": [sample["sample_token"] for sample in samples],
        "source_index": [sample["source_index"] for sample in samples],
        "dense_targets": [sample["dense_targets"] for sample in samples],
    }
