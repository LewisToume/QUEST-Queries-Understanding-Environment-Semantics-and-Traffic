"""Real same-split previous frames, with explicit cold-start masks and context audit."""

import hashlib
import json
import math
import pickle
from pathlib import Path

import torch

from .depth_geometry import pack_distortion
from .depth_teacher import sensor_to_lidar
from .map_teacher import _validated_transform
from .seg_depth_dataset import SegDepthDataset, read_hashed_record
from .teacher_adapters import NAVFORMER_CLASS_SUPPORT, validate_canonical_agent


def causal_history_plans(infos, targets, allowed_indices, options):
    if not all(math.isfinite(options[k]) and options[k] > 0 for k in ("timestamp_scale", "max_gap_s")):
        raise ValueError("Invalid temporal timestamp scale/gap")
    allowed = set(allowed_indices)
    if not set(targets) <= allowed or any(not isinstance(i, int) or i < 0 or i >= len(infos) for i in allowed):
        raise ValueError("History targets/context must belong to the selected manifest split")
    scenes, seen = {}, set()
    for index in allowed:
        info = infos[index]
        stamp, token = float(info["timestamp"]), str(info["token"])
        if not math.isfinite(stamp) or token in seen:
            raise ValueError("Nonfinite timestamp or duplicate token in temporal split")
        seen.add(token)
        scenes.setdefault(str(info["scene_token"]), []).append(index)
    previous = {}
    for scene, indices in scenes.items():
        indices.sort(key=lambda i: (float(infos[i]["timestamp"]), i))
        for position, index in enumerate(indices):
            prev = indices[position - 1] if position else None
            if prev is not None and float(infos[prev]["timestamp"]) >= float(infos[index]["timestamp"]):
                raise ValueError("Non-increasing timestamps within scene=" + scene)
            previous[index] = prev
    plans = []
    for index in targets:
        current, prev = infos[index], previous[index]
        dt = ((float(current["timestamp"]) - float(infos[prev]["timestamp"])) * options["timestamp_scale"]
              if prev is not None else 0.)
        reason = "scene_start" if prev is None else ("gap" if dt > options["max_gap_s"] else "available")
        if reason != "available":
            prev, dt = None, 0.
        current_pose = _validated_transform(current["lidar2global"], "current lidar2global")
        relative = (torch.linalg.inv(current_pose) @ _validated_transform(infos[prev]["lidar2global"], "previous lidar2global")
                    if prev is not None else torch.eye(4, dtype=torch.float64))
        plans.append({"source_index": index, "token": str(current["token"]),
                      "scene_token": str(current["scene_token"]), "timestamp": float(current["timestamp"]),
                      "history_index": prev, "history_token": str(infos[prev]["token"]) if prev is not None else None,
                      "history_timestamp": float(infos[prev]["timestamp"]) if prev is not None else None,
                      "dt_s": dt, "reason": reason, "previous_to_current": relative.float().tolist()})
    return plans


class TemporalSegDepthDataset(SegDepthDataset):
    def __init__(self, *args, temporal_options, history_source_indices, agent_labels=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.required_depth_geometry != "raw_opencv" or self.tasks != ("depth",):
            raise ValueError("Temporal dataset requires the explicit raw-geometry Depth-only contract")
        with self.images.metadata_path.open("rb") as stream:
            infos = pickle.load(stream)["infos"]  # Same trusted metadata, not external annotations.
        self.history_plans = causal_history_plans(infos, self.source_indices, history_source_indices, temporal_options)
        self.agent_labels = Path(agent_labels) if agent_labels is not None else None
        self.agent_digests = {}
        used = {row["history_index"] for row in self.history_plans if row["history_index"] is not None}
        self.history_infos = {index: infos[index] for index in used}
        for info in self.history_infos.values():
            if not self.images._is_complete_frame(info):
                raise ValueError("Previous frame has missing cameras/calibration/JPEG: " + str(info["token"]))
            for name in self.images.camera_names:
                camera = info["cams"][name]
                pack_distortion(camera["distortion"])
                _validated_transform(sensor_to_lidar(camera), "previous sensor2lidar")
        self.temporal_audit = {
            "policy": "same_split_same_scene_strict_past_v1", "frames": len(self.history_plans),
            "with_history": sum(row["history_index"] is not None for row in self.history_plans),
            "scene_start": sum(row["reason"] == "scene_start" for row in self.history_plans),
            "gap_masked": sum(row["reason"] == "gap" for row in self.history_plans),
            "max_dt_s": max((row["dt_s"] for row in self.history_plans), default=0.),
            "context_sha256": hashlib.sha256(json.dumps(self.history_plans, sort_keys=True).encode()).hexdigest(),
        }

    def _agent_target(self, position, establish=False):
        index, token = self.source_indices[position], str(self.images.infos[position]["token"])
        record, digest = read_hashed_record(self.agent_labels / (token + ".pt"))
        if str(record.get("token")) != token or ("sample_index" in record and record["sample_index"] != index):
            raise ValueError("Temporal Agent token/index mismatch")
        agent = record["agent"]
        validate_canonical_agent(agent, self.images.max_agent_instances)
        if not torch.equal(agent["class_support_mask"].bool(), NAVFORMER_CLASS_SUPPORT):
            raise ValueError("Temporal Agent must preserve vehicle/pedestrian support only")
        if establish:
            self.agent_digests[index] = digest
        elif self.agent_digests.get(index) != digest:
            raise ValueError("Agent labels changed after preflight")
        return agent

    def audit(self, *args, **kwargs):
        result = {**super().audit(*args, **kwargs), "temporal_context": dict(self.temporal_audit)}
        if self.agent_labels is not None:
            counts = {"vehicle": 0, "pedestrian": 0}
            for position in range(len(self)):
                agent = self._agent_target(position, establish=True)
                for class_id, name in enumerate(counts):
                    counts[name] += int(((agent["labels"] == class_id) & agent["valid_mask"].bool()).sum())
            result["agent_temporal_labels"] = {"frames": len(self), "counts": counts,
                "labels_sha256": hashlib.sha256(json.dumps(sorted(self.agent_digests.items())).encode()).hexdigest(),
                "reference": "NAVFORMER_SUPPORTED_CANONICAL_PSEUDO"}
        return result

    def __getitem__(self, position):
        sample = super().__getitem__(position)
        plan = self.history_plans[position]
        if plan["history_index"] is None:
            # Shape buffers only: NOT a repeated/fabricated historical observation.
            previous = {"images": torch.zeros_like(sample["images"]),
                        "intrinsics": sample["intrinsics"].clone(), "extrinsics": sample["extrinsics"].clone(),
                        "distortion": sample["distortion"].clone()}
        else:
            info = self.history_infos[plan["history_index"]]
            previous = self.images._load_multiview(info)
            previous["distortion"] = torch.stack([pack_distortion(info["cams"][name]["distortion"])
                                                  for name in self.images.camera_names])
        sample["history"] = {**previous, "valid": torch.tensor(plan["history_index"] is not None),
                             "dt_s": torch.tensor(plan["dt_s"], dtype=torch.float32),
                             "previous_to_current": torch.tensor(plan["previous_to_current"], dtype=torch.float32)}
        sample["history_identity"] = {key: plan[key] for key in (
            "source_index", "history_index", "history_token", "history_timestamp", "dt_s", "reason")}
        if self.agent_labels is not None:
            sample["soft_labels"] = {"agent": self._agent_target(position)}
        return sample
