from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from .map_teacher import (
    TEACHER_ALIGNMENT_VERSION, TEACHER_COORDINATE_FRAME, TEACHER_MAP_SCHEMA_VERSION,
    TEACHER_SCORE_KIND, align_teacher_map_to_quest_bev, resolve_lidar2ego,
    validate_teacher_record,
)
from .map_training import validate_vector_record
from .nuplan_map_locator import MAP_LAYER_AUDIT_VERSION
from .nuplan_relation_audit import BASELINE_RELATION_AUDIT_VERSION
from .openscene_dataset import OpenSceneMetadataDataset
from .vector_map_labels import MAP_CLASS_NAMES, MAP_HEIGHT_REFERENCE, VECTOR_SEMANTICS_VERSION


def load_record(path: Path) -> Mapping[str, Any]:
    try:
        record = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        record = torch.load(path, map_location="cpu")
    if not isinstance(record, Mapping):
        raise ValueError(f"label file must contain a mapping: {path}")
    return record


def load_teacher_audit(path: str | Path, expected_split_sha256: str | None = None) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        audit = json.load(stream)
    if audit.get("verified") is not True:
        raise ValueError("teacher channel/orientation audit must be explicitly VERIFIED before KD")
    if expected_split_sha256 is not None and audit.get("stage3_split_sha256") != expected_split_sha256:
        raise ValueError("teacher audit was not made for the current Stage 3 split")
    if audit.get("teacher_score_kind") != TEACHER_SCORE_KIND:
        raise ValueError("teacher audit score semantics differ from Pansegformer mask scores")
    if (audit.get("teacher_coordinate_frame") != TEACHER_COORDINATE_FRAME
            or audit.get("teacher_alignment_version") != TEACHER_ALIGNMENT_VERSION
            or audit.get("teacher_schema_version") != TEACHER_MAP_SCHEMA_VERSION):
        raise ValueError("teacher audit uses old LiDAR-frame alignment; re-export and re-audit")
    if audit.get("vector_map_classes") != list(MAP_CLASS_NAMES):
        raise ValueError("teacher audit was made for a different vector map taxonomy")
    vector_provenance = audit.get("vector_gt_provenance")
    if (not isinstance(vector_provenance, dict)
            or vector_provenance.get("vector_semantics_version") != VECTOR_SEMANTICS_VERSION
            or vector_provenance.get("map_height_reference") != MAP_HEIGHT_REFERENCE
            or vector_provenance.get("map_cast_audit_version") != BASELINE_RELATION_AUDIT_VERSION
            or vector_provenance.get("map_layer_audit_version") != MAP_LAYER_AUDIT_VERSION
            or vector_provenance.get("num_points") != 20
            or not vector_provenance.get("map_version")
            or not isinstance(vector_provenance.get("min_length_m"), (float, int))):
        raise ValueError("teacher audit was made for old or unknown vector GT geometry")
    names = audit.get("teacher_channel_names_or_ids", [])
    support = audit.get("teacher_channel_support_mask", [])
    weights = audit.get("teacher_channel_weights", [])
    if not names or len(names) != len(support) or len(names) != len(weights) or not any(support):
        raise ValueError("teacher audit channel names/support/weights are invalid")
    mapping = audit.get("teacher_channel_mapping", [])
    if len(mapping) != len(names):
        raise ValueError("teacher channel mapping is missing")
    allowed = {*MAP_CLASS_NAMES, "drivable_aux"}
    if any(enabled and mapping[index] not in allowed for index, enabled in enumerate(support)):
        raise ValueError("every supported teacher channel needs a reviewed semantic mapping")
    for key in ("row_axis", "row_direction", "col_direction", "teacher_pc_range",
                "teacher_checkpoint", "teacher_schema_version", "teacher_config_sha256",
                "teacher_checkpoint_size_bytes", "teacher_checkpoint_mtime_ns"):
        if key not in audit:
            raise ValueError(f"teacher audit missing {key}")
    if audit["row_axis"] not in ("x", "y") or audit["row_direction"] not in (-1, 1) or audit["col_direction"] not in (-1, 1):
        raise ValueError("teacher orientation is not fully specified")
    return audit


class Stage3JoinedDataset(Dataset):
    """Metadata raw indices are authoritative; filenames are joined by token."""

    def __init__(self, stage1_dataset_config: Mapping[str, Any], source_indices: list[int],
                 agent_dir: str | Path, vector_dir: str | Path, teacher_dir: str | Path,
                 audit: Mapping[str, Any], quest_range: tuple[float, float, float, float],
                 bev_h: int, bev_w: int, skip_missing_teacher: bool = False) -> None:
        self.vector_dir = Path(vector_dir)
        self.teacher_dir = Path(teacher_dir)
        self.audit = audit
        self.quest_range = quest_range
        self.bev_h, self.bev_w = bev_h, bev_w
        self.source_indices = list(source_indices)
        self.images = OpenSceneMetadataDataset(
            **stage1_dataset_config, soft_labels_root=agent_dir,
            source_indices=self.source_indices,
        )
        if skip_missing_teacher:
            keep = [position for position, info in enumerate(self.images.infos)
                    if (self.teacher_dir / f"{info['token']}.pt").is_file()]
            self.source_indices = [self.source_indices[position] for position in keep]
            self.images.infos = [self.images.infos[position] for position in keep]
        if not self.source_indices:
            raise RuntimeError("Stage 3 has no frames after explicit teacher-missing skip")
        tokens = [str(info["token"]) for info in self.images.infos]
        if len(tokens) != len(set(tokens)):
            raise ValueError("duplicate OpenScene tokens make Stage 3 index joining ambiguous")
        for source_index, info in zip(self.source_indices, self.images.infos):
            token = str(info["token"])
            for directory, label in ((Path(agent_dir), "Agent pseudo"),
                                     (self.vector_dir, "nuPlan vector GT"),
                                     (self.teacher_dir, "Navformer map teacher")):
                if not (directory / f"{token}.pt").is_file():
                    raise FileNotFoundError(f"{label} missing for metadata index={source_index} token={token}")

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        sample = self.images[position]
        token = str(sample["sample_token"])
        source_index = self.source_indices[position]
        if "agent" not in sample.get("soft_labels", {}):
            raise ValueError(f"Agent pseudo label missing for {token}")
        agent_payload = load_record(self.images.soft_labels_root / f"{token}.pt")
        if "sample_index" in agent_payload and agent_payload["sample_index"] != source_index:
            raise ValueError(f"Agent pseudo label index/token mismatch for {token}")
        vector = load_record(self.vector_dir / f"{token}.pt")
        validate_vector_record(vector, token, source_index, self.quest_range,
                               expected_info=self.images.infos[position])
        for key, expected in self.audit["vector_gt_provenance"].items():
            if vector.get(key) != expected:
                raise ValueError(f"vector GT {key} differs from audited labels for {token}")
        teacher = load_record(self.teacher_dir / f"{token}.pt")
        soft = validate_teacher_record(teacher, token, source_index)
        lidar2ego = resolve_lidar2ego(self.images.infos[position])
        if not torch.allclose(teacher["teacher_lidar2ego"], lidar2ego, atol=1e-4, rtol=1e-4):
            raise ValueError(f"teacher lidar2ego differs from OpenScene metadata for {token}")
        if list(teacher["teacher_channel_names_or_ids"]) != self.audit["teacher_channel_names_or_ids"]:
            raise ValueError(f"teacher channels differ from audit for {token}")
        if tuple(teacher["teacher_pc_range"]) != tuple(self.audit["teacher_pc_range"]):
            raise ValueError(f"teacher pc_range differs from audit for {token}")
        if str(teacher["teacher_checkpoint"]) != self.audit["teacher_checkpoint"]:
            raise ValueError(f"teacher checkpoint differs from audit for {token}")
        if teacher["teacher_score_kind"] != self.audit["teacher_score_kind"]:
            raise ValueError(f"teacher score semantics differ from audit for {token}")
        for key in ("teacher_coordinate_frame", "teacher_alignment_version"):
            if teacher[key] != self.audit[key]:
                raise ValueError(f"teacher {key} differs from audit for {token}")
        for key in ("teacher_config_sha256", "teacher_checkpoint_size_bytes",
                    "teacher_checkpoint_mtime_ns"):
            if teacher[key] != self.audit[key]:
                raise ValueError(f"teacher {key} differs from audit for {token}")
        sample["source_index"] = torch.tensor(source_index, dtype=torch.int64)
        sample["vector_target"] = vector
        aligned, valid = align_teacher_map_to_quest_bev(
            soft, teacher["teacher_pc_range"], self.quest_range, self.bev_h, self.bev_w,
            self.audit["row_axis"], self.audit["row_direction"], self.audit["col_direction"],
            lidar2ego=lidar2ego,
        )
        sample["teacher_map_aligned"] = aligned
        sample["teacher_map_valid"] = valid
        return sample


def collate_stage3(samples: list[dict[str, Any]]) -> dict[str, Any]:
    from .dataset import collate_fn

    vectors = [sample.pop("vector_target") for sample in samples]
    batch = collate_fn(samples)
    batch["vector_targets"] = vectors
    return batch
