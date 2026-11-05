"""Join all four targets by the same metadata token; never truncate Map GT."""

import hashlib
import json
from pathlib import Path

import torch

from .map_training import validate_vector_record
from .seg_depth_dataset import collate_seg_depth, read_hashed_record
from .temporal_depth_dataset import TemporalSegDepthDataset


class FourTaskDataset(TemporalSegDepthDataset):
    def __init__(self, *args, vector_dir, vector_provenance, xy_range, map_queries, **kwargs):
        super().__init__(*args, **kwargs)
        if self.tasks != ("seg", "depth") or self.agent_labels is None:
            raise ValueError("Four-task batches require Agent, vector Map, Seg, and Depth labels")
        self.vector_dir = Path(vector_dir)
        self.vector_provenance = dict(vector_provenance)
        self.xy_range, self.map_queries = tuple(xy_range), map_queries
        self.vector_digests = {}

    def vector(self, position, establish=False):
        info, index = self.images.infos[position], self.source_indices[position]
        token = str(info["token"])
        record, digest = read_hashed_record(self.vector_dir / (token + ".pt"))
        validate_vector_record(record, token, index, self.xy_range, expected_info=info)
        if any(record.get(key) != value for key, value in self.vector_provenance.items()):
            raise ValueError("Vector GT provenance differs from certified capacity audit")
        if len(record["class_ids"]) > self.map_queries:
            raise ValueError("Vector GT exceeds shared query capacity; truncation is forbidden")
        if establish:
            self.vector_digests[index] = digest
        elif self.vector_digests.get(index) != digest:
            raise ValueError("Vector GT changed after four-task preflight")
        return record

    def audit(self, *args, **kwargs):
        result = super().audit(*args, **kwargs)
        counts, scene_locations = [0, 0, 0], {}
        for position, info in enumerate(self.images.infos):
            record = self.vector(position, establish=True)
            city = scene_locations.setdefault(str(info["scene_token"]), record["map_location"])
            if city != record["map_location"]:
                raise ValueError("Vector GT map city changes inside a scene")
            for i in range(3):
                counts[i] += int((record["class_ids"] == i).sum())
        result["vector_gt"] = {"counts": counts, "frames": len(self),
            "sha256": hashlib.sha256(json.dumps(sorted(self.vector_digests.items())).encode()).hexdigest(),
            "provenance": self.vector_provenance}
        return result

    def __getitem__(self, position):
        return {**super().__getitem__(position), "vector_target": self.vector(position)}


def collate_four_tasks(samples):
    return {**collate_seg_depth(samples), "vector_targets": [sample["vector_target"] for sample in samples]}
