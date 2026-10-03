import unittest

import torch

from quest.map_teacher import (
    TEACHER_ALIGNMENT_VERSION, TEACHER_COORDINATE_FRAME,
    TEACHER_MAP_SCHEMA_VERSION, TEACHER_RAW_SCORE_SEMANTICS,
    TEACHER_SCORE_KIND, TEACHER_SCORE_TRANSFORM, validate_teacher_record,
)
from scripts.export_navformer_map_soft import extract_teacher_soft_scores, temporal_export_plan


class Stage3MapExportTest(unittest.TestCase):
    def test_panseg_mask_scores_keep_zero_background(self):
        lanes = torch.zeros(3, 2, 2)
        lanes[1, 0, 0] = 0.75
        masks = torch.zeros(101, 2, 2)
        masks[-1, 0, 1] = 1.25
        masks[-1, 1, 0] = -0.25
        soft, count = extract_teacher_soft_scores({"lane_score": lanes, "score_list": masks})
        self.assertEqual(count, 3)
        self.assertEqual(tuple(soft.shape), (4, 2, 2))
        self.assertEqual(float(soft[0, 1, 1]), 0.0)
        self.assertEqual(float(soft[1, 0, 0]), 0.75)
        self.assertEqual(float(soft[-1, 0, 1]), 1.0)
        self.assertEqual(float(soft[-1, 1, 0]), 0.0)
        with self.assertRaisesRegex(ValueError, "must be a tensor"):
            extract_teacher_soft_scores({"lane_score": lanes, "score_list": [masks]})

    def test_temporal_plan_preserves_original_target_slice(self):
        infos = [
            {"scene_token": "A", "token": "a2", "timestamp": 2},
            {"scene_token": "B", "token": "b1", "timestamp": 1},
            {"scene_token": "A", "token": "a0", "timestamp": 0},
            {"scene_token": "A", "token": "a1", "timestamp": 1},
        ]
        plan = temporal_export_plan(infos, 0, 2)
        self.assertEqual({item["sample_index"] for item in plan if item["target"]}, {0, 1})
        self.assertEqual([item["info"]["token"] for item in plan], ["a0", "a1", "a2", "b1"])
        self.assertFalse(plan[0]["target"])
        self.assertFalse(plan[1]["target"])
        self.assertTrue(plan[2]["target"])
        self.assertEqual(plan[2]["scene_start_token"], "a0")

    def test_teacher_schema_requires_temporal_and_score_provenance(self):
        record = {
            "schema_version": TEACHER_MAP_SCHEMA_VERSION,
            "sample_index": 7, "token": "sample-7",
            "teacher_map_soft": torch.zeros(4, 2, 2),
            "teacher_map_shape": (4, 2, 2),
            "teacher_channel_names_or_ids": ["a", "b", "c", "drivable"],
            "teacher_pc_range": (-51.2, -51.2, 51.2, 51.2),
            "teacher_coordinate_frame": TEACHER_COORDINATE_FRAME,
            "teacher_alignment_version": TEACHER_ALIGNMENT_VERSION,
            "teacher_lidar2ego": torch.eye(4, dtype=torch.float64),
            "teacher_config": "config.py", "teacher_config_sha256": "config-digest",
            "teacher_checkpoint": "teacher.pth", "teacher_checkpoint_size_bytes": 100,
            "teacher_checkpoint_mtime_ns": 12345,
            "teacher_score_kind": TEACHER_SCORE_KIND,
            "teacher_raw_score_semantics": TEACHER_RAW_SCORE_SEMANTICS,
            "teacher_score_transform": TEACHER_SCORE_TRANSFORM,
            "temporal_mode": "scene_start_to_target",
            "temporal_history_sha256": "digest",
            "temporal_scene_start_token": "sample-0",
        }
        validate_teacher_record(record, "sample-7", 7)
        record["teacher_score_transform"] = "sigmoid"
        with self.assertRaisesRegex(ValueError, "score transform mismatch"):
            validate_teacher_record(record, "sample-7", 7)
        record["teacher_score_transform"] = TEACHER_SCORE_TRANSFORM
        record["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "regenerate old LiDAR-frame labels"):
            validate_teacher_record(record, "sample-7", 7)


if __name__ == "__main__":
    unittest.main()
