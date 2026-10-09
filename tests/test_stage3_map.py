import unittest

import numpy as np
import torch

from quest.map_teacher import align_teacher_map_to_quest_bev, soft_map_distillation_loss
from quest.map_training import equivalent_point_orders, match_vector_queries, validate_vector_record
from quest.nuplan_map_locator import MAP_LAYER_AUDIT_VERSION, MAP_EXPORT_REQUIRED_LAYERS
from quest.nuplan_relation_audit import BASELINE_RELATION_AUDIT_VERSION
from quest.vector_map_labels import (
    COORDINATE_FRAME, MAP_HEIGHT_REFERENCE, VECTOR_GT_SCHEMA_VERSION,
    VECTOR_SEMANTICS_VERSION, resample_polyline,
)


class Stage3MapTest(unittest.TestCase):
    def test_open_and_closed_arc_length_resampling(self):
        open_points = resample_polyline(np.array([[0, 0], [10, 0]]), 20, False)
        self.assertEqual(open_points.shape, (20, 2))
        np.testing.assert_allclose(open_points[0], [0, 0])
        np.testing.assert_allclose(open_points[-1], [10, 0])
        closed = resample_polyline(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]), 20, True)
        self.assertEqual(closed.shape, (20, 2))
        self.assertFalse(np.allclose(closed[0], closed[-1]))

    def test_point_order_equivalence(self):
        points = torch.stack((torch.arange(20).float(), torch.zeros(20)), dim=-1)
        open_variants = equivalent_point_orders(points, False)
        torch.testing.assert_close(open_variants[1], points.flip(0))
        closed_variants = equivalent_point_orders(points, True)
        self.assertEqual(closed_variants.shape, (40, 20, 2))
        self.assertTrue(any(torch.equal(item, torch.roll(points, 7, dims=0)) for item in closed_variants))

    def test_hungarian_reversed_line(self):
        target_points = torch.stack((torch.linspace(-10, 10, 20), torch.zeros(20)), dim=-1)
        target = {"class_ids": torch.tensor([0]), "points_xy_m": target_points[None],
                  "is_closed": torch.tensor([False])}
        predicted = torch.stack((torch.linspace(0.4, 0.6, 20).flip(0), torch.full((20,), 0.5)), dim=-1)
        logits = torch.tensor([[9.0, 0.0, 0.0, -9.0]])
        matches = match_vector_queries(logits, predicted[None], target,
                                       (-50, -50, 50, 50), 2.0, 5.0)
        self.assertEqual(len(matches), 1)
        torch.testing.assert_close(matches[0][2], predicted)

    def test_metric_alignment_and_orientation(self):
        teacher = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 15
        aligned, valid = align_teacher_map_to_quest_bev(
            teacher, (-2, -2, 2, 2), (-2, -2, 2, 2), 4, 4, "y", 1, 1,
            lidar2ego=torch.eye(4),
        )
        torch.testing.assert_close(aligned, teacher, atol=1e-6, rtol=0)
        self.assertTrue(bool(valid.all()))
        flipped, _ = align_teacher_map_to_quest_bev(
            teacher, (-2, -2, 2, 2), (-2, -2, 2, 2), 4, 4, "y", -1, 1,
            lidar2ego=torch.eye(4),
        )
        torch.testing.assert_close(flipped, teacher.flip(1), atol=1e-6, rtol=0)

    def test_unsupported_teacher_channel_does_not_enter_kd(self):
        logits = torch.zeros(1, 2, 2, 2)
        teacher = torch.stack((torch.zeros(2, 2), torch.ones(2, 2)))[None]
        loss = soft_map_distillation_loss(logits, teacher, torch.ones(1, 2, 2, dtype=torch.bool),
                                          torch.tensor([True, False]),
                                          torch.ones(2))
        torch.testing.assert_close(loss, torch.tensor(0.69314718), atol=1e-6, rtol=0)

    def test_vector_token_index_validation(self):
        record = {"schema_version": VECTOR_GT_SCHEMA_VERSION, "sample_index": 4, "token": "token-4",
                  "coordinate_frame": COORDINATE_FRAME, "xy_range_m": (-50, -50, 50, 50),
                  "num_points": 20, "min_length_m": 1.0, "map_version": "test-map",
                  "map_height_reference": MAP_HEIGHT_REFERENCE, "map_reference_global_z_m": 0.0,
                  "map_location": "test-city", "scene_token": "scene-4",
                  "map_cast_audit_version": BASELINE_RELATION_AUDIT_VERSION,
                  "map_layer_audit_version": MAP_LAYER_AUDIT_VERSION,
                  "map_layer_diagnostics": {
                      "version": MAP_LAYER_AUDIT_VERSION, "status": "verified_for_frame_roi",
                      "per_layer": {name: {"source_rows": 0, "api_rows": 0,
                                           "source_invalid_rows": 0, "api_invalid_rows": 0,
                                           "preexisting_invalid_outside_roi": 0,
                                           "new_invalid_rows": 0, "missing_rows": 0}
                                    for name in MAP_EXPORT_REQUIRED_LAYERS},
                  },
                  "map_cast_diagnostics": {
                      "version": BASELINE_RELATION_AUDIT_VERSION,
                      "status": "verified_no_baseline_relation_omission_in_roi", "roi_baseline_rows": 0,
                      "candidate_objects": {"LANE": 0, "LANE_CONNECTOR": 0},
                      "fields": {key: {"null_rows": 0, "invalid_non_null_rows": 0,
                                       "invalid_non_null_roi_rows": 0, "valid_association_rows": 0}
                                 for key in ("lane_fid", "lane_connector_fid")},
                      "unresolved_roi_rows": [], "missing_candidate_relations": [],
                      "unresolved_global_fields": [], "invalid_cast_warnings": [],
                      "invalid_cast_warning_count": 0,
                  },
                  "vector_semantics_version": VECTOR_SEMANTICS_VERSION,
                  "class_ids": torch.empty(0, dtype=torch.long),
                  "points_xy_m": torch.empty(0, 20, 2),
                  "is_closed": torch.empty(0, dtype=torch.bool), "length_m": torch.empty(0)}
        validate_vector_record(record, "token-4", 4, (-50, -50, 50, 50))
        with self.assertRaisesRegex(ValueError, "index/token mismatch"):
            validate_vector_record(record, "token-4", 5, (-50, -50, 50, 50))
        with self.assertRaisesRegex(ValueError, "map_version differs"):
            validate_vector_record(record, "token-4", 4, (-50, -50, 50, 50),
                                   expected_map_version="another-map")
        with self.assertRaisesRegex(ValueError, "min_length_m differs"):
            validate_vector_record(record, "token-4", 4, (-50, -50, 50, 50),
                                   expected_min_length_m=2.0)
        record["vector_semantics_version"] = "old_polygon_crop"
        with self.assertRaisesRegex(ValueError, "geometry semantics mismatch"):
            validate_vector_record(record, "token-4", 4, (-50, -50, 50, 50))
        record["vector_semantics_version"] = VECTOR_SEMANTICS_VERSION
        record["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "regenerate old vector labels"):
            validate_vector_record(record, "token-4", 4, (-50, -50, 50, 50))
        record["schema_version"] = VECTOR_GT_SCHEMA_VERSION
        record["map_cast_diagnostics"]["status"] = "unverified"
        with self.assertRaisesRegex(ValueError, "relation audit"):
            validate_vector_record(record, "token-4", 4, (-50, -50, 50, 50))
        record["map_cast_diagnostics"]["status"] = "verified_no_baseline_relation_omission_in_roi"
        info = {"scene_token": "scene-4", "lidar2global": np.eye(4)}
        validate_vector_record(record, "token-4", 4, (-50, -50, 50, 50), expected_info=info)
        record["map_reference_global_z_m"] = 100.0
        with self.assertRaisesRegex(ValueError, "reference height differs"):
            validate_vector_record(record, "token-4", 4, (-50, -50, 50, 50), expected_info=info)


if __name__ == "__main__":
    unittest.main()
