import json
import tempfile
import unittest
from pathlib import Path

from quest.map_training import load_vector_capacity_audit
from quest.nuplan_map_locator import MAP_LAYER_AUDIT_VERSION
from quest.nuplan_relation_audit import BASELINE_RELATION_AUDIT_VERSION
from quest.vector_map_labels import MAP_HEIGHT_REFERENCE, VECTOR_GT_SCHEMA_VERSION, VECTOR_SEMANTICS_VERSION
from scripts.audit_vector_gt_capacity import recommended_capacity, summary


class VectorGTCapacityTest(unittest.TestCase):
    def test_distribution_and_capacity(self):
        result = summary([50, 58, 101])
        self.assertEqual(result["frames_exceeding"]["50"], 2)
        self.assertEqual(result["frames_exceeding"]["64"], 1)
        self.assertEqual(recommended_capacity(result["max"]), 128)

    def test_full_split_audit_gate(self):
        config = {"train": {"start_index": 0, "num_samples": 5000},
                  "eval": {"start_index": 5000, "num_samples": 100},
                  "map": {"map_query_count": None}}
        provenance = {"num_points": 20, "min_length_m": 1.0, "map_version": "test",
                      "vector_semantics_version": VECTOR_SEMANTICS_VERSION,
                      "map_height_reference": MAP_HEIGHT_REFERENCE,
                      "map_cast_audit_version": BASELINE_RELATION_AUDIT_VERSION,
                      "map_layer_audit_version": MAP_LAYER_AUDIT_VERSION}
        report = {"capacity_certified": True, "schema_version": VECTOR_GT_SCHEMA_VERSION,
                  "vector_semantics_version": VECTOR_SEMANTICS_VERSION,
                  "vector_gt_provenance": provenance, "recommended_map_query_count": 128,
                  "splits": {"train": {"start_index": 0, "num_samples": 5000,
                                       "available": 5000, "total": {"max": 103}},
                             "eval": {"start_index": 5000, "num_samples": 100,
                                      "available": 100, "total": {"max": 87}}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            count, loaded = load_vector_capacity_audit(path, config)
            self.assertEqual(count, 128)
            self.assertEqual(loaded, provenance)
            report["capacity_certified"] = False
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or obsolete"):
                load_vector_capacity_audit(path, config)


if __name__ == "__main__":
    unittest.main()
