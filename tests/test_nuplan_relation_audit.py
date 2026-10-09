import unittest

import numpy as np
import pandas as pd

from quest.nuplan_relation_audit import (
    BASELINE_RELATION_AUDIT_VERSION, _relation_values, validate_relation_audit,
)


class NuPlanRelationAuditTest(unittest.TestCase):
    def test_null_other_association_is_not_invalid(self):
        values, valid, invalid = _relation_values(pd.Series([10.0, np.nan, 11.0]))
        self.assertEqual(valid.tolist(), [True, False, True])
        self.assertEqual(invalid.tolist(), [False, False, False])
        self.assertTrue(np.isnan(values[1]))

    def test_non_null_fractional_or_text_id_is_invalid(self):
        _, valid, invalid = _relation_values(pd.Series([10.5, "unknown", None, 11]))
        self.assertEqual(valid.tolist(), [False, False, False, True])
        self.assertEqual(invalid.tolist(), [True, True, False, False])

    def test_verified_warning_requires_impact_audit(self):
        fields = {column: {"null_rows": 1, "invalid_non_null_rows": 0,
                           "invalid_non_null_roi_rows": 0, "valid_association_rows": 1}
                  for column in ("lane_fid", "lane_connector_fid")}
        audit = {"version": BASELINE_RELATION_AUDIT_VERSION,
                 "status": "verified_no_baseline_relation_omission_in_roi", "roi_baseline_rows": 1,
                 "candidate_objects": {"LANE": 1, "LANE_CONNECTOR": 0},
                 "fields": fields, "unresolved_roi_rows": [],
                 "missing_candidate_relations": [], "unresolved_global_fields": [],
                 "invalid_cast_warnings": [{"source": "nuplan_map/utils.py", "line": 372,
                                            "message": "invalid value encountered in cast", "count": 1}],
                 "invalid_cast_warning_count": 1}
        validate_relation_audit(audit)
        audit["unresolved_roi_rows"] = ["baseline-2"]
        with self.assertRaisesRegex(ValueError, "missing or unresolved"):
            validate_relation_audit(audit)
        audit["unresolved_roi_rows"] = []
        audit["fields"]["lane_fid"]["invalid_non_null_roi_rows"] = 1
        with self.assertRaisesRegex(ValueError, "invalid IDs"):
            validate_relation_audit(audit)


if __name__ == "__main__":
    unittest.main()
