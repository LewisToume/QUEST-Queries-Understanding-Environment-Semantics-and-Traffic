import unittest

import torch

from quest.map_diagnostics import map_candidate_audit
from quest.map_training import class_aware_vector_matches


def lines(ys):
    ys = torch.tensor(ys, dtype=torch.float32)
    return torch.stack((torch.linspace(0., 5., 20).expand(len(ys), -1),
                        ys[:, None].expand(-1, 20)), dim=-1)


class MapCandidateAuditTest(unittest.TestCase):
    def test_partitions_misses_and_reproduces_class_aware_matching(self):
        target = {
            "class_ids": torch.tensor([0, 0, 0, 0, 0, 0, 0, 2]),
            "points_xy_m": lines([0, 10, 20, 30, 40, 40.2, 70, 80]),
            "is_closed": torch.zeros(8, dtype=torch.bool),
        }
        points = lines([0, 10, 20, 30, 40, 80])
        labels = torch.tensor([0, 3, 1, 0, 0, 2])
        scores = torch.tensor([.9, .8, .9, .5, .9, .9])
        result = map_candidate_audit(labels, scores, points, target, distance_m=.5, confidence=.6)
        self.assertEqual(result["centerline"], {
            "gt": 7, "predictions": 2, "matched": 2,
            "no_ordered_geometry": 1, "nearby_background_only": 1, "nearby_wrong_class": 1,
            "below_confidence": 1, "one_to_one_conflict": 1, "chamfer_coverage": 6,
            "ordered_2x_coverage": 6,
        })
        self.assertEqual(result["ped_crossing"]["gt"], 0)
        self.assertEqual(result["ped_crossing"]["predictions"], 1)
        self.assertEqual(result["road_boundary"]["matched"], 1)
        keep = (labels != 3) & (scores >= .6)
        expected = class_aware_vector_matches(labels[keep], points[keep], target, .5)
        self.assertEqual(sum(row["matched"] for row in result.values()), len(expected))

    def test_empty_queries_are_geometry_misses(self):
        target = {"class_ids": torch.tensor([1]), "points_xy_m": lines([0]),
                  "is_closed": torch.tensor([False])}
        result = map_candidate_audit(torch.empty(0, dtype=torch.long), torch.empty(0), lines([]),
                                     target, distance_m=1.5, confidence=.25)
        self.assertEqual(result["ped_crossing"]["no_ordered_geometry"], 1)
        self.assertEqual(result["ped_crossing"]["matched"], 0)


if __name__ == "__main__":
    unittest.main()
