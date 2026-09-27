import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "scripts/evaluate_pseudo_vs_gt.py"
spec = importlib.util.spec_from_file_location("evaluate_pseudo_vs_gt_test", SCRIPT)
evaluation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluation
spec.loader.exec_module(evaluation)


class EvaluatePseudoVsGTTest(unittest.TestCase):
    def test_normalized_center_to_metric(self):
        centers = torch.tensor([[0.0, 0.5, 1.0], [1.0, 0.0, 0.5]])
        metric = evaluation.normalized_center_to_metric(centers)
        self.assertTrue(
            torch.equal(
                metric,
                torch.tensor([[-50.0, 0.0, 5.0], [50.0, -50.0, 0.0]]),
            )
        )

    def test_matching_is_class_aware_and_thresholded(self):
        pseudo_centers = torch.tensor(
            [[0.0, 0.0, 0.0], [1.9, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        pseudo_labels = torch.tensor([0, 1, 2])
        gt_centers = torch.tensor(
            [[0.1, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        gt_labels = torch.tensor([0, 1, 3])
        matches = evaluation.class_aware_matches(
            pseudo_centers,
            pseudo_labels,
            gt_centers,
            gt_labels,
            distance_threshold=2.0,
        )
        self.assertEqual([(pseudo, gt) for pseudo, gt, _ in matches], [(0, 0)])

    def test_threshold_metrics(self):
        metrics = evaluation.ThresholdMetrics(2.0)
        metrics.update([(0, 0, 0.5), (2, 1, 1.5)], torch.tensor([0, 1, 2]))
        summary = metrics.summary(
            pseudo_count=4,
            gt_count=5,
            pseudo_class_count=[1, 1, 1, 1],
            gt_class_count=[2, 1, 1, 1],
        )
        self.assertEqual(summary["matched"], 2)
        self.assertAlmostEqual(summary["precision"], 0.5)
        self.assertAlmostEqual(summary["recall"], 0.4)
        self.assertAlmostEqual(summary["mean_center_error"], 1.0)
        self.assertEqual(summary["per_class"]["vehicle"]["matched"], 1)
        self.assertEqual(summary["per_class"]["traffic_cone"]["matched"], 1)

    def test_coordinate_statistics(self):
        accumulator = evaluation.CoordinateAccumulator()
        accumulator.update(torch.tensor([[0.0, 2.0, 4.0], [2.0, 4.0, 6.0]]))
        summary = accumulator.summary()
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["mean"], [1.0, 3.0, 5.0])
        self.assertEqual(summary["min"], [0.0, 2.0, 4.0])
        self.assertEqual(summary["max"], [2.0, 4.0, 6.0])

    def test_raw_scores_are_aligned_by_class_and_center(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "token.pt"
            torch.save(
                {
                    "token": "token",
                    "boxes_3d": torch.tensor(
                        [
                            [10, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                        ],
                        dtype=torch.float32,
                    ),
                    "scores_3d": torch.tensor([0.8, 0.9]),
                    "labels_3d": torch.tensor([8, 0]),
                },
                path,
            )
            scores = evaluation.recover_raw_scores(
                path,
                pseudo_centers=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
                pseudo_labels=torch.tensor([0, 1]),
            )
        self.assertTrue(torch.allclose(scores, torch.tensor([0.9, 0.8])))


if __name__ == "__main__":
    unittest.main()
