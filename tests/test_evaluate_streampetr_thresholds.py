import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "scripts/evaluate_streampetr_thresholds.py"
spec = importlib.util.spec_from_file_location(
    "evaluate_streampetr_thresholds_test", SCRIPT
)
evaluation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluation
spec.loader.exec_module(evaluation)


class EvaluateStreamPETRThresholdsTest(unittest.TestCase):
    def test_confidence_filtering_and_class_mapping(self):
        boxes = torch.zeros(4, 9)
        scores = torch.tensor([0.05, 0.10, 0.20, 0.90])
        raw_labels = torch.tensor([0, 8, 9, 5])
        filtered = evaluation.filter_raw_predictions(
            boxes, scores, raw_labels, confidence_threshold=0.10
        )
        self.assertTrue(
            torch.allclose(filtered["scores"], torch.tensor([0.10, 0.20, 0.90]))
        )
        self.assertEqual(filtered["labels"].tolist(), [1, 2, 3])

    def test_converter_class_mapping_is_complete(self):
        self.assertEqual(
            evaluation.STREAM_PETR_TO_QUEST.tolist(),
            [0, 0, 0, 0, 0, 3, 0, 0, 1, 2],
        )

    def test_strict_class_aware_hungarian(self):
        matches = evaluation.strict_class_aware_hungarian(
            prediction_centers=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
            ),
            prediction_labels=torch.tensor([0, 1, 2]),
            gt_centers=torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]),
            gt_labels=torch.tensor([1, 1]),
            distance_threshold=2.0,
        )
        self.assertEqual([(prediction, gt) for prediction, gt, _ in matches], [(1, 1)])

    def test_precision_and_recall(self):
        metrics = evaluation.ThresholdMetrics()
        metrics.update(
            prediction_labels=torch.tensor([0, 1, 1]),
            gt_labels=torch.tensor([0, 1, 2, 3]),
            matches=[(0, 0, 0.5), (1, 1, 1.0)],
        )
        summary = metrics.summary()
        self.assertAlmostEqual(summary["precision"], 2 / 3)
        self.assertAlmostEqual(summary["recall"], 2 / 4)
        self.assertAlmostEqual(summary["mean_bev_center_error"], 0.75)
        self.assertEqual(summary["per_class"]["pedestrian"]["predictions"], 2)
        self.assertEqual(summary["per_class"]["pedestrian"]["matched"], 1)


if __name__ == "__main__":
    unittest.main()
