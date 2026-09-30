import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "scripts/evaluate_agent.py"
spec = importlib.util.spec_from_file_location("evaluate_agent_test", SCRIPT)
evaluation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluation
spec.loader.exec_module(evaluation)


class EvaluateAgentTest(unittest.TestCase):
    def test_checkpoint_loads_model_state_dict(self):
        source = torch.nn.Linear(2, 1)
        target = torch.nn.Linear(2, 1)
        with torch.no_grad():
            source.weight.fill_(3.0)
            source.bias.fill_(2.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(
                {
                    "architecture_version": 3,
                    "model_state_dict": source.state_dict(),
                    "epoch": 5,
                    "trained_class_support_mask": torch.tensor(
                        [True, True, False, False]
                    ),
                },
                path,
            )
            epoch = evaluation.load_model_checkpoint(
                target, path, torch.device("cpu")
            )
        self.assertEqual(epoch, 5)
        self.assertTrue(torch.equal(target.weight, source.weight))
        self.assertTrue(torch.equal(target.bias, source.bias))
        self.assertEqual(
            target.trained_class_support_mask.tolist(),
            [True, True, False, False],
        )

    def test_normalized_center_to_metric_center(self):
        centers = torch.tensor([[0.0, 0.5, 1.0], [1.0, 0.0, 0.5]])
        metric = evaluation.normalized_center_to_metric(centers)
        self.assertTrue(
            torch.equal(
                metric,
                torch.tensor([[-50.0, 0.0, 5.0], [50.0, -50.0, 0.0]]),
            )
        )

    def test_confidence_and_background_filtering(self):
        logits = torch.tensor(
            [
                [5.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        boxes = torch.zeros(3, 8)
        filtered = evaluation.filter_predictions(logits, boxes, 0.25)
        self.assertEqual(filtered["labels"].tolist(), [0])
        self.assertEqual(tuple(filtered["centers_m"].shape), (1, 3))

    def test_unsupported_class_cannot_be_prediction_even_with_highest_logit(self):
        logits = torch.tensor([[1.0, 0.0, 100.0, 99.0, -1.0]])
        filtered = evaluation.filter_predictions(
            logits,
            torch.zeros(1, 8),
            confidence_threshold=0.25,
            trained_class_support_mask=torch.tensor([True, True, False, False]),
        )
        self.assertEqual(filtered["labels"].tolist(), [0])

    def test_unsupported_gt_is_excluded_from_supported_metrics(self):
        agent_gt = {
            "labels": torch.tensor([[0, 2, 3, -1]]),
            "boxes_metric": torch.zeros(1, 4, 7),
            "velocity_mps": torch.zeros(1, 4, 3),
            "scores": torch.ones(1, 4),
            "class_support_mask": torch.ones(1, 4, dtype=torch.bool),
            "valid_mask": torch.tensor([[True, True, True, False]]),
        }
        filtered = evaluation.filter_agent_gt_by_class_support(
            agent_gt, torch.tensor([True, True, False, False])
        )
        self.assertEqual(filtered["valid_mask"].tolist(), [[True, False, False, False]])
        self.assertEqual(
            filtered["class_support_mask"].tolist(),
            [[True, True, False, False]],
        )
        valid = filtered["valid_mask"][0]
        metrics = evaluation.AgentMetrics()
        metrics.update(
            prediction_labels=torch.tensor([0]),
            gt_labels=filtered["labels"][0][valid],
            matches=[(0, 0, 0.25)],
            agent_loss=1.0,
        )
        summary = metrics.summary()
        self.assertEqual(summary["total_gt"], 1)
        self.assertEqual(summary["per_class"]["traffic_cone"]["gt"], 0)
        self.assertEqual(summary["per_class"]["generic_object"]["gt"], 0)
        self.assertEqual(summary["recall"], 1.0)

    def test_hungarian_matches_within_each_class(self):
        prediction_centers = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        prediction_labels = torch.tensor([0, 1])
        gt_centers = torch.tensor([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]])
        gt_labels = torch.tensor([1, 0])
        matches = evaluation.match_agents(
            prediction_centers,
            prediction_labels,
            gt_centers,
            gt_labels,
            distance_threshold=2.0,
        )
        self.assertEqual([(pred, gt) for pred, gt, _ in matches], [(0, 1), (1, 0)])

    def test_wrong_class_is_never_matched(self):
        matches = evaluation.match_agents(
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([0]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([1]),
            distance_threshold=2.0,
        )
        self.assertEqual(matches, [])

    def test_distance_threshold_rejects_match_over_two_meters(self):
        matches = evaluation.match_agents(
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([0]),
            torch.tensor([[2.01, 0.0, 0.0]]),
            torch.tensor([0]),
            distance_threshold=2.0,
        )
        self.assertEqual(matches, [])

    def test_precision_recall_and_per_class_metrics(self):
        metrics = evaluation.AgentMetrics()
        metrics.update(
            prediction_labels=torch.tensor([0, 1, 3]),
            gt_labels=torch.tensor([0, 2]),
            matches=[(0, 0, 0.5), (1, 1, 1.0)],
            agent_loss=2.0,
        )
        summary = metrics.summary()
        self.assertAlmostEqual(summary["precision"], 1 / 3)
        self.assertAlmostEqual(summary["recall"], 1 / 2)
        self.assertAlmostEqual(summary["class_accuracy_on_matched"], 1.0)
        self.assertAlmostEqual(summary["mean_center_error_m"], 0.5)
        self.assertEqual(summary["per_class"]["vehicle"]["matched"], 1)
        self.assertEqual(summary["per_class"]["traffic_cone"]["matched"], 0)

    def test_validation_range_starts_after_training_frames(self):
        dataset = list(range(250))
        subset = evaluation.validation_subset(dataset, start=100, count=5)
        self.assertEqual(list(subset.indices), [100, 101, 102, 103, 104])
        self.assertEqual([subset[index] for index in range(len(subset))], list(range(100, 105)))


if __name__ == "__main__":
    unittest.main()
