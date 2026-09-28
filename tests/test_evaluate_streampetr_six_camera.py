import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "scripts/evaluate_streampetr_six_camera.py"
spec = importlib.util.spec_from_file_location(
    "evaluate_streampetr_six_camera_test", SCRIPT
)
evaluation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluation
spec.loader.exec_module(evaluation)


class EvaluateStreamPETRSixCameraTest(unittest.TestCase):
    def test_fixed_camera_order_and_threshold(self):
        self.assertEqual(
            evaluation.SIX_CAMERA_NAMES,
            ("CAM_F0", "CAM_B0", "CAM_L0", "CAM_L2", "CAM_R0", "CAM_R2"),
        )
        self.assertEqual(evaluation.SCORE_THRESHOLD, 0.25)

    def test_camera_visibility_uses_depth_and_image_bounds(self):
        camera = {
            "sensor2lidar_rotation": torch.eye(3),
            "sensor2lidar_translation": torch.zeros(3),
            "cam_intrinsic": torch.tensor(
                [[10.0, 0.0, 50.0], [0.0, 10.0, 40.0], [0.0, 0.0, 1.0]]
            ),
        }
        centers = torch.tensor(
            [
                [0.0, 0.0, 5.0],
                [30.0, 0.0, 5.0],
                [0.0, 0.0, -5.0],
            ]
        )
        visible = evaluation.camera_visibility_mask(centers, camera, (100, 80))
        self.assertEqual(visible.tolist(), [True, False, False])

    def test_prediction_filter_is_fixed_at_point_two_five(self):
        boxes = torch.zeros(5, 9)
        boxes[4, 0] = 50.01
        scores = torch.tensor([0.24, 0.25, 0.50, 0.90, 0.90])
        raw_labels = torch.tensor([0, 0, 8, 9, 8])
        filtered = evaluation.filter_predictions(boxes, scores, raw_labels)
        self.assertEqual(filtered["scores"].tolist(), [0.25, 0.5])
        self.assertEqual(filtered["labels"].tolist(), [0, 1])

    def test_metrics_report_class_precision_recall_and_pedestrian_bins(self):
        metrics = evaluation.SixCameraMetrics()
        prediction_labels = torch.tensor([0, 0, 1, 1])
        gt_centers = torch.tensor(
            [
                [5.0, 0.0, 0.0],
                [15.0, 0.0, 0.0],
                [25.0, 0.0, 0.0],
                [45.0, 0.0, 0.0],
            ]
        )
        gt_labels = torch.tensor([0, 1, 1, 1])
        matches = [(0, 0, 0.5), (2, 1, 0.5), (3, 3, 0.5)]
        metrics.update(prediction_labels, gt_centers, gt_labels, matches)
        summary = metrics.summary()

        self.assertAlmostEqual(summary["per_class"]["vehicle"]["precision"], 0.5)
        self.assertAlmostEqual(summary["per_class"]["vehicle"]["recall"], 1.0)
        self.assertAlmostEqual(summary["per_class"]["pedestrian"]["precision"], 1.0)
        self.assertAlmostEqual(summary["per_class"]["pedestrian"]["recall"], 2 / 3)
        bins = summary["pedestrian_distance_recall"]
        self.assertEqual([item["gt"] for item in bins], [0, 1, 1, 0, 1])
        self.assertEqual([item["matched"] for item in bins], [0, 1, 0, 0, 1])

    def test_distance_bin_boundaries(self):
        self.assertEqual(evaluation.pedestrian_distance_bin(0.0), 0)
        self.assertEqual(evaluation.pedestrian_distance_bin(9.999), 0)
        self.assertEqual(evaluation.pedestrian_distance_bin(10.0), 1)
        self.assertEqual(evaluation.pedestrian_distance_bin(50.0), 4)
        self.assertIsNone(evaluation.pedestrian_distance_bin(50.01))


if __name__ == "__main__":
    unittest.main()
