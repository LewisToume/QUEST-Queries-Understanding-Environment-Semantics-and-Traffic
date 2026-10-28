"""Test sources for server execution only; no assets, GPU or real data needed."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from quest.seg_depth_diagnostics import (DenseEvaluationDiagnostics, class_agreement, color_depth,
                                        preview_positions, rgb_from_normalized, save_dense_previews)
from quest.seg_depth_losses import empty_dense_metrics, summarize_dense_metrics, update_dense_metrics


SETTINGS = {"depth": {"min_depth_m": .1, "max_depth_m": 80., "exclude_sky": True}}


def example():
    depth = torch.tensor([.1, 10., 20., 30., 40., 50., 80., 100.]).reshape(1, 1, 2, 4).repeat(2, 1, 1, 1)
    target = {"seg": {"labels": torch.zeros(2, 2, 4, dtype=torch.uint8),
                      "valid_mask": torch.ones(2, 2, 4, dtype=torch.bool)},
              "depth": {"depth_m": depth, "valid_mask": torch.ones_like(depth, dtype=torch.bool)}}
    output = {"seg_logits": torch.zeros(1, 2, 19, 2, 4), "depth": (depth * 1.1)[None]}
    return output, target


class SegDepthDiagnosticsTest(unittest.TestCase):
    def test_per_class_counts_confusion_direction_and_empty_class(self):
        confusion = torch.zeros(19, 19, dtype=torch.int64)
        confusion[0, 0] = confusion[0, 1] = confusion[1, 1] = confusion[1, 2] = confusion[2, 2] = 1
        rows = class_agreement(confusion)
        self.assertEqual(rows[0]["teacher_pixels"], 2)
        self.assertEqual(rows[0]["student_pixels"], 1)
        self.assertAlmostEqual(rows[0]["precision"], 1.)
        self.assertAlmostEqual(rows[0]["recall"], .5)
        self.assertAlmostEqual(rows[1]["iou"], 1 / 3)
        self.assertIsNone(rows[3]["iou"])
        confusion[4, 3] = 1  # False positive for an absent teacher class is IoU=0, not N/A.
        self.assertEqual(class_agreement(confusion)[3]["iou"], 0.)

    def test_original_metrics_unchanged_and_bins_partition_evaluated_pixels(self):
        output, target = example()
        old = empty_dense_metrics()
        update_dense_metrics(old, output, [target], SETTINGS, ("seg", "depth"))
        diagnostic = DenseEvaluationDiagnostics(("CAM_F0", "CAM_B0"), SETTINGS, ("seg", "depth"))
        diagnostic.update(output, [target], [{"token": "a", "scene_token": "s"}], [15165], ["a"])
        report = diagnostic.summarize()
        self.assertEqual(report["overall"], summarize_dense_metrics(old))
        self.assertEqual([row["pixels"] for row in report["depth_distance_bins"]], [2, 2, 2, 2, 2, 4])
        self.assertEqual(report["depth_overall"]["pixels"], 14)
        self.assertAlmostEqual(report["depth_overall"]["absrel"], .1, places=5)
        self.assertAlmostEqual(report["depth_overall"]["geometric_mean_student_teacher_ratio"], 1.1, places=5)
        self.assertGreater(report["depth_overall"]["signed_bias_m"], 0.)
        self.assertEqual(sum(row["depth_pixels"] for row in report["per_camera"].values()), 14)
        self.assertEqual(report["per_scene"]["s"]["frames"], 1)
        self.assertEqual(report["frames"][0]["source_index"], 15165)
        self.assertEqual(report["depth_mask_counts"]["excluded_by_range_or_nonfinite"], 2)
        json.dumps(report, allow_nan=False)

    def test_sky_invalid_and_empty_bins_are_not_fake_perfect_metrics(self):
        output, target = example()
        target["seg"]["labels"][:] = 10
        diagnostic = DenseEvaluationDiagnostics(("F", "B"), SETTINGS, ("seg", "depth"))
        diagnostic.update(output, [target], [{"token": "a", "scene_token": "s"}], [7], ["a"])
        report = diagnostic.summarize()
        self.assertEqual(report["depth_mask_counts"]["excluded_by_sky_or_seg_validity"], 14)
        self.assertEqual(report["depth_overall"]["pixels"], 0)
        self.assertIsNone(report["depth_overall"]["absrel"])
        self.assertTrue(all(row["absrel"] is None for row in report["depth_distance_bins"]))
        json.dumps(report, allow_nan=False)

    def test_two_scenes_remain_separate_and_duplicate_tokens_rejected(self):
        output, target = example()
        diagnostic = DenseEvaluationDiagnostics(("F", "B"), SETTINGS, ("seg", "depth"))
        for index, scene in ((9, "scene_a"), (31, "scene_b")):
            token = str(index)
            diagnostic.update(output, [target], [{"token": token, "scene_token": scene}], [index], [token])
        report = diagnostic.summarize()
        self.assertEqual(set(report["per_scene"]), {"scene_a", "scene_b"})
        self.assertEqual(report["depth_overall"]["pixels"], 28)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            diagnostic.update(output, [target], [{"token": "9", "scene_token": "scene_a"}], [9], ["9"])

    def test_geometry_and_seg_invalid_pixels_are_excluded_from_all_bins(self):
        output, target = example()
        target["depth"]["valid_mask"][1, 0, 0, 0] = False
        target["seg"]["valid_mask"][0, 0, 1] = False
        diagnostic = DenseEvaluationDiagnostics(("F", "B"), SETTINGS, ("seg", "depth"))
        diagnostic.update(output, [target], [{"token": "a", "scene_token": "s"}], [0], ["a"])
        report = diagnostic.summarize()
        self.assertEqual(report["depth_mask_counts"]["export_valid_pixels"], 15)
        self.assertEqual(report["depth_mask_counts"]["excluded_by_sky_or_seg_validity"], 1)
        self.assertEqual(report["depth_overall"]["pixels"], 12)
        self.assertEqual(sum(row["pixels"] for row in report["depth_distance_bins"]), 12)

    def test_previews_prefer_distinct_scenes_and_never_overwrite(self):
        infos = [{"scene_token": scene} for scene in ("a", "a", "a", "b", "b", "c")]
        self.assertEqual(preview_positions(infos, 3), [0, 3, 5])
        self.assertEqual(preview_positions(infos, 0), [])
        self.assertEqual(len(preview_positions(infos, 100)), len(infos))
        output, target = example()
        images = torch.zeros(2, 3, 2, 4)
        prediction = {key: value[0] for key, value in output.items()}
        with tempfile.TemporaryDirectory() as directory:
            paths = save_dense_previews(directory, 15165, "a", "scene_a", images, prediction, target,
                                        ("F", "B"), SETTINGS, ("seg", "depth"))
            self.assertEqual(len(paths), 2)
            for name in paths:
                with Image.open(Path(directory) / name) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreater(image.width, image.height)
            with self.assertRaises(FileExistsError):
                save_dense_previews(directory, 15165, "a", "scene_a", images, prediction, target,
                                    ("F", "B"), SETTINGS, ("seg", "depth"))

    def test_preview_color_scale_is_absolute_and_rgb_undoes_imagenet_normalization(self):
        color_a = color_depth(np.array([[10., 20.]]), np.ones((1, 2), dtype=bool), .1, 80.)
        color_b = color_depth(np.array([[10., 50.]]), np.ones((1, 2), dtype=bool), .1, 80.)
        np.testing.assert_array_equal(color_a[0, 0], color_b[0, 0])
        black = color_depth(np.ones((1, 1)), np.zeros((1, 1), dtype=bool), .1, 80.)
        self.assertEqual(int(black.sum()), 0)
        value = torch.tensor([.5, .4, .3])[:, None, None]
        normalized = (value - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
        np.testing.assert_allclose(rgb_from_normalized(normalized)[0, 0], [128, 102, 76], atol=1)


if __name__ == "__main__":
    unittest.main()
