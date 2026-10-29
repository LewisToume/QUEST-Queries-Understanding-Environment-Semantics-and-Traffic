"""Server-run test sources for bounded Seg balance; no assets/GPU required."""

import copy
import math
import unittest

import torch
import torch.nn.functional as F

from quest.seg_depth_losses import compute_seg_depth_loss
from quest.seg_depth_training import resolve_seg_class_balance, validate_dense_data_join


def settings():
    return {"weights": {"seg_ce": 1., "depth_log": 1., "depth_relative": .5, "depth_gradient": .5},
            "depth": {"min_depth_m": .1, "max_depth_m": 80., "exclude_sky": True},
            "seg": {"class_balance": {"mode": "inverse_sqrt_train_pixels", "min_weight": .5, "max_weight": 5.}}}


class SegClassBalanceTest(unittest.TestCase):
    def audit(self):
        counts = [10000000] + [10000] * 17 + [1]
        return {"seg_class_pixel_counts": counts, "labels_sha256": "labels",
                "tasks": {"seg": {"valid_pixels": sum(counts)}}}

    def sample(self):
        logits = torch.zeros(1, 1, 19, 1, 2, requires_grad=True)
        prediction = {"seg_logits": logits, "depth": torch.ones(1, 1, 1, 1, 2)}
        targets = [{"seg": {"labels": torch.tensor([[[0, 18]]], dtype=torch.uint8),
                             "valid_mask": torch.ones(1, 1, 2, dtype=torch.bool)}, "depth": None}]
        return prediction, targets

    def test_actual_training_counts_give_bounded_weights(self):
        config = settings()
        result = resolve_seg_class_balance(config, self.audit())
        self.assertEqual(result["source"], "TRAIN_ONLY")
        self.assertEqual(result["class_weights"][0], .5)
        self.assertEqual(result["class_weights"][-1], 5.)
        self.assertTrue(all(.5 <= value <= 5. for value in result["class_weights"]))
        resolve_seg_class_balance(config, self.audit(), saved_weights_required=True)
        config["seg"]["class_weights"][0] = .75
        with self.assertRaisesRegex(ValueError, "Saved Seg class weights"):
            resolve_seg_class_balance(config, self.audit(), saved_weights_required=True)

    def test_missing_fractional_or_inconsistent_training_counts_fail(self):
        for counts in ([], [1] * 18, [-1] * 19, [0] * 19, [.5] * 19):
            audit = self.audit()
            audit["seg_class_pixel_counts"] = counts
            with self.assertRaises(ValueError):
                resolve_seg_class_balance(settings(), audit)
        audit = self.audit()
        audit["tasks"]["seg"]["valid_pixels"] += 1
        with self.assertRaises(ValueError):
            resolve_seg_class_balance(settings(), audit)

    def test_weighted_mean_matches_pytorch_and_rare_pixel_has_larger_gradient(self):
        config = settings()
        weights = resolve_seg_class_balance(config, self.audit())["class_weights"]
        prediction, targets = self.sample()
        losses = compute_seg_depth_loss(prediction, targets, config, ("seg",))
        expected = F.cross_entropy(prediction["seg_logits"][0], targets[0]["seg"]["labels"].long(),
                                   weight=torch.tensor(weights))
        torch.testing.assert_close(losses["seg_ce"], expected)
        self.assertAlmostEqual(float(losses["seg_ce"]), math.log(19), places=5)
        losses["total_loss"].backward()
        gradient = prediction["seg_logits"].grad
        common = float(gradient[0, 0, :, 0, 0].abs().sum())
        rare = float(gradient[0, 0, :, 0, 1].abs().sum())
        self.assertAlmostEqual(rare / common, 10., places=4)
        self.assertEqual(losses["seg_pixels"], 2)

    def test_common_weight_scaling_cancels_and_invalid_pixels_are_masked(self):
        prediction, targets = self.sample()
        config = settings()
        config["seg"]["class_weights"] = [.5] * 18 + [5.]
        first = compute_seg_depth_loss(prediction, targets, config, ("seg",))["seg_ce"]
        config["seg"]["class_weights"] = [2.] * 18 + [20.]
        second = compute_seg_depth_loss(prediction, targets, config, ("seg",))["seg_ce"]
        torch.testing.assert_close(first, second)
        targets[0]["seg"]["valid_mask"][:] = False
        losses = compute_seg_depth_loss(prediction, targets, config, ("seg",))
        self.assertEqual(float(losses["total_loss"]), 0.)
        self.assertTrue(bool(torch.isfinite(losses["total_loss"])))

    def test_old_unweighted_loss_is_preserved(self):
        prediction, targets = self.sample()
        config = settings()
        config.pop("seg")
        result = compute_seg_depth_loss(prediction, targets, config, ("seg",))
        expected = F.cross_entropy(prediction["seg_logits"][0], targets[0]["seg"]["labels"].long())
        torch.testing.assert_close(result["seg_ce"], expected)

    def test_finetune_accepts_derived_counts_but_rejects_changed_labels(self):
        manifest = {"split_sha256": "split", "metadata_sha256": "metadata"}
        model_config = {"dense_frontend": {"version": "shared_image_pyramid_v1"}}
        audits = {"train": {"labels_sha256": "train", "frames": 5000},
                  "validation": {"labels_sha256": "val", "frames": 120}}
        checkpoint = {"stage": "seg_depth_offline", "dense_checkpoint_version": 1, "architecture_version": 3,
                      "spatial_preservation_verified": True, "bev_use_ego_state": False,
                      "trained_class_support_mask": torch.tensor([True, True, False, False]),
                      "depth_geometry_policy": "experimental_image_plane", "trained_dense_tasks": ["seg", "depth"],
                      "training_config": {"depth_geometry_policy": "experimental_image_plane", "tasks": ["seg", "depth"]},
                      "dense_frontend": model_config["dense_frontend"], "model_config": model_config,
                      **manifest, "initialization": {"epoch": 12}, "label_audits": copy.deepcopy(audits)}
        audits["train"]["seg_class_pixel_counts"] = [1] * 19
        validate_dense_data_join(checkpoint, manifest, audits, model_config, {"epoch": 12})
        audits["train"]["labels_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "label hashes"):
            validate_dense_data_join(checkpoint, manifest, audits, model_config, {"epoch": 12})


if __name__ == "__main__":
    unittest.main()
