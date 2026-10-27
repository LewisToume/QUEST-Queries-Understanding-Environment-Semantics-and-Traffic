"""Server-run CPU test sources; no external weights, datasets or GPU required."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from quest.dense_perception import SHARED_IMAGE_FRONTEND_VERSION
from quest.model import QUESTModel, load_quest_v3_checkpoint
from quest.openscene_dataset import OPENSCENE_CAMERA_NAMES
from quest.seg_depth_dataset import SegDepthDataset, validate_depth_training_scope
from quest.seg_depth_losses import (compute_seg_depth_loss, depth_valid_mask, empty_dense_metrics,
                                    update_dense_metrics, summarize_dense_metrics)
from quest.seg_depth_training import (configure_dense_training, frozen_spatial_fingerprint,
                                      load_dense_model, save_dense_checkpoint)
from quest.seg_teacher import CITYSCAPES_CLASSES, SEG_PIXEL_ALIGNMENT


SETTINGS = {"weights": {"seg_ce": 1., "depth_log": 1., "depth_relative": .5, "depth_gradient": .5},
            "depth": {"min_depth_m": .1, "max_depth_m": 80., "exclude_sky": True}}


def targets():
    return [{"seg": {"labels": torch.zeros(8, 8, 16, dtype=torch.uint8),
                      "valid_mask": torch.ones(8, 8, 16, dtype=torch.bool)},
             "depth": {"depth_m": torch.full((8, 1, 8, 16), 12.),
                       "valid_mask": torch.ones(8, 1, 8, 16, dtype=torch.bool)}}]


class TinyBackbone(nn.Module):
    hidden_dim, patch_size = 16, 2

    def __init__(self, *args):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(.1, 1., 16), requires_grad=False)
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        value = F.avg_pool2d(images.mean(1, keepdim=True), 2).flatten(2).transpose(1, 2)
        tokens = value * self.weight[None, None, :]
        return torch.cat((tokens[:, :1], tokens), 1).detach()

    def forward_intermediates(self, images, layer_indices):
        value = self(images)
        return value, tuple(value * scale for scale in (.25, .5, .75, 1.))


def model_settings():
    return dict(hidden_dim=16, bev_h=4, bev_w=4, bev_layers=1, bev_attention_heads=4, bev_ffn_dim=32,
                agent_decoder_layers=1, map_decoder_layers=1, decoder_attention_heads=4,
                decoder_ffn_dim=32, N_agent=4, N_map=3, C_map=3, dropout=0.,
                x_range=(-4., 4.), y_range=(-4., 4.), C_seg=19, seg_size=[8, 16], depth_size=[8, 16],
                dense_frontend={"version": SHARED_IMAGE_FRONTEND_VERSION,
                                "preserve_spatial_features": True, "layer_indices": [3, 6, 9, 12],
                                "channels": 8, "output_size": [8, 16], "min_depth_m": .1})


class DenseTrainingTest(unittest.TestCase):
    def model(self):
        with patch("quest.model.FrozenDINOv2Backbone", TinyBackbone):
            return QUESTModel(**model_settings())

    def inputs(self):
        return {"images": torch.randn(1, 8, 3, 8, 16),
                "intrinsics": torch.tensor([[8., 0, 8], [0, 8., 4], [0, 0, 1.]])[None, None].repeat(1, 8, 1, 1),
                "extrinsics": torch.eye(4)[None, None].repeat(1, 8, 1, 1), "ego_state": torch.zeros(1, 9)}

    def test_four_task_forward_uses_one_dino_extraction(self):
        model = self.model().eval()
        result = model(**self.inputs())
        self.assertEqual(model.backbone.calls, 1)
        self.assertEqual(result["agent_boxes"].shape, (1, 4, 8))
        self.assertEqual(result["map_points"].shape, (1, 3, 20, 2))
        self.assertEqual(result["seg_logits"].shape, (1, 8, 19, 8, 16))
        self.assertEqual(result["depth"].shape, (1, 8, 1, 8, 16))

    def test_backward_updates_only_dense_neck_and_heads(self):
        model, inputs = self.model(), self.inputs()
        parameters = configure_dense_training(model, ("seg", "depth"))
        baseline = frozen_spatial_fingerprint(model)
        before = model.dense_frontend.lateral[0][0].weight.detach().clone()
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)
        prediction = model.forward_dense(inputs["images"], inputs["intrinsics"])
        loss = compute_seg_depth_loss(prediction, targets(), SETTINGS)["total_loss"]
        loss.backward()
        for module in (model.dense_frontend.lateral, model.dense_frontend.seg_decoder, model.dense_frontend.depth_decoder):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)
        self.assertFalse(model.dense_frontend.bev_gate.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.dense_frontend.lateral[0][0].weight))
        self.assertEqual(baseline, frozen_spatial_fingerprint(model))

    def test_invalid_sky_saturation_and_missing_targets_do_not_train_depth(self):
        target = targets()
        target[0]["seg"]["labels"][:, 0] = 10
        target[0]["depth"]["depth_m"][:, :, 1] = 171.2375
        valid = depth_valid_mask(target[0]["depth"], target[0]["seg"], SETTINGS["depth"])
        self.assertFalse(bool(valid[:, :, :2].any()))
        prediction = {"seg_logits": torch.zeros(1, 8, 19, 8, 16, requires_grad=True),
                      "depth": torch.full((1, 8, 1, 8, 16), 12., requires_grad=True)}
        losses = compute_seg_depth_loss(prediction, target, SETTINGS)
        self.assertEqual(float(losses["depth_log"]), 0)
        self.assertEqual(float(losses["depth_gradient"]), 0)
        empty = compute_seg_depth_loss(prediction, [{"seg": None, "depth": None}], SETTINGS)
        self.assertEqual(float(empty["total_loss"]), 0)
        empty["total_loss"].backward()
        self.assertEqual(float(prediction["depth"].grad.abs().sum()), 0)

    def test_metric_agreement_is_explicitly_not_real_gt(self):
        target = targets()
        prediction = {"seg_logits": torch.zeros(1, 8, 19, 8, 16), "depth": target[0]["depth"]["depth_m"][None] * 2}
        metrics = empty_dense_metrics()
        update_dense_metrics(metrics, prediction, target, SETTINGS, ("seg", "depth"))
        result = summarize_dense_metrics(metrics)
        self.assertAlmostEqual(result["depth_teacher_absrel"], 1.)
        self.assertIn("NOT_REAL_GT", result["reference"])

    def test_versioned_checkpoint_roundtrip_and_no_legacy_conversion(self):
        model = self.model()
        optimizer = torch.optim.AdamW(configure_dense_training(model, ("seg", "depth")), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
        config = {"depth_geometry_policy": "experimental_image_plane", "tasks": ["seg", "depth"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense.pt"
            save_dense_checkpoint(path, model, optimizer, scheduler, 1, config, model_settings(),
                                  {"split_sha256": "split", "metadata_sha256": "metadata"}, {},
                                  frozen_spatial_fingerprint(model), {"epoch": 12})
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            with patch("quest.model.FrozenDINOv2Backbone", TinyBackbone):
                restored = load_dense_model(checkpoint)
            self.assertEqual(frozen_spatial_fingerprint(model), frozen_spatial_fingerprint(restored))
            self.assertEqual(checkpoint["epoch"], 1)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            legacy = dict(checkpoint)
            legacy.pop("dense_frontend")
            with self.assertRaisesRegex(ValueError, "profile mismatch"):
                load_quest_v3_checkpoint(model, legacy)


class DenseDatasetTest(unittest.TestCase):
    def test_candidate_depth_never_silently_becomes_verified_supervision(self):
        record = {"geometry_verified": False, "training_label_scope": "image_plane_candidate"}
        with self.assertRaisesRegex(ValueError, "DEPTH_GEOMETRY_UNVERIFIED"):
            validate_depth_training_scope(record, False)
        validate_depth_training_scope(record, True)
        record["training_label_scope"] = "diagnostic_only"
        with self.assertRaises(ValueError):
            validate_depth_training_scope(record, True)
        validate_depth_training_scope({"geometry_verified": True, "training_label_scope": "verified_geometry"}, False)

    def test_noncontiguous_token_join_hash_change_and_missing_policy(self):
        class Images:
            camera_names, image_size = OPENSCENE_CAMERA_NAMES, (252, 448)

            def __init__(self, source_indices, **kwargs):
                self.infos = [{"token": f"t{index}", "scene_token": "s"} for index in source_indices]

        def record(index):
            return {"schema_version": 1, "token": f"t{index}", "source_index": index, "scene_token": "s",
                    "metadata_sha256": "m", "split_sha256": "split", "camera_names": list(OPENSCENE_CAMERA_NAMES),
                    "class_names": CITYSCAPES_CLASSES, "label_space": "cityscapes19", "pixel_alignment": SEG_PIXEL_ALIGNMENT,
                    "provenance": {"checkpoint_sha256": "a" * 64},
                    "seg": {"labels": torch.zeros(8, 252, 448, dtype=torch.uint8),
                            "confidence": torch.ones(8, 252, 448, dtype=torch.float16),
                            "valid_mask": torch.ones(8, 252, 448, dtype=torch.bool)}}

        with tempfile.TemporaryDirectory() as directory, patch("quest.seg_depth_dataset.OpenSceneMetadataDataset", Images):
            root = Path(directory)
            for index in (19, 7):
                torch.save(record(index), root / f"t{index}.pt")
            dataset = SegDepthDataset({}, [19, 7], root, root, "m", "split", tasks=("seg",))
            self.assertEqual(dataset.audit()["frames"], 2)
            self.assertEqual(dataset.source_indices, [19, 7])
            invalid = record(7)
            invalid["token"] = "t19"
            torch.save(invalid, root / "t7.pt")
            with self.assertRaisesRegex(ValueError, "token/original source"):
                dataset._load_targets(1)
            torch.save(record(7), root / "t7.pt")
            dataset.audit()
            changed = record(7)
            changed["seg"]["labels"][:, 0, 0] = 1
            torch.save(changed, root / "t7.pt")
            with self.assertRaisesRegex(ValueError, "changed after preflight"):
                dataset._load_targets(1)
            (root / "t7.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                dataset.audit()
            masked = SegDepthDataset({}, [19, 7], root, root, "m", "split", tasks=("seg",), missing_policy="mask")
            self.assertEqual(masked.audit()["tasks"]["seg"]["missing"], 1)
            self.assertIsNone(masked._load_targets(1)["seg"])


if __name__ == "__main__":
    unittest.main()
