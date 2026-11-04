"""Server CPU fixtures: no external DINO/teacher weights, dataset or GPU required."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from quest.model import QUESTModel
from quest.seg_on_v3 import (SEG_ON_V3_STAGE, DINOSegFormerBranch, SegOnFrozenV3, full_base_fingerprint,
                             initialize_segformer_decoder, segmentation_loss, load_seg_on_v3)
from quest.seg_teacher import CITYSCAPES_CLASSES
from quest.seg_depth_training import frozen_spatial_fingerprint
from test_geometry_depth_v2 import TinyBackbone, inputs, model_config, training_config
from test_temporal_depth_v3 import history, temporal_options


def teacher_config():
    return dict(model_type="segformer", num_labels=19, num_encoder_blocks=4, depths=[1, 1, 1, 1],
        hidden_sizes=[8, 16, 24, 32], decoder_hidden_size=16, reshape_last_stage=True,
        classifier_dropout_prob=0., id2label={str(i): name for i, name in enumerate(CITYSCAPES_CLASSES)},
        label2id={name: i for i, name in enumerate(CITYSCAPES_CLASSES)})


class SegOnV3Test(unittest.TestCase):
    def make_model(self):
        config = model_config()
        config["dense_frontend"] = temporal_options()
        with patch("quest.model.FrozenDINOv2Backbone", TinyBackbone):
            base = QUESTModel(**config).eval()
        return SegOnFrozenV3(base, DINOSegFormerBranch(16, teacher_config())), config

    def test_teacher_decoder_strict_load(self):
        branch = DINOSegFormerBranch(16, teacher_config())
        state = {"decode_head." + name: tensor.clone() for name, tensor in branch.decode_head.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pytorch_model.bin"
            torch.save(state, path)
            self.assertEqual(initialize_segformer_decoder(branch, directory), len(state))
            state.pop(next(iter(state)))
            torch.save(state, path)
            with self.assertRaises(RuntimeError):
                initialize_segformer_decoder(branch, directory)

    def test_seg_update_preserves_all_base_weights_outputs_and_eval_modes(self):
        torch.manual_seed(42)
        model, config = self.make_model()
        batch = inputs()
        batch["history"] = history(batch)
        model.eval()
        with torch.no_grad():
            reference_depth = model.base(**batch)["depth"]
            before = model(**batch)
        torch.testing.assert_close(before["depth"], reference_depth, atol=0, rtol=0)
        self.assertEqual(before["seg_logits"].shape, (1, 8, 19, 8, 16))
        base_hash = full_base_fingerprint(model.base)
        model.train()
        self.assertFalse(model.base.training)
        self.assertFalse(model.base.dense_frontend.training)
        self.assertTrue(model.seg_branch.training)
        self.assertTrue(all(not p.requires_grad for p in model.base.parameters()))
        optimizer = torch.optim.AdamW(model.seg_branch.parameters(), lr=1e-3)
        targets = [{"seg": {"labels": torch.randint(0, 19, (8, 8, 16), dtype=torch.uint8),
                            "valid_mask": torch.ones(8, 8, 16, dtype=torch.bool)}}]
        prediction = model.forward_seg(batch["images"])
        loss = segmentation_loss(prediction["seg_logits"], targets, torch.ones(19))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))
        for module in (model.seg_branch.projects, model.seg_branch.rgb_detail, model.seg_branch.decode_head):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)
        optimizer.step()
        self.assertEqual(base_hash, full_base_fingerprint(model.base))
        model.eval()
        with torch.no_grad():
            after = model(**batch)
        for key in ("depth", "agent_boxes", "agent_cls_logits", "agent_velocity", "map_cls_logits", "map_points"):
            torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)

        training = training_config()
        training["frontend"] = config["dense_frontend"]
        metadata = dict(stage="seg_depth_offline", dense_checkpoint_version=3, architecture_version=3,
            dense_frontend=config["dense_frontend"], model_config=config, training_config=training,
            spatial_preservation_verified=True, bev_use_ego_state=False, trained_dense_tasks=["depth"],
            trained_temporal_agent=False, trained_class_support_mask=torch.tensor([True, True, False, False]),
            depth_geometry_policy="experimental_image_plane", frozen_spatial_sha256=frozen_spatial_fingerprint(model.base))
        payload = dict(stage=SEG_ON_V3_STAGE, base_metadata=metadata, teacher_config=teacher_config(),
                       model_state_dict=model.state_dict(), base_sha256=base_hash)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combined.pt"
            torch.save(payload, path)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            with patch("quest.model.FrozenDINOv2Backbone", TinyBackbone):
                restored = load_seg_on_v3(saved).eval()
            with torch.no_grad():
                output = restored(**batch)
            for key in ("depth", "seg_logits", "agent_boxes", "map_points"):
                torch.testing.assert_close(output[key], after[key], atol=0, rtol=0)

    def test_seg_loss_masks_invalid_pixels_and_uses_weighted_normalizer(self):
        logits = torch.randn(1, 8, 19, 2, 4, requires_grad=True)
        labels = torch.zeros(8, 2, 4, dtype=torch.uint8)
        valid = torch.ones_like(labels, dtype=torch.bool)
        valid[:, 0] = False
        labels[:, 0] = 255
        target = [{"seg": {"labels": labels, "valid_mask": valid}}]
        a = segmentation_loss(logits, target, torch.ones(19))
        b = segmentation_loss(logits, target, torch.ones(19) * 3)
        torch.testing.assert_close(a, b)
        a.backward()
        self.assertEqual(float(logits.grad[:, :, :, 0].abs().sum()), 0)
        with self.assertRaises(ValueError):
            segmentation_loss(logits, [{"seg": {"labels": labels, "valid_mask": valid & False}}], torch.ones(19))


if __name__ == "__main__":
    unittest.main()
