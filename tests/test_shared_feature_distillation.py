"""Server-only CPU fixtures: no downloads, assets, real labels, or GPU."""

import copy
import unittest
from unittest.mock import patch

import torch

from quest.model import QUESTModel
from quest.seg_on_v3 import DINOSegFormerBranch, SegOnFrozenV3, full_base_fingerprint
from quest.shared_feature_distillation import (FrozenDenseGuide, SharedFeatureDistiller,
    deployment_payload, load_deployment, restore_student)
from quest.unified_scene import UNIFIED_SCENE_VERSION, UnifiedQUEST
from quest.unified_training import four_task_losses
from scripts.train_unified_feature_kd import predict
from test_geometry_depth_v2 import TinyBackbone, inputs
from test_seg_on_v3 import teacher_config
from test_temporal_depth_v3 import history
from test_unified_scene import fixtures, targets


class SharedFeatureDistillationTest(unittest.TestCase):
    def make(self):
        torch.manual_seed(37)
        config, profile = fixtures()
        with patch("quest.model.FrozenDINOv2Backbone", TinyBackbone):
            base = QUESTModel(**config).eval()
        baseline = SegOnFrozenV3(base, DINOSegFormerBranch(16, teacher_config())).eval()
        student = UnifiedQUEST(config, profile, backbone=TinyBackbone())
        student.warm_start(baseline)
        guide = FrozenDenseGuide(baseline)
        distiller = SharedFeatureDistiller(8, guide.seg_channels, guide.depth_channels)
        data = inputs()
        data["history"] = history(data)
        batch, settings = targets(data)
        batch.update(data)
        options = dict(feature_grid=[2, 4], min_valid_fraction=.5, seg_weight=1., depth_weight=.1)
        return student, guide, distiller, batch, settings, options

    def test_guide_shares_only_frozen_input_features_and_does_not_change_predictions(self):
        student, guide, distiller, batch, settings, options = self.make()
        student.eval()
        guide.train(True)
        self.assertFalse(guide.training)
        self.assertTrue(all(not p.requires_grad for p in guide.parameters()))
        for absent in ("backbone", "agent_decoder", "map_decoder", "base"):
            self.assertFalse(hasattr(guide, absent))
        teacher_hash = full_base_fingerprint(guide)
        data = {k: batch[k] for k in ("images", "intrinsics", "extrinsics", "distortion", "history", "ego_state")}
        with torch.no_grad():
            expected = student(**data)
            with patch.object(student.backbone, "forward_intermediates", wraps=student.backbone.forward_intermediates) as call:
                result, kd = predict(student, guide, distiller, batch, torch.device("cpu"), settings, options)
                self.assertEqual(call.call_count, 2)
            for key, value in expected.items():
                torch.testing.assert_close(result[key], value, rtol=0, atol=0)
            control, zeros = predict(student, None, distiller, batch, torch.device("cpu"), settings, options)
            for key, value in expected.items():
                torch.testing.assert_close(control[key], value, rtol=0, atol=0)
            self.assertTrue(all(float(value) == 0 for value in zeros.values()))
            self.assertTrue(all(bool(torch.isfinite(value)) for value in kd.values()))
        self.assertEqual(full_base_fingerprint(guide), teacher_hash)

    def test_feature_loss_updates_common_geometry_and_adapter_but_not_frozen_guides(self):
        student, guide, distiller, batch, settings, options = self.make()
        student.train()
        teacher_hash = full_base_fingerprint(guide)
        dino_hash = full_base_fingerprint(student.backbone)
        optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad] + list(distiller.parameters()), lr=1e-3)
        old = student.trunk.geometry.level_projections[-1].weight.detach().clone()
        result, kd = predict(student, guide, distiller, batch, torch.device("cpu"), settings, options)
        feature_loss = kd["seg_feature_kd"] + .1 * kd["depth_feature_kd"]
        gradient = torch.autograd.grad(feature_loss, student.trunk.geometry.level_projections[-1].weight, retain_graph=True)[0]
        self.assertGreater(float(gradient.norm()), 0)
        raw, _ = four_task_losses(student, result, batch, settings, torch.device("cpu"))
        (sum(raw.values()) + feature_loss).backward()
        self.assertGreater(float(distiller.semantic_adapter.weight.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in guide.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(old, student.trunk.geometry.level_projections[-1].weight))
        self.assertEqual(full_base_fingerprint(guide), teacher_hash)
        self.assertEqual(full_base_fingerprint(student.backbone), dino_hash)

    def test_masking_teacher_stop_gradient_empty_and_shape_checks(self):
        torch.manual_seed(7)
        value = torch.randn(1, 4, 2, 2, requires_grad=True)
        teacher = torch.randn_like(value, requires_grad=True)
        mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
        mask[0, 0, 0, 0] = True
        loss = SharedFeatureDistiller.distance(value, teacher, mask)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(float(value.grad[:, :, 1, :].abs().sum()), 0.)
        with self.assertRaises(ValueError):
            SharedFeatureDistiller.distance(value, teacher, torch.zeros_like(mask))
        with self.assertRaises(ValueError):
            SharedFeatureDistiller.distance(value, teacher[:, :2], mask)
        with self.assertRaises(ValueError):
            SharedFeatureDistiller(8, 16, 12)

    def test_strict_parent_restore_and_standalone_deployment_roundtrip(self):
        student, guide, distiller, batch, settings, options = self.make()
        student.eval()
        parent = dict(stage=UNIFIED_SCENE_VERSION, model_config=student.model_config, profile=student.profile,
                      model_state_dict=copy.deepcopy(student.state_dict()), trained_tasks=["agent", "map", "seg", "depth"])
        restored = UnifiedQUEST(student.model_config, student.profile, backbone=TinyBackbone()).eval()
        restore_student(parent, restored)
        data = {k: batch[k] for k in ("images", "intrinsics", "extrinsics", "distortion", "history", "ego_state")}
        payload = deployment_payload(restored, parent, 1, {}, {"all_pass": False})
        self.assertNotIn("training_distiller_state_dict", payload)
        self.assertFalse(any("adapter" in name or "guide" in name for name in payload["model_state_dict"]))
        deployed = load_deployment(payload, backbone=TinyBackbone()).eval()
        with torch.no_grad():
            expected, actual = student(**data), deployed(**data)
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
        bad = dict(parent, profile={**parent["profile"], "query_layers": 3})
        with self.assertRaises(ValueError):
            restore_student(bad, restored)
        bad = dict(parent, trained_tasks=["depth"])
        with self.assertRaises(ValueError):
            restore_student(bad, restored)
        with self.assertRaises(ValueError):
            load_deployment({**payload, "inference_guides": 1}, backbone=TinyBackbone())


if __name__ == "__main__":
    unittest.main()
