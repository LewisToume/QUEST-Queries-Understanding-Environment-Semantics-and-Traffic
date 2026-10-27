import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from quest.backbone import FrozenDINOv2Backbone
from quest.dense_perception import SharedDensePerception


class SharedDensePerceptionTest(unittest.TestCase):
    def inputs(self):
        levels = [torch.randn(1, 8, 16, 4, 8) for _ in range(4)]
        intrinsic = torch.tensor([[20., 0, 16.], [0, 20., 8.], [0, 0, 1.]])
        return levels, intrinsic[None, None].expand(1, 8, -1, -1).clone()

    def test_rectangular_dense_and_spatial_output_shapes(self):
        module = SharedDensePerception(hidden_dim=16, channels=8, output_size=(16, 32))
        levels, intrinsic = self.inputs()
        result = module(levels, intrinsic, (16, 32))
        self.assertEqual(result["seg_logits"].shape, (1, 8, 19, 16, 32))
        self.assertEqual(result["depth"].shape, (1, 8, 1, 16, 32))
        self.assertEqual(result["image_feature_maps"].shape, (1, 8, 16, 4, 8))
        self.assertEqual(len(result["image_pyramid"]), 4)
        self.assertTrue(bool((result["depth"] > 0).all()))

    def test_each_task_sends_gradients_into_shared_neck(self):
        for task in ("seg_logits", "depth", "image_feature_maps"):
            torch.manual_seed(42)
            module = SharedDensePerception(hidden_dim=16, channels=8, output_size=(16, 32))
            levels, intrinsic = self.inputs()
            module(levels, intrinsic, (16, 32))[task].square().mean().backward()
            for lateral in module.lateral:
                gradient = lateral[0].weight.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(bool(torch.isfinite(gradient).all()))
                self.assertGreater(float(gradient.abs().sum()), 0)

    def test_camera_conditioning_changes_depth_not_seg_logits(self):
        module = SharedDensePerception(hidden_dim=16, channels=8, output_size=(16, 32)).eval()
        levels, intrinsic = self.inputs()
        first = module(levels, intrinsic, (16, 32))
        changed = intrinsic.clone()
        changed[..., 0, 0] *= 2
        second = module(levels, changed, (16, 32))
        torch.testing.assert_close(first["seg_logits"], second["seg_logits"])
        self.assertGreater(float((first["depth"] - second["depth"]).abs().sum()), 0)

    def test_bad_calibration_or_legacy_class_count_rejected(self):
        with self.assertRaises(ValueError):
            SharedDensePerception(num_classes=6)
        module = SharedDensePerception(hidden_dim=16, channels=8, output_size=(16, 32))
        levels, intrinsic = self.inputs()
        intrinsic[..., 0, 0] = 0
        with self.assertRaises(ValueError):
            module(levels, intrinsic, (16, 32))

    def test_optional_dino_intermediates_are_frozen_normalized_and_do_not_change_keys(self):
        class TinyDINO(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(num_hidden_layers=12, hidden_size=16, patch_size=14)
                self.layernorm = nn.LayerNorm(16)

            def forward(self, pixel_values, output_hidden_states=False):
                state = pixel_values.mean((1, 2, 3))[:, None, None] * torch.arange(16)[None, None].float()
                states = tuple(state.expand(-1, 9, -1) + index for index in range(13))
                return SimpleNamespace(last_hidden_state=self.layernorm(states[-1]), hidden_states=states)

        with patch("quest.backbone.AutoModel.from_pretrained", return_value=TinyDINO()):
            backbone = FrozenDINOv2Backbone(local_backbone_dir="not-an-asset-fixture")
        keys = set(backbone.state_dict())
        last, levels = backbone.forward_intermediates(torch.ones(2, 3, 28, 56))
        self.assertEqual(len(levels), 4)
        torch.testing.assert_close(last, levels[-1])
        self.assertFalse(any(value.requires_grad for value in levels))
        self.assertFalse(any(parameter.requires_grad for parameter in backbone.parameters()))
        self.assertEqual(keys, set(backbone.state_dict()))
        with self.assertRaises(ValueError):
            backbone.forward_intermediates(torch.ones(2, 3, 28, 56), (1, 1, 2, 3))


if __name__ == "__main__":
    unittest.main()
