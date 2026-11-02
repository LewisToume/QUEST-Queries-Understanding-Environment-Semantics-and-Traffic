"""Server CPU tests; native source required, no model weights/download/GPU required."""

import copy
import unittest
from pathlib import Path

import torch

from quest.small_dvgt_depth import SmallDVGTDepthFrontend, validate_small_dvgt
from quest.utils import project_root, load_yaml_config


class SmallDVGTTest(unittest.TestCase):
    def options(self):
        value = copy.deepcopy(load_yaml_config(project_root() / "configs/depth_small_dvgt_v5.yaml")["frontend"])
        value.update(geometry_dim=32, geometry_layers=4, geometry_levels=[0, 1, 2, 3],
                     heads=4, dpt_features=8, gradient_checkpointing=False, amp_bf16=False)
        return value

    def test_invalid_width_and_levels(self):
        value = self.options()
        validate_small_dvgt(value)
        value["geometry_dim"] = 30
        with self.assertRaises(ValueError):
            validate_small_dvgt(value)
        value = self.options()
        value["geometry_levels"] = [0, 1, 2, 4]
        with self.assertRaises(ValueError):
            validate_small_dvgt(value)

    @unittest.skipUnless((project_root() / "third_party/DVGT/dvgt").is_dir(), "Pinned server DVGT source required")
    def test_real_native_attention_backward_and_cold_start(self):
        torch.manual_seed(42)
        model = SmallDVGTDepthFrontend(16, self.options()).eval()
        self.assertFalse(hasattr(model.geometry_blocks, "patch_embed"))
        self.assertEqual(len(model.geometry_blocks.cross_frame_blocks), 4)
        current = [torch.randn(1, 8, 16, 2, 4) for _ in range(4)]
        history = {"feature_levels": [torch.randn_like(x) for x in current],
                   "valid": torch.tensor([True]), "dt_s": torch.tensor([.5])}
        result = model(current, None, None, None, (28, 56), history=history)
        self.assertEqual(result["depth"].shape, (1, 8, 1, 28, 56))
        self.assertTrue(bool(torch.isfinite(result["depth"]).all()))
        self.assertGreater(float(result["depth"].min()), 0)
        result["depth"].mean().backward()
        self.assertTrue(any(p.grad is not None and bool((p.grad != 0).any())
                            for p in model.temporal_blocks.parameters()))
        history.update(valid=torch.tensor([False]), dt_s=torch.tensor([0.]))
        with torch.no_grad():
            first = model(current, None, None, None, (28, 56), history=history)["depth"]
            history["feature_levels"] = [torch.randn_like(x) * 100 for x in current]
            second = model(current, None, None, None, (28, 56), history=history)["depth"]
        torch.testing.assert_close(first, second, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
