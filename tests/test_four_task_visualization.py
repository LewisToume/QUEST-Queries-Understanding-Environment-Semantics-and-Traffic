"""Synthetic CPU checks; no real checkpoint, teacher, dataset, or GPU."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from scripts.visualize_quest_four_tasks import box_outline, decode_outputs, render


class FourTaskVisualizationTest(unittest.TestCase):
    def test_metric_boxes_and_supported_decoding(self):
        line = box_outline([10, 20, 0], [4, 2, 1], 0)
        np.testing.assert_allclose(line[0], [8, 19])
        np.testing.assert_allclose(line[-1], line[0])
        rotated = box_outline([0, 0, 0], [4, 2, 1], np.pi / 2)
        np.testing.assert_allclose(rotated[0], [1, -2], atol=1e-6)
        output = {"agent_cls_logits": torch.tensor([[[10., -10, -10, -10, -10],
                                                     [-10., -10, -10, -10, 10.]]]),
                  "agent_boxes": torch.tensor([[[.5, .5, .5, .2, .2, .2, 0, 1]] * 2]),
                  "agent_velocity": torch.zeros(1, 2, 3),
                  "map_cls_logits": torch.tensor([[[10., -10, -10, -10], [-10., -10, -10, 10.]]]),
                  "map_points": torch.full((1, 2, 20, 2), .5)}
        base = SimpleNamespace(geometry_lift=SimpleNamespace(x_range=(-50, 50), y_range=(-50, 50)))
        meta = {"training_config": {"temporal_agent": {"recipe": {
            "agent_loss": {"z_range": [-5, 5], "size_norm": [20, 10, 8]}}}}}
        agent, vectors, extent = decode_outputs(output, base, meta, .25, .25)
        self.assertEqual(agent["labels"].tolist(), [0])
        self.assertEqual(vectors["labels"].tolist(), [0])
        np.testing.assert_allclose(agent["centers"], [[0, 0, 0]])
        np.testing.assert_allclose(vectors["points_xy_m"], 0)
        self.assertEqual(extent, (-50, -50, 50, 50))
        output.update(seg_logits=torch.zeros(1, 2, 19, 14, 28), depth=torch.full((1, 2, 1, 14, 28), 10.))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "four_tasks.png"
            render(path, torch.zeros(2, 3, 14, 28), output, ["CAM_F0", "CAM_B0"],
                   agent, vectors, extent, "Synthetic four-task fixture")
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"))
                self.assertGreater(image.width, 500)
                self.assertGreater(image.height, 500)
                self.assertGreater(float(pixels.std()), 10)


if __name__ == "__main__":
    unittest.main()
