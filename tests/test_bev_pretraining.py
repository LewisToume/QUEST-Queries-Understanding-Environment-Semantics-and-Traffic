import tempfile
import unittest
from pathlib import Path

import torch

from quest.bev_pretraining import (
    BEVAuxiliaryHead,
    compute_bev_pretraining_loss,
    configure_bev_pretraining,
    load_bev_pretrain_checkpoint,
    rasterize_agent_centers,
    save_bev_pretrain_checkpoint,
    topk_center_hits,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.geometry_lift = torch.nn.Linear(2, 2)
        self.bev_encoder = torch.nn.Linear(2, 2)
        self.agent_head = torch.nn.Linear(2, 2)


class BEVPretrainingTest(unittest.TestCase):
    def test_auxiliary_head_shapes(self):
        head = BEVAuxiliaryHead(hidden_dim=8, intermediate_dim=4)
        output = head(torch.randn(2, 8, 3, 5))
        self.assertEqual(tuple(output["foreground_logits"].shape), (2, 3, 5))
        self.assertEqual(tuple(output["class_logits"].shape), (2, 3, 5, 2))

    def test_rasterization_keeps_highest_score_on_collision(self):
        target = {
            "labels": torch.tensor([[0, 1, 2, 0]]),
            "boxes_metric": torch.tensor(
                [[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                  [0.1, 0.1, 0.0, 1.0, 1.0, 1.0, 0.0],
                  [5.0, 5.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                  [-9.0, -9.0, 0.0, 1.0, 1.0, 1.0, 0.0]]]
            ),
            "scores": torch.tensor([[0.4, 0.9, 1.0, 0.8]]),
            "class_support_mask": torch.tensor([[True, True, False, False]]),
            "valid_mask": torch.ones(1, 4, dtype=torch.bool),
        }
        raster = rasterize_agent_centers(target, (-10, 10), (-10, 10), 4, 4)
        self.assertEqual(int(raster["occupied_bev_cells"]), 2)
        self.assertEqual(int(raster["target_collision_count"]), 1)
        self.assertEqual(int(raster["class_target"][0, 2, 2]), 1)
        self.assertEqual(int(raster["vehicle_target_count"]), 2)
        self.assertEqual(int(raster["pedestrian_target_count"]), 1)

    def test_loss_separates_positive_negative_and_handles_empty_positive(self):
        predictions = {
            "foreground_logits": torch.zeros(1, 2, 2),
            "class_logits": torch.zeros(1, 2, 2, 2),
        }
        target = {
            "foreground_target": torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
            "class_target": torch.tensor([[[0, -1], [-1, -1]]]),
            "positive_mask": torch.tensor([[[True, False], [False, False]]]),
        }
        losses = compute_bev_pretraining_loss(predictions, target)
        self.assertTrue(torch.isfinite(losses["total_loss"]))
        self.assertAlmostEqual(float(losses["positive_loss"]), 0.693147, places=5)
        self.assertAlmostEqual(float(losses["negative_loss"]), 0.693147, places=5)
        self.assertAlmostEqual(
            float(losses["foreground_loss"]),
            float(losses["positive_loss"] + 3.0 * losses["negative_loss"]),
            places=5,
        )
        unweighted = compute_bev_pretraining_loss(
            predictions, target, negative_loss_weight=1.0
        )
        self.assertAlmostEqual(
            float(unweighted["foreground_loss"]),
            float(unweighted["positive_loss"] + unweighted["negative_loss"]),
            places=5,
        )
        empty = dict(target)
        empty["foreground_target"] = torch.zeros(1, 2, 2)
        empty["class_target"] = torch.full((1, 2, 2), -1)
        empty["positive_mask"] = torch.zeros(1, 2, 2, dtype=torch.bool)
        empty_losses = compute_bev_pretraining_loss(predictions, empty)
        self.assertTrue(torch.isfinite(empty_losses["total_loss"]))
        self.assertEqual(float(empty_losses["positive_loss"]), 0.0)
        self.assertEqual(float(empty_losses["class_loss"]), 0.0)

    def test_topk_recall_accepts_neighbor_cell(self):
        logits = torch.zeros(1, 4, 4)
        logits[0, 1, 2] = 10.0
        target = torch.zeros(1, 4, 4, dtype=torch.bool)
        target[0, 2, 2] = True
        self.assertEqual(topk_center_hits(logits, target, 1, 1), (1, 1))
        self.assertEqual(topk_center_hits(logits, target, 1, 0), (0, 1))

    def test_parameter_scope_and_checkpoint_component_restore(self):
        model = TinyModel()
        head = BEVAuxiliaryHead(hidden_dim=2, intermediate_dim=2)
        trainable = configure_bev_pretraining(model, head)
        self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.agent_head.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.geometry_lift.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.bev_encoder.parameters()))
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        expected = model.geometry_lift.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bev.pt"
            save_bev_pretrain_checkpoint(path, model, head, optimizer, 3, {"test": True})
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                checkpoint = torch.load(path, map_location="cpu")
            with torch.no_grad():
                model.geometry_lift.weight.zero_()
            epoch = load_bev_pretrain_checkpoint(model, checkpoint, head)
        self.assertEqual(epoch, 3)
        torch.testing.assert_close(model.geometry_lift.weight, expected)
        self.assertEqual(checkpoint["stage"], "bev_pretrain")


if __name__ == "__main__":
    unittest.main()
