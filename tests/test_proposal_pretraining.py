import tempfile
import unittest
from pathlib import Path

import torch

from quest.proposal_pretraining import (
    compute_proposal_pretraining_loss,
    configure_proposal_pretraining,
    load_proposal_pretrain_checkpoint,
    proposal_center_hits,
    rasterize_proposal_targets,
    save_proposal_pretrain_checkpoint,
)


class TinyProposalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.geometry_lift = torch.nn.Linear(2, 2)
        self.bev_encoder = torch.nn.Linear(2, 2)
        self.agent_proposal_head = torch.nn.Linear(2, 2)
        self.agent_decoder = torch.nn.Linear(2, 2)


class ProposalPretrainingTest(unittest.TestCase):
    def test_collision_uses_highest_score_for_offset_but_binary_objectness(self):
        target = {
            "labels": torch.tensor([[0, 1, 2]]),
            "boxes_metric": torch.tensor([[
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [0.5, 0.5, 0.0, 1.0, 1.0, 1.0, 0.0],
                [-5.0, -5.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            ]]),
            "scores": torch.tensor([[0.25, 0.75, 0.99]]),
            "valid_mask": torch.ones(1, 3, dtype=torch.bool),
            "class_support_mask": torch.tensor([[True, True, False, False]]),
        }
        result = rasterize_proposal_targets(target, (-10, 10), (-10, 10), 4, 4)
        flat_index = 2 * 4 + 2
        self.assertEqual(result["positive_cell_count"], 1)
        self.assertEqual(result["collision_count"], 1)
        self.assertEqual(float(result["objectness_target"][0, flat_index]), 1.0)
        self.assertAlmostEqual(float(result["positive_weight"][0, flat_index]), 0.75)
        torch.testing.assert_close(
            result["offset_target"][0, flat_index], torch.tensor([-0.4, -0.4])
        )
        self.assertEqual(len(result["target_centers_metric"][0]), 2)

    def test_separate_weighted_losses_and_empty_positive(self):
        target = {
            "objectness_target": torch.tensor([[1.0, 0.0]]),
            "offset_target": torch.tensor([[[0.25, -0.25], [0.0, 0.0]]]),
            "positive_mask": torch.tensor([[True, False]]),
            "positive_weight": torch.tensor([[0.8, 0.0]]),
        }
        logits = torch.zeros(1, 2, requires_grad=True)
        offsets = torch.zeros(1, 2, 2, requires_grad=True)
        losses = compute_proposal_pretraining_loss(logits, offsets, target)
        torch.testing.assert_close(
            losses["objectness_loss"],
            losses["positive_objectness_loss"]
            + 3.0 * losses["negative_objectness_loss"],
        )
        torch.testing.assert_close(
            losses["proposal_loss"],
            2.0 * losses["objectness_loss"] + losses["offset_loss"],
        )
        empty = dict(target)
        empty["objectness_target"] = torch.zeros(1, 2)
        empty["positive_mask"] = torch.zeros(1, 2, dtype=torch.bool)
        empty_losses = compute_proposal_pretraining_loss(logits, offsets, empty)
        self.assertEqual(float(empty_losses["positive_objectness_loss"]), 0.0)
        self.assertEqual(float(empty_losses["offset_loss"]), 0.0)

    def test_metric_recall_uses_predicted_offset(self):
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
        offsets = torch.zeros(1, 4, 2)
        offsets[0, 0] = torch.tensor([0.4, 0.0])
        centers = [torch.tensor([[-0.1, -0.5]])]
        self.assertEqual(
            proposal_center_hits(logits, offsets, centers, 1, 0.15, (-1, 1), (-1, 1), 2, 2),
            (1, 1),
        )
        offsets[0, 0] = 0.0
        self.assertEqual(
            proposal_center_hits(logits, offsets, centers, 1, 0.15, (-1, 1), (-1, 1), 2, 2),
            (0, 1),
        )

    def test_only_proposal_is_trainable_and_checkpoint_restores_components(self):
        model = TinyProposalModel()
        trainable = configure_proposal_pretraining(model)
        self.assertTrue(all(parameter.requires_grad for parameter in trainable))
        self.assertEqual(
            {name.split(".", 1)[0] for name, parameter in model.named_parameters()
             if parameter.requires_grad},
            {"agent_proposal_head"},
        )
        optimizer = torch.optim.AdamW(trainable)
        expected = model.agent_proposal_head.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposal.pt"
            save_proposal_pretrain_checkpoint(path, model, optimizer, 2, {"train": {}})
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                checkpoint = torch.load(path, map_location="cpu")
            with torch.no_grad():
                model.agent_proposal_head.weight.zero_()
            self.assertEqual(load_proposal_pretrain_checkpoint(model, checkpoint), 2)
        torch.testing.assert_close(model.agent_proposal_head.weight, expected)
        self.assertEqual(checkpoint["stage"], "proposal_pretrain")
        self.assertFalse(checkpoint["bev_use_ego_state"])


if __name__ == "__main__":
    unittest.main()
