import tempfile
import unittest
from pathlib import Path

import torch

from quest.agent_training import (
    class_aware_center_matches,
    configure_agent_training,
    decode_supported_agent_predictions,
    load_agent_stage2_checkpoint,
    prepare_navformer_agent_batch,
    save_agent_stage2_checkpoint,
)
from quest.losses import compute_agent_decoder_loss


class TinyAgentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.geometry_lift = torch.nn.Module()
        self.geometry_lift.ego_mlp = torch.nn.Linear(2, 2)
        self.geometry_lift.weight_layer = torch.nn.Linear(2, 2)
        self.bev_encoder = torch.nn.Linear(2, 2)
        self.agent_proposal_head = torch.nn.Linear(2, 2)
        self.agent_decoder = torch.nn.Linear(2, 2)
        self.agent_head = torch.nn.Linear(2, 2)
        self.map_decoder = torch.nn.Linear(2, 2)
        self.map_head = torch.nn.Linear(2, 2)
        self.seg_head = torch.nn.Linear(2, 2)
        self.depth_head = torch.nn.Linear(2, 2)


class Stage2AgentTrainingTest(unittest.TestCase):
    def test_optimizer_scope_and_checkpoint_metadata(self):
        model = TinyAgentModel()
        bev, agent = configure_agent_training(model)
        self.assertTrue(bev and agent)
        trainable = {
            name.split(".", 1)[0]
            for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertEqual(trainable, {
            "geometry_lift", "bev_encoder", "agent_proposal_head",
            "agent_decoder", "agent_head",
        })
        self.assertFalse(any(p.requires_grad for p in model.geometry_lift.ego_mlp.parameters()))
        optimizer = torch.optim.AdamW([
            {"params": bev, "lr": 2e-5}, {"params": agent, "lr": 1e-4},
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.pt"
            save_agent_stage2_checkpoint(path, model, optimizer, 2, {"train": {}})
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                checkpoint = torch.load(path, map_location="cpu")
            self.assertEqual(load_agent_stage2_checkpoint(model, checkpoint), 2)
            self.assertEqual(checkpoint["stage"], "agent_end_to_end")
            self.assertFalse(checkpoint["bev_use_ego_state"])
            self.assertEqual(checkpoint["trained_class_support_mask"].tolist(),
                             [True, True, False, False])
            checkpoint["bev_use_ego_state"] = True
            with self.assertRaises(ValueError):
                load_agent_stage2_checkpoint(model, checkpoint)

    def test_teacher_batch_ignores_unsupported_classes(self):
        agent = {
            "labels": torch.tensor([[0, 1, 2]]),
            "boxes_metric": torch.zeros(1, 3, 7),
            "velocity_mps": torch.zeros(1, 3, 3),
            "scores": torch.ones(1, 3),
            "class_support_mask": torch.tensor([[True, True, False, False]]),
            "valid_mask": torch.ones(1, 3, dtype=torch.bool),
        }
        result = prepare_navformer_agent_batch(
            {"soft_labels": {"agent": agent}, "sample_token": ["example"]},
            torch.device("cpu"),
        )
        self.assertEqual(result["valid_mask"].tolist(), [[True, True, False]])

    def test_unsupported_prediction_is_masked_and_matching_is_class_aware(self):
        prediction = {
            "agent_cls_logits": torch.tensor([[0.0, 0.0, 30.0, 0.0, -10.0]]),
            "agent_boxes": torch.tensor([[0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.0, 1.0]]),
            "agent_velocity": torch.zeros(1, 3),
        }
        decoded = decode_supported_agent_predictions(
            prediction, torch.tensor([True, True, False, False]), 0.25,
            (-50, 50), (-50, 50), (-5, 5), (20, 10, 8),
        )
        self.assertEqual(decoded["labels"].tolist(), [0])
        self.assertEqual(class_aware_center_matches(
            torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([0]),
            torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([1]), 2.0,
        ), [])

    def test_teacher_score_weights_matched_regression(self):
        cls = torch.zeros(1, 2, 5)
        boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.0, 1.0],
                               [0.8, 0.5, 0.5, 0.4, 0.4, 0.4, 1.0, 0.0]]])
        velocity = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        predictions = {
            "agent_cls_logits_layers": cls.unsqueeze(0),
            "agent_boxes_layers": boxes.unsqueeze(0),
            "agent_velocity_layers": velocity.unsqueeze(0),
        }
        target = {
            "labels": torch.tensor([[0, 0]]),
            "boxes_metric": torch.tensor([[[0.0, 0.0, 0.0, 4.0, 2.0, 1.6, 0.0],
                                            [20.0, 0.0, 0.0, 4.0, 2.0, 1.6, 0.0]]]),
            "velocity_mps": torch.zeros(1, 2, 3),
            "scores": torch.tensor([[1.0, 0.1]]),
            "valid_mask": torch.ones(1, 2, dtype=torch.bool),
            "class_support_mask": torch.tensor([[True, True, False, False]]),
        }
        settings = {
            "xy_range": (-50.0, 50.0), "z_range": (-5.0, 5.0),
            "size_norm": (20.0, 10.0, 8.0),
            "match_cls_weight": 1.0, "match_center_weight": 5.0,
            "match_size_weight": 2.0, "match_yaw_weight": 1.0,
            "background_weight": 0.5,
            "lambda_cls": 1.0, "lambda_center": 5.0, "lambda_size": 2.0,
            "lambda_yaw": 1.0, "lambda_velocity": 0.5,
            "aux_layer_weights": (1.0,),
        }
        unweighted = compute_agent_decoder_loss(predictions, target, settings)
        weighted = compute_agent_decoder_loss(
            predictions, target, {**settings, "teacher_confidence_weighting": True}
        )
        for key in ("agent_center_loss", "agent_size_loss", "agent_yaw_loss",
                    "agent_velocity_loss"):
            self.assertLess(float(weighted[key]), float(unweighted[key]))


if __name__ == "__main__":
    unittest.main()
