import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from quest.agent_training import load_checkpoint_cpu
from quest.map_diagnostics import evenly_spaced_indices, oracle_check
from quest.map_training import (
    STAGE3_MODULES, class_aware_vector_matches, configure_stage3, denormalize_points,
    frozen_agent_fingerprint, match_vector_queries, normalize_points,
    pairwise_vector_distances, stage3_trainable_modules, vector_map_loss,
    mixed_stage3_loss, STAGE3_GT_ONLY_NAME,
    load_stage3_checkpoint, save_stage3_checkpoint,
)
from quest.nuplan_map_locator import MAP_LAYER_AUDIT_VERSION
from quest.nuplan_relation_audit import BASELINE_RELATION_AUDIT_VERSION
from quest.queries import MapDecoder
from quest.vector_map_labels import MAP_HEIGHT_REFERENCE, VECTOR_SEMANTICS_VERSION


XY_RANGE = (-50, -50, 50, 50)


def line_target():
    points = torch.stack((torch.linspace(-10, 10, 20), torch.zeros(20)), dim=-1)[None]
    return {"class_ids": torch.tensor([0]), "points_xy_m": points,
            "is_closed": torch.tensor([False]), "length_m": torch.tensor([20.0])}


def loss_settings():
    return {"map_cls_cost_weight": 2.0, "map_point_cost_weight": 5.0,
            "map_cls_loss_weight": 2.0, "map_point_loss_weight": 5.0,
            "map_direction_loss_weight": 0.005, "background_weight": 0.5,
            "point_cost_reduction": "sum", "point_loss_type": "l1_sum"}


def tiny_model():
    model = nn.Module()
    for name in STAGE3_MODULES:
        setattr(model, name, nn.Linear(2, 2))
    model.geometry_lift.ego_mlp = nn.Linear(2, 2)
    return model


class Stage3GeometryObjectiveTest(unittest.TestCase):
    def test_sum_cost_can_reject_confident_wrong_geometry(self):
        target = line_target()
        normalized = normalize_points(target["points_xy_m"], XY_RANGE)[0]
        predictions = torch.stack((normalized, normalized + torch.tensor([0.1, 0.0])))
        logits = torch.tensor([[0., 0., 0., 0.], [9., 0., 0., 0.]])
        legacy = match_vector_queries(logits, predictions, target, XY_RANGE, 2., 5., "mean")
        corrected = match_vector_queries(logits, predictions, target, XY_RANGE, 2., 5., "sum")
        self.assertEqual(legacy[0][0], 1)
        self.assertEqual(corrected[0][0], 0)

    def test_l1_sums_points_per_instance_and_reports_meters(self):
        target = line_target()
        points = normalize_points(target["points_xy_m"], XY_RANGE) + torch.tensor([0.1, 0.])
        points = points.unsqueeze(0).requires_grad_()
        logits = torch.tensor([[[9., 0., 0., 0.]]], requires_grad=True)
        loss = vector_map_loss(logits, points, [target], loss_settings(), XY_RANGE)
        self.assertAlmostEqual(float(loss["map_point_loss"]), 2.0, places=5)
        self.assertAlmostEqual(float(loss["map_matched_point_error_m"]), 10.0, places=4)
        loss["vector_map_gt_loss"].backward()
        self.assertGreater(float(points.grad.abs().sum()), 0)

    def test_oracle_and_reverse_equivalence(self):
        target = line_target()
        oracle_check(target, XY_RANGE)
        reversed_line = target["points_xy_m"].flip(1)
        matches = class_aware_vector_matches(torch.tensor([0]), reversed_line, target, 0.01)
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(class_aware_vector_matches(torch.tensor([1]), reversed_line, target, 1.5)), 0)

    def test_chamfer_does_not_silently_replace_ordered_metric(self):
        target = line_target()
        order = torch.tensor(list(range(0, 20, 2)) + list(range(1, 20, 2)))
        scrambled = target["points_xy_m"][:, order]
        self.assertGreater(float(pairwise_vector_distances(scrambled, target, "ordered")[0, 0]), 1)
        self.assertAlmostEqual(float(pairwise_vector_distances(scrambled, target, "chamfer")[0, 0]), 0, places=5)

    def test_closed_cyclic_reverse_oracle(self):
        angles = torch.arange(20) * (2 * torch.pi / 20)
        points = torch.stack((angles.cos(), angles.sin()), -1)[None]
        target = {"points_xy_m": points, "class_ids": torch.tensor([1]), "is_closed": torch.tensor([True])}
        oracle_check(target, XY_RANGE)
        transformed = points.flip(1).roll(7, 1)
        self.assertEqual(len(class_aware_vector_matches(torch.tensor([1]), transformed, target, 0.01)), 1)

    def test_asymmetric_normalization_round_trip_and_original_indices(self):
        points = torch.tensor([[[-20., -3.], [35., 12.]]])
        bounds = (-30., -10., 50., 20.)
        self.assertTrue(torch.allclose(denormalize_points(normalize_points(points, bounds), bounds), points))
        frames = [{"index": value} for value in (17, 21, 44, 92, 105)]
        self.assertEqual(evenly_spaced_indices(frames, 3), [17, 44, 105])

    def test_intermediate_map_outputs_preserve_old_final_forward_and_state(self):
        decoder = MapDecoder(hidden_dim=8, num_queries=3, num_layers=2, num_heads=2,
                             ffn_dim=16, dropout=0).eval()
        memory = torch.randn(1, 16, 8)
        queries = decoder.query_embedding[None]
        expected = decoder.norm(decoder.decoder(tgt=queries, memory=memory))
        self.assertTrue(torch.allclose(decoder(memory), expected, atol=1e-6))
        self.assertEqual(len(decoder.forward_layers(memory)), 2)
        copy = MapDecoder(hidden_dim=8, num_queries=3, num_layers=2, num_heads=2, ffn_dim=16, dropout=0)
        copy.load_state_dict(decoder.state_dict(), strict=True)

    def test_auxiliary_map_loss_reaches_both_layers(self):
        layers = torch.tensor([[[[1., 1., 1., 1.]]], [[[2., 2., 2., 2.]]]], requires_grad=True)
        points = torch.zeros(2, 1, 1, 20, 2)
        predictions = {"map_cls_logits_layers": layers, "map_points_layers": points,
                       "map_cls_logits": layers[-1], "map_points": points[-1]}
        model = SimpleNamespace(geometry_lift=SimpleNamespace(x_range=(-50, 50), y_range=(-50, 50)))
        settings = {"stage": STAGE3_GT_ONLY_NAME, "train": {"freeze_agent_and_bev": True},
                    "loss_weights": {"lambda_map_gt": 1., "lambda_map_kd": 0., "lambda_agent": 0.},
                    "map_loss": {"aux_layer_weights": [0.5, 1.]}, "agent_train": {}, "agent_loss": {}}
        with patch("quest.map_training.vector_map_loss", side_effect=lambda logits, *args: {
            "vector_map_gt_loss": logits.mean(),
        }), patch("quest.map_training.compute_stage2_agent_loss", return_value={"total_agent_loss": torch.tensor(0.)}):
            losses = mixed_stage3_loss(model, predictions, [], None, None, None, None, {}, settings)
        self.assertAlmostEqual(float(losses["total_loss"]), 2.5)
        losses["total_loss"].backward()
        self.assertGreater(float(layers.grad[0].abs().sum()), 0)
        self.assertGreater(float(layers.grad[1].abs().sum()), 0)

    def test_frozen_agent_and_bev_do_not_change_during_map_update(self):
        model = tiny_model()
        settings = {"freeze_agent_and_bev": True, "map_lr": 1e-3, "weight_decay": 1e-4}
        optimizer = configure_stage3(model, None, settings)
        before = frozen_agent_fingerprint(model)
        self.assertEqual(stage3_trainable_modules(settings), ("map_decoder", "map_head"))
        self.assertEqual(len(optimizer.param_groups), 1)
        model.map_head(model.map_decoder(torch.ones(1, 2))).sum().backward()
        optimizer.step()
        self.assertEqual(frozen_agent_fingerprint(model), before)
        self.assertTrue(all(p.grad is None for p in model.agent_head.parameters()))
        with torch.no_grad():
            model.agent_head.weight.add_(1)
        self.assertNotEqual(frozen_agent_fingerprint(model), before)

    def test_recovery_checkpoint_round_trip_and_tamper_detection(self):
        model = tiny_model()
        settings = {"stage": STAGE3_GT_ONLY_NAME,
                    "loss_weights": {"lambda_map_gt": 1., "lambda_map_kd": 0., "lambda_agent": 0.},
                    "train": {"freeze_agent_and_bev": True, "map_lr": 1e-3, "weight_decay": 0.},
                    "map": {**loss_settings(), "objective_version": "ordered_l1_sum_v2"}}
        optimizer = configure_stage3(model, None, settings["train"])
        settings["agent_baseline_sha256"] = frozen_agent_fingerprint(model)
        provenance = {"num_points": 20, "min_length_m": 1., "map_version": "test-map",
                      "vector_semantics_version": VECTOR_SEMANTICS_VERSION,
                      "map_height_reference": MAP_HEIGHT_REFERENCE,
                      "map_cast_audit_version": BASELINE_RELATION_AUDIT_VERSION,
                      "map_layer_audit_version": MAP_LAYER_AUDIT_VERSION}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recovery.pt"
            save_stage3_checkpoint(path, model, None, optimizer, 1, settings, None, provenance)
            checkpoint = load_checkpoint_cpu(path)
        self.assertTrue(checkpoint["agent_preservation_verified"])
        self.assertEqual(load_stage3_checkpoint(tiny_model(), None, checkpoint), 1)
        checkpoint["agent_head_state_dict"]["weight"] = checkpoint["agent_head_state_dict"]["weight"] + 1
        with self.assertRaisesRegex(ValueError, "preservation checksum mismatch"):
            load_stage3_checkpoint(tiny_model(), None, checkpoint)


if __name__ == "__main__":
    unittest.main()
