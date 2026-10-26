import unittest
from unittest.mock import patch

import torch
from torch import nn

from quest.map_training import STAGE3_MODULES
from scripts.check_stage3_batch import (
    check_parameter_updates, evaluate_fit_map, overfit_lr_factor,
    select_training_indices, set_check_mode, snapshot_trainable,
)


def tiny_model():
    model = nn.Module()
    for name in STAGE3_MODULES:
        layer = nn.Linear(2, 2)
        layer.bias.requires_grad_(False)
        setattr(model, name, layer)
    return model


class Stage3BatchCheckTest(unittest.TestCase):
    def test_overfit_eval_mode_disables_dropout_without_disabling_gradients(self):
        model = tiny_model()
        model.map_decoder = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.9))
        set_check_mode(model, ("map_decoder", "map_head"), overfit=True)
        self.assertFalse(model.map_decoder[1].training)
        self.assertFalse(model.map_head.training)
        self.assertTrue(model.map_decoder[0].weight.requires_grad)
        model.map_decoder(torch.ones(1, 2)).sum().backward()
        self.assertIsNotNone(model.map_decoder[0].weight.grad)
        set_check_mode(model, ("map_decoder", "map_head"), overfit=False)
        self.assertTrue(model.map_decoder[1].training)
        self.assertFalse(model.geometry_lift.training)

    def test_overfit_cosine_lr_decays_to_explicit_minimum(self):
        factors = [overfit_lr_factor(step, 100, 0.05) for step in (0, 10, 50, 100)]
        self.assertAlmostEqual(factors[0], 1.)
        self.assertAlmostEqual(factors[-1], 0.05)
        self.assertTrue(all(a >= b for a, b in zip(factors, factors[1:])))
        with self.assertRaises(ValueError):
            overfit_lr_factor(0, 0, 0.05)

    def test_fit_eval_is_strict_class_aware_and_preserves_modes_and_gradients(self):
        model = tiny_model()
        model.geometry_lift.x_range = model.geometry_lift.y_range = (-50., 50.)
        model.map_decoder.eval()
        modes = [module.training for module in model.modules()]
        model.map_head.weight.grad = torch.ones_like(model.map_head.weight)
        gradient = model.map_head.weight.grad.clone()
        line = torch.stack((torch.linspace(-10, 10, 20), torch.zeros(20)), -1)
        target_points = torch.stack((line, line + torch.tensor([0., 10.])))
        target = {"class_ids": torch.tensor([0, 1]), "points_xy_m": target_points,
                  "is_closed": torch.tensor([False, False])}
        predictions = {"map_cls_logits": torch.tensor([[[8., 0., 0., -8.], [8., 0., 0., -8.]]]),
                       "map_points": ((target_points + 50) / 100)[None]}
        loss = {"map_cls_cost_weight": 2., "map_point_cost_weight": 5.,
                "map_cls_loss_weight": 2., "map_point_loss_weight": 5.,
                "map_direction_loss_weight": 0.005, "background_weight": 0.5,
                "point_cost_reduction": "sum", "point_loss_type": "l1_sum"}
        config = {"map": loss, "eval": {"point_error_thresholds_m": [0.5, 1., 1.5],
                                         "confidence_threshold": 0.25}}
        with patch("scripts.check_stage3_batch.stage3_forward", return_value=predictions) as forward:
            result = evaluate_fit_map(model, [{"vector_targets": [target]}], torch.device("cpu"), config)
        self.assertFalse(forward.call_args.kwargs["include_dense"])
        self.assertEqual(result["classes"]["centerline"]["matched"], 1)
        self.assertEqual(result["classes"]["ped_crossing"]["gt"], 1)
        self.assertEqual(result["classes"]["ped_crossing"]["matched"], 0)
        self.assertAlmostEqual(result["point_error_m"], 0., places=4)
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertTrue(torch.equal(gradient, model.map_head.weight.grad))

    def test_steps_follow_manifest_indices_without_reindexing(self):
        self.assertEqual(select_training_indices([0, 5, 11, 20], 5, 2), [5, 11])
        self.assertEqual(select_training_indices([0, 5, 11], None, 2), [0, 5])

    def test_invalid_step_ranges_never_wrap_or_use_validation(self):
        for start, steps in ((None, 0), (None, -1), (5, 3), (99, 1)):
            with self.subTest(start=start, steps=steps):
                with self.assertRaises(ValueError):
                    select_training_indices([0, 5, 11], start, steps)

    def test_snapshot_is_independent_and_excludes_frozen_parameters(self):
        model = tiny_model()
        before = snapshot_trainable(model)
        for name in STAGE3_MODULES:
            self.assertEqual(set(before[name]), {"weight"})
            self.assertEqual(before[name]["weight"].device.type, "cpu")
        original = before["map_head"]["weight"].clone()
        with torch.no_grad():
            model.map_head.weight.add_(1)
        self.assertTrue(torch.equal(before["map_head"]["weight"], original))

    def test_update_check_requires_every_module_to_change(self):
        model = tiny_model()
        before = snapshot_trainable(model)
        with torch.no_grad():
            for name in STAGE3_MODULES:
                getattr(model, name).weight.add_(1)
        check_parameter_updates(model, before, verbose=False)
        before["map_head"]["weight"] = model.map_head.weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "did not change any parameter in map_head"):
            check_parameter_updates(model, before, verbose=False)

    def test_nonfinite_parameter_cannot_pass_update_check(self):
        model = tiny_model()
        before = snapshot_trainable(model)
        with torch.no_grad():
            model.geometry_lift.weight.fill_(float("nan"))
        with self.assertRaisesRegex(RuntimeError, "non-finite parameter"):
            check_parameter_updates(model, before, verbose=False)


if __name__ == "__main__":
    unittest.main()
