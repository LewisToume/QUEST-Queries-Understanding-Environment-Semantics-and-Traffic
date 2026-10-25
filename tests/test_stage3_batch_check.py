import unittest

import torch
from torch import nn

from quest.map_training import STAGE3_MODULES
from scripts.check_stage3_batch import check_parameter_updates, select_training_indices, snapshot_trainable


def tiny_model():
    model = nn.Module()
    for name in STAGE3_MODULES:
        layer = nn.Linear(2, 2)
        layer.bias.requires_grad_(False)
        setattr(model, name, layer)
    return model


class Stage3BatchCheckTest(unittest.TestCase):
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
        check_parameter_updates(model, before)
        before["map_head"]["weight"] = model.map_head.weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "did not change any parameter in map_head"):
            check_parameter_updates(model, before)

    def test_nonfinite_parameter_cannot_pass_update_check(self):
        model = tiny_model()
        before = snapshot_trainable(model)
        with torch.no_grad():
            model.geometry_lift.weight.fill_(float("nan"))
        with self.assertRaisesRegex(RuntimeError, "non-finite parameter"):
            check_parameter_updates(model, before)


if __name__ == "__main__":
    unittest.main()
