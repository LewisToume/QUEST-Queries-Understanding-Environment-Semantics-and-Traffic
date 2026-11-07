"""CPU GradNorm math, real shared-gradient and independent-export fixtures."""

import copy
import unittest

import torch

from quest.shared_feature_distillation import deployment_payload, load_deployment
from quest.unified_gradnorm import GRADNORM_STAGE, TaskGradNorm
from quest.unified_scene import UnifiedQUEST
from quest.unified_training import TASKS, four_task_losses
from test_geometry_depth_v2 import TinyBackbone, inputs
from test_temporal_depth_v3 import history
from test_unified_scene import fixtures, targets


class GradNormTest(unittest.TestCase):
    def test_targets_detached_weights_not_trained_by_task_sum_and_normalized_update(self):
        x = torch.nn.Parameter(torch.tensor(1.))
        losses = dict(zip(TASKS, (10*x, .1*x, x, x)))
        balancer = TaskGradNorm({k: float(v.detach()) for k,v in losses.items()}, alpha=1.5)
        weighted = balancer.weighted_losses(losses)
        sum(weighted.values()).backward(retain_graph=True)
        self.assertIsNone(balancer.weights.grad)
        optimizer = torch.optim.SGD(balancer.parameters(), lr=.01)
        stats = balancer.prepare_update(losses, x)
        torch.testing.assert_close(torch.tensor(list(stats["relative_training_rates"].values())), torch.ones(4))
        self.assertGreater(float(balancer.weights.grad[0]), 0.)
        self.assertLess(float(balancer.weights.grad[1]), 0.)
        optimizer.step()
        balancer.renormalize()
        self.assertLess(float(balancer.weights[0]), 1.)
        self.assertGreater(float(balancer.weights[1]), 1.)
        self.assertAlmostEqual(float(balancer.weights.sum()), 4., places=5)
        self.assertTrue(bool((balancer.weights > 0).all()))

    def test_invalid_and_zero_gradient_reference_rejected(self):
        with self.assertRaises(ValueError):
            TaskGradNorm(dict.fromkeys(TASKS, 0.))
        with self.assertRaises(ValueError):
            TaskGradNorm(dict.fromkeys(TASKS, 1.), alpha=float("nan"))
        balancer = TaskGradNorm(dict.fromkeys(TASKS, 1.))
        x = torch.nn.Parameter(torch.tensor(1.))
        with self.assertRaises(RuntimeError):
            balancer.prepare_update(dict.fromkeys(TASKS, x * 0), x)

    def test_real_four_task_gradients_weight_step_and_state_roundtrip(self):
        torch.manual_seed(11)
        config, profile = fixtures()
        model = UnifiedQUEST(config, profile, backbone=TinyBackbone()).train()
        data = inputs()
        data["history"] = history(data)
        batch, settings = targets(data)
        losses, _ = four_task_losses(model, model(**data), batch, settings, torch.device("cpu"))
        balancer = TaskGradNorm({k: max(float(v.detach()), 1e-6) for k,v in losses.items()})
        weight_optimizer = torch.optim.Adam(balancer.parameters(), lr=1e-3)
        stats = balancer.prepare_update(losses, model.trunk.pyramid_fusion.fuse_refine[0].layers[-1].weight)
        self.assertTrue(all(value > 0 for value in stats["base_gradient_norms"].values()))
        sum(balancer.weighted_losses(losses).values()).backward()
        self.assertGreater(float(model.heads.seg.weight.grad.norm()), 0.)
        weight_optimizer.step()
        balancer.renormalize()
        restored = TaskGradNorm(dict.fromkeys(TASKS, 1.))
        restored.load_state_dict(copy.deepcopy(balancer.state_dict()), strict=True)
        torch.testing.assert_close(restored.weights, balancer.weights, rtol=0, atol=0)
        torch.testing.assert_close(restored.reference_losses, balancer.reference_losses, rtol=0, atol=0)

    def test_gradnorm_not_in_student_export_or_required_at_inference(self):
        config, profile = fixtures()
        model = UnifiedQUEST(config, profile, backbone=TinyBackbone()).eval()
        parent = dict(trained_tasks=list(TASKS), epoch=12)
        exported = deployment_payload(model, parent, 1, {}, {"all_pass": False}, experiment=GRADNORM_STAGE)
        restored = load_deployment(exported, backbone=TinyBackbone()).eval()
        self.assertFalse(any("gradnorm" in k for k in exported["model_state_dict"]))
        self.assertEqual(exported["total_student_epochs"], 13)
        data = inputs()
        data["history"] = history(data)
        with torch.no_grad():
            before, after = model(**data), restored(**data)
            for key in before:
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
