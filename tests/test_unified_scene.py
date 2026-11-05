"""Server CPU fixtures for shared four-task paths; no assets, CUDA or downloads."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from quest.geometry import GeometryAwareBEVLift
from quest.geometry_depth import DPTMetricDecoder
from quest.model import DEFAULT_CAMERA_NAMES, QUESTModel
from quest.seg_on_v3 import full_base_fingerprint
from quest.teacher_adapters import NAVFORMER_CLASS_SUPPORT, empty_canonical_agent
from quest.unified_scene import UNIFIED_SCENE_VERSION, UnifiedQUEST
from quest.unified_training import (four_task_losses, joint_backward, pcgrad_vectors,
                                    quality_gate, scaled_task_losses, shared_gradient_audit)
from quest.utils import load_yaml_config, project_root
from test_geometry_depth_v2 import TinyBackbone, inputs, model_config
from test_stage3_spatial_map import architecture, line_target, loss_settings
from test_temporal_depth_v3 import history, temporal_options


def fixtures():
    options = temporal_options()
    config = model_config()
    spatial = architecture()
    spatial.update(num_heads=4, ffn_dim=32)
    config.update(backbone_name="fixture", local_backbone_dir="fixture", camera_names=DEFAULT_CAMERA_NAMES,
                  x_range=(-5., 5.), y_range=(-5., 5.), z_anchors=(1., 2.),
                  dense_frontend=options, map_architecture=spatial, P=20)
    profile = dict(version=UNIFIED_SCENE_VERSION, bev_h=8, bev_w=8, bev_layers=2,
        window_size=4, bev_levels=2, heads=4, ffn_dim=32, query_layers=2, sampling_points=2,
        agent_queries=4, map_queries=3, map_points=20, dropout=0., gradient_checkpointing=True)
    return config, profile


def targets(data, layers=2):
    b, n, _, h, w = data["images"].shape
    agent = empty_canonical_agent(4)
    agent["class_support_mask"] = NAVFORMER_CLASS_SUPPORT.clone()
    agent["labels"][0] = 0
    agent["valid_mask"][0] = True
    agent["boxes_metric"][0] = torch.tensor([0., 0., 1., 2., 1., 1., 0.])
    agent["scores"][0] = .9
    recipe = load_yaml_config(project_root() / "configs/stage2_agent.yaml")
    recipe["agent_loss"]["xy_range"] = [-5., 5.]
    recipe["agent_loss"]["aux_layer_weights"] = [1.] * layers
    settings = {"agent": recipe, "map": loss_settings(),
        "dense": {"weights": {"seg_ce": 1., "depth_log": 1., "depth_relative": .5, "depth_gradient": .5},
                  "depth": {"min_depth_m": .1, "max_depth_m": 80., "exclude_sky": False}}}
    settings["map"]["aux_layer_weights"] = [.5] * (layers - 1) + [1.]
    batch = {"sample_token": ["fixture"], "soft_labels": [{"agent": agent}],
        "vector_targets": [line_target()], "dense_targets": [{
            "seg": {"labels": torch.zeros(n, h, w, dtype=torch.long), "valid_mask": torch.ones(n, h, w, dtype=torch.bool)},
            "depth": {"depth_m": torch.full((n, 1, h, w), 8.), "valid_mask": torch.ones(n, 1, h, w, dtype=torch.bool)}}]}
    return batch, settings


class UnifiedSceneTest(unittest.TestCase):
    def make_model(self):
        config, profile = fixtures()
        return UnifiedQUEST(config, profile, backbone=TinyBackbone()), config, profile

    def test_four_real_losses_reach_common_geometry_fusion_bev_query_and_update(self):
        torch.manual_seed(17)
        model, _, _ = self.make_model()
        data = inputs()
        data["history"] = history(data)
        batch, settings = targets(data)
        before = full_base_fingerprint(model.backbone)
        for absent in ("base", "agent_decoder", "map_decoder", "seg_branch", "dense_frontend"):
            self.assertFalse(hasattr(model, absent))
        model.train()
        output = model(**data)
        self.assertEqual(output["seg_logits"].shape, (1, 8, 19, 8, 16))
        self.assertEqual(output["map_points"].shape, (1, 3, 20, 2))
        self.assertTrue(bool((output["depth"] > 0).all()))
        raw, _ = four_task_losses(model, output, batch, settings, torch.device("cpu"))
        weighted = scaled_task_losses(raw, dict.fromkeys(raw, 1.), dict.fromkeys(raw, 1.))
        audit = shared_gradient_audit(model, weighted)
        self.assertEqual(set(audit["task_to_shared_gradient_norms"]), {"agent", "map", "seg", "depth"})
        self.assertEqual(len(audit["sampled_shared_gradient_cosines"]), 6)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        old = model.trunk.geometry.level_projections[-1].weight.detach().clone()
        joint_backward(model, weighted)
        for head in (model.heads.agent.cls_head[-1], model.heads.map_class[-1], model.heads.seg, model.heads.depth[-1]):
            self.assertIsNotNone(head.weight.grad)
            self.assertGreater(float(head.weight.grad.abs().sum()), 0)
        optimizer.step()
        self.assertFalse(torch.equal(old, model.trunk.geometry.level_projections[-1].weight))
        self.assertEqual(before, full_base_fingerprint(model.backbone))
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))

    def test_shared_history_affects_all_tasks_and_cold_start_masks_arbitrary_history(self):
        torch.manual_seed(29)
        model, _, _ = self.make_model()
        model.eval()
        with torch.no_grad():
            for block in model.trunk.geometry.temporal_blocks:
                block.gate.fill_(.4)
            data = inputs()
            data["history"] = history(data)
            first = model(**data)
            data["history"]["images"] = torch.randn_like(data["images"]) * 3
            second = model(**data)
            for key in ("agent_cls_logits", "map_points", "seg_logits", "depth"):
                self.assertGreater(float((first[key] - second[key]).abs().sum()), 1e-7, key)
            data["history"]["valid"].zero_()
            data["history"]["dt_s"].zero_()
            masked = model(**data)
            data["history"]["images"] = torch.randn_like(data["images"]) * 10
            other = model(**data)
            for key in ("agent_cls_logits", "map_points", "seg_logits", "depth"):
                torch.testing.assert_close(masked[key], other[key], rtol=0, atol=0)
            data["history"]["valid"].fill_(True)
            data["history"]["dt_s"].fill_(-1)
            with self.assertRaisesRegex(ValueError, "causal"):
                model(**data)

    def test_strict_partial_warm_start_roundtrip_and_old_base_unchanged(self):
        config, profile = fixtures()
        with patch("quest.model.FrozenDINOv2Backbone", TinyBackbone):
            base = QUESTModel(**config).eval()
        source_hash = full_base_fingerprint(base)
        model = UnifiedQUEST(config, profile, backbone=TinyBackbone()).eval()
        report = model.warm_start(SimpleNamespace(base=base))
        self.assertFalse(report["prediction_identity"])
        self.assertEqual(full_base_fingerprint(base), source_hash)
        self.assertEqual(model.heads.depth.state_dict().keys(), base.dense_frontend.depth_decoder.output.state_dict().keys())
        data = inputs()
        data["history"] = history(data)
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            expected = model(**data)
            path = Path(directory) / "unified.pt"
            torch.save(model.state_dict(), path)
            restored = UnifiedQUEST(config, profile, backbone=TinyBackbone()).eval()
            restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
            actual = restored(**data)
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)

    def test_raw_lift_default_unchanged_and_distortion_actually_changes_projection(self):
        lift = GeometryAwareBEVLift(16, 4, 4, (-2, 2), (-2, 2), (5.,))
        data = inputs()
        old = lift.project_reference_points(data["intrinsics"], data["extrinsics"], (8, 16))
        zero = lift.project_reference_points(data["intrinsics"], data["extrinsics"], (8, 16), data["distortion"])
        torch.testing.assert_close(old[0][old[1]], zero[0][old[1]], rtol=0, atol=1e-6)
        self.assertTrue(torch.equal(old[1], zero[1]))
        data["distortion"][..., 0] = -.2
        changed = lift.project_reference_points(data["intrinsics"], data["extrinsics"], (8, 16), data["distortion"])
        self.assertGreater(float((changed[0][changed[1]] - zero[0][changed[1]]).abs().sum()), 0)

    def test_dpt_shared_feature_api_preserves_native_depth_function(self):
        decoder = DPTMetricDecoder(16, 8, .1).eval()
        levels = [torch.randn(8, 16, 2, 4) for _ in range(4)]
        with torch.no_grad():
            expected = decoder(levels, (8, 16))
            value = decoder.forward_features(levels)
            logits = torch.nn.functional.interpolate(decoder.output(value), (8, 16), mode="bilinear", align_corners=False)
            torch.testing.assert_close(expected, torch.nn.functional.softplus(logits) + .1, rtol=0, atol=0)

    def test_pcgrad_conflict_nonconflict_private_heads_and_loss_scales(self):
        generator = torch.Generator().manual_seed(1)
        value = pcgrad_vectors(torch.tensor([[1., 0.], [-1., 0.]]), generator)
        torch.testing.assert_close(value, torch.zeros(2))
        aligned = torch.tensor([[1., 2.], [2., 4.]])
        torch.testing.assert_close(pcgrad_vectors(aligned, generator), aligned.sum(0))
        model = SimpleNamespace(trunk=torch.nn.Linear(1, 1, bias=False))
        head = torch.nn.Parameter(torch.ones(()))
        x = model.trunk.weight.sum()
        losses = {"a": x + head, "b": -x + head * 2}
        joint_backward(model, losses, mode="pcgrad", generator=generator)
        torch.testing.assert_close(model.trunk.weight.grad, torch.zeros_like(model.trunk.weight))
        torch.testing.assert_close(head.grad, torch.tensor(3.))
        raw = {k: torch.tensor(v) for k, v in zip(("agent", "map", "seg", "depth"), (2., 20., .2, .02))}
        result = scaled_task_losses(raw, {k: float(v) for k, v in raw.items()}, dict.fromkeys(raw, 1.))
        self.assertTrue(all(float(value) == 1. for value in result.values()))
        with self.assertRaises(ValueError):
            scaled_task_losses(raw, dict.fromkeys(raw, 0.), dict.fromkeys(raw, 1.))

    def test_quality_gate_rejects_one_regressed_task_even_if_others_improve(self):
        baseline = {"dense": {"seg_teacher_mIoU": .46, "depth_teacher_absrel": .1, "depth_teacher_delta1": .88},
                    "agent": {k: {"precision": .3, "recall": .4} for k in ("vehicle", "pedestrian")},
                    "map": {k: {"precision": .1, "recall": .1} for k in ("centerline", "ped_crossing", "road_boundary")}}
        limits = dict(seg_miou_drop=.01, depth_absrel_increase_ratio=.05, depth_delta1_drop=.01,
                      agent_precision_recall_drop=.02, map_precision_recall_drop=.01)
        current = copy.deepcopy(baseline)
        self.assertTrue(quality_gate(current, baseline, limits)["all_pass"])
        current["dense"].update(seg_teacher_mIoU=.6, depth_teacher_absrel=.08)
        current["agent"]["pedestrian"]["recall"] = .3
        self.assertFalse(quality_gate(current, baseline, limits)["all_pass"])


if __name__ == "__main__":
    unittest.main()
