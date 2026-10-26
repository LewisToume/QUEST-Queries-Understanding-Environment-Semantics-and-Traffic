"""Test sources for server execution; no pretrained weights or GPU required."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from quest.agent_training import AGENT_END_TO_END_STAGE, load_checkpoint_cpu
from quest.map_objectives import (SPATIAL_MAP_OBJECTIVE, dense_map_gt_loss,
                                  focal_matching_cost, rasterize_vector_targets)
from quest.map_training import (
    STAGE3_GT_ONLY_NAME, configure_stage3, frozen_agent_fingerprint, load_stage2_for_stage3,
    load_stage3_checkpoint, match_vector_queries, mixed_stage3_loss, normalize_points,
    save_stage3_checkpoint, stage3_forward, stage3_kd_enabled, stage3_model_config,
    vector_map_loss,
)
from quest.model import DEFAULT_CAMERA_NAMES, QUESTModel
from quest.nuplan_map_locator import MAP_LAYER_AUDIT_VERSION
from quest.nuplan_relation_audit import BASELINE_RELATION_AUDIT_VERSION
from quest.spatial_map import (SPATIAL_MAP_VERSION, MultiScalePointAttention,
                               SpatialMapDecoder, SpatialMapHead, WindowBEVLayer)
from quest.vector_map_labels import MAP_HEIGHT_REFERENCE, VECTOR_SEMANTICS_VERSION


def architecture():
    return {"version": SPATIAL_MAP_VERSION, "bev_h": 8, "bev_w": 8,
            "encoder_layers": 2, "window_size": 4, "decoder_layers": 6,
            "num_heads": 2, "ffn_dim": 16, "num_levels": 2,
            "sampling_points": 2, "dropout": 0.0}


def loss_settings():
    return {"objective_version": SPATIAL_MAP_OBJECTIVE, "classification_loss": "softmax_focal",
            "point_loss_type": "metric_l1", "point_cost_reduction": "sum",
            "map_cls_cost_weight": 2., "map_point_cost_weight": 1.,
            "map_cls_loss_weight": 2., "map_point_loss_weight": 1.,
            "map_direction_loss_weight": 0.1, "background_weight": 0.5,
            "class_weights": [1., 3., 2.], "focal_gamma": 2.,
            "aux_layer_weights": [0.5, 0.5, 0.5, 0.5, 0.5, 1.],
            "dense_focal_weight": 1., "dense_dice_weight": 1.,
            "dense_negative_weight": 1., "dense_line_width_m": 0.05}


def settings():
    loss = loss_settings()
    return {"stage": STAGE3_GT_ONLY_NAME, "map_architecture": architecture(), "map": loss,
            "map_loss": loss, "train": {"freeze_agent_and_bev": True, "map_spatial_enabled": True,
                                         "map_lr": 1e-3, "weight_decay": 0.},
            "loss_weights": {"lambda_map_gt": 1., "lambda_map_kd": 0., "lambda_agent": 0.},
            "agent_train": {}, "agent_loss": {}}


def line_target(class_id=0):
    points = torch.stack((torch.linspace(-3, 3, 20), torch.ones(20)), -1)[None]
    return {"class_ids": torch.tensor([class_id]), "points_xy_m": points,
            "is_closed": torch.tensor([False]), "length_m": torch.tensor([6.])}


class TinyBackbone(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.hidden_dim, self.patch_size = 8, 14
        self.projection = nn.Linear(3, 8).requires_grad_(False)

    def forward(self, images):
        patches = F.adaptive_avg_pool2d(images, (2, 4)).flatten(2).transpose(1, 2)
        features = self.projection(patches)
        return torch.cat((features[:, :1] * 0, features), 1)


def tiny_model():
    with patch("quest.model.FrozenDINOv2Backbone", TinyBackbone):
        return QUESTModel(hidden_dim=8, camera_names=DEFAULT_CAMERA_NAMES, bev_h=4, bev_w=4,
                          x_range=(-5, 5), y_range=(-5, 5), bev_layers=1,
                          bev_attention_heads=2, bev_ffn_dim=16, agent_decoder_layers=1,
                          decoder_attention_heads=2, decoder_ffn_dim=16, dropout=0,
                          N_agent=3, N_map=3, C_map=3, map_architecture=architecture())


def batch():
    intrinsic = torch.tensor([[20., 0, 28.], [0, 20., 14.], [0, 0, 1.]])
    extrinsic = torch.eye(4)
    extrinsic[2, 3] = -10
    return {"images": torch.randn(1, 8, 3, 28, 56),
            "intrinsics": intrinsic[None, None].expand(1, 8, -1, -1).clone(),
            "extrinsics": extrinsic[None, None].expand(1, 8, -1, -1).clone(),
            "ego_state": torch.zeros(1, 9)}


def provenance():
    return {"num_points": 20, "min_length_m": 1., "map_version": "test-map",
            "vector_semantics_version": VECTOR_SEMANTICS_VERSION,
            "map_height_reference": MAP_HEIGHT_REFERENCE,
            "map_cast_audit_version": BASELINE_RELATION_AUDIT_VERSION,
            "map_layer_audit_version": MAP_LAYER_AUDIT_VERSION}


class Stage3SpatialMapTest(unittest.TestCase):
    def test_shifted_window_shape_and_gradient(self):
        layer = WindowBEVLayer(8, 8, 8, 2, 16, 4, 2, 0)
        value = torch.randn(2, 8, 8, 8, requires_grad=True)
        result = layer(value)
        self.assertEqual(result.shape, value.shape)
        self.assertTrue(bool(torch.isfinite(result).all()))
        result.square().mean().backward()
        self.assertGreater(float(value.grad.abs().sum()), 0)

    def test_out_of_range_sampling_has_no_padding_feature_or_nan(self):
        attention = MultiScalePointAttention(8, 2, 2, 2)
        query = torch.randn(1, 2, 20, 8)
        references = torch.full((1, 2, 20, 2), 100.)
        result = attention(query, references, [torch.ones(1, 8, 8, 8), torch.ones(1, 8, 4, 4)])
        self.assertTrue(bool(torch.isfinite(result).all()))
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_spatial_attention_uses_visual_features_at_both_scales(self):
        attention = MultiScalePointAttention(8, 2, 2, 2)
        features = [torch.randn(1, 8, size, size, requires_grad=True) for size in (8, 4)]
        query = torch.randn(1, 2, 20, 8)
        references = torch.full((1, 2, 20, 2), 0.5)
        result = attention(query, references, features)
        result.square().mean().backward()
        for feature in features:
            self.assertIsNotNone(feature.grad)
            self.assertGreater(float(feature.grad.abs().sum()), 0)
        self.assertGreater(float(attention.offsets.weight.grad.abs().sum()), 0)

    def test_six_layers_and_per_point_iterative_refinement(self):
        decoder = SpatialMapDecoder(8, 3, 20, architecture())
        head = SpatialMapHead(8, 3, 6)
        outputs = decoder([torch.randn(1, 8, 8, 8), torch.randn(1, 8, 4, 4)], head)
        self.assertEqual(outputs["map_points_layers"].shape, (6, 1, 3, 20, 2))
        self.assertEqual(outputs["map_cls_logits"].shape, (1, 3, 4))
        self.assertTrue(bool(((outputs["map_points"] > 0) & (outputs["map_points"] < 1)).all()))
        total = outputs["map_points_layers"].mean() + outputs["map_cls_logits_layers"].square().mean()
        total.backward()
        for cls, point in zip(head.cls_branches, head.point_branches):
            self.assertIsNotNone(cls[-1].weight.grad)
            self.assertIsNotNone(point[-1].weight.grad)

    def test_metric_loss_does_not_change_with_roi_normalization(self):
        target = line_target()
        values = []
        for bounds in ((-50, -50, 50, 50), (-10, -20, 30, 60)):
            points = normalize_points(target["points_xy_m"] + torch.tensor([2., 0.]), bounds)
            losses = vector_map_loss(torch.tensor([[[9., 0., 0., -9.]]]), points[None],
                                      [target], loss_settings(), bounds)
            values.append(float(losses["map_point_loss"]))
        self.assertAlmostEqual(values[0], 2., places=4)
        self.assertAlmostEqual(values[1], 2., places=4)

    def test_more_gt_than_queries_raises_instead_of_truncating(self):
        target = line_target()
        target = {key: value.repeat((4, 1, 1) if key == "points_xy_m" else 4)
                  for key, value in target.items()}
        with self.assertRaisesRegex(ValueError, "only 3 queries"):
            vector_map_loss(torch.zeros(1, 3, 4), torch.zeros(1, 3, 20, 2),
                            [target], loss_settings(), (-5, -5, 5, 5))

    def test_focal_cost_prefers_supported_gt_class_when_geometry_equal(self):
        logits = torch.tensor([[0., 9., 0., -9.], [9., 0., 0., -9.]])
        costs = focal_matching_cost(logits, torch.tensor([0]), loss_settings())
        self.assertLess(float(costs[1, 0]), float(costs[0, 0]))
        target = line_target()
        points = normalize_points(target["points_xy_m"], (-5, -5, 5, 5)).expand(2, -1, -1)
        matches = match_vector_queries(logits, points, target, (-5, -5, 5, 5), 2, 1,
                                       objective_config=loss_settings())
        self.assertEqual(matches[0][0], 1)

    def test_dense_raster_never_closes_an_open_line(self):
        points = torch.cat((torch.stack((torch.linspace(0, 6, 10), torch.zeros(10)), -1),
                            torch.stack((torch.full((10,), 6.), torch.linspace(0, 6, 10)), -1)), 0)[None]
        target = {"points_xy_m": points, "class_ids": torch.tensor([1]), "is_closed": torch.tensor([False])}
        opened = rasterize_vector_targets([target], 32, 32, (-1, -1, 7, 7), points, 0.05)
        target["is_closed"][0] = True
        closed = rasterize_vector_targets([target], 32, 32, (-1, -1, 7, 7), points, 0.05)
        self.assertEqual(float(opened[0, 1, 16, 16]), 0)
        self.assertEqual(float(closed[0, 1, 16, 16]), 1)

    def test_dense_gt_loss_backpropagates_for_all_classes_and_empty_frames(self):
        logits = torch.zeros(1, 3, 8, 8, requires_grad=True)
        targets = [line_target(1)]
        losses = dense_map_gt_loss(logits, targets, loss_settings(), (-5, -5, 5, 5))
        sum(losses.values()).backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        empty = {"points_xy_m": torch.empty(0, 20, 2), "class_ids": torch.empty(0, dtype=torch.long),
                 "is_closed": torch.empty(0, dtype=torch.bool)}
        self.assertTrue(all(bool(torch.isfinite(v)) for v in
                            dense_map_gt_loss(logits, [empty], loss_settings(), (-5, -5, 5, 5)).values()))

    def test_full_map_forward_backward_preserves_agent_and_shared_bev(self):
        model, config = tiny_model(), settings()
        optimizer = configure_stage3(model, None, config["train"])
        model.eval()
        for name in ("map_spatial", "map_decoder", "map_head"):
            getattr(model, name).train()
        before = frozen_agent_fingerprint(model)
        predictions = stage3_forward(model, None, batch(), torch.device("cpu"))
        self.assertEqual(predictions["map_dense_gt_logits"].shape, (1, 3, 8, 8))
        with patch("quest.map_training.compute_stage2_agent_loss", return_value={"total_agent_loss": torch.tensor(0.)}):
            losses = mixed_stage3_loss(model, predictions, [line_target()], None, None, None, None, {}, config)
        self.assertGreater(float(losses["map_dense_gt_loss"]), 0)
        losses["total_loss"].backward()
        for name in ("map_spatial", "map_decoder", "map_head"):
            gradients = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
            self.assertTrue(any(bool(g.abs().sum() > 0) for g in gradients))
        optimizer.step()
        self.assertEqual(frozen_agent_fingerprint(model), before)
        self.assertTrue(all(p.grad is None for p in model.map_spatial.geometry_lift.ego_mlp.parameters()))

    def test_stage2_initialization_copies_visual_lift_without_loading_wrong_grids(self):
        model = tiny_model()
        before = frozen_agent_fingerprint(model)
        modules = (("geometry_lift", "geometry_aware_bev_lift_state_dict"),
                   ("bev_encoder", "bev_encoder_state_dict"),
                   ("agent_proposal_head", "agent_proposal_head_state_dict"),
                   ("agent_decoder", "agent_decoder_state_dict"), ("agent_head", "agent_head_state_dict"))
        checkpoint = {"architecture_version": 3, "stage": AGENT_END_TO_END_STAGE, "epoch": 5,
                      "bev_use_ego_state": False, "trained_class_support_mask": torch.tensor([True, True, False, False]),
                      **{key: copy.deepcopy(getattr(model, name).state_dict()) for name, key in modules}}
        self.assertEqual(load_stage2_for_stage3(model, checkpoint), 5)
        self.assertEqual(model.map_spatial.geometry_lift.bev_embedding.shape, (64, 8))
        self.assertEqual(frozen_agent_fingerprint(model), before)
        torch.testing.assert_close(model.map_spatial.geometry_lift.weight_mlp[0].weight,
                                   model.geometry_lift.weight_mlp[0].weight)

    def test_checkpoint_round_trip_and_legacy_cannot_silently_load_into_spatial_map(self):
        model, config = tiny_model(), settings()
        optimizer = configure_stage3(model, None, config["train"])
        config["agent_baseline_sha256"] = frozen_agent_fingerprint(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spatial.pt"
            save_stage3_checkpoint(path, model, None, optimizer, 2, config, None, provenance())
            checkpoint = load_checkpoint_cpu(path)
        self.assertEqual(checkpoint["map_architecture"], architecture())
        self.assertIn("map_spatial_state_dict", checkpoint)
        self.assertEqual(load_stage3_checkpoint(tiny_model(), None, checkpoint), 2)
        bad = dict(checkpoint, map_architecture=None)
        with self.assertRaisesRegex(ValueError, "Map architecture mismatch"):
            load_stage3_checkpoint(tiny_model(), None, bad)

    def test_evaluation_uses_saved_architecture_not_new_yaml_defaults(self):
        base = {"hidden_dim": 8, "map_decoder_layers": 2}
        self.assertTrue(stage3_kd_enabled(settings()) is False)
        self.assertEqual(stage3_model_config(base, settings())["map_decoder_layers"], 6)
        old = stage3_model_config(base, settings(), {"map_objective_version": "ordered_l1_sum_v2"})
        self.assertNotIn("map_architecture", old)
        self.assertEqual(old["map_decoder_layers"], 2)
        broken = settings()
        broken["train"]["map_spatial_enabled"] = False
        with self.assertRaisesRegex(ValueError, "must agree"):
            stage3_kd_enabled(broken)


if __name__ == "__main__":
    unittest.main()
