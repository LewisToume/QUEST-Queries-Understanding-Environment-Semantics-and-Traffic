"""Server-run test sources only. No teacher/DINO assets or GPU required."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from quest.depth_geometry import pack_distortion, raw_camera_rays, project_raw, sampling_geometry
from quest.geometry_depth import GEOMETRY_DEPTH_VERSION, DPTMetricDecoder, GeometryDepthFrontend
from quest.model import QUESTModel, load_quest_v3_checkpoint
from quest.seg_depth_dataset import collate_seg_depth
from quest.seg_depth_diagnostics import DenseEvaluationDiagnostics
from quest.seg_depth_losses import compute_seg_depth_loss
from quest.seg_depth_training import (configure_dense_training, frozen_spatial_fingerprint,
                                      load_dense_model, save_dense_checkpoint, validate_dense_checkpoint)


def options():
    return dict(version=GEOMETRY_DEPTH_VERSION, preserve_spatial_features=True, image_geometry="raw_opencv",
                layer_indices=[3, 6, 9, 12], output_size=[252, 448], min_depth_m=.1, geometry_layers=4,
                heads=4, attention_dim=16, ffn_dim=32, depth_samples=4, ray_min_depth_m=.5,
                ray_max_depth_m=80., neighbors=2, dpt_features=8, dropout=0.,
                amp_bf16=False, gradient_checkpointing=False)


def inputs(h=8, w=16):
    k = torch.tensor([[8., 0, w / 2], [0, 8., h / 2], [0, 0, 1.]])[None, None].repeat(1, 8, 1, 1)
    return dict(images=torch.randn(1, 8, 3, h, w), intrinsics=k,
                extrinsics=torch.eye(4)[None, None].repeat(1, 8, 1, 1),
                ego_state=torch.zeros(1, 9), distortion=torch.zeros(1, 8, 8))


class TinyBackbone(nn.Module):
    hidden_dim, patch_size = 16, 2

    def __init__(self, *args):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(.1, 1., 16), requires_grad=False)

    def forward_intermediates(self, images, indices):
        value = F.avg_pool2d(images.mean(1, keepdim=True), 2).flatten(2).transpose(1, 2)
        value = value * self.weight[None, None]
        tokens = torch.cat((value[:, :1], value), 1).detach()
        return tokens, tuple(tokens * scale for scale in (.25, .5, .75, 1.))


def model_config():
    return dict(hidden_dim=16, bev_h=4, bev_w=4, bev_layers=1, bev_attention_heads=4, bev_ffn_dim=32,
                agent_decoder_layers=1, map_decoder_layers=1, decoder_attention_heads=4,
                decoder_ffn_dim=32, N_agent=4, N_map=3, C_map=3, dropout=0., C_seg=19,
                seg_size=[252, 448], depth_size=[252, 448], dense_frontend=options())


def training_config():
    return dict(stage="seg_depth_offline", tasks=["depth"], frontend=options(),
                depth_geometry_policy="experimental_image_plane",
                train=dict(epochs=5, num_samples=5000, batch_size=1, log_interval=100, num_workers=0,
                           lr=1e-4, weight_decay=1e-4, gradient_clip_norm=1., missing_labels="error"),
                loss=dict(weights=dict(seg_ce=0., depth_log=1., depth_relative=.5, depth_gradient=.5),
                          depth=dict(min_depth_m=.1, max_depth_m=80., exclude_sky=False)))


class GeometryDepthTest(unittest.TestCase):
    def test_raw_rays_roundtrip_zero_and_nonzero_distortion(self):
        data = inputs()
        for coefficients in (torch.zeros(8), torch.tensor([-.2, .04, .001, -.001, .01, 0, 0, 0])):
            d = coefficients[None, None].repeat(1, 8, 1)
            rays, valid = raw_camera_rays(data["intrinsics"], d, (4, 8), (8, 16))
            grid, projected = project_raw(rays * 10, data["intrinsics"], d, (8, 16))
            y, x = torch.meshgrid(torch.arange(4), torch.arange(8), indexing="ij")
            expected = torch.stack((2 * (x.flatten() + .5) / 8 - 1,
                                    2 * (y.flatten() + .5) / 4 - 1), -1)
            self.assertTrue(bool(valid.all() & projected.all()))
            torch.testing.assert_close(grid[0, 0], expected, atol=.001, rtol=0)
        self.assertEqual(pack_distortion([1, 2, 3, 4]).shape, (8,))
        with self.assertRaises(ValueError):
            pack_distortion([0] * 12)

    def test_positive_depth_boundaries_and_sensor2lidar_translation(self):
        k = torch.eye(3)[None, None]
        points = torch.tensor([[[[0., 0., 1.], [0., 0., -1.], [100., 0., 1.]]]])
        _, valid = project_raw(points, k, torch.zeros(1, 1, 8), (8, 16))
        self.assertEqual(valid.tolist(), [[[True, False, False]]])
        data = inputs()
        data["extrinsics"][:, 1, 0, 3] = 1.
        geometry = sampling_geometry(data["intrinsics"], data["extrinsics"], data["distortion"],
                                     torch.ones(1, 8, dtype=torch.bool), (4, 8), (8, 16), torch.tensor([10.]), 7)
        rank = geometry[0][0, 0].tolist().index(1)
        # source camera translated +1m: its image coordinate moves left by fx/10.
        expected_shift = -2 * 8 / 10 / 16
        torch.testing.assert_close(geometry[1][rank][0, 0, 10, 0, 0], torch.tensor(-.375 + expected_shift))
        data["extrinsics"][:, 0, 0, 0] = 2.
        with self.assertRaisesRegex(ValueError, "rigid sensor2lidar"):
            sampling_geometry(data["intrinsics"], data["extrinsics"], data["distortion"],
                              torch.ones(1, 8, dtype=torch.bool), (4, 8), (8, 16), torch.tensor([10.]), 2)

    def test_full_rotation_projection_not_yaw_only(self):
        data = inputs()
        # Camera 1 rotated 180 degrees relative to camera 0.
        data["extrinsics"][:, 1, :3, :3] = torch.diag(torch.tensor([-1., 1., -1.]))
        result = sampling_geometry(data["intrinsics"], data["extrinsics"], data["distortion"],
                                   torch.ones(1, 8, dtype=torch.bool), (4, 8), (8, 16), torch.tensor([10.]), 7)
        rank = result[0][0, 0].tolist().index(1)
        self.assertFalse(bool(result[2][rank][0, 0].any()))

    def test_dpt_learned_reassembly_rectangular_shape_and_gradient(self):
        module = DPTMetricDecoder(16, 8, .1)
        levels = [torch.randn(2, 16, 2, 4, requires_grad=True) for _ in range(4)]
        result = module(levels, (16, 32))
        self.assertEqual(result.shape, (2, 1, 16, 32))
        self.assertTrue(bool(torch.isfinite(result).all() & (result > 0).all()))
        result.mean().backward()
        self.assertTrue(all(value.grad is not None and bool((value.grad != 0).any()) for value in levels))

    def test_cross_view_dependency_and_all_masked_source_fallback(self):
        torch.manual_seed(3)
        module, data = GeometryDepthFrontend(16, options()).eval(), inputs()
        levels = [torch.randn(1, 8, 16, 2, 4) for _ in range(4)]
        with torch.no_grad():
            first = module(levels, data["intrinsics"], data["extrinsics"], data["distortion"], (8, 16))
            changed = [value.clone() for value in levels]
            for value in changed:
                value[:, 1:] = torch.randn_like(value[:, 1:]) * 5
            second = module(changed, data["intrinsics"], data["extrinsics"], data["distortion"], (8, 16))
            self.assertGreater(float((first["depth"][:, 0] - second["depth"][:, 0]).abs().sum()), 1e-5)
            mask = torch.zeros(1, 8, dtype=torch.bool)
            mask[:, 0] = True
            a = module(levels, data["intrinsics"], data["extrinsics"], data["distortion"], (8, 16), mask)
            b = module(changed, data["intrinsics"], data["extrinsics"], data["distortion"], (8, 16), mask)
            torch.testing.assert_close(a["depth"][:, 0], b["depth"][:, 0])
            self.assertEqual(float(a["depth_cross_view_coverage"]), 0.)
            self.assertTrue(bool(torch.isfinite(a["depth"]).all()))
        self.assertNotIn("seg_logits", first)
        with self.assertRaisesRegex(ValueError, "distortion"):
            module(levels, data["intrinsics"], data["extrinsics"], None, (8, 16))

    def test_checkpointed_blocks_match_non_checkpointed_gradients(self):
        torch.manual_seed(5)
        normal, data = GeometryDepthFrontend(16, options()), inputs()
        reduced = copy.deepcopy(normal)
        reduced.options["gradient_checkpointing"] = True
        levels = [torch.randn(1, 8, 16, 2, 4) for _ in range(4)]
        for module in (normal, reduced):
            module(levels, data["intrinsics"], data["extrinsics"], data["distortion"], (8, 16))["depth"].mean().backward()
        torch.testing.assert_close(normal.geometry_blocks[0].cross_view.query.weight.grad,
                                   reduced.geometry_blocks[0].cross_view.query.weight.grad)

    def test_depth_only_loss_diagnostics_and_true_absrel(self):
        depth = torch.full((1, 8, 1, 4, 8), 11., requires_grad=True)
        target = [{"seg": None, "depth": {"depth_m": torch.full_like(depth[0], 10.),
                                          "valid_mask": torch.ones_like(depth[0], dtype=torch.bool)}}]
        settings = training_config()["loss"]
        losses = compute_seg_depth_loss({"depth": depth}, target, settings, ("depth",))
        self.assertAlmostEqual(float(losses["depth_absrel"]), .1, places=5)
        self.assertLess(float(losses["depth_relative"]), .01)
        losses["total_loss"].backward()
        diagnostics = DenseEvaluationDiagnostics(tuple(str(i) for i in range(8)), settings, ("depth",))
        diagnostics.update({"depth": depth.detach()}, target, [{"token": "t", "scene_token": "s"}], [5], ["t"])
        self.assertEqual(diagnostics.summarize()["overall"]["seg_teacher_mIoU"], None)

    def test_model_freezes_spatial_and_versioned_checkpoint_roundtrip(self):
        with patch("quest.model.FrozenDINOv2Backbone", TinyBackbone):
            model = QUESTModel(**model_config()).eval()
        data = inputs()
        with torch.no_grad():
            before = model(**data)
        baseline = frozen_spatial_fingerprint(model)
        optimizer = torch.optim.AdamW(configure_dense_training(model, ("depth",)), lr=1e-4)
        prediction = model.forward_dense(data["images"], data["intrinsics"], data["extrinsics"], data["distortion"])
        prediction["depth"].mean().backward()
        self.assertGreater(float(model.dense_frontend.geometry_blocks[0].cross_view.key_value.weight.grad.abs().sum()), 0)
        optimizer.step()
        self.assertEqual(frozen_spatial_fingerprint(model), baseline)
        model.eval()
        with torch.no_grad():
            after = model(**data)
        for key in ("agent_cls_logits", "agent_boxes", "agent_velocity", "map_cls_logits", "map_points"):
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v2.pt"
            save_dense_checkpoint(path, model, optimizer, scheduler, 1, training_config(), model_config(),
                                  {"split_sha256": "s", "metadata_sha256": "m"}, {}, baseline, {})
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(saved["dense_checkpoint_version"], 2)
            with patch("quest.model.FrozenDINOv2Backbone", TinyBackbone):
                restored = load_dense_model(saved)
            self.assertEqual(frozen_spatial_fingerprint(restored), baseline)
            wrong = dict(saved, dense_checkpoint_version=1)
            with self.assertRaises(ValueError):
                validate_dense_checkpoint(wrong)
            wrong = copy.deepcopy(saved)
            wrong["training_config"]["frontend"]["neighbors"] = 1
            with self.assertRaisesRegex(ValueError, "training profile mismatch"):
                validate_dense_checkpoint(wrong)
            with self.assertRaisesRegex(ValueError, "profile mismatch"):
                load_quest_v3_checkpoint(model, dict(saved, dense_frontend=None))

    def test_calibration_collation_preserves_camera_order(self):
        data = inputs()
        sample = {key: value[0] for key, value in data.items()}
        sample.update(sample_token="t", source_index=5, dense_targets={"seg": None, "depth": None})
        sample["distortion"][:, 0] = torch.arange(8)
        batch = collate_seg_depth([sample])
        torch.testing.assert_close(batch["distortion"][0, :, 0], torch.arange(8).float())


if __name__ == "__main__":
    unittest.main()
