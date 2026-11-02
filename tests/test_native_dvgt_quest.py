"""CPU unit tests with small stand-ins; actual native DVGT is checked on the server."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from quest.dvgt_depth import DVGT_SCALE
from quest.native_dvgt import (NATIVE_TO_QUEST, QUESTNativeDVGT, audit_native_architecture,
                               native_camera_depth, publish_native_checkpoint, load_native_checkpoint_state)
from quest.native_dvgt_inputs import causal_native_window
from quest.openscene_dataset import OPENSCENE_CAMERA_NAMES


class ToyCore(nn.Module):
    """Contract fixture, deliberately not evidence of a real native model forward."""
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.aggregator = SimpleNamespace(embed_dim=1024, patch_size=16, depth=24,
            aa_order=["intra_view", "cross_view", "cross_frame"],
            intra_view_blocks=[None] * 24, cross_view_blocks=[None] * 24, cross_frame_blocks=[None] * 24)
        self.point_head = SimpleNamespace(intermediate_layer_idx=[4, 11, 17, 23])
        self.ego_pose_head = nn.Identity()


class ToySpatial(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.))
        self.camera_names = OPENSCENE_CAMERA_NAMES
        self.dense_frontend_config = {"version": "geometry_dpt_depth_v2", "layer_indices": [3, 6, 9, 12]}
        self.dense_frontend = nn.Linear(2, 2)
        self.seg_head = self.depth_head = None
        self.requested_layers = None

    def extract_camera_features(self, images, *, layer_indices):
        self.requested_layers = layer_indices
        return {"x": images * self.weight}

    def encode_camera_features(self, features, *args, **kwargs):
        if kwargs["use_ego_state"]:
            raise AssertionError("Preserved spatial route must not add ego conditioning")
        return {"bev_tokens": features["x"], "bev_features": features["x"]}

    def forward_agent(self, tokens, features):
        return {"agent_boxes": tokens + features}

    def forward_map(self, encoded, *args, **kwargs):
        return {"map_points": encoded["bev_tokens"] + 1}


def fixture_geometry():
    h, w = 3, 4
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    z = torch.arange(1, 9).float()[None, :, None, None].expand(1, 8, h, w)
    points = torch.stack((xx * z, yy * z, z), -1) * DVGT_SCALE
    geometry = {"camera_from_first_ego": torch.eye(4).expand(1, 8, 4, 4).clone(),
                "native_intrinsics": torch.eye(3).expand(1, 8, 3, 3).clone(),
                "raw_native_uv": torch.stack((xx, yy), -1).float().expand(1, 8, h, w, 2).clone()}
    return points, torch.ones_like(z), geometry


class NativeDVGTQuestTest(unittest.TestCase):
    def test_camera_order_scale_z_and_gradients(self):
        points, confidence, geometry = fixture_geometry()
        points.requires_grad_(True)
        result = native_camera_depth(points, confidence, **geometry)
        expected = torch.arange(1, 9).float()[list(NATIVE_TO_QUEST)]
        torch.testing.assert_close(result["depth"][0, :, 0, 1, 1], expected)
        self.assertTrue(bool(result["depth_valid_mask"].all()))
        result["depth"].sum().backward()
        self.assertTrue(bool(torch.isfinite(points.grad).all()))
        self.assertGreater(float(points.grad.abs().sum()), 0)

    def test_invalid_projection_confidence_and_range_are_not_filled(self):
        points, confidence, geometry = fixture_geometry()
        confidence[:, 0] = 0
        geometry["raw_native_uv"][:, 1] = -5
        result = native_camera_depth(points, confidence, **geometry, max_depth=5)
        for canonical, native in enumerate(NATIVE_TO_QUEST):
            if canonical == 1 or native == 0 or native + 1 > 5:
                self.assertFalse(bool(result["depth_valid_mask"][:, canonical].any()))
                self.assertEqual(float(result["depth"][:, canonical].sum()), 0)

    def test_reject_nonfinite_and_mismatched_points(self):
        points, confidence, geometry = fixture_geometry()
        with self.assertRaises(ValueError):
            native_camera_depth(points[:, :7], confidence, **geometry)
        points[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            native_camera_depth(points, confidence, **geometry)

    def test_native_counts_cannot_silently_shrink(self):
        core = ToyCore()
        core.aggregator.cross_frame_blocks.pop()
        with self.assertRaises(ValueError):
            audit_native_architecture(core)

    def test_spatial_route_preserved_and_old_depth_removed(self):
        spatial = ToySpatial()
        model = QUESTNativeDVGT(spatial, ToyCore())
        image = torch.ones(1, 8, 3, 2, 2)
        output = model.forward_spatial(image, torch.eye(3).expand(1, 8, 3, 3),
                                       torch.eye(4).expand(1, 8, 4, 4), torch.zeros(1, 9))
        torch.testing.assert_close(output["agent_boxes"], image * 4, rtol=0, atol=0)
        torch.testing.assert_close(output["map_points"], image * 2 + 1, rtol=0, atol=0)
        self.assertEqual(spatial.requested_layers, (3, 6, 9, 12))
        self.assertIsNone(spatial.dense_frontend)
        self.assertFalse(any(key.startswith("spatial.dense_frontend.") for key in model.state_dict()))
        self.assertEqual(model.parameter_summary()["trainable"], 0)

    def test_reject_old_temporal_agent_source(self):
        spatial = ToySpatial()
        spatial.dense_frontend_config["version"] = "temporal_geometry_dpt_depth_v3"
        with self.assertRaises(ValueError):
            QUESTNativeDVGT(spatial, ToyCore())

    def test_causal_window_has_no_duplicate_or_future_padding(self):
        cold = {"source_index": 10, "history_index": None, "dt_s": 0, "reason": "scene_start"}
        self.assertEqual(causal_native_window(cold), [10])
        real = {"source_index": 8, "history_index": 12, "dt_s": .5, "reason": "available",
                "history_timestamp": 100, "timestamp": 101}
        self.assertEqual(causal_native_window(real), [12, 8])  # Order is time, not metadata index.
        real["history_timestamp"] = 102
        with self.assertRaises(ValueError):
            causal_native_window(real)

    def test_checkpoint_roundtrip_and_no_overwrite(self):
        model = QUESTNativeDVGT(ToySpatial(), ToyCore())
        audit = {"native_strict_load": True, "runtime_checks_passed": True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "init.pt"
            with self.assertRaises(ValueError):
                publish_native_checkpoint(path, model, {})
            publish_native_checkpoint(path, model, audit)
            with self.assertRaises(FileExistsError):
                publish_native_checkpoint(path, model, audit)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            self.assertFalse(checkpoint["training_started"])
            self.assertFalse(checkpoint["shared_backbone"])
            restored = QUESTNativeDVGT(ToySpatial(), ToyCore())
            load_native_checkpoint_state(restored, checkpoint)
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
            checkpoint["profile"] = "legacy_dense"
            with self.assertRaises(ValueError):
                load_native_checkpoint_state(restored, checkpoint)


if __name__ == "__main__":
    unittest.main()
