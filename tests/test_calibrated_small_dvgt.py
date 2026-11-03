"""Server-only native CPU fixtures for V5-preserving calibrated V6 migration."""

import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from quest.calibrated_small_dvgt import CalibratedSmallDVGTDepthFrontend, validate_calibrated_small_dvgt
from quest.small_dvgt_depth import SmallDVGTDepthFrontend, validate_native_package_origin
from quest.geometry_depth import SMALL_DVGT_VERSION, CALIBRATED_SMALL_DVGT_VERSION
from quest.seg_depth_training import frozen_spatial_fingerprint, load_small_dvgt_warm_start
from quest.utils import project_root, load_yaml_config


def options():
    value = copy.deepcopy(load_yaml_config(project_root() / "configs/depth_calibrated_small_dvgt_v6.yaml")["frontend"])
    value.update(geometry_dim=32, geometry_layers=4, geometry_levels=[0, 1, 2, 3],
                 heads=4, dpt_features=8, gradient_checkpointing=False, amp_bf16=False)
    value["calibration"].update(round_indices=[0, 1, 2, 3], attention_dim=16, depth_samples=4)
    return value


def fixture():
    levels = [torch.randn(1, 8, 16, 2, 4) for _ in range(4)]
    k = torch.tensor([[28., 0., 28.], [0., 28., 14.], [0., 0., 1.]])[None, None].repeat(1, 8, 1, 1)
    extrinsic = torch.eye(4)[None, None].repeat(1, 8, 1, 1)
    extrinsic[0, :, 0, 3] = torch.linspace(-.7, .7, 8)
    distortion = torch.zeros(1, 8, 8)
    relative = torch.eye(4)[None]
    relative[:, 0, 3] = .1
    history = {"feature_levels": [torch.randn_like(x) for x in levels], "intrinsics": k.clone(),
               "extrinsics": extrinsic.clone(), "distortion": distortion.clone(),
               "previous_to_current": relative, "valid": torch.tensor([True]), "dt_s": torch.tensor([.5])}
    return levels, k, extrinsic, distortion, history


class Container(nn.Module):
    def __init__(self, frontend, profile):
        super().__init__()
        self.dense_frontend = frontend
        self.dense_frontend_config = profile
        self.frozen = nn.Linear(2, 2)


class CalibratedSmallDVGTTest(unittest.TestCase):
    def test_regular_and_namespace_package_origins(self):
        repo = project_root() / "third_party/DVGT"
        package = repo / "dvgt"
        # The server's second constructor sees an already imported namespace
        # package with __file__=None. This must not call Path(None).
        namespace = SimpleNamespace(__file__=None, __path__=[str(package)])
        validate_native_package_origin(namespace, repo)
        validate_native_package_origin(namespace, repo)
        validate_native_package_origin(SimpleNamespace(__path__=[str(package)]), repo)
        validate_native_package_origin(SimpleNamespace(__file__=str(package / "__init__.py"),
                                                       __path__=[str(package)]), repo)
        for bad in (
            SimpleNamespace(__file__=None),
            SimpleNamespace(__file__=None, __path__=[str(project_root() / "other/dvgt")]),
            SimpleNamespace(__file__=None, __path__=[str(package), str(project_root() / "other/dvgt")]),
            SimpleNamespace(__file__=str(project_root() / "other/dvgt/__init__.py"), __path__=[str(package)]),
        ):
            with self.assertRaises(RuntimeError):
                validate_native_package_origin(bad, repo)

    def test_invalid_calibration_rounds(self):
        value = options()
        validate_calibrated_small_dvgt(value)
        value["calibration"]["round_indices"] = [0, 1, 1, 3]
        with self.assertRaises(ValueError):
            validate_calibrated_small_dvgt(value)

    @unittest.skipUnless((project_root() / "third_party/DVGT/dvgt").is_dir(), "Pinned native source required")
    def test_warm_start_identity_multilevel_pose_and_cold_start(self):
        torch.manual_seed(42)
        profile = options()
        old_profile = dict(profile, version=SMALL_DVGT_VERSION)
        old_profile.pop("calibration")
        old = Container(SmallDVGTDepthFrontend(16, old_profile), old_profile).eval()
        new = Container(CalibratedSmallDVGTDepthFrontend(16, profile), profile).eval()
        source = {"dense_frontend": old_profile, "model_state_dict": old.state_dict(),
                  "frozen_spatial_sha256": frozen_spatial_fingerprint(old)}
        self.assertGreater(load_small_dvgt_warm_start(new, source), 0)
        data = fixture()
        levels, k, extrinsic, distortion, history = data
        with torch.no_grad():
            before = old.dense_frontend(levels, k, extrinsic, distortion, (28, 56), history=history)["depth"]
            after = new.dense_frontend(levels, k, extrinsic, distortion, (28, 56), history=history)["depth"]
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertEqual(frozen_spatial_fingerprint(old), frozen_spatial_fingerprint(new))

        dense = new.dense_frontend
        with torch.no_grad():
            for gate in (dense.injection_gates, dense.position_gates, dense.spatial_gates):
                gate.fill_(.2)
            for block in dense.calibrated_temporal:
                block.gate.fill_(.2)
        levels = [x.detach().requires_grad_() for x in levels]
        output = dense(levels, k, extrinsic, distortion, (28, 56), history=history)
        output["depth"].mean().backward()
        for level in levels:
            self.assertIsNotNone(level.grad)
            self.assertGreater(float(level.grad.abs().sum()), 0)
        for module in (dense.calibrated_spatial, dense.calibrated_temporal):
            self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in module.parameters()))
        with torch.no_grad():
            changed = dict(history, previous_to_current=history["previous_to_current"].clone())
            changed["previous_to_current"][:, 0, 3] += 1.
            pose_output = dense(levels, k, extrinsic, distortion, (28, 56), history=changed)["depth"]
            self.assertGreater(float((output["depth"] - pose_output).abs().max()), 0)
            shifted = extrinsic.clone()
            shifted[:, 0, 0, 3] += 1.
            camera_output = dense(levels, k, shifted, distortion, (28, 56), history=history)["depth"]
            self.assertGreater(float((output["depth"] - camera_output).abs().max()), 0)
            cold = dict(history, valid=torch.tensor([False]), dt_s=torch.tensor([0.]))
            first = dense(levels, k, extrinsic, distortion, (28, 56), history=cold)
            cold["feature_levels"] = [torch.randn_like(x) * 100 for x in levels]
            second = dense(levels, k, extrinsic, distortion, (28, 56), history=cold)
            torch.testing.assert_close(first["depth"], second["depth"], rtol=0, atol=0)
            self.assertEqual(float(first["depth_temporal_coverage"]), 0.)
        with self.assertRaises(ValueError):
            dense(levels, None, extrinsic, distortion, (28, 56), history=history)
        # A saved V6 profile must never masquerade as the V5 migration source.
        with self.assertRaises(ValueError):
            load_small_dvgt_warm_start(new, dict(source, dense_frontend={"version": CALIBRATED_SMALL_DVGT_VERSION}))


if __name__ == "__main__":
    unittest.main()
