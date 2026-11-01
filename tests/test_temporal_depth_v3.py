"""User-run CPU unit fixtures; no real labels, teachers, DINO download or GPU."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from quest.geometry_depth import GeometryDepthFrontend, TEMPORAL_DEPTH_VERSION
from quest.model import QUESTModel
from quest.seg_depth_dataset import SegDepthDataset, collate_seg_depth
from quest.seg_depth_training import (configure_dense_training, frozen_spatial_fingerprint,
    load_dense_model, load_depth_warm_start, save_dense_checkpoint, validate_depth_warm_start)
from quest.temporal_depth import TemporalGeometryDepthFrontend, temporal_sampling_geometry
from quest.temporal_depth_dataset import TemporalSegDepthDataset, causal_history_plans
from test_geometry_depth_v2 import TinyBackbone, inputs, model_config, options, training_config


def temporal_options():
    result = options()
    result.update(version=TEMPORAL_DEPTH_VERSION, temporal=dict(frames=2, causal=True,
        missing_history="mask", timestamp_scale=1e-6, max_gap_s=1.1, source_cameras=3))
    return result


def history(data):
    return {key: data[key].clone() for key in ("images", "intrinsics", "extrinsics", "distortion")} | {
        "valid": torch.tensor([True]), "dt_s": torch.tensor([1.]), "previous_to_current": torch.eye(4)[None]}


def info(token, scene, timestamp, x=0.):
    pose = torch.eye(4)
    pose[0, 3] = x
    return dict(token=token, scene_token=scene, timestamp=timestamp, lidar2global=pose)


class TemporalDepthTest(unittest.TestCase):
    def test_pairing_is_chronological_same_scene_and_same_split(self):
        infos = [info("a1", "A", 1000000, 1.), info("b0", "B", 0),
                 info("a0", "A", 0), info("a2", "A", 2000000, 2.)]
        plans = causal_history_plans(infos, [0, 1, 2, 3], [0, 1, 2, 3], temporal_options()["temporal"])
        self.assertEqual([row["history_index"] for row in plans], [2, None, None, 0])
        self.assertEqual(plans[0]["previous_to_current"][0][3], -1.)
        restricted = causal_history_plans(infos, [0, 3], [0, 3], temporal_options()["temporal"])
        self.assertIsNone(restricted[0]["history_index"])
        with self.assertRaises(ValueError):
            causal_history_plans(infos, [0], [1, 2], temporal_options()["temporal"])
        infos[3]["timestamp"] = 5000000
        gap = causal_history_plans(infos, [3], [0, 2, 3], temporal_options()["temporal"])[0]
        self.assertEqual((gap["history_index"], gap["dt_s"], gap["reason"]), (None, 0., "gap"))
        infos[3]["timestamp"] = infos[0]["timestamp"]
        with self.assertRaisesRegex(ValueError, "timestamps"):
            causal_history_plans(infos, [3], [0, 2, 3], temporal_options()["temporal"])

    def test_relative_motion_projection_sign_rotation_and_future_rejection(self):
        data = inputs()
        previous = history(data)
        previous["previous_to_current"][:, 0, 3] = -1.
        geometry = temporal_sampling_geometry(data["intrinsics"], data["extrinsics"], data["distortion"],
            previous, (4, 8), (8, 16), torch.tensor([10.]), 8, 1.1)
        rank = geometry[0][0, 0].tolist().index(0)
        torch.testing.assert_close(geometry[1][rank][0, 0, 10, 0, 0], torch.tensor(-.375 + .1))
        previous["previous_to_current"][0] = torch.diag(torch.tensor([-1., 1., -1., 1.]))
        geometry = temporal_sampling_geometry(data["intrinsics"], data["extrinsics"], data["distortion"],
            previous, (4, 8), (8, 16), torch.tensor([10.]), 3, 1.1)
        self.assertFalse(bool(torch.stack(geometry[2]).any()))
        previous["dt_s"][:] = -1
        with self.assertRaisesRegex(ValueError, "causal"):
            temporal_sampling_geometry(data["intrinsics"], data["extrinsics"], data["distortion"],
                previous, (4, 8), (8, 16), torch.tensor([10.]), 3, 1.1)

    def test_zero_gates_preserve_v2_then_real_history_changes_output(self):
        torch.manual_seed(9)
        data, base = inputs(), GeometryDepthFrontend(16, options()).eval()
        module = TemporalGeometryDepthFrontend(16, temporal_options()).eval()
        result = module.load_state_dict(base.state_dict(), strict=False)
        self.assertTrue(all(key.startswith("temporal_blocks.") for key in result.missing_keys))
        levels = [torch.randn(1, 8, 16, 2, 4) for _ in range(4)]
        previous = history(data) | {"feature_levels": [torch.randn_like(x) for x in levels]}
        def run():
            return module(levels, data["intrinsics"], data["extrinsics"], data["distortion"], (8, 16), history=previous)
        with torch.no_grad():
            reference = base(levels, data["intrinsics"], data["extrinsics"], data["distortion"], (8, 16))
            torch.testing.assert_close(run()["depth"], reference["depth"], rtol=0, atol=0)
            for block in module.temporal_blocks:
                block.gate.fill_(.2)
            first = run()["depth"]
            previous["feature_levels"] = [torch.randn_like(x) * 3 for x in levels]
            second = run()["depth"]
            self.assertGreater(float((first - second).abs().sum()), 1e-5)
            previous["valid"].zero_()
            previous["dt_s"].zero_()
            masked = run()
            torch.testing.assert_close(masked["depth"], reference["depth"], rtol=0, atol=0)
            self.assertEqual(float(masked["depth_temporal_coverage"]), 0.)

    def test_gate_then_attention_receive_gradients_with_checkpointing(self):
        torch.manual_seed(11)
        settings, data = temporal_options(), inputs()
        settings["gradient_checkpointing"] = True
        module = TemporalGeometryDepthFrontend(16, settings)
        levels = [torch.randn(1, 8, 16, 2, 4) for _ in range(4)]
        previous = history(data) | {"feature_levels": [torch.randn_like(x) for x in levels]}
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            result = module(levels, data["intrinsics"], data["extrinsics"], data["distortion"], (8, 16), history=previous)
            result["depth"].mean().backward()
            self.assertGreater(float(module.temporal_blocks[0].gate.grad.abs().sum()), 0.)
            if step:
                self.assertGreater(float(module.temporal_blocks[0].attention.key_value.weight.grad.abs().sum()), 0.)
            optimizer.step()

    def test_missing_history_is_zero_buffer_not_repeated_frame_and_collates(self):
        data = inputs()
        sample = {key: value[0] for key, value in data.items()}
        sample.update(sample_token="t", source_index=0, dense_targets={"depth": None, "seg": None})
        dataset = TemporalSegDepthDataset.__new__(TemporalSegDepthDataset)
        dataset.agent_labels = None
        dataset.history_plans = [{"history_index": None, "source_index": 0, "history_token": None,
            "history_timestamp": None, "dt_s": 0., "reason": "scene_start", "previous_to_current": torch.eye(4).tolist()}]
        with patch.object(SegDepthDataset, "__getitem__", return_value=sample):
            result = dataset[0]
        self.assertFalse(bool(result["history"]["valid"]))
        self.assertEqual(float(result["history"]["images"].abs().sum()), 0.)
        batch = collate_seg_depth([result])
        self.assertEqual(batch["history"]["images"].shape, (1, 8, 3, 8, 16))
        self.assertEqual(batch["history_identity"][0]["reason"], "scene_start")

    def test_explicit_warm_start_frozen_spatial_and_v3_checkpoint_roundtrip(self):
        with patch("quest.model.FrozenDINOv2Backbone", TinyBackbone):
            base = QUESTModel(**model_config()).eval()
        baseline = frozen_spatial_fingerprint(base)
        optimizer = torch.optim.AdamW(configure_dense_training(base, ("depth",)), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
        manifest, initialization = {"split_sha256": "s", "metadata_sha256": "m"}, {"source": "spatial"}
        audits = {split: {"frames": 1, "labels_sha256": split, "teacher_signatures": {"depth": "d"}}
                  for split in ("train", "validation")}
        new_config, new_training = model_config(), training_config()
        new_config["dense_frontend"] = temporal_options()
        new_training["frontend"] = temporal_options()
        temporal_audits = {key: value | {"temporal_context": {"with_history": 1}} for key, value in audits.items()}
        data = inputs()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v2.pt"
            save_dense_checkpoint(path, base, optimizer, scheduler, 5, training_config(), model_config(),
                                  manifest, audits, baseline, initialization)
            source = torch.load(path, map_location="cpu", weights_only=True)
            validate_depth_warm_start(source, new_training, new_config, manifest, temporal_audits, initialization)
            changed = copy.deepcopy(temporal_audits)
            changed["train"]["labels_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "label hashes"):
                validate_depth_warm_start(source, new_training, new_config, manifest, changed, initialization)
            with patch("quest.model.FrozenDINOv2Backbone", TinyBackbone):
                model = QUESTModel(**new_config).eval()
            self.assertGreater(load_depth_warm_start(model, source), 0)
            with torch.no_grad():
                before = base(**data)
                after = model(**data, history=history(data))
            for key in ("agent_cls_logits", "agent_boxes", "map_cls_logits", "map_points", "depth"):
                torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
            trained = torch.optim.AdamW(configure_dense_training(model, ("depth",)), lr=1e-4)
            model.forward_dense(data["images"], data["intrinsics"], data["extrinsics"], data["distortion"],
                                history=history(data))["depth"].mean().backward()
            trained.step()
            self.assertEqual(frozen_spatial_fingerprint(model), baseline)
            model.eval()
            with torch.no_grad():
                after = model(**data, history=history(data))
            for key in ("agent_cls_logits", "agent_boxes", "map_cls_logits", "map_points"):
                torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
            output = Path(directory) / "v3.pt"
            save_dense_checkpoint(output, model, trained, torch.optim.lr_scheduler.CosineAnnealingLR(trained, 5),
                1, new_training, new_config, manifest, temporal_audits, baseline, initialization,
                validation_history=[{"epoch": 1}])
            saved = torch.load(output, map_location="cpu", weights_only=True)
            self.assertEqual(saved["dense_checkpoint_version"], 3)
            self.assertEqual(saved["validation_history"], [{"epoch": 1}])
            with patch("quest.model.FrozenDINOv2Backbone", TinyBackbone):
                restored = load_dense_model(saved)
            self.assertEqual(frozen_spatial_fingerprint(restored), baseline)


if __name__ == "__main__":
    unittest.main()
