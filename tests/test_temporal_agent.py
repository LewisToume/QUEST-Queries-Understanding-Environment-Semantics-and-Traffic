"""Synthetic user-run tests for selective Agent temporal integration; no teacher runtime."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from quest.model import QUESTModel
from quest.seg_depth_training import (configure_dense_training, frozen_spatial_fingerprint,
    load_depth_warm_start, load_dense_model, save_dense_checkpoint, validate_depth_warm_start)
from quest.temporal_agent import TemporalAgentBEV
from quest.temporal_agent_training import (empty_agent_agreement, summarize_agent_agreement,
    temporal_agent_loss, update_agent_agreement)
from quest.utils import load_yaml_config
from test_geometry_depth_v2 import TinyBackbone, inputs, model_config, training_config
from test_temporal_depth_v3 import history, temporal_options


def agent_options():
    return dict(enabled=True, layers=1, heads=4, attention_dim=16)


class TemporalAgentTest(unittest.TestCase):
    def test_motion_sampling_zero_gate_real_history_and_masked_fallback(self):
        module, data = TemporalAgentBEV(16, agent_options(), 1.1).eval(), inputs()
        previous, current = torch.randn(1, 16, 2, 4), torch.randn(1, 16, 2, 4)
        context = history(data)
        context["previous_to_current"][:, 0, 3] = -1.
        with torch.no_grad():
            with patch.object(module.attention[0], "forward", wraps=module.attention[0].forward) as call:
                _, output, _ = module(current, previous, context, (-2., 2.), (-1., 1.))
                geometry = call.call_args.args[2]
            torch.testing.assert_close(geometry[1][0][0, 0, 0, 4, 0], torch.tensor(-.25))
            torch.testing.assert_close(output, current, atol=0, rtol=0)
            module.gates.fill_(.2)
            _, a, _ = module(current, previous, context, (-2., 2.), (-1., 1.))
            _, b, _ = module(current, torch.randn_like(previous) * 3, context, (-2., 2.), (-1., 1.))
            self.assertGreater(float((a - b).abs().sum()), 1e-5)
            context["valid"].zero_()
            context["dt_s"].zero_()
            _, output, coverage = module(current, previous, context, (-2., 2.), (-1., 1.))
            torch.testing.assert_close(output, current, atol=0, rtol=0)
            self.assertEqual(float(coverage), 0.)

    def test_joint_loss_map_preservation_and_checkpoint_roundtrip(self):
        with patch("quest.model.FrozenDINOv2Backbone", TinyBackbone):
            base = QUESTModel(**model_config()).eval()
        baseline = frozen_spatial_fingerprint(base)
        source_optimizer = torch.optim.AdamW(configure_dense_training(base, ["depth"]), lr=1e-4)
        source_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(source_optimizer, 5)
        new_model, config = model_config(), training_config()
        settings = temporal_options()
        settings["temporal"]["agent_bev"] = agent_options()
        new_model["dense_frontend"], config["frontend"] = settings, settings
        recipe = load_yaml_config(Path(__file__).resolve().parents[1] / "configs/stage2_agent.yaml")
        recipe["agent_loss"]["aux_layer_weights"] = [1.]
        config["temporal_agent"] = dict(enabled=True, loss_weight=.5, recipe=recipe)
        manifest = dict(split_sha256="s", metadata_sha256="m")
        audits = {split: dict(frames=1, labels_sha256=split, teacher_signatures={"depth": "d"})
                  for split in ("train", "validation")}
        joined = {key: value | {"temporal_context": {}, "agent_temporal_labels": {"frames": 1}}
                  for key, value in audits.items()}
        data, context = inputs(), history(inputs())
        agent = dict(labels=torch.full((64,), -1, dtype=torch.long), boxes_metric=torch.zeros(64, 7),
            velocity_mps=torch.zeros(64, 3), scores=torch.zeros(64),
            class_support_mask=torch.tensor([True, True, False, False]), valid_mask=torch.zeros(64, dtype=torch.bool))
        agent["labels"][:2] = torch.tensor([0, 1])
        agent["valid_mask"][:2] = True
        agent["scores"][:2] = 1.
        agent["boxes_metric"][:2, 3:6] = 1.
        batch = dict(sample_token=["t"], soft_labels=[dict(agent=agent)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            save_dense_checkpoint(path, base, source_optimizer, source_scheduler, 5, training_config(),
                                  model_config(), manifest, audits, baseline, {})
            source = torch.load(path, map_location="cpu", weights_only=True)
            validate_depth_warm_start(source, config, new_model, manifest, joined, {})
            with patch("quest.model.FrozenDINOv2Backbone", TinyBackbone):
                model = QUESTModel(**new_model).eval()
            load_depth_warm_start(model, source)
            with torch.no_grad():
                before, initial = base(**data), model(**data, history=context)
            for key in ("agent_cls_logits", "agent_boxes", "map_cls_logits", "map_points", "depth"):
                torch.testing.assert_close(before[key], initial[key], rtol=0, atol=0)
            optimizer = torch.optim.AdamW(configure_dense_training(model, ["depth"], train_temporal_agent=True), lr=1e-3)
            prediction = model(**data, history=context)
            loss = temporal_agent_loss(model, prediction, batch, config, torch.device("cpu")) + prediction["depth"].mean()
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            self.assertGreater(float(model.dense_frontend.agent_temporal.gates.grad.abs().sum()), 0.)
            optimizer.step()
            self.assertEqual(frozen_spatial_fingerprint(model), baseline)
            model.eval()
            masked = {**context, "valid": torch.tensor([False]), "dt_s": torch.tensor([0.])}
            with torch.no_grad():
                after, fallback = model(**data, history=context), model(**data, history=masked)
            for key in ("map_cls_logits", "map_points"):
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
            for key in ("agent_cls_logits", "agent_boxes", "agent_velocity"):
                torch.testing.assert_close(before[key], fallback[key], rtol=0, atol=0)
            metrics = empty_agent_agreement()
            update_agent_agreement(metrics, model, after, batch, config, torch.device("cpu"))
            self.assertEqual(summarize_agent_agreement(metrics, config)["classes"]["vehicle"]["teacher_count"], 1)
            output = Path(directory) / "joint.pt"
            save_dense_checkpoint(output, model, optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5),
                1, config, new_model, manifest, joined, baseline, {})
            saved = torch.load(output, map_location="cpu", weights_only=True)
            self.assertTrue(saved["trained_temporal_agent"])
            with patch("quest.model.FrozenDINOv2Backbone", TinyBackbone):
                restored = load_dense_model(saved)
            self.assertIsNotNone(restored.dense_frontend.agent_temporal)


if __name__ == "__main__":
    unittest.main()
