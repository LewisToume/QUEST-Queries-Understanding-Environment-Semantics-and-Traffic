import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from quest.agent_training import load_checkpoint_cpu
from quest.map_teacher import TEACHER_ALIGNMENT_VERSION, TEACHER_COORDINATE_FRAME, TEACHER_MAP_SCHEMA_VERSION
from quest.map_training import (
    STAGE3_GT_ONLY_NAME, STAGE3_NAME, configure_stage3, load_stage3_checkpoint,
    mixed_stage3_loss, save_stage3_checkpoint, stage3_kd_enabled,
)
from quest.nuplan_map_locator import MAP_LAYER_AUDIT_VERSION
from quest.nuplan_relation_audit import BASELINE_RELATION_AUDIT_VERSION
from quest.stage3_dataset import Stage3JoinedDataset
from quest.vector_map_labels import MAP_HEIGHT_REFERENCE, VECTOR_SEMANTICS_VERSION


def config(kd_enabled):
    return {
        "stage": STAGE3_NAME if kd_enabled else STAGE3_GT_ONLY_NAME,
        "loss_weights": {"lambda_map_gt": 1.0, "lambda_map_kd": 1.0 if kd_enabled else 0.0,
                         "lambda_agent": 0.5},
        "train": {"bev_lr": 1e-5, "map_lr": 1e-4, "agent_lr": 3e-5,
                  "weight_decay": 1e-4},
        "map_loss": {}, "agent_train": {}, "agent_loss": {},
    }


class FakeLift(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2)
        self.ego_mlp = nn.Linear(2, 2)
        self.x_range = (-50.0, 50.0)
        self.y_range = (-50.0, 50.0)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.geometry_lift = FakeLift()
        for name in ("bev_encoder", "agent_proposal_head", "agent_decoder",
                     "agent_head", "map_decoder", "map_head"):
            setattr(self, name, nn.Linear(2, 2))


class FakeImages:
    def __init__(self, agent_dir):
        self.infos = [{"token": "tok"}]
        self.soft_labels_root = agent_dir

    def __getitem__(self, index):
        return {"sample_token": "tok", "soft_labels": {"agent": {}}}


def provenance():
    return {"vector_semantics_version": VECTOR_SEMANTICS_VERSION,
            "num_points": 20, "map_version": "test-map", "min_length_m": 1.0,
            "map_height_reference": MAP_HEIGHT_REFERENCE,
            "map_cast_audit_version": BASELINE_RELATION_AUDIT_VERSION,
            "map_layer_audit_version": MAP_LAYER_AUDIT_VERSION}


class Stage3GTOnlyTest(unittest.TestCase):
    def test_gt_only_skips_map_teacher_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent_dir, vector_dir = root / "agent", root / "vector"
            agent_dir.mkdir()
            vector_dir.mkdir()
            (agent_dir / "tok.pt").touch()
            (vector_dir / "tok.pt").touch()
            images = FakeImages(agent_dir)
            with patch("quest.stage3_dataset.OpenSceneMetadataDataset", return_value=images):
                dataset = Stage3JoinedDataset({}, [5000], agent_dir, vector_dir, None, None,
                                              (-50, -50, 50, 50), 32, 32,
                                              vector_provenance=provenance())
            self.assertEqual(len(dataset), 1)
            self.assertIsNone(dataset.teacher_dir)
            vector = provenance()
            with patch("quest.stage3_dataset.load_record", side_effect=[{"token": "tok"}, vector]), \
                    patch("quest.stage3_dataset.validate_vector_record"):
                sample = dataset[0]
            self.assertIn("vector_target", sample)
            self.assertNotIn("teacher_map_aligned", sample)
            with patch("quest.stage3_dataset.load_record", return_value={"token": "other"}):
                with self.assertRaisesRegex(ValueError, "index/token mismatch"):
                    dataset[0]

    def test_mode_and_optimizer_have_no_raster_group(self):
        settings = config(False)
        self.assertFalse(stage3_kd_enabled(settings))
        model = FakeModel()
        optimizer = configure_stage3(model, None, settings["train"])
        self.assertEqual(len(optimizer.param_groups), 3)
        self.assertFalse(any(p.requires_grad for p in model.geometry_lift.ego_mlp.parameters()))
        settings["stage"] = STAGE3_NAME
        with self.assertRaisesRegex(ValueError, "stage/lambda_map_kd mismatch"):
            stage3_kd_enabled(settings)

    def test_zero_kd_never_calls_teacher_loss(self):
        prediction = {"map_cls_logits": torch.ones(1, 1, 4, requires_grad=True),
                      "map_points": torch.ones(1, 1, 20, 2)}
        vector_loss = prediction["map_cls_logits"].mean()
        with patch("quest.map_training.vector_map_loss", return_value={
            "vector_map_gt_loss": vector_loss,
        }), patch("quest.map_training.compute_stage2_agent_loss", return_value={
            "total_agent_loss": vector_loss * 2,
        }), patch("quest.map_training.soft_map_distillation_loss") as kd:
            losses = mixed_stage3_loss(FakeModel(), prediction, [], None, None, None, None,
                                       {}, config(False))
        kd.assert_not_called()
        self.assertEqual(float(losses["map_kd_loss"]), 0.0)
        self.assertEqual(float(losses["total_loss"]), 2.0)
        losses["total_loss"].backward()
        self.assertIsNotNone(prediction["map_cls_logits"].grad)

    def test_hybrid_still_requires_teacher_targets(self):
        prediction = {"map_cls_logits": torch.ones(1, 1, 4),
                      "map_points": torch.ones(1, 1, 20, 2)}
        with patch("quest.map_training.vector_map_loss", return_value={
            "vector_map_gt_loss": torch.tensor(1.0),
        }), patch("quest.map_training.soft_map_distillation_loss") as kd:
            with self.assertRaisesRegex(ValueError, "requires a raster head"):
                mixed_stage3_loss(FakeModel(), prediction, [], None, None, None, None,
                                  {}, config(True))
        kd.assert_not_called()

    def test_checkpoint_modes_are_explicit_and_old_hybrid_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel()
            gt_config = config(False)
            gt_optimizer = configure_stage3(model, None, gt_config["train"])
            path = Path(directory) / "gt.pt"
            save_stage3_checkpoint(path, model, None, gt_optimizer, 1, gt_config, None, provenance())
            checkpoint = load_checkpoint_cpu(path)
            self.assertFalse(checkpoint["map_kd_enabled"])
            self.assertNotIn("teacher_channel_metadata", checkpoint)
            self.assertEqual(load_stage3_checkpoint(FakeModel(), None, checkpoint,
                                                    expected_map_kd_enabled=False), 1)
            with self.assertRaisesRegex(ValueError, "KD mode differs"):
                load_stage3_checkpoint(FakeModel(), nn.Linear(2, 2), checkpoint,
                                       expected_map_kd_enabled=True)

            hybrid_config = config(True)
            raster_head = nn.Linear(2, 2)
            hybrid_optimizer = configure_stage3(model, raster_head, hybrid_config["train"])
            teacher = {"teacher_channel_support_mask": [True],
                       "teacher_schema_version": TEACHER_MAP_SCHEMA_VERSION,
                       "teacher_alignment_version": TEACHER_ALIGNMENT_VERSION,
                       "teacher_coordinate_frame": TEACHER_COORDINATE_FRAME}
            save_stage3_checkpoint(path, model, raster_head, hybrid_optimizer, 2,
                                   hybrid_config, teacher, provenance())
            old_checkpoint = load_checkpoint_cpu(path)
            del old_checkpoint["map_kd_enabled"]
            self.assertEqual(load_stage3_checkpoint(FakeModel(), nn.Linear(2, 2), old_checkpoint,
                                                    expected_map_kd_enabled=True), 2)


if __name__ == "__main__":
    unittest.main()
