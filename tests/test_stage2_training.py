import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quest.teacher_adapters import empty_canonical_agent
from quest.utils import load_yaml_config


SCRIPT = ROOT / "scripts/train_stage2_distill.py"
spec = importlib.util.spec_from_file_location("train_stage2_distill_test", SCRIPT)
stage2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage2)


class TinyAgentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = torch.nn.Parameter(torch.zeros(5))
        self.box = torch.nn.Parameter(torch.zeros(8))
        self.velocity = torch.nn.Parameter(torch.zeros(3))
        self.objectness = torch.nn.Parameter(torch.zeros(16))
        self.offset = torch.nn.Parameter(torch.zeros(16, 2))
        self.forward_calls = 0

    def forward(self, images, intrinsics, extrinsics, ego_state):
        del intrinsics, extrinsics, ego_state
        self.forward_calls += 1
        batch_size = images.shape[0]
        cls = self.cls.reshape(1, 1, 5).expand(batch_size, 4, 5)
        boxes = torch.sigmoid(self.box).reshape(1, 1, 8).expand(batch_size, 4, 8)
        velocity = self.velocity.reshape(1, 1, 3).expand(batch_size, 4, 3)
        zero = self.cls.sum() * 0.0
        return {
            "seg_logits": zero.expand(batch_size, 1),
            "depth": zero.expand(batch_size, 1),
            "proposal_objectness_logits": self.objectness.unsqueeze(0).expand(batch_size, -1),
            "proposal_xy_offsets": self.offset.unsqueeze(0).expand(batch_size, -1, -1),
            "proposal_spatial_std": torch.ones(batch_size),
            "agent_cls_logits": cls,
            "agent_boxes": boxes,
            "agent_velocity": velocity,
            "agent_cls_logits_layers": cls.unsqueeze(0),
            "agent_boxes_layers": boxes.unsqueeze(0),
            "agent_velocity_layers": velocity.unsqueeze(0),
            "map_cls_logits": zero.expand(batch_size, 1, 5),
            "map_points": zero.expand(batch_size, 1, 2, 2),
            "bev_visible_ratio": torch.ones(batch_size),
        }


def make_agent(valid=True):
    agent = empty_canonical_agent(64)
    if valid:
        agent["labels"][0] = 0
        agent["boxes_metric"][0] = torch.tensor([0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0])
        agent["scores"][0] = 0.9
        agent["valid_mask"][0] = True
    agent["class_support_mask"] = torch.tensor([True, True, False, False])
    return {key: value.unsqueeze(0) for key, value in agent.items()}


def make_batch(token):
    return {
        "images": torch.zeros(1, 8, 3, 2, 2),
        "intrinsics": torch.eye(3).reshape(1, 1, 3, 3).expand(1, 8, 3, 3),
        "extrinsics": torch.eye(4).reshape(1, 1, 4, 4).expand(1, 8, 4, 4),
        "ego_state": torch.zeros(1, 9),
        "sample_token": [token],
        "agent_gt": make_agent(valid=False),
        "agent_valid": torch.tensor([False]),
        "soft_labels": {"agent": make_agent(valid=True)},
    }


class Stage2TrainingTest(unittest.TestCase):
    def test_default_training_config(self):
        config = load_yaml_config(ROOT / "configs/stage2_distill.yaml")
        self.assertEqual(config["distill"]["num_samples"], 100)
        self.assertEqual(config["distill"]["epochs"], 5)
        self.assertEqual(config["supervision_source"], "teacher_only")
        self.assertEqual(config["distill"]["proposal_warmup_epochs"], 1)
        self.assertEqual(
            stage2.initial_trained_class_support_mask("teacher_only").tolist(),
            [True, True, False, False],
        )
        self.assertTrue(
            stage2.initial_trained_class_support_mask("hard_gt_only").all()
        )
        self.assertTrue(stage2.initial_trained_class_support_mask("hybrid").all())

    def test_epoch_traverses_loader_and_updates_model(self):
        model = TinyAgentModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        initial = model.objectness.detach().clone()
        average_total, average_agent = stage2.train_one_epoch(
            model=model,
            loader=[make_batch("token-1"), make_batch("token-2")],
            optimizer=optimizer,
            device=torch.device("cpu"),
            loss_config={
                "tasks": {"seg": False, "depth": False, "agent": True, "map": False},
                "task_weights": {"seg": 0.0, "depth": 0.0, "agent": 1.0, "map": 0.0},
                "agent": {"bev_h": 4, "bev_w": 4, "aux_layer_weights": [1.0]},
            },
            supervision_source="teacher_only",
            proposal_warmup_epochs=1,
            epoch=2,
            epochs=2,
        )
        self.assertEqual(model.forward_calls, 2)
        self.assertGreater(average_total, 0.0)
        self.assertGreater(average_agent, 0.0)
        self.assertFalse(torch.equal(initial, model.objectness.detach()))

    def test_checkpoint_contains_v3_training_state(self):
        model = TinyAgentModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/quest_stage2_agent.pt"
            stage2.save_training_checkpoint(
                path,
                model,
                optimizer,
                epoch=5,
                trained_class_support_mask=torch.tensor([True, True, False, False]),
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 5)
            self.assertEqual(checkpoint["architecture_version"], 3)
            self.assertIn("model_state_dict", checkpoint)
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertEqual(
                checkpoint["trained_class_support_mask"].tolist(),
                [True, True, False, False],
            )


if __name__ == "__main__":
    unittest.main()
