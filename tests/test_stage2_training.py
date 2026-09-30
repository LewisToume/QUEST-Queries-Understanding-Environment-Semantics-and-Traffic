import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
        self.forward_calls = 0

    def forward(self, images, intrinsics, extrinsics, ego_state):
        del intrinsics, extrinsics, ego_state
        self.forward_calls += 1
        batch_size = images.shape[0]
        zero = self.cls.sum() * 0.0
        return {
            "seg_logits": zero.expand(batch_size, 1),
            "depth": zero.expand(batch_size, 1),
            "agent_cls_logits": self.cls.reshape(1, 1, 5).expand(
                batch_size, 4, 5
            ),
            "agent_boxes": torch.sigmoid(self.box).reshape(1, 1, 8).expand(
                batch_size, 4, 8
            ),
            "agent_velocity": self.velocity.reshape(1, 1, 3).expand(
                batch_size, 4, 3
            ),
            "map_cls_logits": zero.expand(batch_size, 1, 5),
            "map_points": zero.expand(batch_size, 1, 2, 2),
        }


def make_batch(token):
    labels = torch.full((1, 64), -1, dtype=torch.long)
    labels[0, 0] = 0
    boxes = torch.zeros(1, 64, 8)
    boxes[0, 0, :6] = 0.5
    boxes[0, 0, 7] = 1.0
    velocity = torch.zeros(1, 64, 3)
    return {
        "images": torch.zeros(1, 8, 3, 2, 2),
        "intrinsics": torch.eye(3).reshape(1, 1, 3, 3).expand(1, 8, 3, 3),
        "extrinsics": torch.eye(4).reshape(1, 1, 4, 4).expand(1, 8, 4, 4),
        "ego_state": torch.zeros(1, 9),
        "sample_token": [token],
        "agent_gt": {
            "labels": torch.full((1, 64), -1, dtype=torch.long),
            "boxes": torch.zeros(1, 64, 8),
            "velocity": torch.zeros(1, 64, 3),
        },
        "agent_valid": torch.tensor([False]),
        "soft_labels": {
            "agent": {"labels": labels, "boxes": boxes, "velocity": velocity}
        },
    }


class Stage2TrainingTest(unittest.TestCase):
    def test_default_training_config(self):
        config = load_yaml_config(ROOT / "configs/stage2_distill.yaml")
        self.assertEqual(config["distill"]["num_samples"], 100)
        self.assertEqual(config["distill"]["epochs"], 5)
        self.assertEqual(
            config["paths"]["checkpoint_path"],
            "checkpoints/quest_stage2_agent.pt",
        )

    def test_epoch_traverses_loader_and_updates_model(self):
        model = TinyAgentModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        initial_cls = model.cls.detach().clone()
        average_total, average_agent = stage2.train_one_epoch(
            model=model,
            loader=[make_batch("token-1"), make_batch("token-2")],
            optimizer=optimizer,
            device=torch.device("cpu"),
            loss_config={
                "tasks": {
                    "seg": False,
                    "depth": False,
                    "agent": True,
                    "map": False,
                },
                "task_weights": {
                    "seg": 0.0,
                    "depth": 0.0,
                    "agent": 1.0,
                    "map": 0.0,
                },
            },
            use_hard_gt=False,
            epoch=1,
            epochs=1,
        )
        self.assertEqual(model.forward_calls, 2)
        self.assertGreater(average_total, 0.0)
        self.assertGreater(average_agent, 0.0)
        self.assertFalse(torch.equal(initial_cls, model.cls.detach()))

    def test_checkpoint_contains_training_state(self):
        model = TinyAgentModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/quest_stage2_agent.pt"
            stage2.save_training_checkpoint(path, model, optimizer, epoch=5)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 5)
            self.assertIn("model_state_dict", checkpoint)
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
