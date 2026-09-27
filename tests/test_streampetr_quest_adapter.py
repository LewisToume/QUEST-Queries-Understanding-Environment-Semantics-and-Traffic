import importlib.util
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quest.heads import AgentHead
from quest.losses import compute_agent_loss
from quest.openscene_dataset import OPENSCENE_CAMERA_NAMES, OpenSceneMetadataDataset


ADAPTER_SCRIPT = ROOT / "scripts/convert_streampetr_to_quest.py"
TRAIN_SCRIPT = ROOT / "scripts/train_stage2_distill.py"


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_script("convert_streampetr_to_quest", ADAPTER_SCRIPT)
stage2 = load_script("train_stage2_distill", TRAIN_SCRIPT)


def raw_payload(token, boxes, scores, labels):
    return {
        "token": token,
        "boxes_3d": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 9),
        "scores_3d": torch.tensor(scores, dtype=torch.float32),
        "labels_3d": torch.tensor(labels, dtype=torch.long),
    }


class StreamPETRQuestAdapterTest(unittest.TestCase):
    def test_all_classes_map_to_quest_taxonomy(self):
        expected = [0, 0, 0, 0, 0, 3, 0, 0, 1, 2]
        self.assertEqual(adapter.STREAM_PETR_TO_QUEST.tolist(), expected)

    def test_box_velocity_threshold_range_and_padding(self):
        payload = raw_payload(
            "token-a",
            [
                [0, 0, 0, 10, 5, 4, np.pi / 2, 4, -2],
                [1, 2, 0, 1, 1, 1, 0, 0, 0],
                [51, 0, 0, 1, 1, 1, 0, 0, 0],
            ],
            [0.9, 0.24, 0.99],
            [0, 8, 9],
        )
        converted = adapter.convert_predictions(payload, score_threshold=0.25)
        agent = converted["agent"]
        self.assertEqual(converted["token"], "token-a")
        self.assertEqual(tuple(agent["labels"].shape), (64,))
        self.assertEqual(tuple(agent["boxes"].shape), (64, 8))
        self.assertEqual(tuple(agent["velocity"].shape), (64, 3))
        self.assertEqual(agent["labels"][0].item(), 0)
        self.assertTrue(
            torch.allclose(
                agent["boxes"][0],
                torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 0.0]),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(agent["velocity"][0], torch.tensor([0.2, -0.1, 0.0]))
        )
        self.assertTrue((agent["labels"][1:] == -1).all())
        self.assertEqual(torch.count_nonzero(agent["boxes"][1:]).item(), 0)
        self.assertEqual(torch.count_nonzero(agent["velocity"][1:]).item(), 0)

    def test_top64_are_sorted_by_score(self):
        count = 70
        boxes = torch.zeros(count, 9)
        boxes[:, 0] = torch.arange(count, dtype=torch.float32) / 100.0
        scores = torch.arange(count, dtype=torch.float32) / 100.0
        payload = {
            "token": "token-b",
            "boxes_3d": boxes,
            "scores_3d": scores,
            "labels_3d": torch.zeros(count, dtype=torch.long),
        }
        agent = adapter.convert_predictions(payload, score_threshold=0.0)["agent"]
        expected_x = (torch.arange(69, 5, -1, dtype=torch.float32) / 100.0 + 50) / 100
        self.assertTrue(torch.allclose(agent["boxes"][:, 0], expected_x))
        self.assertTrue((agent["labels"] == 0).all())

    def test_score_threshold_is_inclusive(self):
        payload = raw_payload(
            "threshold-token",
            [
                [0, 0, 0, 1, 1, 1, 0, 0, 0],
                [1, 0, 0, 1, 1, 1, 0, 0, 0],
            ],
            [0.25, 0.2499],
            [0, 0],
        )
        labels = adapter.convert_predictions(payload, score_threshold=0.25)["agent"][
            "labels"
        ]
        self.assertEqual((labels >= 0).sum().item(), 1)

    def test_convert_file_rejects_token_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "filename-token.pt"
            torch.save(
                raw_payload("payload-token", [], [], []),
                input_path,
            )
            with self.assertRaisesRegex(ValueError, "token mismatch"):
                adapter.convert_file(input_path, root / "output", 0.25)

    def test_dataset_reads_converted_agent_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera_root = root / "cameras"
            cams = {}
            for name in OPENSCENE_CAMERA_NAMES:
                raw_path = "dataset/openscene-v1.0/sensor_blobs/scene/{}/image.jpg".format(
                    name
                )
                image_path = (
                    camera_root
                    / "openscene-v1.0/sensor_blobs/mini/scene"
                    / name
                    / "image.jpg"
                )
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 6)).save(image_path)
                cams[name] = {
                    "data_path": raw_path,
                    "cam_intrinsic": np.eye(3),
                    "sensor2lidar_rotation": np.eye(3),
                    "sensor2lidar_translation": np.zeros(3),
                }
            info = {
                "token": "dataset-token",
                "cams": cams,
                "ego2global": np.eye(4),
                "can_bus": np.zeros(9),
            }
            metadata_path = root / "metadata.pkl"
            with metadata_path.open("wb") as stream:
                pickle.dump({"infos": [info]}, stream)
            soft_root = root / "soft"
            converted = adapter.convert_predictions(
                raw_payload(
                    "dataset-token",
                    [[0, 0, 0, 2, 2, 2, 0, 1, 2]],
                    [0.9],
                    [8],
                )
            )
            soft_root.mkdir()
            torch.save(converted, soft_root / "dataset-token.pt")
            dataset = OpenSceneMetadataDataset(
                metadata_path,
                camera_root,
                image_size=(6, 8),
                max_agent_instances=64,
                soft_labels_root=soft_root,
            )
            sample = dataset[0]
            self.assertEqual(sample["soft_labels"]["agent"]["labels"][0].item(), 1)
            self.assertEqual(
                tuple(sample["soft_labels"]["agent"]["boxes"].shape), (64, 8)
            )

    def test_soft_agent_overrides_hard_gt_and_backpropagates(self):
        soft_agent = adapter.convert_predictions(
            raw_payload(
                "loss-token",
                [[0, 0, 0, 2, 2, 2, 0, 1, 2]],
                [0.9],
                [9],
            )
        )["agent"]
        batch = {
            "images": torch.zeros(1, 8, 3, 2, 2),
            "agent_gt": {
                "labels": torch.zeros(1, 64, dtype=torch.long),
                "boxes": torch.zeros(1, 64, 8),
                "velocity": torch.zeros(1, 64, 3),
            },
            "agent_valid": torch.tensor([True]),
            "soft_labels": {
                "agent": {key: value.unsqueeze(0) for key, value in soft_agent.items()}
            },
        }
        targets = stage2.build_targets(batch, torch.device("cpu"), use_hard_gt=False)
        self.assertEqual(targets["agent_gt"]["labels"][0, 0].item(), 2)
        head = AgentHead(hidden_dim=16, C_agent=4, D_box=8)
        cls_logits, boxes, velocity = head(torch.randn(1, 100, 16))
        losses = compute_agent_loss(cls_logits, boxes, velocity, targets["agent_gt"])
        self.assertTrue(torch.isfinite(losses["agent_loss"]))
        losses["agent_loss"].backward()
        gradients = [parameter.grad for parameter in head.parameters()]
        self.assertTrue(
            any(
                gradient is not None
                and torch.isfinite(gradient).all()
                and torch.count_nonzero(gradient).item() > 0
                for gradient in gradients
            )
        )

    def test_disabled_hard_gt_does_not_fall_back_when_soft_label_is_missing(self):
        batch = {
            "images": torch.zeros(1, 8, 3, 2, 2),
            "agent_gt": {
                "labels": torch.zeros(1, 64, dtype=torch.long),
                "boxes": torch.zeros(1, 64, 8),
                "velocity": torch.zeros(1, 64, 3),
            },
            "agent_valid": torch.tensor([True]),
            "soft_labels": {},
        }
        targets = stage2.build_targets(batch, torch.device("cpu"), use_hard_gt=False)
        self.assertNotIn("agent_gt", targets)
        self.assertFalse(targets["agent_valid"].any())


if __name__ == "__main__":
    unittest.main()
