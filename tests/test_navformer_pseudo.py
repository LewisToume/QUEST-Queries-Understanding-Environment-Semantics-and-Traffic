from pathlib import Path

import torch

from quest.openscene_dataset import OpenSceneMetadataDataset
from quest.teachers import navformer_output_to_quest
from quest.utils import load_yaml_config
from scripts.export_navformer_pseudo import build_payload, save_payload


def visible_context(camera_count=8):
    matrix = torch.zeros(4, 4)
    matrix[0, 3] = 50.0
    matrix[1, 3] = 50.0
    matrix[2, 3] = 1.0
    matrix[3, 3] = 1.0
    return matrix.unsqueeze(0).expand(camera_count, 4, 4), [(100, 100, 3)] * camera_count


def convert(boxes, scores, labels):
    lidar2img, image_shapes = visible_context()
    return navformer_output_to_quest(
        {"boxes_3d": boxes, "scores_3d": scores, "labels_3d": labels},
        lidar2img=lidar2img,
        image_shapes=image_shapes,
    )


def test_navformer_keeps_metric_targets_scores_and_verified_classes_only():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 0.0, 10.0, 5.0, 4.0, 0.0, 2.0, -4.0],
            [10.0, -20.0, 2.0, 2.0, 1.0, 2.0, 1.5, 0.0, 0.0],
            [1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    result = convert(boxes, torch.tensor([0.8, 0.7, 0.9]), torch.tensor([0, 2, 3]))
    assert result["labels"][:2].tolist() == [0, 1]
    torch.testing.assert_close(result["boxes_metric"][:2], boxes[:2, :7])
    torch.testing.assert_close(result["velocity_mps"][0], torch.tensor([2.0, -4.0, 0.0]))
    torch.testing.assert_close(result["scores"][:2], torch.tensor([0.8, 0.7]))
    assert result["class_support_mask"].tolist() == [True, True, False, False]
    assert result["valid_mask"][:2].all()
    assert result["labels"][2:].eq(-1).all()


def test_navformer_filters_low_score_distance_z_and_unobservable_boxes():
    boxes = torch.zeros(7, 9)
    boxes[:, 3:6] = 1.0
    boxes[2, :2] = torch.tensor([30.0, 40.0])
    boxes[3, :2] = torch.tensor([30.0, 40.1])
    boxes[4, 2] = 5.0
    boxes[5, 2] = 5.1
    lidar2img, image_shapes = visible_context()
    result = navformer_output_to_quest(
        {
            "boxes_3d": boxes,
            "scores_3d": torch.tensor([0.24, 0.25, 0.9, 0.9, 0.9, 0.9, 0.8]),
            "labels_3d": torch.tensor([0, 0, 2, 2, 0, 2, 4]),
        },
        lidar2img=lidar2img,
        image_shapes=image_shapes,
    )
    assert int(result["valid_mask"].sum()) == 3
    assert sorted(result["labels"][:3].tolist()) == [0, 0, 1]


def test_navformer_keeps_top_64_by_score():
    boxes = torch.zeros(70, 9)
    boxes[:, 2] = 1.0
    boxes[:, 3:6] = 1.0
    boxes[:, 7] = torch.arange(70, dtype=torch.float32)
    scores = 0.3 + torch.arange(70, dtype=torch.float32) / 100.0
    result = convert(boxes, scores, torch.zeros(70, dtype=torch.long))
    assert int(result["valid_mask"].sum()) == 64
    assert result["labels"][result["valid_mask"]].eq(0).all()
    assert result["scores"][0] > result["scores"][1]


def test_payload_is_canonical_token_aligned_and_dataset_compatible(tmp_path: Path):
    lidar2img, image_shapes = visible_context()
    payload = build_payload(
        "token-a",
        torch.empty(0, 9),
        torch.empty(0),
        torch.empty(0, dtype=torch.long),
        lidar2img,
        image_shapes,
    )
    path = save_payload(payload, tmp_path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["token"] == "token-a"
    assert tuple(loaded["agent"]["boxes_metric"].shape) == (64, 7)
    assert tuple(loaded["agent"]["velocity_mps"].shape) == (64, 3)
    assert tuple(loaded["agent"]["scores"].shape) == (64,)
    dataset = OpenSceneMetadataDataset.__new__(OpenSceneMetadataDataset)
    dataset.soft_labels_root = tmp_path
    dataset.max_agent_instances = 64
    loaded_by_dataset = dataset._load_soft_labels("token-a")
    assert loaded_by_dataset["agent"]["class_support_mask"].tolist() == [True, True, False, False]


def test_stage2_uses_teacher_only_canonical_navformer_labels():
    config = load_yaml_config(
        Path(__file__).resolve().parents[1] / "configs/stage2_distill.yaml"
    )
    assert config["supervision_source"] == "teacher_only"
    assert config["distill"]["proposal_warmup_epochs"] == 1
    assert config["paths"]["soft_labels_dir"] == "data/soft_labels_navformer"
