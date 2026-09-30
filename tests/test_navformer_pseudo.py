from pathlib import Path

import torch

from quest.openscene_dataset import OpenSceneMetadataDataset
from quest.teachers import navformer_output_to_quest
from quest.utils import load_yaml_config
from scripts.export_navformer_pseudo import build_payload, save_payload


def test_navformer_class_mapping_and_box_conversion():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 0.0, 10.0, 5.0, 4.0, 0.0, 2.0, -4.0],
            [10.0, -20.0, 2.0, 2.0, 1.0, 2.0, 1.5707963, 0.0, 0.0],
            [20.0, 30.0, -2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [4.0, 5.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    result = navformer_output_to_quest(
        {
            "boxes_3d": boxes,
            "scores_3d": torch.arange(7, dtype=torch.float32),
            "labels_3d": torch.arange(7),
        }
    )

    assert result["labels"][:7].tolist() == [3, 3, 3, 2, 1, 0, 0]
    torch.testing.assert_close(result["boxes"][6, :6], torch.tensor([0.5] * 6))
    torch.testing.assert_close(result["boxes"][6, 6:], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(result["velocity"][6], torch.tensor([0.1, -0.2, 0.0]))
    assert result["labels"][7:].eq(-1).all()
    assert result["boxes"][7:].eq(0).all()
    assert result["velocity"][7:].eq(0).all()


def test_navformer_conversion_filters_range_and_keeps_top_64():
    boxes = torch.zeros(70, 9)
    boxes[:, 3:6] = 1.0
    boxes[:, 7] = torch.arange(70, dtype=torch.float32)
    boxes[0, 0] = 51.0
    scores = torch.arange(70, dtype=torch.float32)
    labels = torch.zeros(70, dtype=torch.long)

    result = navformer_output_to_quest(
        {"boxes_3d": boxes, "scores_3d": scores, "labels_3d": labels}
    )

    assert int((result["labels"] >= 0).sum()) == 64
    assert result["labels"].eq(0).all()
    assert abs(result["velocity"][0, 0].item() - 69.0 / 20.0) < 1e-6
    assert abs(result["velocity"][-1, 0].item() - 6.0 / 20.0) < 1e-6


def test_payload_is_token_aligned_and_stage2_compatible(tmp_path: Path):
    payload = build_payload(
        "token-a",
        torch.empty(0, 9),
        torch.empty(0),
        torch.empty(0, dtype=torch.long),
    )
    path = save_payload(payload, tmp_path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)

    assert loaded["token"] == "token-a"
    assert tuple(loaded["agent"]["labels"].shape) == (64,)
    assert tuple(loaded["agent"]["boxes"].shape) == (64, 8)
    assert tuple(loaded["agent"]["velocity"].shape) == (64, 3)
    assert not path.with_suffix(".pt.tmp").exists()

    dataset = OpenSceneMetadataDataset.__new__(OpenSceneMetadataDataset)
    dataset.soft_labels_root = tmp_path
    dataset.max_agent_instances = 64
    loaded_by_dataset = dataset._load_soft_labels("token-a")
    assert loaded_by_dataset["agent"]["labels"].shape == (64,)


def test_stage2_uses_dedicated_navformer_soft_label_directory():
    config = load_yaml_config(
        Path(__file__).resolve().parents[1] / "configs/stage2_distill.yaml"
    )
    assert config["use_hard_gt"] is False
    assert config["paths"]["soft_labels_dir"] == "data/soft_labels_navformer"
