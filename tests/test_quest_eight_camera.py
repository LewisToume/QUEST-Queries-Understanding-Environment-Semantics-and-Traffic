import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from quest.model import DEFAULT_CAMERA_NAMES, QUESTModel
from quest.openscene_dataset import OPENSCENE_CAMERA_NAMES, OpenSceneMetadataDataset
from quest.utils import load_yaml_config


EXPECTED_CAMERAS = (
    "CAM_F0",
    "CAM_L0",
    "CAM_R0",
    "CAM_L1",
    "CAM_R1",
    "CAM_L2",
    "CAM_R2",
    "CAM_B0",
)


class PatchTokenBackbone(torch.nn.Module):
    hidden_dim = 32
    patch_size = 14

    def forward(self, images):
        batch, _, height, width = images.shape
        patch_count = (height // self.patch_size) * (width // self.patch_size)
        return torch.zeros(
            batch,
            patch_count + 1,
            self.hidden_dim,
            device=images.device,
            dtype=images.dtype,
        )


def make_metadata(root: Path) -> tuple[Path, Path, list[np.ndarray]]:
    camera_root = root / "camera_root"
    cameras = {}
    extrinsics = []
    for index, name in enumerate(EXPECTED_CAMERAS):
        relative = "dataset/sensor_blobs/scene/{}/image.jpg".format(name)
        image_path = (
            camera_root
            / "sensor_blobs/mini/scene"
            / name
            / "image.jpg"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (200, 100), color=(index, index, index)).save(image_path)
        rotation = np.eye(3, dtype=np.float32)
        translation = np.array([index, index + 1, index + 2], dtype=np.float32)
        cameras[name] = {
            "data_path": relative,
            "cam_intrinsic": np.array(
                [[100.0, 0.0, 50.0], [0.0, 200.0, 25.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            "sensor2lidar_rotation": rotation,
            "sensor2lidar_translation": translation,
        }
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        extrinsics.append(matrix)

    metadata_path = root / "metadata.pkl"
    with metadata_path.open("wb") as stream:
        pickle.dump(
            {
                "infos": [
                    {
                        "token": "sample",
                        "cams": cameras,
                        "ego2global": np.eye(4),
                        "can_bus": np.zeros(9),
                    }
                ]
            },
            stream,
        )
    return metadata_path, camera_root, extrinsics


def test_dataset_returns_native_eight_camera_order_and_rectangular_images(tmp_path):
    metadata_path, camera_root, expected_extrinsics = make_metadata(tmp_path)
    dataset = OpenSceneMetadataDataset(metadata_path, camera_root)

    sample = dataset[0]

    assert OPENSCENE_CAMERA_NAMES == EXPECTED_CAMERAS
    assert sample["camera_names"] == list(EXPECTED_CAMERAS)
    assert tuple(sample["images"].shape) == (8, 3, 252, 448)
    expected_intrinsic = torch.tensor(
        [[224.0, 0.0, 112.0], [0.0, 504.0, 63.0], [0.0, 0.0, 1.0]]
    )
    torch.testing.assert_close(sample["intrinsics"][0], expected_intrinsic)
    torch.testing.assert_close(
        sample["extrinsics"], torch.from_numpy(np.stack(expected_extrinsics))
    )
    assert sample["images"].shape[-2:] == (252, 448)


def test_student_configs_use_eight_cameras_and_16_by_9_input():
    root = Path(__file__).resolve().parents[1]
    model_config = load_yaml_config(root / "configs/model.yaml")["model"]
    dataset_config = load_yaml_config(root / "configs/stage1.yaml")["dataset"]

    assert tuple(model_config["camera_names"]) == EXPECTED_CAMERAS
    assert model_config["architecture_version"] == 2
    assert model_config["hidden_dim"] == 384
    assert model_config["bev_layers"] == 4
    assert model_config["agent_decoder_layers"] == 4
    assert model_config["map_decoder_layers"] == 2
    assert tuple(dataset_config["camera_names"]) == EXPECTED_CAMERAS
    assert dataset_config["image_size"] == [252, 448]


def test_quest_forward_uses_eight_cameras_and_18_by_32_patch_grid():
    with patch(
        "quest.model.FrozenDINOv2Backbone", return_value=PatchTokenBackbone()
    ):
        model = QUESTModel(
            hidden_dim=32,
            bev_h=4,
            bev_w=4,
            x_range=(-2.0, 2.0),
            y_range=(-2.0, 2.0),
            z_anchors=(1.0,),
            bev_layers=1,
            bev_attention_heads=4,
            bev_ffn_dim=64,
            agent_decoder_layers=1,
            map_decoder_layers=1,
            decoder_attention_heads=4,
            decoder_ffn_dim=64,
            dropout=0.0,
            N_agent=4,
            N_map=3,
            seg_size=(8, 12),
            depth_size=(8, 12),
            C_agent=4,
            C_map=4,
            P=5,
        ).eval()

    images = torch.zeros(1, 8, 3, 252, 448)
    intrinsic = torch.tensor(
        [[20.0, 0.0, 224.0], [0.0, 20.0, 126.0], [0.0, 0.0, 1.0]]
    )
    intrinsics = intrinsic.reshape(1, 1, 3, 3).expand(1, 8, 3, 3)
    extrinsics = torch.eye(4).reshape(1, 1, 4, 4).expand(1, 8, 4, 4)
    ego_state = torch.zeros(1, 9)
    with torch.no_grad():
        encoded = model.encode_image(images, intrinsics, extrinsics, ego_state)
        outputs = model(images, intrinsics, extrinsics, ego_state)

    assert DEFAULT_CAMERA_NAMES == EXPECTED_CAMERAS
    assert model.num_cameras == 8
    assert not hasattr(model, "fusion")
    assert model.agent_decoder is not model.map_decoder
    assert encoded["patch_grid_size"] == (18, 32)
    assert tuple(reversed(encoded["patch_grid_size"])) == (32, 18)
    assert tuple(encoded["backbone_patch_tokens"].shape) == (1, 8, 576, 32)
    assert tuple(encoded["backbone_feature_maps"].shape) == (1, 8, 32, 18, 32)
    assert tuple(encoded["lifted_bev_tokens"].shape) == (1, 16, 32)
    assert tuple(outputs["seg_logits"].shape) == (1, 8, 6, 8, 12)
    assert tuple(outputs["depth"].shape) == (1, 8, 1, 8, 12)
    assert tuple(outputs["agent_cls_logits"].shape) == (1, 4, 5)
    assert tuple(outputs["map_points"].shape) == (1, 3, 5, 2)
