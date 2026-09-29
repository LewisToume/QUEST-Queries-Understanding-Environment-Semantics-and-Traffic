import numpy as np

from scripts.run_navformer_openscene_teacher import (
    build_camera_geometry,
    build_preprocess_transforms,
    find_transform,
    make_can_bus,
    resolve_camera_path,
)


class FakeCV2:
    @staticmethod
    def getOptimalNewCameraMatrix(intrinsic, distortion, image_size, alpha):
        assert image_size == (1920, 1080)
        assert alpha == 1
        return np.asarray(intrinsic), (0, 0, image_size[0], image_size[1])


def test_resolve_camera_path_maps_sensor_blobs_mini(tmp_path):
    image = tmp_path / "sensor_blobs" / "mini" / "CAM_F0" / "frame.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"jpeg")

    result = resolve_camera_path(
        "dataset/sensor_blobs/CAM_F0/frame.jpg", tmp_path
    )

    assert result == image


def test_camera_geometry_preserves_metadata_order_and_builds_lidar2img(tmp_path):
    cameras = {}
    expected_names = ["CAM_{}".format(index) for index in range(8)]
    intrinsic = np.diag([2.0, 3.0, 1.0])
    for index, name in enumerate(expected_names):
        image = tmp_path / "sensor_blobs" / "mini" / name / "frame.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"jpeg")
        cameras[name] = {
            "data_path": "sensor_blobs/mini/{}/frame.jpg".format(name),
            "sensor2lidar_rotation": np.eye(3),
            "sensor2lidar_translation": np.array([float(index), 0.0, 0.0]),
            "cam_intrinsic": intrinsic,
            "distortion": np.zeros(5),
        }

    geometry = build_camera_geometry({"cams": cameras}, tmp_path, FakeCV2)

    assert geometry["camera_names"] == expected_names
    np.testing.assert_allclose(geometry["lidar2cam"][3][:3, 3], [-3.0, 0.0, 0.0])
    np.testing.assert_allclose(
        geometry["lidar2img"][3],
        np.diag([2.0, 3.0, 1.0, 1.0]) @ geometry["lidar2cam"][3],
    )


def test_make_can_bus_uses_real_pose_and_preserves_motion_channels():
    angle = np.pi / 2.0
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]]
    )
    ego2global = np.eye(4)
    ego2global[:3, :3] = rotation
    ego2global[:3, 3] = [1.0, 2.0, 3.0]
    original = np.arange(18, dtype=np.float32)

    can_bus = make_can_bus({"can_bus": original}, ego2global)

    np.testing.assert_allclose(can_bus[:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(can_bus[7:16], original[7:16])
    np.testing.assert_allclose(can_bus[-2], angle, atol=1e-6)
    np.testing.assert_allclose(can_bus[-1], 90.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(can_bus[3:7]), 1.0, atol=1e-6)


def test_find_transform_reads_nested_test_pipeline():
    pipeline = [
        {"type": "NormalizeMultiviewImage", "mean": [1, 2, 3]},
        {
            "type": "MultiScaleFlipAug3D",
            "transforms": [
                {"type": "RandomScaleImageMultiViewImage", "scales": [0.5]}
            ],
        },
    ]

    transform = find_transform(pipeline, "RandomScaleImageMultiViewImage")

    assert transform == {
        "type": "RandomScaleImageMultiViewImage",
        "scales": [0.5],
    }


def test_build_preprocess_transforms_uses_official_order():
    class Config:
        test_pipeline = [
            {
                "type": "NormalizeMultiviewImage",
                "mean": [103.53, 116.28, 123.675],
                "std": [1.0, 1.0, 1.0],
                "to_rgb": False,
            },
            {"type": "PadMultiViewImage", "size_divisor": 32},
            {
                "type": "MultiScaleFlipAug3D",
                "transforms": [
                    {"type": "RandomScaleImageMultiViewImage", "scales": [0.5]}
                ],
            },
        ]

    registry = object()
    build_calls = []

    def fake_build_from_cfg(config, received_registry):
        assert received_registry is registry
        build_calls.append(config.copy())
        return config["type"]

    transforms, configs = build_preprocess_transforms(
        Config(), fake_build_from_cfg, registry
    )

    expected_order = [
        "NormalizeMultiviewImage",
        "RandomScaleImageMultiViewImage",
        "PadMultiViewImage",
    ]
    assert transforms == expected_order
    assert [config["type"] for config in configs] == expected_order
    assert [config["type"] for config in build_calls] == expected_order
    assert configs[1]["scales"] == [0.5]
    assert configs[2]["size_divisor"] == 32
