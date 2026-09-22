import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_streampetr_openscene.py"
spec = importlib.util.spec_from_file_location("run_streampetr_openscene", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class StreamPETROpenSceneAdapterTest(unittest.TestCase):
    def test_native_config_uses_available_base_files(self):
        class FakeConfig:
            @staticmethod
            def fromstring(source, file_format):
                return source, file_format

        args = SimpleNamespace(
            stream_petr_root=module.STREAM_PETR_ROOT,
            mmdet3d_config_root=None,
        )
        source, file_format = module.load_native_config(
            args, SimpleNamespace(Config=FakeConfig)
        )
        self.assertEqual(file_format, ".py")
        self.assertNotIn("../../../mmdetection3d/", source)
        self.assertIn("stream_petr_r50_flash_704_bs2_seq_90e", str(module.CONFIG_RELATIVE))

    def test_geometry_and_camera_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cameras = {}
            for index, name in enumerate(module.CAMERAS):
                image = root / "openscene-v1.0/sensor_blobs/mini/scene" / name / "image.jpg"
                image.parent.mkdir(parents=True, exist_ok=True)
                image.touch()
                cameras[name] = {
                    "data_path": "dataset/openscene-v1.0/sensor_blobs/scene/{}/image.jpg".format(
                        name
                    ),
                    "sensor2lidar_rotation": np.eye(3),
                    "sensor2lidar_translation": [float(index), 0.0, 0.0],
                    "cam_intrinsic": np.diag([100.0, 120.0, 1.0]),
                }
            ego_pose = np.eye(4)
            ego_pose[:3, 3] = [665000.0, 4000000.0, 500.0]
            frame = {
                "token": "sample-token",
                "scene_token": "scene-token",
                "timestamp": 1620000000000000,
                "lidar2global": ego_pose,
                "cams": cameras,
            }
            data = module.make_input(frame, root)
            self.assertEqual(len(data["img_filename"]), 6)
            self.assertEqual(
                [Path(path).parent.name for path in data["img_filename"]],
                list(module.CAMERAS),
            )
            self.assertFalse(data["prev_exists"])
            self.assertEqual(data["timestamp"], 1620000000.0)
            for index in range(6):
                self.assertTrue(
                    np.allclose(
                        data["intrinsics"][index] @ data["extrinsics"][index],
                        data["lidar2img"][index],
                    )
                )
                self.assertAlmostEqual(data["extrinsics"][index][0, 3], -index)
            self.assertTrue(
                np.allclose(data["ego_pose"] @ data["ego_pose_inv"], np.eye(4))
            )


if __name__ == "__main__":
    unittest.main()
