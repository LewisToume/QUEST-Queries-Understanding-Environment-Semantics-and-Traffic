"""CPU test sources for server execution; no real teacher/DINO weights are needed."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from quest.depth_teacher import (
    DEPTH_ALIGNMENT_VERSION, DEPTH_SCHEMA_VERSION, cross_view_audit,
    depth_to_original_pixels, pixel_rays, prepare_metric3d_input,
    rectified_teacher_image, resize_depth_label, restore_metric_depth,
    scale_intrinsic, sensor_to_lidar, validate_depth_record, validate_image_geometry,
)
from quest.openscene_dataset import OPENSCENE_CAMERA_NAMES
from scripts.export_metric3d_depth import (
    build_teacher, inference_policy, save_geometry_preview, select_indices, teacher_dependency_blockers,
)


def camera():
    return {"cam_intrinsic": np.array([[500., 0, 28], [0, 500., 14], [0, 0, 1]]),
            "distortion": np.zeros(5), "sensor2lidar_rotation": np.eye(3),
            "sensor2lidar_translation": np.array([1., 2., 3.])}


class Metric3DDepthAdapterTest(unittest.TestCase):
    def test_unverified_diagnostic_inference_never_enables_training_label_export(self):
        policy = inference_policy("raw_opencv", "", True)
        self.assertFalse(policy["write_labels"])
        self.assertIsNone(policy["blocker"])
        self.assertIn("NOT_CALIBRATION_APPROVAL", policy["evidence"])
        with self.assertRaises(ValueError):
            inference_policy("unverified", "", True)
        self.assertIsNotNone(inference_policy("raw_opencv", "", False)["blocker"])
        self.assertIsNotNone(inference_policy("unverified", "", False)["blocker"])
        self.assertTrue(inference_policy("raw_opencv", "verified source fixture", False)["write_labels"])

    def test_mmengine_alone_does_not_satisfy_official_native_imports(self):
        versions = {"cv2_importable_spec": True, "numpy": "1.26.4", "mmengine": "0.10.7",
                    "mmcv": "NOT_INSTALLED", "mmcv-full": "NOT_INSTALLED", "timm": "0.9.16",
                    "matplotlib": "3.9.4"}
        self.assertTrue(any("MMCV1_MISSING" in item for item in teacher_dependency_blockers(versions)))
        versions["mmcv"] = "1.7.2"
        self.assertEqual(teacher_dependency_blockers(versions), [])
        versions["numpy"] = "2.2.6"
        self.assertTrue(any("LEGACY_NUMPY_PROFILE" in item for item in teacher_dependency_blockers(versions)))

    def test_geometry_preview_never_saves_depth_labels_or_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.jpg"
            Image.fromarray(np.full((28, 56, 3), 100, np.uint8)).save(source)
            item = camera()
            cameras = [{"name": name, "path": source, "intrinsic": item["cam_intrinsic"],
                        "distortion": item["distortion"]} for name in OPENSCENE_CAMERA_NAMES]
            output = root / "preview.png"
            save_geometry_preview(cameras, output)
            self.assertTrue(output.is_file())
            self.assertEqual(list(root.glob("*.pt")), [])
            with self.assertRaises(FileExistsError):
                save_geometry_preview(cameras, output)

    def test_original_index_selection_and_noncontiguous_split(self):
        infos = [{"token": f"token{index}"} for index in range(30)]
        manifest = {"validation": {"frames": [{"index": 21}, {"index": 7}]}}
        self.assertEqual(select_indices(infos, 1, 1, manifest=manifest, split="validation"), [7])
        self.assertEqual(select_indices(infos, 0, 1, explicit_indices=[0, 25, 29]), [0, 25, 29])
        self.assertEqual(select_indices(infos, 3, 2), [3, 4])
        with self.assertRaises(ValueError):
            select_indices(infos, 0, 1, [1, 1])

    def test_intrinsic_resize_and_extrinsic_direction(self):
        item = camera()
        intrinsic = item["cam_intrinsic"].copy()
        resized = scale_intrinsic(intrinsic, (1080, 1920), (252, 448))
        np.testing.assert_allclose(resized[0], intrinsic[0] * 448 / 1920)
        np.testing.assert_allclose(resized[1], intrinsic[1] * 252 / 1080)
        np.testing.assert_equal(item["cam_intrinsic"], intrinsic)
        matrix = sensor_to_lidar(item)
        np.testing.assert_allclose((matrix @ [0, 0, 10, 1])[:3], [1, 2, 13])

    def test_geometry_must_be_declared_not_guessed_from_coefficients(self):
        item = camera()
        with self.assertRaisesRegex(ValueError, "UNVERIFIED"):
            validate_image_geometry(item, "unverified")
        item["distortion"][0] = 0.1
        with self.assertRaisesRegex(ValueError, "corrected K"):
            validate_image_geometry(item, "rectified_pinhole")
        self.assertEqual(len(validate_image_geometry(item, "raw_opencv")), 5)
        item["distortion"] = np.zeros(6)
        with self.assertRaisesRegex(ValueError, "ordering"):
            validate_image_geometry(item, "raw_opencv")

    def test_official_resize_padding_normalization_and_metric_scale(self):
        image = np.zeros((28, 56, 3), dtype=np.uint8)
        tensor, resize = prepare_metric3d_input(image, camera()["cam_intrinsic"], (84, 112), 1000.)
        self.assertEqual(tuple(tensor.shape), (1, 3, 84, 112))
        self.assertEqual(resize["resized_hw"], (56, 112))
        self.assertEqual(resize["padding_tblr"], (14, 14, 0, 0))
        self.assertEqual(resize["metric_scale"], 1.)
        self.assertAlmostEqual(tensor[0, 0, 20, 20].item(), -123.675 / 58.395, places=5)
        restored = restore_metric_depth(torch.full((1, 1, 84, 112), 10.), resize)
        self.assertEqual(restored.shape, (28, 56))
        np.testing.assert_allclose(restored, 10, atol=1e-5)
        resize["metric_scale"] = .5
        np.testing.assert_allclose(restore_metric_depth(torch.full((1, 1, 84, 112), 10.), resize), 5)

    def test_unexpected_native_depth_shape_and_nonfinite_are_rejected(self):
        _, resize = prepare_metric3d_input(np.zeros((28, 56, 3), np.uint8),
                                           camera()["cam_intrinsic"], (56, 112), 1000.)
        with self.assertRaises(ValueError):
            restore_metric_depth(torch.zeros(1, 1, 28, 56), resize)
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            restore_metric_depth(torch.full((1, 1, 56, 112), float("nan")), resize)

    def test_zero_distortion_rectify_and_return_to_original_pixels(self):
        item = camera()
        rgb = np.full((28, 56, 3), 120, dtype=np.uint8)
        rectified, image_mask = rectified_teacher_image(rgb, item["cam_intrinsic"], item["distortion"], "raw_opencv")
        np.testing.assert_equal(rectified[2:-2, 2:-2], rgb[2:-2, 2:-2])
        depth, valid = depth_to_original_pixels(np.full((28, 56), 12., np.float32), image_mask,
                                                 item["cam_intrinsic"], item["distortion"], "raw_opencv")
        np.testing.assert_allclose(depth[2:-2, 2:-2], 12)
        self.assertTrue(valid[2:-2, 2:-2].all())

    def test_nonzero_distortion_uses_inverse_pixel_map_not_flip(self):
        import cv2

        item = camera()
        item["distortion"][0] = 0.2
        item["cam_intrinsic"][:2, :2] = np.diag([35., 35.])
        yy, xx = np.indices((28, 56), dtype=np.float32)
        depth = 2 + xx / 10 + yy / 20
        restored, valid = depth_to_original_pixels(depth, np.ones_like(depth, dtype=bool),
                                                   item["cam_intrinsic"], item["distortion"], "raw_opencv")
        uv = np.array([[[8., 7.]]], dtype=np.float64)
        uv_rect = cv2.undistortPoints(uv, item["cam_intrinsic"], item["distortion"], P=item["cam_intrinsic"])[0, 0]
        self.assertTrue(valid[7, 8])
        self.assertAlmostEqual(float(restored[7, 8]), 2 + uv_rect[0] / 10 + uv_rect[1] / 20, delta=.01)

    def test_invalid_values_are_masked_not_clipped_to_fake_depth(self):
        depth = np.array([[1., np.nan], [500., -2.]], dtype=np.float32)
        values, mask = resize_depth_label(depth, np.ones_like(depth, bool), (2, 2), .1, 200.)
        self.assertEqual(mask.sum().item(), 1)
        self.assertEqual(values[0, 0, 0].item(), 1)
        self.assertTrue((values[~mask] == 0).all())

    def test_camera_z_not_radial_range(self):
        intrinsic = np.array([[100., 0, 0], [0, 100., 0], [0, 0, 1]])
        ray = pixel_rays(np.array([[100., 0]]), intrinsic, np.zeros(5), "rectified_pinhole")[0]
        point = ray * 10
        np.testing.assert_allclose(point, [10, 0, 10])
        self.assertGreater(np.linalg.norm(point), point[2])

    def test_identical_camera_reprojection_and_no_overlap(self):
        depth = np.full((2, 8, 8), 10., np.float32)
        valid = np.ones_like(depth, bool)
        intrinsic = np.array([[100., 0, 4], [0, 100., 4], [0, 0, 1]])
        matrices = np.stack([np.eye(4), np.eye(4)])
        reports = cross_view_audit(depth, valid, np.stack([intrinsic] * 2), matrices,
                                  [np.zeros(5)] * 2, ["A", "B"], "rectified_pinhole", 2)
        self.assertEqual(reports[0]["median_abs_z_error_m"], 0)
        matrices[1, 0, 3] = 10000
        self.assertEqual(cross_view_audit(depth, valid, np.stack([intrinsic] * 2), matrices,
                                         [np.zeros(5)] * 2, ["A", "B"], "rectified_pinhole", 2), [])
        valid[:] = False
        self.assertEqual(cross_view_audit(depth, valid, np.stack([intrinsic] * 2), matrices,
                                         [np.zeros(5)] * 2, ["A", "B"], "raw_opencv", 2), [])

    def test_partial_checkpoint_cannot_produce_labels(self):
        fake_model = torch.nn.Linear(2, 1)
        with patch("torch.hub.load", return_value=fake_model), patch(
                "torch.load", return_value={"model_state_dict": {"weight": fake_model.weight}}):
            with self.assertRaisesRegex(ValueError, "Incomplete/mismatched"):
                build_teacher(Path("/unused/metric3d"), Path("/unused/checkpoint.pth"), "cpu")

    def test_label_token_order_units_shape_and_mask_are_validated(self):
        record = {"schema_version": DEPTH_SCHEMA_VERSION, "alignment_version": DEPTH_ALIGNMENT_VERSION,
                  "token": "abc", "source_index": 12, "camera_names": list(OPENSCENE_CAMERA_NAMES),
                  "adaptation_verified": False, "metadata_sha256": "a" * 64,
                  "provenance": {"checkpoint_sha256": "b" * 64, "geometry_evidence": "test fixture"},
                  "depth": {"depth_m": torch.ones(8, 1, 252, 448),
                            "valid_mask": torch.ones(8, 1, 252, 448, dtype=torch.bool),
                            "min_depth_m": .1, "max_depth_m": 200.,
                            "intrinsics": torch.eye(3).expand(8, 3, 3).clone(),
                            "sensor2lidar": torch.eye(4).expand(8, 4, 4).clone(),
                            "distortion": [torch.zeros(5) for _ in range(8)],
                            "units": "m", "definition": "camera_z", "image_geometry": "raw_opencv"}}
        validate_depth_record(record, "abc", 12, OPENSCENE_CAMERA_NAMES)
        for expected_token, expected_index in (("wrong", 12), ("abc", 0)):
            with self.assertRaisesRegex(ValueError, "index/token"):
                validate_depth_record(record, expected_token, expected_index, OPENSCENE_CAMERA_NAMES)
        wrong = copy.deepcopy(record)
        wrong["camera_names"].reverse()
        with self.assertRaisesRegex(ValueError, "order"):
            validate_depth_record(wrong, "abc", 12, OPENSCENE_CAMERA_NAMES)
        wrong = copy.deepcopy(record)
        wrong["depth"]["definition"] = "radial_range"
        with self.assertRaisesRegex(ValueError, "camera Z"):
            validate_depth_record(wrong, "abc", 12, OPENSCENE_CAMERA_NAMES)
        wrong = copy.deepcopy(record)
        wrong["depth"]["valid_mask"][0] = False
        with self.assertRaisesRegex(ValueError, "Invalid canonical"):
            validate_depth_record(wrong, "abc", 12, OPENSCENE_CAMERA_NAMES)


if __name__ == "__main__":
    unittest.main()
