"""CPU test SOURCES for server execution. No real teacher, GPU or DINO required."""

import copy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from quest.depth_teacher import validate_depth_record
from quest.dvgt_depth import (
    DINOV3_REVISION, DVGT_ALIGNMENT_VERSION, DVGT_CHECKPOINT_SHA256, DVGT_REVISION,
    DVGT_SCHEMA_VERSION, DVGT_TEACHER_CAMERAS, FLU_FROM_RDF,
    build_sequence_plans, build_teacher, camera_from_first_ego, load_strict_state,
    points_to_camera_z, prepare_image, restore_depth,
)
from quest.openscene_dataset import OPENSCENE_CAMERA_NAMES
from quest.seg_depth_dataset import SegDepthDataset, validate_depth_training_scope
from scripts.export_dvgt_depth import check_native_predictions, parse_args, publish_record


def info(index, scene="A", time=None):
    return {"token": f"token{index}", "scene_token": scene,
            "timestamp": index * 500000 if time is None else time,
            "ego2global": np.eye(4), "lidar2global": np.eye(4)}


def depth_record():
    return {"schema_version": DVGT_SCHEMA_VERSION, "alignment_version": DVGT_ALIGNMENT_VERSION,
            "token": "token1", "source_index": 1, "scene_token": "A",
            "adaptation_verified": False, "geometry_verified": False,
            "training_label_scope": "image_plane_candidate", "camera_names": list(OPENSCENE_CAMERA_NAMES),
            "metadata_sha256": "a" * 64,
            "sequence": {"source_indices": [0, 1], "tokens": ["token0", "token1"],
                         "scene_token": "A", "target_position": 1, "timestamps": [0., 500000.],
                         "uses_future": False},
            "provenance": {"teacher": "DVGT-1", "repo_revision": DVGT_REVISION,
                           "dinov3_revision": DINOV3_REVISION, "checkpoint_sha256": DVGT_CHECKPOINT_SHA256,
                           "gt_scale_factor": .1, "native_coordinate_frame": "first_ego_rdf",
                           "pose_source": "openscene_metadata", "geometry_evidence": "unverified fixture",
                           "teacher_domain": "trained_and_evaluated_on_openscene",
                           "checkpoint_load": {"strict": True, "missing_keys": [],
                                               "unexpected_keys": [], "mismatched_shapes": []},
                           "native_saturation_policy": "unbounded_points_finite_positive_range_reprojection"},
            "depth": {"depth_m": torch.full((8, 1, 252, 448), 10.),
                      "valid_mask": torch.ones(8, 1, 252, 448, dtype=torch.bool),
                      "definition": "camera_z", "units": "m", "image_geometry": "raw_opencv",
                      "min_depth_m": .1, "max_depth_m": 80.,
                      "intrinsics": torch.eye(3).repeat(8, 1, 1),
                      "sensor2lidar": torch.eye(4).repeat(8, 1, 1),
                      "distortion": [torch.zeros(5) for _ in range(8)]}}


class DVGTDepthAdapterTest(unittest.TestCase):
    def test_relevant_scenes_are_built_in_one_metadata_scan(self):
        class CountedInfos(list):
            scans = 0

            def __iter__(self):
                self.scans += 1
                return super().__iter__()

        infos = CountedInfos([info(index) for index in range(12)])
        build_sequence_plans(infos, [3, 5, 9])
        self.assertEqual(infos.scans, 1)

    def test_original_noncontiguous_targets_survive_temporal_sorting(self):
        infos = [info(0, time=1000000), info(1, time=0), info(2, time=500000), info(3, "B")]
        plans = build_sequence_plans(infos, [0, 2])
        self.assertEqual([row["targets"] for row in plans], [[0], [2]])
        self.assertEqual([row["indices"] for row in plans], [[2, 0], [1, 2]])

    def test_boundary_future_is_explicit_real_and_grouped(self):
        infos = [info(0), info(1), info(2)]
        with self.assertRaisesRegex(ValueError, "insufficient past"):
            build_sequence_plans(infos, [0])
        plans = build_sequence_plans(infos, [0, 1], allow_future=True)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]["indices"], [0, 1])
        self.assertEqual(plans[0]["targets"], [0, 1])

    def test_scene_gap_duplicate_and_manifest_isolation(self):
        with self.assertRaisesRegex(ValueError, "not enough real frames"):
            build_sequence_plans([info(0), info(1, "B")], [0], allow_future=True)
        with self.assertRaisesRegex(ValueError, "gaps_s"):
            build_sequence_plans([info(0), info(1, time=2000000)], [1])
        with self.assertRaisesRegex(ValueError, "non-increasing"):
            build_sequence_plans([info(0), info(1, time=0)], [1])
        with self.assertRaisesRegex(ValueError, "manifest split"):
            build_sequence_plans([info(0), info(1)], [1], allowed_scenes={"B"})

    def test_ego_rdf_axes_translation_and_metric_units(self):
        first, current = info(0), info(1)
        current["ego2global"][0, 3] = 2.
        current["lidar2global"][0, 3] = 2.
        camera2lidar = FLU_FROM_RDF.copy()  # Front camera, aligned with ego RDF.
        transform = camera_from_first_ego(first, current, camera2lidar)
        # Native .1 * first-ego-RDF [1,2,12] -> current camera [1,2,10].
        z, xyz = points_to_camera_z(np.array([[[.1, .2, 1.2]]]), transform)
        np.testing.assert_allclose(xyz, [[[1., 2., 10.]]], atol=1e-6)
        np.testing.assert_allclose(z, [[10.]])
        self.assertGreater(float(np.linalg.norm(xyz[0, 0])), z[0, 0])
        with self.assertRaisesRegex(ValueError, "scale_factor"):
            points_to_camera_z(np.zeros((2, 2, 3)), np.eye(4), 1.)

    def test_full_rotation_is_used_and_inconsistent_lidar_pose_rejected(self):
        first, current = info(0), info(1)
        angle = .2
        rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                             [0, np.sin(angle), np.cos(angle)]])
        current["ego2global"][:3, :3] = rotation
        current["lidar2global"][:3, :3] = rotation
        transform = camera_from_first_ego(first, current, FLU_FROM_RDF)
        expected = np.linalg.inv(FLU_FROM_RDF) @ np.linalg.inv(current["ego2global"]) @ FLU_FROM_RDF
        np.testing.assert_allclose(transform, expected, atol=1e-8)
        current["lidar2ego"] = np.eye(4)
        current["lidar2ego"][0, 3] = 3.
        with self.assertRaisesRegex(ValueError, "disagrees"):
            camera_from_first_ego(first, current, FLU_FROM_RDF)

    def test_raw_pixels_reprojection_and_invalid_range_mask(self):
        k = np.array([[10., 0, 1.], [0, 10., 1.], [0, 0, 1.]])
        y, x = np.indices((3, 4))
        points = np.stack(((x - 1) / 10, (y - 1) / 10, np.ones_like(x)), -1)  # scaled native Z=1 -> 10m
        depth, mask, stats = restore_depth(points, np.ones((3, 4)), np.eye(4), k,
                                          np.zeros(5), k, (3, 4), (3, 4), "raw_opencv")
        np.testing.assert_allclose(depth.numpy(), 10.)
        self.assertTrue(mask.all())
        self.assertLess(stats["native_reprojection_median_px"], 1e-6)
        points[1, 1] = [0., 0., -1.]
        points[1, 2] *= 100
        depth, mask, _ = restore_depth(points, np.ones((3, 4)), np.eye(4), k,
                                       np.zeros(5), k, (3, 4), (3, 4), "rectified_pinhole")
        self.assertFalse(mask[0, 1, 1])
        self.assertFalse(mask[0, 1, 2])
        self.assertTrue((depth[~mask] == 0).all())

    def test_outside_crop_is_invalid_not_fake_zero_target(self):
        k = np.eye(3)
        native_k = k.copy()
        native_k[0, 2] = -100.
        points = np.zeros((2, 2, 3))
        points[..., 2] = 1.
        depth, mask, _ = restore_depth(points, np.ones((2, 2)), np.eye(4), k, np.zeros(5),
                                       native_k, (2, 2), (2, 2), "rectified_pinhole")
        self.assertFalse(mask.any())
        self.assertEqual(float(depth.sum()), 0.)

    def test_preprocessing_calls_native_chain_and_does_not_normalize_twice(self):
        rgb = np.full((288, 512, 3), 255, np.uint8)
        k = np.eye(3)
        native_image = Mock()
        native_image.process_image_sequentially.return_value = (rgb, k, None)
        native_image.resize_to_aspect_ratio.return_value = (rgb, k)
        transforms = Mock()
        transforms.crop_image_depth_and_intrinsic_by_pp.return_value = (rgb, None, k, None)
        transforms.resize_image_depth_and_intrinsic.return_value = (rgb, None, k, None)
        tensor, _ = prepare_image(rgb, k, np.zeros(5), "raw_opencv", (native_image, transforms))
        self.assertEqual(tuple(tensor.shape), (3, 288, 512))
        self.assertTrue((tensor == 1.).all())
        self.assertEqual(transforms.crop_image_depth_and_intrinsic_by_pp.call_count, 2)
        self.assertFalse(transforms.resize_image_depth_and_intrinsic.call_args.kwargs["enable_random_resize"])
        self.assertTrue(transforms.crop_image_depth_and_intrinsic_by_pp.call_args.kwargs["strict"])

    def test_full_checkpoint_strict_load_no_omission_allowlist(self):
        model = torch.nn.Linear(2, 1)
        self.assertTrue(load_strict_state(model, {"model": model.state_dict()})["strict"])
        for state in ({"weight": model.weight}, {**model.state_dict(), "other": torch.zeros(1)},
                      {"weight": torch.zeros(3, 2), "bias": model.bias}):
            with self.assertRaisesRegex(ValueError, "Incomplete/mismatched"):
                load_strict_state(model, state)

    def test_backbone_build_is_local_code_only_no_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            package = types.ModuleType("dvgt")
            package.__file__ = str(root / "dvgt/__init__.py")
            module = types.ModuleType("dvgt.models.architectures.dvgt1")

            def construct(**kwargs):
                self.assertIsNone(kwargs["dino_v3_weight_path"])
                return torch.hub.load("third_party/dinov3", "dinov3_vitl16", source="local")

            module.DVGT1 = construct
            model = torch.nn.Linear(2, 1)
            dinov3 = types.ModuleType("dinov3")
            dinov3.__file__ = str(root / "third_party/dinov3/dinov3/__init__.py")
            backbones = types.ModuleType("dinov3.hub.backbones")
            backbones.dinov3_vitl16 = Mock(return_value=model)
            old_path = sys.path.copy()
            try:
                with patch.dict(sys.modules, {
                        "dvgt": package, module.__name__: module, "dinov3": dinov3,
                        "dinov3.hub": types.ModuleType("dinov3.hub"),
                        backbones.__name__: backbones, "dinov3.eval": None, "torchmetrics": None,
                }), patch(
                        "torch.hub.load", side_effect=AssertionError("Task hubconf must not load")) as hub, patch(
                        "torch.load", return_value=model.state_dict()):
                    loaded, _ = build_teacher(root, root / "weights.pt", "cpu")
                self.assertIs(loaded, model)
                hub.assert_not_called()
                backbones.dinov3_vitl16.assert_called_once_with(pretrained=False)
                self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
            finally:
                sys.path[:] = old_path

    def test_backbone_build_rejects_other_hub_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            package = types.ModuleType("dvgt")
            package.__file__ = str(root / "dvgt/__init__.py")
            module = types.ModuleType("dvgt.models.architectures.dvgt1")
            requests = (("third_party/dinov3", "dinov3_vitb16", {"source": "local"}),
                        ("other/dinov3", "dinov3_vitl16", {"source": "local"}),
                        ("third_party/dinov3", "dinov3_vitl16", {"source": "github"}))
            old_path = sys.path.copy()
            try:
                for repo_or_dir, model_name, kwargs in requests:
                    with self.subTest(repo=repo_or_dir, model=model_name, source=kwargs["source"]):
                        def construct(**unused):
                            return torch.hub.load(repo_or_dir, model_name, **kwargs)

                        module.DVGT1 = construct
                        with patch.dict(sys.modules, {"dvgt": package, module.__name__: module}), patch(
                                "torch.load") as load:
                            with self.assertRaisesRegex(RuntimeError, "Unexpected native hub request"):
                                build_teacher(root, root / "weights.pt", "cpu")
                            load.assert_not_called()
            finally:
                sys.path[:] = old_path

    def test_native_outputs_are_checked_before_any_label_write(self):
        output = {"points": torch.zeros(1, 2, 8, 2, 3, 3), "points_conf": torch.ones(1, 2, 8, 2, 3),
                  "absolute_ego_pose_enc": torch.zeros(1, 2, 7)}
        self.assertIn("points", check_native_predictions(output, 2, (2, 3)))
        output["points"][0, 0, 0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            check_native_predictions(output, 2, (2, 3))

    def test_new_schema_and_scope_remain_explicit_and_teacher_mixing_rejected(self):
        record = depth_record()
        validate_depth_record(record, "token1", 1, OPENSCENE_CAMERA_NAMES)
        validate_depth_training_scope(record, True)
        with self.assertRaisesRegex(ValueError, "UNVERIFIED"):
            validate_depth_training_scope(record, False)
        wrong = copy.deepcopy(record)
        wrong["provenance"]["teacher"] = "Metric3Dv2"
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_depth_record(wrong, "token1", 1, OPENSCENE_CAMERA_NAMES)
        wrong = copy.deepcopy(record)
        wrong["sequence"]["target_position"] = 0
        with self.assertRaisesRegex(ValueError, "sequence/target"):
            validate_depth_record(wrong, "token1", 1, OPENSCENE_CAMERA_NAMES)

    def test_save_is_target_named_exclusive_and_diagnostic_is_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publish_record({"token": "target", "depth": {}}, root)
            self.assertEqual({path.name for path in root.iterdir()}, {"target.pt"})
            with self.assertRaises(FileExistsError):
                publish_record({"token": "target"}, root)
        self.assertFalse(parse_args([]).export_candidate_labels)
        self.assertEqual(set(DVGT_TEACHER_CAMERAS), set(OPENSCENE_CAMERA_NAMES))
        self.assertNotEqual(DVGT_TEACHER_CAMERAS, OPENSCENE_CAMERA_NAMES)

    def test_existing_dataset_reads_dvgt_without_seg_or_metric_saturation_claim(self):
        record = depth_record()
        record.update(split_sha256="b" * 64, source_hw=[(252, 448)] * 8)
        sample = info(1)
        sample["cams"] = {name: {"cam_intrinsic": np.eye(3), "sensor2lidar_rotation": np.eye(3),
                                  "sensor2lidar_translation": np.zeros(3), "distortion": np.zeros(5)}
                          for name in OPENSCENE_CAMERA_NAMES}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(record, root / "token1.pt")
            dataset = SegDepthDataset.__new__(SegDepthDataset)
            dataset.tasks = dataset.load_tasks = ("depth",)
            dataset.directories = {"depth": root}
            dataset.missing_policy, dataset.experimental_depth = "error", True
            dataset.images = types.SimpleNamespace(infos=[sample], image_size=(252, 448))
            dataset.source_indices = [1]
            dataset.metadata_sha256, dataset.split_sha256 = "a" * 64, "b" * 64
            dataset.teacher_signatures, dataset.digests = {}, {}
            loaded = dataset._load_targets(0, establish=True)
            self.assertIsNone(loaded["seg"])
            self.assertEqual(tuple(loaded["depth"]["depth_m"].shape), (8, 1, 252, 448))
            record["provenance"]["confidence_min"] = 1.
            torch.save(record, root / "token1.pt")
            with self.assertRaisesRegex(ValueError, "Mixed teacher"):
                dataset._load_targets(0, establish=True)


if __name__ == "__main__":
    unittest.main()
