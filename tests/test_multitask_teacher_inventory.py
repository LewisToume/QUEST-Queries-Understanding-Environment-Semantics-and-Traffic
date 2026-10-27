import tempfile
import unittest
from pathlib import Path

from scripts.inspect_multitask_teachers import inspect_files, inspect_teachers, parse_args


class MultitaskTeacherInventoryTest(unittest.TestCase):
    def test_missing_files_are_explicit_and_not_runtime_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.pth"
            report = inspect_files("depth", "teacher", ((path,),), "UNVERIFIED")
            self.assertEqual(report["asset_status"], "ASSETS_MISSING")
            self.assertEqual(report["missing"], [[str(path)]])
            self.assertEqual(report["runtime_status"], "NOT_CHECKED")

    def test_existing_files_do_not_certify_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(b"file presence fixture, not a real checkpoint")
            report = inspect_files("seg", "teacher", ((path,),), "UNVERIFIED")
            self.assertEqual(report["asset_status"], "FILES_PRESENT")
            self.assertEqual(report["runtime_status"], "NOT_CHECKED")
            self.assertEqual(report["openscene_alignment"], "UNVERIFIED")

    def test_empty_file_is_not_an_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.pth"
            path.touch()
            report = inspect_files("map", "teacher", ((path,),), "UNVERIFIED")
            self.assertEqual(report["asset_status"], "ASSETS_MISSING")

    def test_explicit_weight_alternatives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"file presence fixture")
            report = inspect_files("seg", "teacher", (
                (root / "model.safetensors", root / "pytorch_model.bin"),
            ), "UNVERIFIED")
            self.assertEqual(report["missing"], [])

    def test_roster_has_four_tasks_but_map_requires_gt_not_maptr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = inspect_teachers(root, root, root, root, root / "depth.pth")
            self.assertEqual([item["task"] for item in reports], ["agent", "map", "seg", "depth"])
            map_report = reports[1]
            self.assertIn("stage3_split.json", str(map_report["missing"]))
            self.assertNotIn("maptrv2", str(map_report["missing"]))
            self.assertEqual(map_report["runtime_status"], "NOT_CHECKED")

    def test_map_gt_does_not_require_raster_or_vector_teacher_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/vector_map_gt").mkdir(parents=True)
            (root / "data/stage3_split.json").write_text("{}")
            (root / "data/vector_map_capacity_audit.json").write_text("{}")
            reports = inspect_teachers(root, root, root, root, root / "depth.pth")
            self.assertEqual(reports[1]["asset_status"], "FILES_PRESENT")
            self.assertIn("raster_KD=DISABLED", reports[1]["openscene_alignment"])
            self.assertIn("full_GT_coverage=NOT_CHECKED", reports[1]["openscene_alignment"])

    def test_inventory_defaults_do_not_select_a_training_action(self):
        args = parse_args([])
        self.assertFalse(hasattr(args, "map_profile"))
        self.assertFalse(args.json)
        self.assertFalse(hasattr(args, "train"))

    def test_partial_depth_download_is_not_a_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "depth.pth.part").write_bytes(b"partial")
            report = inspect_teachers(root, root, root, root, root / "depth.pth")[-1]
            self.assertEqual(report["asset_status"], "ASSETS_MISSING")
            self.assertEqual(report["partial_download_bytes"], 7)


if __name__ == "__main__":
    unittest.main()
