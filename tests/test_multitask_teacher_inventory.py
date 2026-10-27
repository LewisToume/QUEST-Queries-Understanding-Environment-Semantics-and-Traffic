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

    def test_roster_is_exactly_four_tasks_and_default_map_requires_centerline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = inspect_teachers(root, root, root / "map.pth", root, root, root / "depth.pth")
            self.assertEqual([item["task"] for item in reports], ["agent", "map", "seg", "depth"])
            map_report = reports[1]
            self.assertIn("w_centerline.py", str(map_report["missing"]))
            self.assertEqual(map_report["runtime_status"], "NOT_CHECKED")

    def test_three_class_profile_discloses_missing_centerline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = inspect_teachers(root, root, root / "map.pth", root, root, root / "depth.pth",
                                       map_profile="three_class")
            self.assertIn("NO centerline", reports[1]["openscene_alignment"])
            self.assertNotIn("w_centerline.py", str(reports[1]["missing"]))

    def test_inventory_defaults_do_not_select_a_training_action(self):
        args = parse_args([])
        self.assertEqual(args.map_profile, "centerline")
        self.assertFalse(args.json)
        self.assertFalse(hasattr(args, "train"))


if __name__ == "__main__":
    unittest.main()
