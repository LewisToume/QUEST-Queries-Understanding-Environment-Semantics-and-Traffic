from __future__ import annotations

import json
import unittest

import numpy as np

from quest.stage3_split import SPLIT_SCHEMA_VERSION, load_stage3_split, validate_split
from quest.stage3_dataset import load_teacher_audit
from scripts.build_stage3_split import (
    build_split, choose_scenes, save_split_manifest,
)
from scripts.export_navformer_map_soft import temporal_export_plan


def synthetic_split(directory):
    from pathlib import Path

    class CityLocator:
        def resolve(self, info):
            return info["city"]

    infos = []
    for index in range(5120):
        if index < 5000:
            city = "A" if index < 2500 else "B"
            scene = f"train-{city}"
            x = 0.0
        else:
            scene_number = (index - 5000) // 40
            city = "A"
            scene = f"held-{scene_number}"
            x = scene_number * 2.0 + (index - 5000) % 40
        matrix = np.eye(4)
        matrix[0, 3] = x
        infos.append({"token": f"token-{index}", "scene_token": scene,
                      "timestamp": index, "city": city, "lidar2global": matrix})
    path = Path(directory) / "metadata.pkl"
    path.write_bytes(b"synthetic")
    return infos, path, build_split(infos, CityLocator(), path)


class Stage3SplitTest(unittest.TestCase):
    def test_noncontiguous_temporal_targets_keep_original_indices(self):
        infos = [{"token": f"a{n}", "scene_token": "A", "timestamp": n}
                 for n in range(5)]
        plan = temporal_export_plan(infos, 0, 1, [1, 4])
        self.assertEqual([item["sample_index"] for item in plan if item["target"]], [1, 4])
        self.assertEqual([item["sample_index"] for item in plan if not item["target"]], [0, 2, 3])


    def test_scene_selection_is_seeded_and_prefers_training_cities(self):
        scenes = [{"scene": name, "city": city, "motion_m": motion}
                  for name, city, motion in (("A", "train", 0.0), ("B", "train", 10.0),
                                             ("C", "train", 20.0), ("D", "other", 5.0))]
        selected = choose_scenes(scenes, {"train"}, seed=42)
        self.assertEqual({item["scene"] for item in selected}, {"A", "B", "C"})
        self.assertEqual([item["scene"] for item in selected],
                         [item["scene"] for item in choose_scenes(list(reversed(scenes)), {"train"}, seed=42)])
        self.assertIn("C", {item["scene"] for item in selected})

    def test_one_high_motion_scene_even_when_training_city_scenes_are_static(self):
        scenes = [{"scene": name, "city": city, "motion_m": motion}
                  for name, city, motion in (("A", "train", 0.0), ("B", "train", 0.0),
                                             ("C", "train", 0.0), ("D", "other", 100.0))]
        self.assertIn("D", {item["scene"] for item in choose_scenes(scenes, {"train"})})

    def test_scene_selection_falls_back_to_other_cities(self):
        scenes = [{"scene": name, "city": city, "motion_m": motion}
                  for name, city, motion in (("A", "train", 10.0), ("B", "other", 0.0),
                                             ("C", "other", 20.0))]
        self.assertEqual({item["scene"] for item in choose_scenes(scenes, {"train"})},
                         {"A", "B", "C"})
        with self.assertRaisesRegex(ValueError, "three complete"):
            choose_scenes(scenes[:2], {"train"})


    def test_split_hash_and_metadata_token_checks(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            infos, metadata, manifest = synthetic_split(directory)
            validate_split(manifest, infos, metadata)
            split_path = Path(directory) / "split.json"
            split_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(load_stage3_split(split_path, infos, metadata)["split_sha256"],
                             manifest["split_sha256"])
            changed = json.loads(json.dumps(manifest))
            changed["validation"]["frames"][0]["token"] = "wrong"
            with self.assertRaisesRegex(ValueError, "index/token/scene"):
                validate_split(changed, infos, metadata)
            changed = json.loads(json.dumps(manifest))
            changed["validation"]["frames"][0]["scene_token"] = "train-A"
            with self.assertRaisesRegex(ValueError, "index/token/scene"):
                validate_split(changed, infos, metadata)
            changed = json.loads(json.dumps(manifest))
            changed["validation"]["scenes"][0]["indices"].reverse()
            with self.assertRaisesRegex(ValueError, "scene description"):
                validate_split(changed, infos, metadata)
            changed = json.loads(json.dumps(manifest))
            changed["validation"]["scenes"][0]["motion_extent_m"] = -1.0
            with self.assertRaisesRegex(ValueError, "stale motion_extent_m"):
                validate_split(changed, infos, metadata)
            changed = json.loads(json.dumps(manifest))
            changed["validation"]["frames"][0]["token"] = changed["train"]["frames"][0]["token"]
            with self.assertRaisesRegex(ValueError, "index/token/scene"):
                validate_split(changed, infos, metadata)
            changed = json.loads(json.dumps(manifest))
            index = changed["validation"]["frames"][0]["index"]
            infos[index]["token"] = changed["train"]["frames"][0]["token"]
            changed["validation"]["frames"][0]["token"] = infos[index]["token"]
            with self.assertRaisesRegex(ValueError, "duplicate token"):
                validate_split(changed, infos, metadata)
            infos[index]["token"] = f"token-{index}"
            changed = json.loads(json.dumps(manifest))
            infos[index]["scene_token"] = "train-A"
            changed["validation"]["frames"][0]["scene_token"] = "train-A"
            with self.assertRaisesRegex(ValueError, "overlaps training scene"):
                validate_split(changed, infos, metadata)
            infos[index]["scene_token"] = f"held-{(index - 5000) // 40}"
            changed = json.loads(json.dumps(manifest))
            changed["schema_version"] = 1
            with self.assertRaisesRegex(ValueError, "schema is unsupported"):
                validate_split(changed, infos, metadata)
            changed = json.loads(json.dumps(manifest))
            changed["split_sha256"] = "old-audit-hash"
            with self.assertRaisesRegex(ValueError, "frame-list hash mismatch"):
                validate_split(changed, infos, metadata)

    def test_builder_keeps_5000_training_indices_and_three_full_scenes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            _, _, manifest = synthetic_split(directory)
            self.assertEqual(manifest["schema_version"], SPLIT_SCHEMA_VERSION)
            self.assertEqual([row["index"] for row in manifest["train"]["frames"]], list(range(5000)))
            self.assertEqual(len(manifest["validation"]["frames"]), 120)
            self.assertEqual(len(manifest["validation"]["scenes"]), 3)
            self.assertEqual(manifest["validation"]["frames"][0]["index"], 5000)
            self.assertTrue(all(scene["frame_count"] == 40
                                for scene in manifest["validation"]["scenes"]))
            self.assertEqual({row["city"] for row in manifest["validation"]["frames"]}, {"A"})
            self.assertEqual({row["token"] for row in manifest["train"]["frames"]}
                             & {row["token"] for row in manifest["validation"]["frames"]}, set())
            self.assertEqual({row["scene_token"] for row in manifest["train"]["frames"]}
                             & {row["scene_token"] for row in manifest["validation"]["frames"]}, set())

    def test_incomplete_scene_is_not_selected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            infos, metadata, _ = synthetic_split(directory)
            infos[-1]["scene_token"] = "incomplete-single-frame-scene"

            class CityLocator:
                def resolve(self, info):
                    return info["city"]

            with self.assertRaisesRegex(RuntimeError, "need three complete 40-frame scenes"):
                build_split(infos, CityLocator(), metadata)

    def test_v2_manifest_is_backed_up_and_v3_hash_is_distinct(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            _, _, manifest = synthetic_split(directory)
            output = Path(directory) / "stage3_split.json"
            old_bytes = b'{"schema_version": 2, "split_sha256": "old-audit-hash"}\n'
            output.write_bytes(old_bytes)
            backup = save_split_manifest(output, manifest)
            self.assertEqual(backup.name, "stage3_split.v2.backup.json")
            self.assertEqual(backup.read_bytes(), old_bytes)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["split_sha256"],
                             manifest["split_sha256"])
            self.assertNotEqual(manifest["split_sha256"], "old-audit-hash")
            self.assertEqual(save_split_manifest(output, manifest), None)
            output.write_bytes(old_bytes)
            backup.write_bytes(b"different historical split")
            with self.assertRaisesRegex(ValueError, "backup differs"):
                save_split_manifest(output, manifest)
            self.assertEqual(output.read_bytes(), old_bytes)

    def test_teacher_audit_cannot_use_another_split(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps({"verified": True, "stage3_split_sha256": "old"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "current Stage 3 split"):
                load_teacher_audit(path, expected_split_sha256="new")
