from __future__ import annotations

import json
import unittest

import numpy as np

from quest.stage3_split import SPLIT_SCHEMA_VERSION, file_sha256, load_stage3_split, split_hash, validate_split
from quest.stage3_dataset import load_teacher_audit
from scripts.build_stage3_split import build_split, choose_scenes, nearest_same_city
from scripts.export_navformer_map_soft import temporal_export_plan


class Stage3SplitTest(unittest.TestCase):
    def test_city_trees_never_compare_foreign_projection_coordinates(self):
        train = {"las_vegas": np.array([[0.0, 0.0]]),
                 "pittsburgh": np.array([[1000.0, 1000.0]])}
        distances = nearest_same_city(train, ["pittsburgh", "las_vegas"],
                                      np.array([[0.0, 0.0], [10.0, 0.0]]))
        self.assertGreater(distances[0], 1000)
        self.assertAlmostEqual(distances[1], 10)


    def test_noncontiguous_temporal_targets_keep_original_indices(self):
        infos = [{"token": f"a{n}", "scene_token": "A", "timestamp": n}
                 for n in range(5)]
        plan = temporal_export_plan(infos, 0, 1, [1, 4])
        self.assertEqual([item["sample_index"] for item in plan if item["target"]], [1, 4])
        self.assertEqual([item["sample_index"] for item in plan if not item["target"]], [0, 2, 3])


    def test_scene_selection_prefers_full_moving_scenes(self):
        def candidate(scene, count, motion):
            return {"scene": scene, "city": "city", "indices": list(range(count)),
                    "points": np.column_stack((np.arange(count) + ord(scene) * 1000,
                                               np.zeros(count))),
                    "motion_m": motion, "log": f"log-{scene}"}
        scenes = [candidate("A", 40, 39), candidate("B", 40, 39),
                  candidate("C", 40, 39), candidate("D", 35, 34)]
        self.assertEqual({item["scene"] for item in choose_scenes(scenes, set())}, {"A", "B", "C"})


    def test_split_hash_and_metadata_token_checks(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "metadata.pkl"
            metadata.write_bytes(b"test metadata bytes")
            infos = [{"token": f"t{index}", "scene_token": "train" if index < 5000 else "held"}
                     for index in range(5001)]
            train = [{"index": index, "token": f"t{index}", "scene_token": "train", "city": "city"}
                     for index in range(5000)]
            validation = [{"index": 5000, "token": "t5000", "scene_token": "held",
                           "city": "city", "nearest_train_m": 250.0}]
            manifest = {"schema_version": SPLIT_SCHEMA_VERSION, "metadata_sha256": file_sha256(metadata),
                        "split_sha256": split_hash(train, validation),
                        "train": {"frames": train}, "validation": {"frames": validation}}
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
            changed["validation"]["frames"][0]["nearest_train_m"] = 199.9
            with self.assertRaisesRegex(ValueError, "spatially isolated"):
                validate_split(changed, infos, metadata)

    def test_builder_keeps_5000_training_indices_and_three_full_scenes(self):
        import tempfile
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
                city = "A" if scene_number != 1 else "B"
                scene = f"held-{scene_number}"
                x = 1000.0 * (scene_number + 1) + (index - 5000) % 40
            matrix = np.eye(4)
            matrix[0, 3] = x
            infos.append({"token": f"token-{index}", "scene_token": scene,
                          "timestamp": index, "city": city, "lidar2global": matrix})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.pkl"
            path.write_bytes(b"synthetic")
            manifest = build_split(infos, CityLocator(), path)
            self.assertEqual([row["index"] for row in manifest["train"]["frames"]], list(range(5000)))
            self.assertEqual(len(manifest["validation"]["frames"]), 120)
            self.assertEqual(len(manifest["validation"]["scenes"]), 3)
            self.assertTrue(all(row["nearest_train_m"] >= 200
                                for row in manifest["validation"]["frames"]))

    def test_teacher_audit_cannot_use_another_split(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps({"verified": True, "stage3_split_sha256": "old"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "current Stage 3 split"):
                load_teacher_audit(path, expected_split_sha256="new")
