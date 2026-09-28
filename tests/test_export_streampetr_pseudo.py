import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_streampetr_pseudo.py"
spec = importlib.util.spec_from_file_location("export_streampetr_pseudo", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class StreamPETRPseudoExportTest(unittest.TestCase):
    def test_select_range_is_exact_and_validated(self):
        infos = list(range(10))
        self.assertEqual(module.select_range(infos, 3, 4), [3, 4, 5, 6])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            module.select_range(infos, -1, 1)
        with self.assertRaisesRegex(ValueError, "positive"):
            module.select_range(infos, 0, 0)
        with self.assertRaisesRegex(IndexError, "metadata size"):
            module.select_range(infos, 8, 3)

    def test_payload_preserves_all_raw_predictions(self):
        boxes = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        scores = torch.tensor([0.001, 0.2, 0.99])
        labels = torch.tensor([8, 0, 5])
        payload = module.make_payload(
            "sample-token",
            {
                "boxes_3d": SimpleNamespace(tensor=boxes),
                "scores_3d": scores,
                "labels_3d": labels,
            },
        )
        self.assertEqual(payload["token"], "sample-token")
        self.assertTrue(torch.equal(payload["boxes_3d"], boxes))
        self.assertTrue(torch.equal(payload["scores_3d"], scores))
        self.assertTrue(torch.equal(payload["labels_3d"], labels))

    def test_save_payload_writes_complete_pt_file(self):
        payload = {
            "token": "sample-token",
            "boxes_3d": torch.zeros(2, 9),
            "scores_3d": torch.ones(2),
            "labels_3d": torch.zeros(2, dtype=torch.long),
        }
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "nested/sample-token.pt"
            module.save_payload(payload, output_path, torch)
            loaded = torch.load(str(output_path), map_location="cpu")
            self.assertEqual(loaded["token"], "sample-token")
            self.assertEqual(tuple(loaded["boxes_3d"].shape), (2, 9))
            self.assertFalse(output_path.with_suffix(".pt.tmp").exists())

    def test_temporal_prev_exists_within_scene(self):
        state = module.TemporalSequenceState()
        values = []
        for timestamp in (1.0, 2.0, 3.0):
            decision = state.begin("scene-a", timestamp)
            values.append(int(decision["prev_exists"]))
            state.complete(True)
        self.assertEqual(values, [0, 1, 1])

    def test_scene_change_resets_temporal_memory(self):
        state = module.TemporalSequenceState()
        first = state.begin("scene-a", 1.0)
        state.complete(True)
        second = state.begin("scene-b", 1.0)
        self.assertEqual(first["prev_exists"], 0.0)
        self.assertEqual(second["prev_exists"], 0.0)
        self.assertTrue(second["new_scene"])

    def test_non_monotonic_timestamp_resets_temporal_memory(self):
        state = module.TemporalSequenceState()
        state.begin("scene-a", 2.0)
        state.complete(True)
        decision = state.begin("scene-a", 2.0)
        self.assertTrue(decision["non_monotonic"])
        self.assertTrue(decision["reset_required"])
        self.assertEqual(decision["prev_exists"], 0.0)

    def test_failed_frame_forces_next_frame_to_reset(self):
        state = module.TemporalSequenceState()
        state.begin("scene-a", 1.0)
        state.complete(True)
        failed = state.begin("scene-a", 2.0)
        self.assertEqual(failed["prev_exists"], 1.0)
        state.complete(False)
        following = state.begin("scene-a", 3.0)
        self.assertFalse(following["new_scene"])
        self.assertTrue(following["reset_required"])
        self.assertEqual(following["prev_exists"], 0.0)

    def test_different_scenes_never_share_memory(self):
        state = module.TemporalSequenceState()
        state.begin("scene-a", 1.0)
        state.complete(True)
        scene_b = state.begin("scene-b", 1.0)
        state.complete(True)
        scene_a_again = state.begin("scene-a", 2.0)
        self.assertEqual(scene_b["prev_exists"], 0.0)
        self.assertEqual(scene_a_again["prev_exists"], 0.0)

    def test_temporal_output_directory_does_not_overwrite_single_frame(self):
        temporal = module.default_output_dir(True)
        single_frame = module.default_output_dir(False)
        self.assertNotEqual(temporal, single_frame)
        self.assertEqual(temporal.name, "streampetr_temporal")
        self.assertEqual(single_frame.name, "streampetr")


if __name__ == "__main__":
    unittest.main()
