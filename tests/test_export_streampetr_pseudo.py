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
    @staticmethod
    def make_info(token, scene_token, timestamp):
        return {
            "token": token,
            "scene_token": scene_token,
            "timestamp": timestamp,
        }

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

    def test_target_tokens_are_the_original_metadata_slice(self):
        infos = [
            self.make_info("b-late", "scene-b", 20),
            self.make_info("a-target", "scene-a", 30),
            self.make_info("b-target", "scene-b", 10),
            self.make_info("a-early", "scene-a", 5),
        ]
        target_infos = module.select_range(infos, 1, 2)
        self.assertEqual(
            [info["token"] for info in target_infos],
            ["a-target", "b-target"],
        )

        plan = module.build_temporal_execution_plan(infos, target_infos)
        planned_targets = [
            item["info"]["token"] for item in plan if item["target"]
        ]
        self.assertEqual(set(planned_targets), {"a-target", "b-target"})
        self.assertNotIn("b-late", planned_targets)

    def test_relevant_scenes_are_built_with_one_metadata_scan(self):
        class CountingInfos:
            def __init__(self, infos):
                self.infos = infos
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                return iter(self.infos)

        infos = CountingInfos(
            [
                self.make_info("a0", "scene-a", 0),
                self.make_info("b0", "scene-b", 0),
                self.make_info("a1", "scene-a", 1),
            ]
        )
        relevant = module.build_relevant_by_scene(infos, {"scene-a"})
        self.assertEqual(infos.iterations, 1)
        self.assertEqual(
            [info["token"] for info in relevant["scene-a"]], ["a0", "a1"]
        )

    def test_temporal_plan_warms_up_from_scene_start_to_latest_target(self):
        infos = [
            self.make_info("target-late", "scene-a", 30),
            self.make_info("after-target", "scene-a", 40),
            self.make_info("warmup-0", "scene-a", 10),
            self.make_info("target-early", "scene-a", 20),
            self.make_info("other-scene", "scene-b", 5),
        ]
        target_infos = [infos[0], infos[3]]
        plan = module.build_temporal_execution_plan(infos, target_infos)

        self.assertEqual(
            [item["info"]["token"] for item in plan],
            ["warmup-0", "target-early", "target-late"],
        )
        self.assertEqual(
            [item["target"] for item in plan], [False, True, True]
        )

    def test_success_and_failure_logs_include_token(self):
        success = module.format_success_log(
            "scene-a", 123.0, "token-a", 1.0, True, True, 7
        )
        self.assertEqual(
            success,
            "scene=scene-a timestamp=123.0 token=token-a prev_exists=1 "
            "target=true saved=true detections=7",
        )

        error = RuntimeError("inference failed")
        failure = module.format_failure_log(
            "scene-b", 456.0, "token-b", 0.0, False, error
        )
        self.assertEqual(
            failure,
            "scene=scene-b timestamp=456.0 token=token-b prev_exists=0 "
            "target=false saved=false FAILED RuntimeError: inference failed",
        )

    def test_target_output_validation_requires_exact_token_set(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            targets = {"target-a", "target-b"}
            for token in targets:
                (output_dir / "{}.pt".format(token)).touch()

            module.validate_target_outputs(targets, output_dir)
            (output_dir / "target-b.pt").unlink()
            (output_dir / "extra.pt").touch()
            with self.assertRaisesRegex(
                RuntimeError,
                r"missing_files=\['target-b'\] extra_files=\['extra'\]",
            ):
                module.validate_target_outputs(targets, output_dir)


if __name__ == "__main__":
    unittest.main()
