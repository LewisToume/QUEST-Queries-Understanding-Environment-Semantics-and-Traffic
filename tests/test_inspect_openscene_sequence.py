import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/inspect_openscene_sequence.py"
spec = importlib.util.spec_from_file_location("inspect_openscene_sequence", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def info(scene, timestamp, token):
    return {"scene_token": scene, "timestamp": timestamp, "token": token}


class InspectOpenSceneSequenceTest(unittest.TestCase):
    def test_infos_sort_by_scene_then_timestamp(self):
        infos = [
            info("scene-b", 2, "b2"),
            info("scene-a", 3, "a3"),
            info("scene-a", 1, "a1"),
            info("scene-b", 1, "b1"),
        ]
        sorted_infos = module.sort_infos_temporally(infos)
        self.assertEqual(
            [item["token"] for item in sorted_infos], ["a1", "a3", "b1", "b2"]
        )

    def test_analysis_detects_fragmented_and_non_monotonic_scenes(self):
        infos = [
            info("scene-a", 2, "a2"),
            info("scene-b", 1, "b1"),
            info("scene-a", 1, "a1"),
        ]
        analysis = module.analyze_sequence(infos)
        self.assertFalse(analysis["infos_already_grouped_by_scene"])
        self.assertFalse(analysis["infos_already_time_sorted_within_scene"])
        self.assertEqual(analysis["fragmented_scenes"], ["scene-a"])
        self.assertEqual(analysis["non_monotonic_scenes"], ["scene-a"])
        self.assertEqual(analysis["scene_summaries"]["scene-a"]["frame_count"], 2)


if __name__ == "__main__":
    unittest.main()
