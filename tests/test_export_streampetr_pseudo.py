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


if __name__ == "__main__":
    unittest.main()
