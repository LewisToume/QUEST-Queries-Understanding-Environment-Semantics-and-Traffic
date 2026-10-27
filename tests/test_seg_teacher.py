import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from quest.frame_selection import select_indices
from quest.seg_teacher import (
    CITYSCAPES_CLASSES, SEG_PIXEL_ALIGNMENT, SEG_SCHEMA_VERSION, logits_to_seg_label,
    prepare_segformer_input, sliding_window_logits, sliding_windows, validate_seg_record, validate_teacher_config,
)
from scripts.export_segformer_pseudo import save_record_exclusive


class SegTeacherTest(unittest.TestCase):
    def test_native_taxonomy_and_model_profile(self):
        config = {"model_type": "segformer", "id2label": dict(enumerate(CITYSCAPES_CLASSES)),
                  "depths": [3, 6, 40, 3], "hidden_sizes": [64, 128, 320, 512], "decoder_hidden_size": 768}
        config["id2label"] = {str(key): value for key, value in config["id2label"].items()}
        validate_teacher_config(config)
        config["depths"] = [2, 2, 2, 2]
        with self.assertRaises(ValueError):
            validate_teacher_config(config)

    def test_resize_retains_original_aspect_ratio(self):
        rgb = np.zeros((108, 192, 3), dtype=np.uint8)
        tensor, resize = prepare_segformer_input(rgb, scale_wh=(204, 102))
        self.assertEqual(resize["resized_hw"], [102, 181])
        self.assertEqual(tensor.shape, (1, 3, 102, 181))

    def test_sliding_coverage_including_last_edge(self):
        coverage = torch.zeros(7, 13)
        for top, bottom, left, right in sliding_windows(7, 13, (5, 6), (3, 4)):
            coverage[top:bottom, left:right] += 1
        self.assertTrue(bool((coverage > 0).all()))
        with self.assertRaises(ValueError):
            list(sliding_windows(7, 13, (3, 3), (4, 4)))

    def test_overlap_is_averaged_not_summed(self):
        class Constant(nn.Module):
            def forward(self, pixel_values):
                return SimpleNamespace(logits=torch.ones(1, 19, *pixel_values.shape[-2:]))
        output = sliding_window_logits(Constant(), torch.zeros(1, 3, 7, 13), "cpu", (5, 6), (3, 4))
        torch.testing.assert_close(output, torch.ones_like(output))

    def test_label_grid_preserves_native_sky_id(self):
        logits = torch.zeros(1, 19, 4, 8)
        logits[:, 10] = 20
        seg = logits_to_seg_label(logits, (8, 16), (4, 8))
        self.assertEqual(seg["labels"].shape, (4, 8))
        self.assertTrue(bool((seg["labels"] == 10).all()))
        with self.assertRaises(ValueError):
            logits_to_seg_label(torch.zeros(1, 6, 4, 8), (8, 16), (4, 8))

    def test_record_token_camera_order_and_nonfinite_rejected(self):
        record = {"schema_version": SEG_SCHEMA_VERSION, "token": "a", "source_index": 123,
                  "camera_names": ["CAM_F0", "CAM_B0"], "label_space": "cityscapes19",
                  "class_names": CITYSCAPES_CLASSES, "pixel_alignment": SEG_PIXEL_ALIGNMENT,
                  "seg": {"labels": torch.zeros(2, 4, 8, dtype=torch.uint8),
                          "confidence": torch.ones(2, 4, 8), "valid_mask": torch.ones(2, 4, 8, dtype=torch.bool)}}
        validate_seg_record(record, "a", 123, record["camera_names"], (4, 8))
        with self.assertRaises(ValueError):
            validate_seg_record(record, "b", 123, record["camera_names"], (4, 8))
        with self.assertRaises(ValueError):
            validate_seg_record(record, "a", 123, list(reversed(record["camera_names"])), (4, 8))
        record["seg"]["confidence"][0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            validate_seg_record(record, "a", 123, record["camera_names"], (4, 8))

    def test_raw_indices_not_manifest_positions(self):
        infos = [{"token": str(index)} for index in range(20)]
        manifest = {"validation": {"frames": [{"index": 12}, {"index": 7}]}}
        self.assertEqual(select_indices(infos, 0, 2, manifest=manifest, split="validation"), [12, 7])
        with self.assertRaises(ValueError):
            select_indices(infos, 0, 1, explicit_indices=[0], manifest=manifest)

    def test_no_label_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.pt"
            save_record_exclusive({"token": "a"}, path)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_record_exclusive({"token": "b"}, path)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
