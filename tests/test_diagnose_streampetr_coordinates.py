import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "scripts/diagnose_streampetr_coordinates.py"
spec = importlib.util.spec_from_file_location(
    "diagnose_streampetr_coordinates_test", SCRIPT
)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class DiagnoseStreamPETRCoordinatesTest(unittest.TestCase):
    def test_homogeneous_point_transform(self):
        points = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
        transform = torch.eye(4)
        transform[:3, 3] = torch.tensor([10.0, -2.0, 0.5])
        transformed = diagnostics.transform_points(points, transform)
        self.assertTrue(
            torch.allclose(
                transformed,
                torch.tensor([[11.0, 0.0, 3.5], [9.0, -2.0, 1.5]]),
            )
        )

    def test_class_aware_bev_distance(self):
        distances, indices = diagnostics.class_aware_nearest_distances(
            source_centers=torch.tensor([[0.0, 0.0, 100.0], [5.0, 0.0, 0.0]]),
            source_labels=torch.tensor([0, 1]),
            gt_centers=torch.tensor([[3.0, 4.0, -100.0], [6.0, 0.0, 10.0]]),
            gt_labels=torch.tensor([0, 1]),
            dimensions=2,
        )
        self.assertTrue(torch.allclose(distances, torch.tensor([5.0, 1.0])))
        self.assertEqual(indices.tolist(), [0, 1])

    def test_class_aware_xyz_distance(self):
        distances, _ = diagnostics.class_aware_nearest_distances(
            source_centers=torch.tensor([[0.0, 0.0, 0.0]]),
            source_labels=torch.tensor([2]),
            gt_centers=torch.tensor([[1.0, 2.0, 2.0], [0.0, 0.0, 0.0]]),
            gt_labels=torch.tensor([2, 1]),
            dimensions=3,
        )
        self.assertTrue(torch.allclose(distances, torch.tensor([3.0])))

    def test_z_plus_minus_half_height(self):
        values = diagnostics.z_convention_values(
            torch.tensor([2.0, -1.0]), torch.tensor([4.0, 2.0])
        )
        self.assertTrue(torch.equal(values["base"], torch.tensor([2.0, -1.0])))
        self.assertTrue(
            torch.equal(values["plus_half_height"], torch.tensor([4.0, 0.0]))
        )
        self.assertTrue(
            torch.equal(values["minus_half_height"], torch.tensor([0.0, -2.0]))
        )

    def test_transform_field_discovery(self):
        fields = diagnostics.transform_fields(
            {
                "lidar2ego": np.eye(4),
                "lidar2ego_rotation": np.eye(3),
                "ego2global_translation": np.zeros(3),
                "unrelated": np.eye(4),
            }
        )
        self.assertEqual(
            fields,
            ["ego2global_translation", "lidar2ego", "lidar2ego_rotation"],
        )


if __name__ == "__main__":
    unittest.main()
