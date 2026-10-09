import unittest
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
from shapely.geometry import Point

from quest.nuplan_map_locator import (
    MAP_EXPORT_REQUIRED_LAYERS, MAP_LAYER_AUDIT_VERSION, NuPlanMapLocator,
    check_map_layer_counts, projected_map_bounds,
    validate_map_layer_audit,
)
from quest.vector_map_labels import global_to_local_geometry, require_lidar2global


class NuPlanMapLocatorTest(unittest.TestCase):
    def test_preexisting_invalid_geometry_is_checked_again_for_each_frame(self):
        from shapely.geometry import box

        cached_layer = {"source_rows": 1, "api_rows": 1, "source_invalid_rows": 1,
                        "api_invalid_rows": 1,
                        "existing_invalid": {"7": (box(100, 100, 101, 101),
                                                   box(100, 100, 101, 101))}}
        cached = {"city": {layer: dict(cached_layer) for layer in MAP_EXPORT_REQUIRED_LAYERS}}
        first = np.eye(4)
        first[:2, 3] = [0, 0]
        info = {"token": "first", "lidar2global": first}
        safe = check_map_layer_counts(None, "city", info, (-1, -1, 1, 1), cache=cached)
        validate_map_layer_audit(safe)
        self.assertEqual(safe["per_layer"]["baseline_paths"]["preexisting_invalid_outside_roi"], 1)
        second = np.eye(4)
        second[:2, 3] = [100, 100]
        with self.assertRaisesRegex(ValueError, "affects this frame ROI"):
            check_map_layer_counts(None, "city", {"token": "second", "lidar2global": second},
                                   (-1, -1, 1, 1), cache=cached)

    def test_new_invalid_or_missing_rows_are_blocked(self):
        row = {"source_rows": 1, "api_rows": 1, "source_invalid_rows": 0,
               "api_invalid_rows": 0, "preexisting_invalid_outside_roi": 0,
               "preexisting_invalid_examples": [], "new_invalid_rows": 0, "missing_rows": 0}
        safe = {"version": MAP_LAYER_AUDIT_VERSION, "status": "verified_for_frame_roi",
                "per_layer": {layer: dict(row) for layer in MAP_EXPORT_REQUIRED_LAYERS}}
        validate_map_layer_audit(safe)
        safe["per_layer"]["baseline_paths"]["new_invalid_rows"] = 1
        with self.assertRaisesRegex(ValueError, "unresolved"):
            validate_map_layer_audit(safe)
        safe["per_layer"]["baseline_paths"]["new_invalid_rows"] = 0
        safe["per_layer"]["baseline_paths"]["missing_rows"] = 1
        with self.assertRaisesRegex(ValueError, "unresolved"):
            validate_map_layer_audit(safe)

    def test_geographic_gpkg_extent_projects_to_city_utm(self):
        from pyproj import Transformer

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.gpkg"
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE meta (key TEXT, value TEXT)")
                db.execute("INSERT INTO meta VALUES (?, ?)", ("projectedCoordSystem", "EPSG:32611"))
                db.execute("CREATE TABLE gpkg_spatial_ref_sys (srs_id INTEGER, organization TEXT, "
                           "organization_coordsys_id INTEGER, definition TEXT)")
                db.execute("INSERT INTO gpkg_spatial_ref_sys VALUES (4326, 'EPSG', 4326, '')")
                db.execute("CREATE TABLE gpkg_contents (data_type TEXT, srs_id INTEGER, "
                           "min_x REAL, min_y REAL, max_x REAL, max_y REAL)")
                db.execute("INSERT INTO gpkg_contents VALUES ('features', 4326, -115.2, 36.1, -115.1, 36.2)")
            x0, y0, x1, y1 = projected_map_bounds(path)
            x, y = Transformer.from_crs("EPSG:4326", "EPSG:32611", always_xy=True).transform(-115.15, 36.15)
            self.assertLess(x0, x)
            self.assertLess(x, x1)
            self.assertLess(y0, y)
            self.assertLess(y, y1)

    def test_full_3d_rotation_uses_lidar_origin_height_for_planar_map(self):
        pitch = np.deg2rad(12.0)
        yaw = np.deg2rad(30.0)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        rotation = np.array([[cy * cp, -sy, cy * sp],
                             [sy * cp, cy, sy * sp],
                             [-sp, 0.0, cp]])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = [1000.0, 2000.0, 850.0]
        require_lidar2global({"lidar2global": matrix})
        local = global_to_local_geometry(Point(1000.0, 2000.0), matrix)
        np.testing.assert_allclose([local.x, local.y], [0.0, 0.0], atol=1e-9)
        local = global_to_local_geometry(Point(1010.0, 2000.0), matrix)
        expected = rotation.T @ np.array([10.0, 0.0, 0.0])
        np.testing.assert_allclose([local.x, local.y], expected[:2], atol=1e-9)

    def test_reflection_rejected_but_roll_allowed(self):
        matrix = np.eye(4)
        matrix[:3, :3] = np.diag([1.0, -1.0, -1.0])
        require_lidar2global({"lidar2global": matrix})
        matrix[:3, :3] = np.diag([1.0, 1.0, -1.0])
        with self.assertRaisesRegex(ValueError, "determinant"):
            require_lidar2global({"lidar2global": matrix})

    def test_city_resolution_and_scene_consistency(self):
        locator = NuPlanMapLocator.__new__(NuPlanMapLocator)
        locator.bounds = {"city-a": (0.0, 0.0, 10.0, 10.0),
                          "city-b": (20.0, 20.0, 30.0, 30.0)}
        locator.scene_locations = {}
        matrix = np.eye(4)
        matrix[:2, 3] = [5.0, 5.0]
        self.assertEqual(locator.resolve({"lidar2global": matrix, "scene_token": "scene"}), "city-a")
        matrix[:2, 3] = [25.0, 25.0]
        with self.assertRaisesRegex(ValueError, "changed map"):
            locator.resolve({"lidar2global": matrix, "scene_token": "scene"})
        with self.assertRaisesRegex(ValueError, "disagrees"):
            locator.resolve({"lidar2global": matrix, "scene_token": "other",
                             "map_location": "city-a"})
        matrix[:2, 3] = [15.0, 15.0]
        with self.assertRaisesRegex(ValueError, "matches 0"):
            locator.resolve({"lidar2global": matrix, "scene_token": "other"})
        locator.bounds["city-c"] = (10.0, 10.0, 20.0, 20.0)
        matrix[:2, 3] = [10.0, 10.0]
        with self.assertRaisesRegex(ValueError, "matches 2"):
            locator.resolve({"lidar2global": matrix, "scene_token": "other"})


if __name__ == "__main__":
    unittest.main()
