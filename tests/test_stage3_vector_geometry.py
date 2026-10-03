import unittest

import numpy as np
from shapely.geometry import box

from quest.vector_map_labels import process_geometry, process_road_area_polygons


class Stage3VectorGeometryTest(unittest.TestCase):
    def setUp(self):
        self.transform = np.eye(4)
        self.roi = (-1.0, -1.0, 1.0, 1.0)

    def test_polygon_covering_roi_does_not_create_roi_box(self):
        result = process_geometry(box(-2, -2, 2, 2), self.transform, self.roi)
        self.assertEqual(result, [])

    def test_partial_polygon_has_no_artificial_crop_edge(self):
        result = process_geometry(box(-2, -0.5, 0.5, 0.5), self.transform, self.roi)
        self.assertTrue(result)
        self.assertTrue(all(not closed for _, closed, _ in result))
        for points, _, _ in result:
            segments = np.diff(points, axis=0)
            fake_left_edge = (np.isclose(points[:-1, 0], -1) &
                              np.isclose(points[1:, 0], -1) &
                              (np.abs(segments[:, 1]) > 1e-4))
            self.assertFalse(fake_left_edge.any())

    def test_touching_road_polygons_have_no_internal_seam(self):
        road = [box(-1, -0.5, 0, 0.5), box(0, -0.5, 1, 0.5)]
        result = process_road_area_polygons(road, self.transform, (-2, -2, 2, 2), 20, 0.1)
        self.assertEqual(len(result), 1)
        points, closed, _ = result[0]
        self.assertTrue(closed)
        ring = np.concatenate((points, points[:1]), axis=0)
        seam = (np.isclose(ring[:-1, 0], 0) & np.isclose(ring[1:, 0], 0) &
                (np.abs(np.diff(ring[:, 1])) > 1e-4))
        self.assertFalse(seam.any())

    def test_clipped_crosswalk_boundary_is_open(self):
        result = process_geometry(box(-2, -0.5, 0.5, 0.5), self.transform, self.roi)
        self.assertTrue(result)
        self.assertTrue(all(not closed for _, closed, _ in result))

    def test_contained_crosswalk_boundary_is_closed(self):
        result = process_geometry(box(-0.5, -0.5, 0.5, 0.5), self.transform, self.roi)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0][1])


if __name__ == "__main__":
    unittest.main()
