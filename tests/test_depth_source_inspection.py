import unittest

from scripts.inspect_depth_sources import source_fields


class DepthSourceInspectionTest(unittest.TestCase):
    def test_sweep_generic_path_and_timestamp_are_reported(self):
        info = {"sweeps": [{"data_path": "lidar/frame.pcd", "timestamp": 123}]}
        fields = dict(source_fields(info))
        self.assertEqual(fields["sweeps[0].data_path"], "lidar/frame.pcd")
        self.assertEqual(fields["sweeps[0].timestamp"], 123)

    def test_sweep_inspection_is_bounded(self):
        info = {"sweeps": [{"data_path": str(index)} for index in range(4)]}
        self.assertEqual(len(dict(source_fields(info))), 2)

    def test_camera_image_paths_are_not_point_cloud_candidates(self):
        info = {"cams": {"CAM_F0": {"data_path": "image.jpg"}}, "lidar_path": "frame.bin"}
        self.assertEqual(dict(source_fields(info)), {"lidar_path": "frame.bin"})


if __name__ == "__main__":
    unittest.main()
