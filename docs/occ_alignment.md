# OCC Alignment

```json
{
  "occ_config": "C:\\Users\\31722\\Desktop\\Research\\QUEST\\third_party\\OpenScene\\DriveEngine\\projects\\configs\\bevformer\\bev_tiny_occ_r50_nuplan.py",
  "axis_order": "QUEST uses [B, C_occ, X, Y, Z]; OpenScene sparse labels use flat voxel index plus semantic id.",
  "x_definition": "X spans point_cloud_range[0] -> point_cloud_range[3]",
  "y_definition": "Y spans point_cloud_range[1] -> point_cloud_range[4]",
  "z_definition": "Z spans point_cloud_range[2] -> point_cloud_range[5]",
  "voxel_origin": [
    -50.0,
    -50.0,
    -4.0
  ],
  "point_cloud_range": [
    -50.0,
    -50.0,
    -4.0,
    50.0,
    50.0,
    4.0
  ],
  "voxel_size": [
    0.5,
    0.5,
    0.5
  ],
  "expected_grid_from_config": {
    "X": 200,
    "Y": 200,
    "Z": 16
  },
  "semantic_ids": "OpenScene/nuPlan config occupancy_classes=11 in pts_bbox_head; exact ids must be validated against dataset docs before KD.",
  "empty_free_class": "Current QUEST dense loader initializes missing sparse voxels to class 0; verify whether class 0 is free/empty before OCC KD.",
  "quest": {
    "C_occ": 11,
    "X": 200,
    "Y": 200,
    "Z": 16
  },
  "alignment_status": "STRUCTURE_MATCHES",
  "note": "No reshape-based conversion is allowed; KD must align by physical voxel centers and semantic mapping."
}
```
