# Teacher Integration Status

Updated: 2026-09-07T03:23:52

## Summary

| Teacher | Task | Environment | Repo | Config | Checkpoint | Camera count | Camera mapping | Build | Real inference | Raw output | Conversion | GT evaluation | Remaining blocker |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| StreamPETR | agent | BLOCKED | OK | OK | OK | 6 | {"CAM_F0": "FRONT", "CAM_R0": "FRONT_RIGHT", "CAM_R2": "BACK_RIGHT", "CAM_B0": "BACK", "CAM_L2": "BACK_LEFT", "CAM_L0": "FRONT_LEFT"} | BLOCKED | BLOCKED | none | BLOCKED | BLOCKED | missing dependencies: mmcv, mmdet, mmdet3d |
| MapTRv2 | map | BLOCKED | OK | OK | OK | 6 | {"CAM_F0": "FRONT", "CAM_R0": "FRONT_RIGHT", "CAM_R2": "BACK_RIGHT", "CAM_B0": "BACK", "CAM_L2": "BACK_LEFT", "CAM_L0": "FRONT_LEFT"} | BLOCKED | BLOCKED | none | BLOCKED | BLOCKED | missing dependencies: mmcv, mmdet, mmdet3d |
| OccNet | occ | BLOCKED | OK | OK | BLOCKED | 8 | OpenScene native | BLOCKED | BLOCKED | none | BLOCKED | BLOCKED | checkpoint not found; searched: ['C:\\Users\\31722\\Desktop\\Research\\QUEST\\weights', 'C:\\Users\\31722\\Desktop\\Research\\QUEST\\third_party']; missing dependencies: mmcv, mmdet, mmdet3d |
| ViDAR | future_world | BLOCKED | OK | OK | OK | 8 | OpenScene native | BLOCKED | BLOCKED | none | BLOCKED | BLOCKED | missing dependencies: mmcv, mmdet, mmdet3d |

## Details

### StreamPETR -> agent

- environment: `BLOCKED`
- repo_path: `C:\Users\31722\Desktop\Research\QUEST\third_party\StreamPETR`
- config_path: `C:\Users\31722\Desktop\Research\QUEST\third_party\StreamPETR\projects\configs\StreamPETR\stream_petr_r50_flash_704_bs2_seq_90e.py`
- checkpoint_path: `C:\Users\31722\Desktop\Research\QUEST\third_party\StreamPETR-main\ckpts\stream_petr_r50_flash_704_bs2_seq_90e.pth`
- camera_count: `6`
- camera_mapping: `{"CAM_F0": "FRONT", "CAM_R0": "FRONT_RIGHT", "CAM_R2": "BACK_RIGHT", "CAM_B0": "BACK", "CAM_L2": "BACK_LEFT", "CAM_L0": "FRONT_LEFT"}`
- missing_dependencies: `mmcv, mmdet, mmdet3d`
- checkpoint_keys: `state_dict`
- searched_paths: `C:\Users\31722\Desktop\Research\QUEST\weights; C:\Users\31722\Desktop\Research\QUEST\third_party`
- errors: `missing dependencies: mmcv, mmdet, mmdet3d`

### MapTRv2 -> map

- environment: `BLOCKED`
- repo_path: `C:\Users\31722\Desktop\Research\QUEST\third_party\MapTR`
- config_path: `C:\Users\31722\Desktop\Research\QUEST\third_party\MapTR\projects\configs\maptrv2\maptrv2_nusc_r50_24ep.py`
- checkpoint_path: `C:\Users\31722\Desktop\Research\QUEST\weights\teachers\maptrv2\maptrv2_nusc_r50_24ep.pth`
- camera_count: `6`
- camera_mapping: `{"CAM_F0": "FRONT", "CAM_R0": "FRONT_RIGHT", "CAM_R2": "BACK_RIGHT", "CAM_B0": "BACK", "CAM_L2": "BACK_LEFT", "CAM_L0": "FRONT_LEFT"}`
- missing_dependencies: `mmcv, mmdet, mmdet3d`
- checkpoint_keys: `meta, state_dict, optimizer`
- searched_paths: `C:\Users\31722\Desktop\Research\QUEST\weights; C:\Users\31722\Desktop\Research\QUEST\third_party`
- errors: `missing dependencies: mmcv, mmdet, mmdet3d`

### OccNet -> occ

- environment: `BLOCKED`
- repo_path: `C:\Users\31722\Desktop\Research\QUEST\third_party\OccNet`
- config_path: `C:\Users\31722\Desktop\Research\QUEST\third_party\OpenScene\DriveEngine\projects\configs\bevformer\bev_tiny_occ_r50_nuplan.py`
- checkpoint_path: `None`
- camera_count: `8`
- camera_mapping: `{}`
- missing_dependencies: `mmcv, mmdet, mmdet3d`
- checkpoint_keys: `none`
- searched_paths: `C:\Users\31722\Desktop\Research\QUEST\weights; C:\Users\31722\Desktop\Research\QUEST\third_party`
- errors: `checkpoint not found; searched: ['C:\\Users\\31722\\Desktop\\Research\\QUEST\\weights', 'C:\\Users\\31722\\Desktop\\Research\\QUEST\\third_party']; missing dependencies: mmcv, mmdet, mmdet3d`

### ViDAR -> future_world

- environment: `BLOCKED`
- repo_path: `C:\Users\31722\Desktop\Research\QUEST\third_party\ViDAR`
- config_path: `C:\Users\31722\Desktop\Research\QUEST\third_party\ViDAR\projects\configs\vidar_pretrain\OpenScene\vidar_OpenScene_mini_full_3future.py`
- checkpoint_path: `C:\Users\31722\Desktop\Research\QUEST\weights\teachers\vidar\vidar_openscene_mini_full_3future.pth`
- camera_count: `8`
- camera_mapping: `{}`
- missing_dependencies: `mmcv, mmdet, mmdet3d`
- checkpoint_keys: `meta, state_dict`
- searched_paths: `C:\Users\31722\Desktop\Research\QUEST\weights; C:\Users\31722\Desktop\Research\QUEST\third_party`
- errors: `missing dependencies: mmcv, mmdet, mmdet3d`

## Current Assessment

- StreamPETR real inference: BLOCKED_ENVIRONMENT (missing dependencies: mmcv, mmdet, mmdet3d).
- MapTRv2 real inference: BLOCKED_ENVIRONMENT (missing dependencies: mmcv, mmdet, mmdet3d).
- OccNet real inference: BLOCKED_CHECKPOINT (checkpoint not found; searched: ['C:\\Users\\31722\\Desktop\\Research\\QUEST\\weights', 'C:\\Users\\31722\\Desktop\\Research\\QUEST\\third_party']; missing dependencies: mmcv, mmdet, mmdet3d).
- ViDAR real inference: BLOCKED_ENVIRONMENT (missing dependencies: mmcv, mmdet, mmdet3d).
- Flow remains supervised by OpenScene Flow GT. ViDAR is recorded only as the Future World teacher.
- `quest_teacher_legacy` creation was attempted on Windows; pip failed on `torch-1.10.1+cu111` with an invalid wheel error, leaving torch/mmcv/mmdet/mmdet3d unavailable.
- No random soft labels, fake tensors, or dummy teacher outputs are generated.
- Aggregate environment status: BLOCKED. Required modules: torch, mmcv, mmdet, mmdet3d.
