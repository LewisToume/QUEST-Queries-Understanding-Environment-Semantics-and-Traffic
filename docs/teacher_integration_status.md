# Teacher Integration Status

Updated: 2026-09-07T00:36:48

## Summary

| Teacher | Repo | Config | Checkpoint | Load | Deps | Inference | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| StreamPETR | OK | OK | OK | OK | BLOCKED | BLOCKED | BLOCKED |
| MapTRv2 | OK | OK | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| OccNet | OK | OK | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| ViDAR | OK | OK | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |

## Details

### StreamPETR -> agent

- repo_path: `C:\Users\31722\Desktop\Research\QUEST\third_party\StreamPETR`
- config_path: `C:\Users\31722\Desktop\Research\QUEST\third_party\StreamPETR\projects\configs\StreamPETR\stream_petr_r50_flash_704_bs2_seq_90e.py`
- checkpoint_path: `C:\Users\31722\Desktop\Research\QUEST\third_party\StreamPETR-main\ckpts\stream_petr_r50_flash_704_bs2_seq_90e.pth`
- missing_dependencies: `mmcv, mmdet, mmdet3d`
- checkpoint_keys: `state_dict`
- searched_paths: `C:\Users\31722\Desktop\Research\QUEST\weights; C:\Users\31722\Desktop\Research\QUEST\third_party`
- errors: `missing dependencies: mmcv, mmdet, mmdet3d`

### MapTRv2 -> map

- repo_path: `C:\Users\31722\Desktop\Research\QUEST\third_party\MapTR`
- config_path: `C:\Users\31722\Desktop\Research\QUEST\third_party\MapTR\projects\configs\maptr\maptr_tiny_r50_24e.py`
- checkpoint_path: `None`
- missing_dependencies: `mmcv, mmdet, mmdet3d`
- checkpoint_keys: `none`
- searched_paths: `C:\Users\31722\Desktop\Research\QUEST\weights; C:\Users\31722\Desktop\Research\QUEST\third_party`
- errors: `checkpoint not found; searched: ['C:\\Users\\31722\\Desktop\\Research\\QUEST\\weights', 'C:\\Users\\31722\\Desktop\\Research\\QUEST\\third_party']; missing dependencies: mmcv, mmdet, mmdet3d`

### OccNet -> occ

- repo_path: `C:\Users\31722\Desktop\Research\QUEST\third_party\OccNet`
- config_path: `C:\Users\31722\Desktop\Research\QUEST\third_party\OpenScene\DriveEngine\projects\configs\bevformer\bev_tiny_occ_r50_nuplan.py`
- checkpoint_path: `None`
- missing_dependencies: `mmcv, mmdet, mmdet3d`
- checkpoint_keys: `none`
- searched_paths: `C:\Users\31722\Desktop\Research\QUEST\weights; C:\Users\31722\Desktop\Research\QUEST\third_party`
- errors: `checkpoint not found; searched: ['C:\\Users\\31722\\Desktop\\Research\\QUEST\\weights', 'C:\\Users\\31722\\Desktop\\Research\\QUEST\\third_party']; missing dependencies: mmcv, mmdet, mmdet3d`

### ViDAR -> flow

- repo_path: `C:\Users\31722\Desktop\Research\QUEST\third_party\ViDAR`
- config_path: `C:\Users\31722\Desktop\Research\QUEST\third_party\ViDAR\projects\configs\vidar_pretrain\OpenScene\vidar_OpenScene_mini_1_8_3future.py`
- checkpoint_path: `None`
- missing_dependencies: `mmcv, mmdet, mmdet3d`
- checkpoint_keys: `none`
- searched_paths: `C:\Users\31722\Desktop\Research\QUEST\weights; C:\Users\31722\Desktop\Research\QUEST\third_party`
- errors: `checkpoint not found; searched: ['C:\\Users\\31722\\Desktop\\Research\\QUEST\\weights', 'C:\\Users\\31722\\Desktop\\Research\\QUEST\\third_party']; missing dependencies: mmcv, mmdet, mmdet3d`

## Current Assessment

- StreamPETR has a local checkpoint that can be loaded with `torch.load`, but MMDetection3D dependencies are missing, so real model inference is blocked.
- MapTRv2 repo and config are present, but no MapTR checkpoint was found under `weights/` or `third_party/`.
- OccNet repo and OpenScene Occ baseline config are present, but no OccNet/OpenScene checkpoint was found under `weights/` or `third_party/`.
- ViDAR repo and OpenScene config are present, but no ViDAR checkpoint was found under `weights/` or `third_party/`.
- No random soft labels or dummy teacher outputs are generated.

## Commands Run

```powershell
.venv\Scripts\python.exe scripts\check_teachers.py
.venv\Scripts\python.exe scripts\inspect_openscene.py --index 0
.venv\Scripts\python.exe scripts\export_teacher_outputs.py --num-samples 1
.venv\Scripts\python.exe scripts\train_stage2_distill.py
.venv\Scripts\python.exe scripts\eval_teachers_openscene.py --num-samples 1
```

## Actual Results

- `check_teachers.py`: all four teacher repos and configs were identified. StreamPETR checkpoint loaded successfully with top-level key `state_dict`. MapTRv2, OccNet, and ViDAR checkpoints were not found.
- `inspect_openscene.py --index 0`: sample `1d7583b3abc8553b` has all 8 camera images and OpenScene metadata. Flow GT and vector map GT are unavailable in the current extracted sample.
- `export_teacher_outputs.py --num-samples 1`: no tensor outputs were exported because every teacher is currently BLOCKED. The script wrote explicit `.blocked.json` files under `data/soft_labels/`.
- `train_stage2_distill.py`: Stage2 now uses the real 8-camera OpenScene dataset. Teacher KD was skipped for all teachers, `kd_total=0.0000`, and hard GT loss/backward/optimizer.step completed.
- `eval_teachers_openscene.py --num-samples 1`: GT summary was computed on the same OpenScene sample, but teacher evaluation is BLOCKED until real teacher inference works.

## Uncertain Or Blocked Items

- MMDetection3D runtime dependencies are missing in the current `.venv`: `mmcv`, `mmdet`, and `mmdet3d`. StreamPETR/MapTRv2/OccNet/ViDAR cannot build their original models without these packages.
- Only StreamPETR has a discovered local checkpoint: `third_party/StreamPETR-main/ckpts/stream_petr_r50_flash_704_bs2_seq_90e.pth`. This checkpoint loads with `torch.load`, but inference is blocked by missing dependencies.
- No MapTRv2 checkpoint was found under `weights/` or `third_party/`.
- No OccNet/OpenScene occupancy checkpoint was found under `weights/` or `third_party/`.
- No ViDAR checkpoint was found under `weights/` or `third_party/`.
- The current extracted OpenScene local set has only `sample_000` with all 8 camera images. The dataset filters incomplete samples so image/GT mismatch is avoided.
- MapTRv2 class mapping to OpenScene/nuPlan map classes is not verified. Map distillation must remain disabled until taxonomy and vector GT are confirmed.
- OccNet axis order, voxel range, voxel size, and semantic class mapping are documented as required conversion steps, but cannot be validated without actual OccNet inference output.
- ViDAR may not output direct flow tensors. The adapter keeps ViDAR raw future representation separate and does not fabricate flow.
