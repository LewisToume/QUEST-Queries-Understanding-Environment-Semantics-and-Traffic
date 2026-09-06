# QUEST

QUEST is a runnable research prototype for current-scene 8-camera 2D-to-3D
perception on OpenScene with four explicit task lines:

1. agent
2. map
3. occ
4. flow

The current repository is an engineering skeleton, not a final production
system. The goal is to keep the main training and forward paths executable while
making the task semantics clear enough for real-data and distillation
integration. Segmentation code is preserved, but it is not part of the main
Stage1 task set.

## Current Project Layout

```text
QUEST/
  configs/      YAML config for model, stage1, and stage2 distillation
  data/         Placeholder data and future soft-label exports
  docs/         Notes and scratch docs
  quest/        Reusable Python package
  scripts/      Runnable entry points
  third_party/  External teacher-model repos kept out of the core package
  weights/      Local backbone weights, including DINOv2-small
```

## Four Task Lines

- `agent`: class, normalized 3D box, and velocity
- `map`: map class and vector/polyline points
- `occ`: 3D semantic occupancy logits
- `flow`: occupancy/voxel flow

The Stage1 OpenScene path uses fixed camera order:

```text
CAM_F0, CAM_B0, CAM_L0, CAM_L1, CAM_L2, CAM_R0, CAM_R1, CAM_R2
```

The model input is:

```python
images: [B, 8, 3, H, W]
```

## Status

- Stage1 training runs on real extracted OpenScene samples.
- Agent uses real OpenScene 3D boxes, classes, and velocity.
- OCC uses real OpenScene semantic occupancy.
- Map and flow interfaces are present, but their losses stay masked until real
  vector map GT and flow GT are wired.
- Teacher distillation is not active in Stage1.

## Quick Checks

Run from the project root:

```powershell
python scripts/check_env.py
python -m quest.backbone
python scripts/inspect_openscene.py
python -m quest.model
python scripts/train_stage1.py
```

Optional stage2 demo:

```powershell
python scripts/train_stage2_distill.py
```

## Configs

- `configs/model.yaml`: backbone, query counts, and head output dimensions
- `configs/stage1.yaml`: stage1 optimizer, dataset, and loss weights
- `configs/stage2_distill.yaml`: teacher interface settings for future
  distillation

## Notes

- The backbone is one shared frozen DINOv2 encoder reused across all 8 cameras.
- Multi-view fusion is a lightweight camera-aware Transformer using camera
  embeddings plus intrinsic/extrinsic/ego geometry features.
- The decoder uses task-specific query groups for `agent`, `map`, `occ`, and
  `flow`.
- Losses are now centralized in `quest/losses.py`.
- Teacher mapping for later Stage2 is StreamPETR -> Agent, MapTRv2 -> Map,
  OccNet -> OCC, and ViDAR -> Flow / future world.
