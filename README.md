# QUEST

QUEST is a runnable research prototype for a single-frame visual front-end with
four explicit task lines:

1. segmentation
2. agent
3. map
4. occ

The current repository is an engineering skeleton, not a final production
system. The goal is to keep the main training and forward paths executable while
making the task semantics clear enough for future real-data and distillation
integration.

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

- `seg`: image-plane semantic segmentation logits
- `agent`: object-level predictions for future detection/tracking expansion
- `map`: BEV map-element logits such as road, lane, and boundary
- `occ`: 3D occupancy logits

The current dummy dataset and training pipeline already use the future-facing
task interface:

```python
{
    "image": ...,
    "seg_gt": ...,
    "agent_gt": ...,
    "map_gt": ...,
    "occ_gt": ...,
}
```

## Status

- Stage 1 training is runnable with dummy data.
- Stage 2 distillation is still a demo path with placeholder soft labels.
- Real nuPlan data loading is not connected yet.
- Real teacher exports are not connected yet.

## Quick Checks

Run from the project root:

```powershell
python scripts/check_env.py
python -m quest.backbone
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
- `configs/stage2_distill.yaml`: stage2 distillation settings and placeholder
  soft-label path

## Notes

- The backbone is a frozen DINOv2 encoder.
- The decoder uses task-specific query groups for `seg`, `agent`, `map`, and
  `occ`.
- Losses are now centralized in `quest/losses.py`.
- The code is intentionally minimal so the main chain stays runnable while the
  repo transitions from demo naming to clearer research semantics.
