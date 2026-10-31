# QUEST

QUEST V3 is an 8-camera geometry-aware perception model with shared BEV latent and four
outputs: Semantic Segmentation, Depth, Agent, Vector Map.

The central requirement is ONE student learning all four tasks from specialists
and/or real GT, with shared trainable representations and task-specific decoders.
BEV is their spatial representation, not a fifth teacher task. See the
[four-teacher design and readiness audit](docs/four_teacher_student_design.md).
The versioned Seg/Depth profile now has dataset, losses, training, evaluation and
checkpoint integration. Its runtime and accuracy still require server validation;
implemented code is not evidence of a fully trained multi-task model.

Active same-domain supervision is Navformer Agent, real nuPlan Vector Map GT
and OpenScene DVGT-1 candidate depth. Same-domain Seg teacher readiness remains
unresolved; SegFormer Cityscapes and Metric3Dv2 are out-of-domain controls, not
approved nuPlan/OpenScene teachers. MapTRv2 is a structure reference, not a required
teacher checkpoint; unreliable Navformer raster-map KD remains disabled.
See [dense teacher export](docs/dense_teacher_export.md) for asset preparation
and the eight-view semantic export entry point.
See [Seg/Depth student training](docs/seg_depth_training.md) for the runnable
dense training path, preserving the existing Agent/Map checkpoint.
The opt-in [Geometry Depth V2](docs/depth_geometry_v2.md) adds calibrated cross-view
attention and a full DPT-style decoder while freezing DINO/Agent/Map. It reuses
DVGT labels; new runtime and quality are not yet server-verified. This Depth-only
profile does not emit untrained Seg logits or claim four-task completion.

## Model

```text
8 camera images
  -> shared frozen DINOv2
  -> calibrated GeometryAwareBEVLift
  -> 32x32 metric BEV Transformer
  -> image-conditioned Agent proposals and decoder
  -> independent Vector Map query decoder

per-camera DINO patch features
  -> Semantic Segmentation head
  -> Depth head
```

The fixed OpenScene camera order is:

```text
CAM_F0, CAM_L0, CAM_R0, CAM_L1, CAM_R1, CAM_L2, CAM_R2, CAM_B0
```

Stage1 reads the official OpenScene metadata pickle directly. It currently uses
real Agent annotations; Semantic Segmentation, Depth, and Vector Map losses stay
disabled until real labels are available.

Stage2 is offline-only. Teacher inference and label export run separately, and
student training reads token-aligned Navformer Agent labels from
`data/soft_labels_navformer`.

Stage 3 now uses an independent trainable 64x64 Map BEV and six-layer spatial
instance/point decoder, with metric vector loss and dense supervision from real
nuPlan vector GT. The original 32x32 BEV/Agent stays frozen. The Stage 3 config
overrides only the Map architecture; the base Stage 1/2 model remains compatible.
See [Stage 3 spatial Map](docs/stage3_spatial_map.md) for references, loss details,
checkpoint versioning and server-only validation commands. Map teacher KD is off.

Stage 1 BEV pretraining can first supervise the Geometry lift and BEV encoder
directly from rasterized Navformer vehicle/pedestrian centers:

```bash
python scripts/train_stage1_bev.py --num-samples 500 --epochs 5
python scripts/evaluate_stage1_bev.py --start 500 --count 100
python scripts/train_stage2_distill.py \
  --bev-pretrain-checkpoint checkpoints/quest_stage1_bev.pt
```

QUEST V1/V2 checkpoints are architecture-incompatible and are explicitly rejected
by V3 checkpoint loading.

## Checks

```powershell
python -m quest.model
python scripts/check_training_ready.py
python scripts/train_stage1.py
```
