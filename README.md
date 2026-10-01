# QUEST

QUEST V3 is an 8-camera geometry-aware perception model with shared BEV latent and four
outputs: Semantic Segmentation, Depth, Agent, Vector Map.

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
