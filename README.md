# QUEST

QUEST is an 8-camera multi-task perception model with shared BEV latent and four
outputs: Semantic Segmentation, Depth, Agent, Vector Map.

## Model

```text
8 camera images
  -> shared frozen DINOv2
  -> camera-aware multi-view fusion
  -> learned BEV encoder
  -> Agent and Vector Map query decoders

per-camera DINO patch features
  -> Semantic Segmentation head
  -> Depth head
```

The fixed OpenScene camera order is:

```text
CAM_F0, CAM_B0, CAM_L0, CAM_L1, CAM_L2, CAM_R0, CAM_R1, CAM_R2
```

Stage1 reads the official OpenScene metadata pickle directly. It currently uses
real Agent annotations; Semantic Segmentation, Depth, and Vector Map losses stay
disabled until real labels are available.

Stage2 is offline-only. Teacher inference and label export run separately, and
student training reads token-aligned files from `data/soft_labels`.

## Checks

```powershell
python -m quest.model
python scripts/check_training_ready.py
python scripts/train_stage1.py
```
