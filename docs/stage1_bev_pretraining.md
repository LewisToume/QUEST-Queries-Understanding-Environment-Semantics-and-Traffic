# Stage 1 BEV Pretraining

Stage 1 trains only `GeometryAwareBEVLift`, `BEVEncoder`, and a temporary
`BEVAuxiliaryHead`. DINOv2 and every Agent, Map, Segmentation, and Depth module
remain frozen. The forward path ends at the encoded `[B,384,32,32]` BEV and does
not invoke Agent proposals, queries, decoders, or heads.

The auxiliary head predicts foreground logits `[B,32,32]` and two-class logits
`[B,32,32,2]`. Canonical Navformer vehicle and pedestrian centers are mapped
using the `x_range`, `y_range`, `bev_h`, and `bev_w` stored by the active
`GeometryAwareBEVLift`. If targets collide, the higher-confidence teacher target
sets the cell class. Unsupported classes are ignored.

Foreground positive and negative BCE terms are normalized separately. Class
cross entropy is evaluated only at occupied center cells. Epoch logs include
target counts, collision counts, foreground probabilities, and gradient RMS for
all three trainable modules.

```bash
python scripts/train_stage1_bev.py --num-samples 500 --epochs 5
python scripts/evaluate_stage1_bev.py --start 500 --count 100
```

The checkpoint defaults to `checkpoints/quest_stage1_bev.pt` and stores separate
state dictionaries for Geometry lift, BEV encoder, and auxiliary head. Stage 2
loads only the first two components:

```bash
python scripts/train_stage2_distill.py \
  --bev-pretrain-checkpoint checkpoints/quest_stage1_bev.pt
```
