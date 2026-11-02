# Small DVGT-Style Geometry With Unchanged DINOv2

The latest user request supersedes the full DINOv3 native transplant plan.
Do NOT run `check_quest_native_dvgt.py --save-initialized` for this experiment.
V4 remains an optional isolated baseline, not the selected architecture.

V5 retains ONE shared, frozen DINOv2-S/14 (384 channels) and all trained
Agent/Map parameters. A learned projection produces geometry width 256.
Eight rounds each reuse the pinned official DVGT intra-view, cross-view and
cross-frame attention blocks (24 attention blocks total, heads=8, FFN ratio=4).
Native RoPE, register tokens, temporal position embeddings and the native DPT
reassembly/fusion are reused. DPT width is 128, with geometry levels 1/3/5/7.
No DINOv3 model is constructed, imported as an encoder, or downloaded.

Source: [pinned official geometry aggregator](https://github.com/wzzheng/DVGT/blob/51cf3f6d11fdff8bc7e2bbe1a88f71665ccb2236/dvgt/models/backbones/dvgt1_aggregator.py).
The native aggregator's image-backbone constructor is bypassed, because QUEST
has already encoded the real images with DINOv2. Its native attention methods
operate directly on projected real image tokens; they are not shallow BEV
history adapters. No new image encoder is run inside the geometry branch.

Important adaptations, NOT a full teacher reproduction:
- Keep raw JPEG coordinates and existing DINOv2 feature preprocessing, rather
  than silently rectifying only Depth and breaking shared image alignment.
- Direct positive scalar camera-Z DPT output, not unobservable free XYZ/pose
  heads. Existing offline targets supervise depth only; no XYZ/pose GT exists.
- Native confidence is not used as a calibrated probability or new supervision.
- Two actual causal frames when available; native T=1 at scene starts/gaps.
  Coverage in logs denotes available history, NOT correct geometric matches.
- Geometry/DPT parameters are NEW. Dimension/layer changes prevent strict reuse
  of the full DVGT weights or the incompatible V2/V3 depth weights.
- This is a trainable small geometry student/candidate, NOT already a qualified
  pretrained OpenScene teacher. Original DVGT-1 remains the actual Depth teacher.

Reuse existing 5000 train/120 validation labels and the audited scene split.
Agent/Map/DINO remain frozen. Original checkpoints are not overwritten.
Batch=1 and gradient checkpointing are explicit memory constraints; native
global cross-view attention can still cost more activation memory than V3.
Actual parameter counts, memory and learning quality require server evidence.

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -p 'test_small_dvgt_depth.py' -q
CUDA_VISIBLE_DEVICES=3 python -u scripts/train_seg_depth.py --config configs/depth_small_dvgt_v5.yaml --num-samples 2 --smoke-steps 4 --experimental-unverified-depth
CUDA_VISIBLE_DEVICES=3 python -u scripts/train_seg_depth.py --config configs/depth_small_dvgt_v5.yaml --experimental-unverified-depth
```

Smoke never saves. Full run is 5000 targets/12 epochs, new file
`checkpoints/quest_depth_small_dvgt_v5.pt`. A fresh geometry model may initially
perform worse than trained V2; mechanism similarity is not guaranteed accuracy.
The old dense evaluation entry point supports V5, including current-only
ablations. No local Python/tests/model execution was run by Codex.
