# Calibrated Small DVGT V6

## Reason For Change

User-reported validation: V5 AbsRel 0.1141413 vs V3 0.1014647; 50-80m bias
-7.4222m vs -6.7527m. These are DVGT pseudo-label agreement, not LiDAR accuracy.
V5 removed calibrated geometry and ignored three DINO levels while changing
encoder depth/width. It was not a controlled test of native attention alone.
The exact contribution of each difference is not established by these results.

## Changes

- Keep the same shared frozen DINOv2-S, width-256/8-round native DVGT attention,
  native DPT width 128, camera layout, target masks and depth loss.
- Keep final DINO block 12 as the V5 base; add block 3/6/9 features in rounds
  0/2/4. All four real backbone levels participate after the new gates learn.
- Add four calibrated refinement stages at rounds 0/2/4/6: distortion-aware
  cross-camera ray sampling, ray/origin embeddings, and pose-compensated
  current-to-previous-camera sampling, reusing V2/V3 geometry helpers.
- Sampling checks positive camera Z, valid image bounds and real causal history.
  These masks are projection validity, NOT proof of visibility/occlusion or
  motion compensation for moving objects. Native global attention remains.
- New residual channel gates start at zero. Explicitly inherit EVERY existing
  V5 parameter; only whitelisted additions may be missing. Initial eval must
  preserve the V5 function, rather than restarting a trained encoder/decoder.
- Canonical camera IDs, distortion, transforms and split are not modified.
  Missing history uses native T=1. Report calibrated temporal coverage, not
  a fabricated constant 1. Existing teacher labels are reused unchanged.

V6 still predicts camera Z from offline depth targets, not the original
teacher's jointly supervised XYZ/pose outputs. It is a student adaptation,
not an already-qualified small nuPlan teacher. No DINOv3 weights are used.

## Data And Warm Start

Input: existing V5 checkpoint `checkpoints/quest_depth_small_dvgt_v5.pt`.
Full runs verify the saved initialization, model dimensions, loss, manifest,
label/context audits, teacher signatures and frozen Agent/Map/DINO checksum.
Smoke checks allow a subset but never save; full training requires exact data.
The optimizer is NEW, lr=3e-5, 6 additional epochs over 5000 targets. Original
V3/V5 checkpoints are never overwritten. Initialization validation is enabled;
it should reproduce the original V5 metrics up to floating-point roundoff.

Output: `checkpoints/quest_depth_calibrated_small_dvgt_v6.pt` (dense profile 6).
Existing evaluation script supports both full-history/current-only results.
Do not assume that new geometry or additional training will beat V3: measure it.

## Server Verification

No local Python/test/model execution was run by Codex. Static checks only.
Native CPU tests cover V5-to-V6 identity, all four DINO gradients, calibrated
spatial/temporal gradients, pose/calibration sensitivity, cold starts and bad
inputs. Full data/calibration checks reuse the audited existing dataset.

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -p 'test_calibrated_small_dvgt.py' -q
CUDA_VISIBLE_DEVICES=3 python -u scripts/train_seg_depth.py --config configs/depth_calibrated_small_dvgt_v6.yaml --num-samples 2 --smoke-steps 4 --experimental-unverified-depth
CUDA_VISIBLE_DEVICES=3 python -u scripts/train_seg_depth.py --config configs/depth_calibrated_small_dvgt_v6.yaml --experimental-unverified-depth
```
