# Geometry-Aware QUEST Depth V2

Implementation: 2026-10-31. STATIC REVIEW ONLY. No Python/tests/model execution,
export, environment changes or training were run by Codex. This is a new
Depth-only architecture profile; server runtime, memory and quality remain
unverified. Old checkpoints, labels and default training configs are unchanged.

## Architecture

```text
8 x RGB 252x448
 -> ONE frozen DINOv2-S (blocks 3/6/9/12, each 384x18x32)
 -> four trainable geometry stages, each:
      current DINO-level injection + local-LiDAR ray/optical-center embedding
      within-image attention
      calibrated cross-view ray-sweep attention
      FFN (384 -> 1536 -> 384) / residual scales
 -> four geometry-refined feature maps
 -> DPT learned reassembly (72x128, 36x64, 18x32, 9x16)
 -> four skip/residual fusion stages, width 256
 -> positive metric camera-Z [B,8,1,252,448]
```

The geometry stages preserve DVGT's within-view/cross-view interaction principle;
they do NOT reproduce its prior-free 24-block, multi-frame geometry encoder.
The Student uses current-frame images and existing calibration, never future
images or teacher targets at inference. DPT uses learned transpose/strided
convolutions and residual fusion rather than only bilinear interpolation.
Source mechanisms:
[DVGT geometry](https://github.com/wzzheng/DVGT/blob/51cf3f6d11fdff8bc7e2bbe1a88f71665ccb2236/dvgt/models/backbones/dvgt1_aggregator.py),
[DVGT DPT](https://github.com/wzzheng/DVGT/blob/51cf3f6d11fdff8bc7e2bbe1a88f71665ccb2236/dvgt/models/heads/dpt_head.py),
[DINO-compatible DPT](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py).
No teacher code/weights are substituted into the deployed Student.

## Calibrated Cross-View Attention

Use scaled K, sensor2lidar and the actual raw OpenCV distortion coefficients.
Supported coefficient lengths are 4/5/8 in OpenCV rational Brown-Conrady order;
fisheye, thin-prism and tilted-sensor models are explicitly rejected, not guessed.
The Dataset packs actual coefficients to eight entries with trailing zeros.

At DINO feature-cell centers, iteratively invert the raw lens distortion.
Reject inverse solutions with more than 0.25 pixels of forward round-trip error.
Sweep 16 log-spaced camera-Z hypotheses from 0.5m to 80m. Transform points:

```text
target camera ray * Z
 -> target sensor2lidar
 -> inverse(source sensor2lidar)
 -> source raw distorted pixels through its K and lens model
```

Each camera chooses two other cameras by optical-axis similarity computed from
calibration, NOT hardcoded camera positions/order. Sample their actual feature
maps at projected locations, use multi-head query/key weighting over source
views/depth hypotheses, and mix their values. Positive source Z, source image
boundaries, valid inverse rays and camera masks gate attention. A self-feature
candidate keeps all-masked source cases finite. Zero padding is never a valid
cross-view candidate. Visibility here means projectability, NOT independently
verified occlusion visibility. Cross-view coverage is logged; zero/low coverage
requires investigation, not declaring geometric fusion successful.

There is no global absolute ego position input. Original camera order and
extrinsics are unchanged. This fixes projection consistency INSIDE Depth's new
branch under the existing `raw_opencv` hypothesis. The preserved Agent/Map BEV
still uses its original pinhole lift; this release does NOT claim to have fixed
that old geometry route or connected Depth predictions to BEV.

## Supervision And Preservation

- Reuse all existing DVGT schema-2 candidate labels and the same 5000/120 split.
  The new Dataset profile requires DVGT labels and matching raw image geometry.
  No re-export, pseudo-label edits or independent LiDAR accuracy claim.
- Keep the existing log-depth SmoothL1, relative SmoothL1 and log-gradient loss
  weights (1, 0.5, 0.5) to isolate architecture effects. No fake point/normal or
  feature-distillation targets. The log now includes true batch `depth_absrel`;
  `depth_relative` remains a SmoothL1 loss, not an accuracy percentage.
- The backbone, Agent, Map and legacy BEV routes stay frozen and unmodified.
  New Depth modules consume the same frozen image features in a separate branch.
  A frozen-state checksum remains mandatory when saving. Test sources also
  compare unchanged Agent/Map predictions before and after a Depth-only update.
- `geometry_dpt_depth_v2` is a separate component profile with dense checkpoint
  version 2. Whole-model spatial architecture remains V3 for explicit Stage 3
  warm-start; the complete dense profile must match on strict load/resume.
  Version-1 dense checkpoints remain readable. Old shallow Depth parameters are
  NOT silently loaded into this different decoder, and Seg is not claimed trained.
- Initialization loads the existing Stage 3 Agent/Map checkpoint strictly per
  spatial module, while the new Depth modules start new. Output is
  `checkpoints/quest_depth_geometry_v2.pt`, distinct from the baseline. Existing
  output files still require explicit resume or a different output filename.
- BF16 and non-reentrant geometry-block activation checkpointing are explicit
  profile settings. Sampling/geometry and metric depth losses use fp32 where
  needed. Unsupported BF16 errors instead of altering model/resolution. Actual
  3090 memory/runtime must be measured by the user; no fit/quality guarantee.

## Server Commands

Use `quest_env`, NOT the DVGT teacher environment. These are USER-RUN commands.
No new dependencies, environment installation or teacher inference is required.

Read-only label/config/checkpoint preflight (no Student model or GPU):

```bash
conda activate quest_env && cd /home/user/DataDisk/QUEST_WORK/QUEST && git pull --ff-only origin feature/multiview-quest && python scripts/train_seg_depth.py --config configs/depth_geometry_v2.yaml --experimental-unverified-depth --preflight-only
```

One real-sample optimizer-step check, GPU3 if available; NEVER saves a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/train_seg_depth.py --config configs/depth_geometry_v2.yaml --num-samples 1 --smoke-steps 1 --experimental-unverified-depth
```

Expect shape, finite-loss/gradient and parameter-update checks, nonzero geometry
gradients, cross-view coverage, frozen Agent/Map checksum and measured peak
memory. PASS is not claimed before the server executes it. In a Depth-only
profile the available task shapes omit Seg, rather than emitting fake logits.

Full 5000-frame / 5-epoch run, after the new runtime path is checked:

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/train_seg_depth.py --config configs/depth_geometry_v2.yaml --num-samples 5000 --epochs 5 --experimental-unverified-depth
```

Same full held-out protocol and distance-bin diagnostics as the baseline:

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/evaluate_seg_depth.py --checkpoint checkpoints/quest_depth_geometry_v2.pt --output-dir data/diagnostics/depth_geometry_v2_review --preview-frames 0 --log-interval 120
```

Compare against baseline AbsRel=0.1238268, RMSE=6.0619m, delta1=0.8278686 and
50-80m signed bias=-9.4681m, on the same teacher labels and masks. Three scenes
do not establish broad generalization. No performance gain is promised by
adding attention/parameters. The original image geometry and absolute metric
accuracy still require matched real LiDAR evidence.

Server CPU test sources (not executed locally):

```bash
python -m unittest discover -s tests -p 'test_geometry_depth_v2.py' -v
```

Also rerun existing `test_seg_depth_training.py`, `test_shared_dense_perception.py`,
`test_seg_depth_diagnostics.py` and `test_dvgt_depth_adapter.py` for legacy/profile
regression. Tests use small synthetic unit fixtures, not invented training GT.
