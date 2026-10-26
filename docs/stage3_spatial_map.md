# Stage 3 spatial Map redesign

## Evidence and scope

The user-reported `ordered_l1_sum_v2` recovery completed 5000 frames x 5 epochs.
Its epoch-average training point error decreased from 6.50 m to 3.57 m. On the
unchanged 120-frame validation split, ordered 1.5 m recall was 0.032501 for
centerlines, zero for crossings, and 0.007491 for road boundaries. This is not
acceptable map quality. It does not prove that depth alone or parameter count
is the cause; frozen coarse features, spatial decoding, loss design, data
observability and optimization are separate risks.

Primary references inspected:

- [MapTRv2 paper](https://arxiv.org/html/2308.05736v2), especially Sections 4-5:
  instance/point queries, decoupled attention, reference-point refinement,
  ordered matching and dense auxiliary supervision.
- [Official MapTRv2 configuration](https://github.com/hustvl/MapTR/blob/maptrv2/projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py)
  and [decoder](https://github.com/hustvl/MapTR/blob/maptrv2/projects/mmdet3d_plugin/maptr/modules/decoder.py):
  six decoder layers, reference-relative deformable sampling, intermediate
  predictions, Focal classification, L1 geometry, direction and dense tasks.
- [MapTracker](https://github.com/woodfrog/maptracker): temporal map and BEV
  memory is useful, but is not introduced before single-frame fitting works.
- [Navformer OpenScene configuration](https://github.com/OpenDriveLab/Navformer/blob/main/configs/navformer/track_map_nuplan_r50_navtrain.py):
  a trainable spatial encoder and much finer BEV than QUEST's frozen Agent BEV.
  Its unreliable map teacher labels are not used by this redesign.

This is a new portable PyTorch implementation inspired by those mechanisms,
not a MapTRv2 reproduction or pretrained MapTR model. No OpenMMLab extension,
new package, downloaded weight, environment change or Navformer edit is needed.

## Architecture

```
8 RGB cameras [B,8,3,252,448]
  -> frozen DINOv2-S [B,8,384,18,32] (one shared backbone call)
      |-> existing 32x32 lift/BEV/Agent, all frozen and preservation-checked
      `-> independent trainable calibrated GeometryAwareBEVLift at 64x64
           + projected frozen 32x32 BEV context
           -> 4 alternating local/shifted-window Transformer layers
           -> 64x64 / 32x32 / 16x16 Map feature pyramid
           -> 6-layer instance/point decoder
           -> [B,Q,4] logits and [B,Q,20,2] normalized vector points
```

The finer lift samples original DINO feature maps at new metric locations. It
is not merely an upsample of the frozen BEV. Sensor2lidar inversion, scaled
intrinsics, positive camera depth and camera visibility reuse the existing
geometry implementation. Both ego MLPs remain frozen and unused. Ranges stay
at [-50,50] m in each LiDAR axis, giving 1.5625 m Map cells. Finer BEV does not
create new image detail or guarantee 1.5 m vector precision.

Window size is 8x8; alternating shifts allow neighboring windows to exchange
information. This avoids full quadratic self-attention over all 4096 cells.
Existing global 32x32 context is supplied by the frozen Agent BEV. Each decoder
layer performs point attention within an instance, attention among pooled
instances, and multi-head reference-relative sampling over all pyramid scales.
Every point has its own feature and reference. Every layer predicts a residual
in inverse-sigmoid coordinates. References are detached between layers, and all
six layer predictions are supervised. No 4480-by-4480 full point self-attention
is constructed. Out-of-range sampling locations are masked and renormalized;
all-invalid positions produce no sampled feature, not NaN or padding evidence.

Instance references cover the ROI, with small learnable circular point offsets
for nondegenerate initialization. This does not enforce closed predictions or
horizontal roads. Instances also receive features sampled at their reference
centers, so learned identities are not their only input.

The audited Q capacity, 20-point representation, three map classes, open reverse
and closed cyclic/reverse matching, tokens, original indices, camera order and
all existing GT provenance checks are unchanged. No GT is truncated.

## Loss and optimization

Objective version is `metric_focal_dense_v3`:

- Four-way softmax Focal classification (three foreground classes + background),
  with class weights [1,3,2] and background weight 0.5. Classification matching
  uses the Focal cost change from background to each candidate GT class.
  This is explicitly not MapTR's three independent sigmoid outputs.
- Ordered mean L1 **meters per point** for matching and regression. X and Y
  absolute errors are summed, then averaged over 20 points; this is not a mean
  over 40 normalized coordinates. Matching/regression use the same metric units
  and legal point orders, including asymmetric ROI normalization.
- Class-weighted point and direction regression reduces centerline dominance.
- Layer weights [0.5,0.5,0.5,0.5,0.5,1.0]. Hungarian cost is non-differentiable;
  gradients flow through matched predictions, not through assignment.
- A training-only three-channel dense Map head receives real vector GT directly.
  On-device distance-to-segment rasterization respects open/closed contours,
  includes cell footprints, allows class overlap, and processes all instances
  in memory-bounded chunks. It neither draws an ROI border nor writes labels.
  Positive/negative Focal means are balanced separately; Dice adds spatial
  overlap supervision. This head is not Navformer KD and is not the final map.

Vector weights are classification=2, metric L1=1, direction=0.1; dense Focal and
Dice each have weight 1. These are explicit starting settings, not experimentally
optimized coefficients. New loss magnitudes are not comparable to older runs.
Default training is 12 epochs with 500-step linear warmup and cosine decay to
5% of map LR (1e-4). Agent loss is logged only; its weight stays zero.

No unsupported depth targets or image-plane labels are fabricated. Auxiliary
one-to-many queries and temporal memory are not part of this implementation.

## Checkpoints and compatibility

The shared V3 Agent architecture remains version 3. The Map branch has its own
`map_architecture.version=spatial_map_v1` plus complete grid/layer settings and
an additional `map_spatial_state_dict`. The checkpoint includes all Map heads,
optimizer/scheduler state, objective, config, exact split hash, vector provenance
and shared-BEV/Agent preservation fingerprint.

Initialize only from `quest_stage2_agent_5000_e5.pt`, not from the failed Stage 3
map heads. Copy compatible lift visual-weight MLP weights; explicitly interpolate
a *copy* of the learned BEV embedding from 32x32 to 64x64 for initialization.
Recompute metric reference/position buffers at 64x64; never load incompatible
32x32 grids. The source Agent lift/encoder weights are not modified.

New output is `checkpoints/quest_stage3_map_spatial_v1.pt`. Existing checkpoints,
original labels and split/capacity audit are not overwritten. Training refuses
an existing output path. Evaluation, visualization and dependency diagnostics
reconstruct the saved architecture, allowing historical V3 Map checkpoints to
remain inspectable. Loading old Map weights directly into the new Map branch
fails explicitly. The Agent evaluator still loads only the five unchanged
shared/Agent components and ignores the new map state.

## Server validation only

Activate `quest_env` and work in `/home/user/DataDisk/QUEST_WORK/QUEST`.
No commands below were executed during authoring. Select an available GPU;
GPU 0 below is an example, not an assertion of current availability.

New CPU unit tests, with a tiny test backbone and no external weights:

```bash
python -m unittest discover -s tests -p test_stage3_spatial_map.py -q
```

First require a short real-data fitting experiment rather than another blind
full run. This repeats one original training-manifest frame, uses the audited
Q count, executes 200 optimizer steps and saves no checkpoint or labels:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/check_stage3_batch.py --overfit-samples 1 --steps 200
```

Logs are limited to the first step, every 20 steps and the final step, with
initial/final meter errors, module gradients, frozen Agent preservation, updates
and peak CUDA memory. `STAGE3_SHORT_RUN=PASS` certifies execution health only,
not fitting quality; inspect meter error rather than treating PASS as accuracy.
Use `--overfit-samples 4` to check a small varied group. This is not held-out
evaluation and never samples validation frames.

After server tests and fitting results are satisfactory:

```bash
python scripts/train_stage3_map.py --preflight-only
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_stage3_map.py
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_stage3_map.py
```

Formal train/validation remain the exact 5000/120 manifest frames. Ordered
metrics and separately requested sparse Chamfer metrics retain their existing
meaning; neither is official nuScenes mAP. Inspect all classes and an image
overlay, not just the error on the small set of successful matches.

## Remaining server-only assumptions

Static review cannot confirm actual CUDA memory/time, gradients, tiny-set fitting,
generalization, checkpoint round trips or unchanged Agent predictions. Tests
were added but not run. Single-scale 18x32 frozen DINO patches may limit road
detail; denser BEV alone cannot recover missing visual information. Full HD-map
GT includes occluded/virtual structure; BEV projection coverage does not prove
observability or physical calibration accuracy. Class weights and dense-line
width need measured ablation, not claims of guaranteed improvements.
