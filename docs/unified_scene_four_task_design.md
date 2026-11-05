# Shared-Scene QUEST: Architecture and Joint Supervision

Status: implementation and server test sources, NOT a trained replacement.
Keep `quest_seg_on_depth_v3_v1_best.pt` unchanged as the selected baseline.
No existing label, checkpoint, teacher repository, or environment is rewritten.

## Why the Previous Version Did Not Meet the Stronger Sharing Goal

The old combined model was one model/checkpoint and one DINO, but most downstream
learning was task-isolated: Agent/coarse BEV; fine Map lift/encoder/decoder;
Depth geometry/time/DPT; Seg reassembly/SegFormer decoder. Its frozen DINO did
not learn four-task features, and staged training deliberately prevented losses
from changing the other branches. Packaging those paths in one checkpoint does
not prove multi-task interaction or joint learning.

The proposed replacement has ONE trainable shared scene trunk after frozen
DINOv2-S. Native width 384 is retained for compatible learned-weight transfers.

```text
current + strictly previous eight-view RGB
  -> same frozen DINOv2-S (one set of weights, one call per frame)
  -> ONE shared multilevel ray/geometry + motion-compensated temporal encoder
  -> ONE shared DPT feature reassembly/fusion (no task-specific dense decoder)
  -> ONE raw-distortion-aware 64x64 BEV lift + windowed scene encoder
  -> ONE structured scene-query decoder
       object queries + polyline-point queries share attention/FFN weights
       joint instance attention exchanges object/map context
  -> calibrated BEV-to-camera ray-sweep and shared-query feedback
  -> shared camera-plane scene features + shared scene-query features
       -> light Agent class/box/velocity MLPs
       -> light Map class/point MLPs (+ training-only 3-channel raster readout)
       -> 1x1 Seg classifier, resized to 252x448
       -> small positive camera-Z readout, resized to 252x448
```

Camera-plane and BEV coordinates remain two representations of the SAME scene,
not four parallel models. Forcing every output into a BEV grid loses image-plane
boundaries and is not required for a single model. Boxes, polylines, categorical
pixels and positive metric depth necessarily have different final parameterizations.
Task query types/indexing and light readouts are not separate encoders/decoders.

Importantly, Seg/Depth receive BEV/query feedback, while Agent/Map consume the
same fused image pyramid. Every actual task loss is checked for finite/nonzero
gradients to sampled geometry, pyramid, BEV and query weights. This is more than
renaming old private branches as a shared trunk. All four outputs now consume
the causal scene representation, including real past-frame information. There
is no teacher input, target tensor or absolute global pose in the model forward.

Old V3 geometry/time/DPT weights, fine Map window/query weights, and compatible
Agent/Map/Depth readout weights are explicitly loaded with strict checks. The
old SegFormer decoder, old Agent decoder, coarse BEV and temporal Agent adapter
are NOT stored or executed by the new student. Some projections, context gates,
query type embeddings and final Seg/raster classifiers are new. Therefore initial
four-task predictions are not preserved. It requires joint retraining; no claim
that its Seg score already equals 45.96% or its depth error already equals V3.

## Labels and Losses

Current usable references are THREE neural teachers plus real vector Map GT,
not four confirmed on-domain neural teacher checkpoints:

| Task | Reference | Objective reused |
| --- | --- | --- |
| Agent | Navformer supported vehicle/pedestrian pseudo boxes | proposal objectness/offset; Hungarian class/center/size/yaw/velocity; confidence-weighted regression |
| Map | certified nuPlan vector GT | order-equivalent Hungarian matching; metric L1; focal classes; direction; GT-derived raster focal/Dice |
| Seg | SegFormer-B5 Cityscapes19 offline labels | valid-mask weighted CE; bounded inverse-sqrt TRAIN pixel class weights |
| Depth | DVGT-1 OpenScene candidate camera-Z | valid-mask log/relative/gradient objectives |

All four labels join the SAME token/raw metadata index, cameras and current
target frame. Previous-frame observations are selected by the existing
same-split/same-scene/strict-past timestamp rule with SE(3) motion compensation.
Missing/gapped history is masked. No random index-minus-one pairing, no fake
past copied from current, no training/validation scene crossing, no GT truncation.
Files and provenance are hashed; changed records abort the run.

```text
L = wa * La/sa + wm * Lm/sm + ws * Ls/ss + wd * Ld/sd
```

`s` is a FIXED loss normalizer measured before updates on 16 deterministic TRAIN
frames only; `w` is an explicit positive task weight. Normalizers are saved and
not re-estimated on validation/resume. This handles raw unit/scale differences,
NOT a proof of equal gradient magnitudes. This is not GradNorm or an uncertainty
probability model. Raw and scaled losses must remain distinguishable.

Default backpropagation sums the four normalized losses. On the first batch of
each epoch, actual per-task gradients to shared weights are audited and sampled
pairwise cosine similarities recorded. Negative cosine is evidence of local
gradient conflict, not proof of harmful validation behavior.

Optional `train.gradient_mode: pcgrad` applies the projection from the PCGrad
paper to shared-trunk gradients only; output heads keep their ordinary task
gradients. It performs extra backward passes, uses additional VRAM, and is NOT
automatically enabled or promised to improve these data. Use a separate config
and checkpoint to compare it with summed gradients. No task is silently zeroed
to make the total loss look good. DINO stays frozen by explicit design; the
post-DINO common scene trunk learns from all four objectives.

## Accuracy and Positive/Negative Transfer

There is no mathematical guarantee all tasks improve under sharing. The current
checks are regression alarms, not certification of real-world accuracy:

1. Re-evaluate the selected OLD combined checkpoint on the exact same 120 frames,
   masks, class mappings, confidence thresholds and matching distances.
2. Evaluate all four tasks after EVERY epoch, not just training loss or a mean.
3. Inspect Agent vehicle AND pedestrian precision/recall; all three Map classes;
   Seg mIoU AND adequately represented class IoUs; Depth AbsRel/delta1 AND far bin.
4. Store per-class/per-camera/per-scene results in timestamped JSON reports.
5. Save every epoch for recovery, but write `_qualified.pt` only when ALL configured
   no-regression checks pass against the selected baseline. It is the latest
   passing checkpoint, NOT a claim of absolute qualification or best official mAP.
6. Old models and old scores remain available even if the new sharing experiment
   fails. There is no automatic replacement of the selected model.

Default development tolerances are explicit in YAML: Seg mIoU -1 point; Depth
AbsRel +5% relative and delta1 -1 point; Agent class precision/recall -2 points;
Map class precision/recall -1 point; Seg supported class IoU -5 points; far-depth
AbsRel +10% relative. Set stricter/zero tolerances in a NEW config when required.
Weak baseline Map scores cannot make a passing gate equivalent to good mapping.

Potential complementary information (hypotheses, NOT measured improvements here):

- Depth can help distinguish perspective/metric position; Seg can distinguish
  road/sidewalk/vehicle surfaces and support geometric boundaries.
- Object evidence and map layout can provide shared scene context.
- Shared real history can stabilize features under partial occlusion.

Also possible: pseudo-label errors reinforce each other; Map far/unobservable
geometry competes with visible-pixel Seg; static map history and moving objects
conflict; frequent road/building pixels overwhelm rare classes; student causal
context differs from teacher exports that allowed future context. More layers
or sharing alone does not resolve these problems. No hard constraint equates
image lane paint, map centerlines, sidewalks and 3D road boundaries.

To establish "1+1>2", compare JOINT training with SINGLE-TASK training on this SAME
new architecture, initialization, frame list, optimizer schedule and step budget.
The runner's `--only-task agent|map|seg|depth` creates separate `_only_TASK.pt`
controls without changing trunk capacity. Those runs are explicitly NOT marked
four-task qualified. Start with one seed for debugging; use multiple matched
seeds and scene-level uncertainty before a research claim. Comparing new joint
scores only to the old different architecture is a practical baseline comparison,
not an isolated causal demonstration of positive transfer.

Only three held-out scenes currently exist in this split. More independent
held-out scenes and rare-object coverage are needed for reliable generalization
claims. Seg/Depth/Agent teacher agreement is still not real GT accuracy. A small
independently annotated Seg set and aligned LiDAR camera-Z would be useful next;
neither is required to reuse existing labels for this implementation check.

## Server Workflow

Nothing runs locally. First test the new paths and all affected old V3 APIs:

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -p 'test_unified_scene.py' -q
CUDA_VISIBLE_DEVICES=3 python -u scripts/train_unified_scene.py --num-samples 2 --smoke-steps 4 --experimental-unverified-depth
```

The smoke run must show actual updates to common geometry and all active readouts,
four-task gradients reaching common geometry/pyramid/BEV/query, and unchanged
DINO state. It does NOT establish accuracy and writes no checkpoint.

One-command tests -> smoke -> 5000 frames / 12 jointly trained epochs:

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/train_unified_scene.py --run-all --experimental-unverified-depth
```

All subprocess failures stop the workflow. Paths and epochs are in
`configs/unified_scene_v1.yaml`. New main checkpoint: `checkpoints/quest_unified_scene_v1.pt`.
Reports: `data/diagnostics/unified_scene_v1/`. Explicit resume:

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/train_unified_scene.py --resume checkpoints/quest_unified_scene_v1.pt --experimental-unverified-depth
```

Single-task control (a separate run, not needed for the first smoke):

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/train_unified_scene.py --only-task depth --experimental-unverified-depth
```

## Primary Research References and Scope

- [MQTransformer original paper](https://arxiv.org/abs/2205.14354) and
  [authors' code](https://github.com/yangyangxu0/MQTransformer): shared encoder,
  cross-task query reasoning and shared decoder provide a directly relevant
  sharing principle. The paper covers dense prediction, not this four-task
  OpenScene setting; the proposed geometry/BEV/structured-query combination is
  an adaptation, not a claimed reproduction of that architecture.
- [MultiMAE official project](https://multimae.epfl.ch/) and
  [code](https://github.com/EPFL-VILAB/MultiMAE): shared representations with output
  adapters support a single multi-task model; it is not our driving-task recipe.
- [UniAD, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.pdf):
  shared query interfaces/task interaction and staged/end-to-end learning, NOT
  proof that adding tasks helps every perception metric. Its benchmark is
  nuScenes; do not relabel it as a nuPlan checkpoint for this project.
- [PCGrad original paper](https://arxiv.org/abs/2001.06782) and
  [authors' code](https://github.com/tianheyu927/PCGrad): conflicting-gradient
  projection; an optional optimizer experiment, not a validated QUEST result.
- [GradNorm original paper](https://proceedings.mlr.press/v80/chen18a/chen18a.pdf):
  explicitly addresses task gradient/learning-rate imbalance. The simpler fixed
  initial TRAIN-loss normalization here must not be claimed as GradNorm.

The new combination is an engineering adaptation for QUEST. It is not a faithful
reproduction of any one paper, and its performance must be established on server.
