# Shared Feature Distillation V1

Implemented experiment; server tests/training still required. This is an
adaptation of teacher-guided feature transfer, not a reproduction of LFG or the
online single-task-training method. No accuracy improvement is established yet.

## Scope and Honest Teacher Identity

The student is the EXISTING `quest_unified_scene_v1.pt`, normally epoch 12. Its
architecture, all four original losses, masks, class balance, task loss scales,
Map reference behavior, and fixed-weight gradient summation are unchanged.
No GradNorm/PCGrad or extra decoder blocks are added in this experiment.

Training guides are selected Seg/Depth modules from the validated pre-integration
`quest_seg_on_depth_v3_v1_best.pt`. They are frozen PREVIOUS QUEST proxy guides,
NOT the original SegFormer-B5 or DVGT-1 specialist models. Original specialist
pseudo labels and real Map GT continue supervising all four output tasks.
This experiment tests preservation of already-learned dense knowledge during
sharing; it does not establish direct specialist feature distillation.

Only the proxy Seg branch and Depth frontend are held on GPU. Original Agent/Map
stacks and the proxy image backbone are not part of the guide. Exact frozen DINO
weight fingerprints and layer/camera orders are checked; current and previous
DINO features are computed once each and reused by student and guides. Both
receive the same observations, with no added augmentation or future-frame input.

## One Shared Tensor, Two Training Constraints

The student exposes its final shared dense feature tensor only on an explicit
training call. Default inference outputs and state-dict keys are unchanged.

- Semantic target: proxy SegFormer decoder input to its final classifier.
- Geometry target: proxy V3 DPT decoder input to its metric readout.
- Both are adaptively pooled to a fixed 18x32 grid per camera.
- Student features feed a training-only 1x1 semantic dimension adapter. Depth
  uses identity channel alignment, not a learned adapter that could absorb error.
- Each feature objective combines cosine distance and channel-RMS-normalized L1.
- Targets are stop-gradient. Valid masks come from the existing aligned semantic
  and depth labels; depth range filtering is unchanged. At least half the pixels
  contributing to a pooled validity cell must be valid.

```text
L = existing four-task normalized loss
    + 1.0 * semantic_feature_KD
    + 0.1 * geometry_feature_KD
```

These two KD coefficients are explicit experimental hyperparameters, not claimed
paper constants or guarantees. The four task normalizers are inherited from V1,
not silently recalibrated. Scalar-loss balance and gradient conflict remain
separate experiments. KD gradients must reach common geometry and the semantic
adapter; the guides and DINO state must remain unchanged.

## Controlled Continuation

Default budget: six EXTRA epochs over the SAME 5000 frames. Restart AdamW with
shared/new/head/adapter learning rates of 3e-6/3e-5/1e-5/1e-4 and a six-epoch
cosine schedule. This is an explicit new phase, not an exact optimizer resume
from V1. The no-KD control uses the same start, seed, frames, optimizer, rates,
schedule, and student capacity. A gain relative to V1 alone cannot distinguish
KD from extra optimization; run the matched control before making that claim.

The full 5000/120 label/context audits must exactly equal V1's audits. Smoke
sample overrides only limit the iterator AFTER the full audit, so they do not
recompute class weights or weaken label provenance. Missing/changed labels abort.
The selected pre-integration guide file is pinned to V1's saved baseline hash.

## Server Commands

One command inside the server project performs CPU tests, four-step smoke, then
5000-frame/six-epoch continuation, stopping on the first error:

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 python -u scripts/train_unified_feature_kd.py --run-all --experimental-unverified-depth
```

Short checks only:

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -p 'test_shared_feature_distillation.py' -q
CUDA_VISIBLE_DEVICES=3 python -u scripts/train_unified_feature_kd.py --num-samples 2 --smoke-steps 4 --experimental-unverified-depth
```

Matched no-distillation control, a SEPARATE run not automatically doubling the work:

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/train_unified_feature_kd.py --control --experimental-unverified-depth
```

Resume this experiment, preserving optimizer/scheduler/adapter/RNG state:

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/train_unified_feature_kd.py --resume checkpoints/quest_unified_feature_kd_v1.pt --experimental-unverified-depth
```

No new pseudo-label export, LiDAR, installation, or model download is requested.
Training needs additional guide compute/memory; the old 3.2GiB peak is NOT a
prediction of the new peak. Actual memory is printed by the server smoke run.

## Artifacts and Acceptance

- Full resumable training: `checkpoints/quest_unified_feature_kd_v1.pt`.
- Student-only inference: `checkpoints/quest_unified_feature_kd_v1_deploy.pt`.
- Latest all-gates-passing student-only artifact, if any:
  `checkpoints/quest_unified_feature_kd_v1_qualified_deploy.pt`.
- Control filenames append `_control` before the artifact suffixes.
- Per-epoch detailed metrics: `data/diagnostics/unified_feature_kd_v1/`.

Full training checkpoints store the small adapter separately for resume; they
never store the frozen guide state. Deployment payloads contain ONLY the original
UnifiedQUEST architecture/state, validation metadata and experiment identity,
not adapters, guides, or optimizers. Load a deployment payload with
`quest.shared_feature_distillation.load_deployment`, then use the existing unified
model inputs. It requires the local student DINO runtime, not proxy guide files.

Validation runs with student alone. All prior task/class/far-depth gates remain
unchanged against the selected pre-integration baseline, not the weaker V1.
Student-only export does NOT itself mean qualified: only the separately named
qualified artifact passes the configured no-regression checks. Even those checks
are not real-GT certification and do not fix the weak baseline Map quality.

Tests cover feature-mask shape/empty rejection, target stop-gradient, teacher and
DINO immutability, genuine common-trunk KD gradients, no extra DINO forward calls,
unchanged student predictions when merely requesting guide targets, strict parent
loading, and standalone deployment roundtrip with no guide/adapter model keys.
