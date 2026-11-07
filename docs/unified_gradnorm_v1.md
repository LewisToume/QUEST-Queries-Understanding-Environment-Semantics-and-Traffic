# GradNorm-Only Controlled Continuation

Status: implementation and server tests; no improvement claim before execution.

The feature-KD experiment ended with Seg pseudo mIoU 0.429328, versus 0.423683
at its unified-V1 start and 0.459568 in the selected pre-integration baseline.
Depth is essentially unchanged; centerline improves but pedestrian performance
falls relative to unified V1, and crossing matches return to zero. No all-task
gate passes. Extra optimization is a confound until the matched control runs.

This next experiment therefore starts from the SAME `quest_unified_scene_v1.pt`
(epoch 12), NOT from the feature-KD result. It changes task weighting only:

- Student architecture, data, four original objectives and class weights remain.
- Original train-only task loss scales remain fixed.
- No feature KD, online guide, adapter, PCGrad or reference-refinement change.
- Six extra epochs with the SAME student optimizer/LR schedule as feature-KD's
  continuation. A separately named `--control` keeps all four weights at one.
- All task/class/far-depth gates remain against the selected old baseline.

## Algorithm and Explicit Adaptations

Reference: [GradNorm, Chen et al., ICML 2018](https://proceedings.mlr.press/v80/chen18a.html),
[original algorithm](https://proceedings.mlr.press/v80/chen18a/chen18a.pdf).

For normalized task losses `l_i`, trainable weights `w_i`, and the final
high-resolution shared pyramid fusion convolution `W`:

```text
G_i = w_i * ||gradient_W(l_i)||
r_i = (l_i / l_i_reference) / mean_j(l_j / l_j_reference)
G_target_i = stop_gradient(mean_j(G_j) * r_i^alpha)
L_gradnorm = sum_i abs(G_i - G_target_i)
```

Only `L_gradnorm` updates task weights. The student receives the task-loss sum
with detached weights; weights are NOT optimized to trivially minimize that sum.
Targets, measured base gradient norms and relative loss rates are stop-gradient
for the weight optimizer. Weights stay positive and are renormalized to sum four.

This implementation uses alpha=1.5, weight Adam LR=1e-3, and updates every TEN
student steps to bound overhead. Between updates weights stay fixed. The positive
floor before renormalization is 1e-3. These are explicit engineering settings,
not claims of exact reproduction of every-step paper experiments.

Phase reference losses are the parent's final TRAIN epoch mean losses, transformed
by the unchanged original scales/weights; no validation metric sets loss weights.
They are persisted as a buffer, not silently reset on resume. This is a declared
continuation-phase reference estimate rather than the paper's first-step loss.

GradNorm does NOT guarantee lowering Depth's weight: measured gradients AND
relative training rates determine the update. Scalar loss shares alone cannot
predict it. Logs expose weights, base/weighted gradient norms and training rates;
per-epoch records also store the actual mean weighted losses.

## Server

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 python -u scripts/train_unified_gradnorm.py --run-all --experimental-unverified-depth
```

This reuses the audited continuation runner: CPU tests, four-step smoke, then
5000 frames / six extra epochs. Existing files refuse overwrite. Full label/context
audits remain required even for a small smoke iterator.

Only smoke:

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/train_unified_gradnorm.py --num-samples 2 --smoke-steps 4 --experimental-unverified-depth
```

Matched control (a separate run, not automatically doubling training time):

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/train_unified_gradnorm.py --control --experimental-unverified-depth
```

Outputs: `checkpoints/quest_unified_gradnorm_v1.pt` (training),
`checkpoints/quest_unified_gradnorm_v1_deploy.pt` (student only), and an optional
`_qualified_deploy.pt` if ALL original gates pass. Reports are in
`data/diagnostics/unified_gradnorm_v1`. Control filenames append `_control`.

The training checkpoint separately saves GradNorm weights, reference losses and
its optimizer, plus student optimizer/scheduler/RNG state. Neither balancing module
nor its optimizer exists in the deployment model. Use the same
`quest.shared_feature_distillation.load_deployment` loader for student-only exports.

Stored-history review, CPU only:

```bash
python scripts/inspect_unified_training.py --checkpoint checkpoints/quest_unified_gradnorm_v1.pt
```

This stage does not implement the independent Map reference-refinement experiment.
Passing near-zero-baseline Map checks would still not mean acceptable Map quality.
