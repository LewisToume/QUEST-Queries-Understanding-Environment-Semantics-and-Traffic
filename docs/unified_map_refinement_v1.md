# Unified Map reference refinement

Unified V1 shares geometry, BEV and query layers across tasks but samples Map
features at fixed anchors in every query layer. This continuation restores
the iterative update already used by the spatial Map baseline. Each shared
query layer predicts Map points with the existing point readout. The next
layer uses the detached coordinates for position encoding and BEV sampling.
The existing auxiliary Map losses still supervise each layer's prediction.
No weights, task branches, teacher inference or labels are added.

Both variants start from the final Unified V1 checkpoint. They use identical
losses, loss scales, label audits, seed, optimizer schedule and evaluation.
The default variant enables refinement; `--control` keeps fixed references
and saves separate artifacts. There is no feature KD, GradNorm or PCGrad.
Enabling refinement can change predictions before any optimizer step, so the
runner records an initial validation using the selected mode.

```bash
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=3 python -u scripts/train_unified_map_refinement.py --run-all --experimental-unverified-depth
```

`--run-all` runs CPU fixtures, a four-step CUDA smoke test, then six epochs
on the original full split. Compare Map precision and recall by class at
the same thresholds and continuation budget, and check the original four-task
regression gate. That gate compares to the staged baseline; a pass alone
does not establish a gain over fixed references or adequate Map quality.
