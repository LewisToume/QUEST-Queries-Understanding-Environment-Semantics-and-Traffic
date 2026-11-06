# Shared Scene V1 Server Review

Evidence: user-provided epoch-12 validation on 120 frames; all-pass gate false.
No new training or model-quality claims are made by this review.

## Observed Result

- Seg pseudo mIoU: 0.423683; selected prior combined model was 0.459568
  (a 3.59 percentage point decline). Pixel agreement alone conceals this.
- Depth pseudo AbsRel: 0.102585, RMSE 5.126728m, delta1 0.878840;
  overall and far-range baseline checks pass.
- Vehicle precision/recall: 0.257794 / 0.385123; precision check fails.
- Pedestrian precision/recall: 0.259963 / 0.250670; both checks pass.
- Map centerline precision/recall: 0.052503 / 0.066344; both checks fail.
- Map crossing recall remains only 0.002433 and boundary recall 0.018727.
  Passing near-zero-baseline checks cannot establish acceptable Map quality.
- Seg class checks fail for sidewalk, wall, traffic light, traffic sign,
  truck and motorcycle.

The model has demonstrated shared forward/backward paths, but has NOT demonstrated
four-task no-regression or positive transfer. Keep the prior selected checkpoint.
Do not relax gates to call this result qualified, and do not assume more epochs
will solve it without inspecting the full stored history.

## Confounders Confirmed by Code Inspection

- Shared pyramid fusion was initialized from Depth V3. The previous trained
  SegFormer decoder was not transferred; the new Seg classifier is a 1x1 conv.
  This is a representational/readout change, not just multi-task optimization.
- Fine BEV now consumes shared DPT-derived features instead of the old raw DINO
  route. This changes the feature distribution seen by migrated spatial weights.
- The shared structured-query decoder uses fixed initial Map references through
  its layers. Old SpatialMapDecoder updated references after each layer using
  predicted points. Losing iterative reference refinement is a concrete fidelity
  difference, but its causal contribution needs an isolated test.
- Query depth changed from six private Map layers to four shared layers.
- Training loss magnitudes are normalized by initial train-only losses, not by
  live gradient norms. This is not proof of balanced optimization.

The first smoke batch had negative sampled gradient cosines, especially Seg/Depth,
but a single sampled batch is insufficient to diagnose persistent negative transfer.
Likewise, sharing one geometry trunk cannot make teacher pseudo-label errors vanish.

## Next Diagnostic, Without GPU or Uploaded Files

```bash
python scripts/inspect_unified_training.py
```

This reads the existing checkpoint on CPU without constructing DINO or QUEST.
It prints the exact recomputed baseline, all completed epoch scores, best epochs
per task, failed class comparisons and actual normalized mean task losses.
Best historical scores do not imply those epoch weights were retained: main is
the latest epoch; `_qualified` exists only if a saved epoch passed every gate.

Use this to distinguish early-good/late-bad trajectories from optimization that
never recovered. Subsequent changes should isolate reference refinement, dense
shared-feature/readout quality, and gradient balancing instead of changing all
of them again. Preserve the shared-trunk objective; do not silently revert to
four independent heavy decoder models.
