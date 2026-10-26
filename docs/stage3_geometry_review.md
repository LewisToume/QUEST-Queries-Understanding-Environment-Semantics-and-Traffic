# Stage 3 geometry review and recovery

Historical recovery experiment. The current default is documented in
[stage3_spatial_map.md](stage3_spatial_map.md); the architecture and output path
below describe the earlier controlled run, not the latest training defaults.

## Evidence and primary sources

The first server run completed 5000 frames x 5 epochs. On 120 held-out frames,
ordered 1.5 m recall was 0.004163 for centerlines and zero for crossings and
boundaries. Pedestrian recall against the same Agent pseudo labels fell from
0.231457 to 0.182306. These are user-reported server results, not local runs.

- [LFG paper](https://arxiv.org/html/2602.22091v1): dense geometric pretraining
  and frozen downstream representations. Its pseudo-4D/semantic/planning
  results are not vector-map accuracy or a direct QUEST architecture recipe.
- [MapTRv2 config](https://github.com/hustvl/MapTR/blob/maptrv2/projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py),
  [loss/cost code](https://github.com/hustvl/MapTR/blob/maptrv2/projects/mmdet3d_plugin/maptr/losses/map_loss.py),
  [head](https://github.com/hustvl/MapTR/blob/maptrv2/projects/mmdet3d_plugin/maptr/dense_heads/maptr_head.py):
  ordered L1 matching sums 20 x 2 coordinates; point regression sums errors
  and divides by positive instances. It also differs in classification,
  spatial queries, feature resolution and auxiliary tasks. QUEST is not a
  reproduction of MapTRv2.

## Corrections

The previous geometry cost averaged 40 coordinates; regression used default
SmoothL1 averaged across all coordinates. Applying similar numerical weights
to these different reductions weakened geometry relative to classification.
`ordered_l1_sum_v2` explicitly selects sum matching and per-instance L1 sum
regression. Legacy settings remain the default when the new fields are absent.
Open reversal, closed cyclic/reverse equivalence, all GT instances, taxonomy,
coordinates and ROI remain unchanged. New and old loss magnitudes cannot be
compared directly. `map_matched_point_error_m` reports ungated Hungarian-pair
mean Euclidean point error in meters, not precision or recall.

Map decoder layers receive auxiliary supervision with weights `[0.5, 1.0]`.
This exposes existing layer outputs without changing parameters, state-dict
keys, final output shapes or the V3 architecture version.

The recovery config freezes shared BEV and all Agent modules, sets
`lambda_agent: 0`, and optimizes MapDecoder/MapHead only. Agent loss remains
logged as a reference. A SHA-256 fingerprint of shared BEV/Agent parameters
and buffers is checked before saving and after loading a preservation-verified
checkpoint. This protects Stage 2 weights but prevents BEV adaptation to maps;
it is an explicit controlled recovery experiment, not a guaranteed quality fix.
Joint training remains available with `freeze_agent_and_bev: false` and a
positive Agent weight. DINO/ego MLP remain frozen in either mode.

The recovery output is `checkpoints/quest_stage3_map_geometry_v2.pt`; existing
checkpoints and labels are not rewritten. Effective objective, freeze mode,
baseline fingerprint and preservation status are recorded. Stage 2 and legacy
Stage 3 loading remain supported; no automatic resume/conversion is performed.

## Unresolved risks to measure

- Ordered point error differs from Chamfer shape distance. Both are reported
  separately with strict class-aware matching, never as official mAP.
- Instance queries directly regress 20 points: query/line collapse is possible.
  Length and query-center spread should be measured before spatial-query or
  point-decoder redesign.
- The 100 m / 32 cell grid gives 3.125 m cells. Continuous regression can exceed
  cell precision, but feature resolution is a risk at 0.5-1.5 m tolerances.
- Sparse Agent-supervised BEV is not dense geometric pretraining as in LFG.
- Virtual HD-map centerlines are not painted lane dividers. Full-ROI GT may
  include occluded or poorly observed elements. Geometric anchor coverage is
  recorded, but is not an occlusion mask and never removes GT.
- The 2D map has no measured elevation. Camera overlay would require a valid
  ground plane/elevation; diagnostics do not invent sensor heights.
- Classes are imbalanced and validation has only three scenes. Inspect errors
  by class/city before tuning weights or claiming broad generalization.

## Server diagnostic

After updating code, run in `quest_env`:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/diagnose_stage3_map.py --checkpoint checkpoints/quest_stage3_map_gt_only.pt
```

One model diagnoses three evenly spaced original training indices and three
validation indices. No training, label edits or checkpoint writes occur.
`data/diagnostics/stage3_map/report.json` and six PNG figures are produced;
nonempty output directories are not overwritten (use a new `--output-dir`).
The console prints six concise metric rows. Return the JSON and validation PNGs.

Checks include GT-as-prediction and normalization oracles, ordered/Chamfer
matches, ungated nearest distances, lengths, query-center spread, RGB panels,
calibration/pose matrices and zero/other-frame feature dependency. Feature
replacements are temporary forward hooks, never training data or persistent
model edits. Other-frame features use current calibration and preserve camera
order. The first sample has no other-frame result rather than fabricated history.

An optional same-evaluator training check is:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate_stage3_map.py --checkpoint checkpoints/quest_stage3_map_gt_only.pt --split train --diagnostic --count 20
```

Formal evaluation still requires the full validation manifest. Diagnostic
`--start` is a position within the chosen manifest; actual joins always use
original metadata indices and tokens.

After reviewing diagnostics, corrected training is:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_stage3_map.py --num-samples 5000 --epochs 5
```

These are future user-run server commands. No Python, tests, exports, inference
or training were executed locally. Added test sources can be run with
`python -m unittest discover -s tests -p 'test_stage3_*.py' -q` on the server.
