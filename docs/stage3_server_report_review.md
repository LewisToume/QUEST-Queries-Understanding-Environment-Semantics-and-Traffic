# Review of the first Stage 3 server report

## Scope and evidence

The user supplied `report.json` and six camera/BEV figures for the legacy
`quest_stage3_map_gt_only.pt`, epoch 5. These are not results from the corrected
`ordered_l1_sum_v2` objective or its frozen-Agent recovery run.
The JSON was inspected with PowerShell; no Python, model, training or tests
were executed locally. Original reports/images remain outside the repository.

| Original index | split | Centerline ordered nearest (m) | Centerline Chamfer nearest (m) |
| --- | --- | ---: | ---: |
| 0 | train | 7.00 | 3.62 |
| 2500 | train | 7.86 | 3.94 |
| 4999 | train | 12.36 | 4.18 |
| 15165 | validation | 10.98 | 7.09 |
| 20019 | validation | 7.95 | 3.52 |
| 34901 | validation | 11.37 | 4.79 |

These are prediction-to-nearest-same-class diagnostics, not Hungarian recall.
The sampled training frames are also poor, so a pure held-out generalization
explanation is insufficient. The figures show coarse road-like shapes with
incorrect position, endpoints, directions and missing crossings/boundaries.
They do not show every query degenerating to one point.

In frame 15165, 47/57 predicted centerlines have absolute endpoint orientation
below 15 degrees from local X, versus 5/55 GT centerlines. Mean folded absolute
angles are 10.14 degrees and 51.99 degrees respectively. Predicted centerline
scores average about 0.970 despite poor geometry. This supports a directional
template bias and geometry/classification imbalance, not a confidence-threshold
solution. Endpoint angles are only a coarse polyline descriptor, not curvature.

Zero-feature point MAD is 8.21-14.95 m; other-frame feature MAD is 7.93-12.36 m.
The visual path is not completely disconnected. However, these original
statistics average all 224 queries, including unselected background queries
whose point outputs have no direct matched regression target. They do not
establish useful foreground dependence. Future diagnostics also report MAD
for real-foreground query IDs only. No extra run is required before the
already planned corrected-loss experiment.

The six frames have identical `ego2global` and `lidar2global` in the supplied
report. CAM_F0 optical axes point almost along local +X, and resized focal
lengths of 360.5 are consistent with 1545 * 448/1920. Horizontal local-X roads
are therefore not inherently an axis bug. These facts do not validate all
map-to-image alignment, map elevations, distortion handling, or other frames.
GT normalization/oracle checks likewise test arithmetic, not physical alignment.

## Same-data primary-source comparison

- [NAVSIM paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/32768f7faf1995026ef9821c696f3404-Paper-Datasets_and_Benchmarks_Track.pdf)
  describes its OpenScene-based inputs and benchmark. The official
  [Transfuser target builder](https://github.com/autonomousvision/navsim/blob/main/navsim/agents/transfuser/transfuser_features.py),
  [configuration](https://github.com/autonomousvision/navsim/blob/main/navsim/agents/transfuser/transfuser_config.py)
  and [loss](https://github.com/autonomousvision/navsim/blob/main/navsim/agents/transfuser/transfuser_loss.py)
  show metric map-to-ego conversion and dense BEV semantic cross-entropy,
  including lane/connector centerlines. Its sensor inputs and raster task differ
  from QUEST's camera-only vector endpoint, so its scores are not directly comparable.
- [OpenDriveLab Navformer config](https://github.com/OpenDriveLab/Navformer/blob/main/configs/navformer/track_map_nuplan_r50_navtrain.py)
  uses eight cameras, four image feature levels, 200 x 200 BEV, temporal context,
  deformable spatial attention and a Pansegformer head with focal/Dice supervision.
  This is the relevant OpenScene implementation, not the unrelated robot
  target-navigation NavFormer paper. Its known unusable exported map channels
  remain disabled; a stronger architecture does not certify these teacher labels.
- [DriveWorld](https://arxiv.org/abs/2405.04390) uses OpenScene for spatiotemporal
  geometric pretraining. This supports investigating the quality of metric
  features, not automatically adding a world model to QUEST's present task.
- [MapTRv2](https://arxiv.org/abs/2308.05736) is not an OpenScene result. It is
  useful as a vector-map architecture reference: hierarchical instance/point
  queries, spatial references, iterative refinement and auxiliary dense training
  are substantially richer than two instance-level decoder layers plus a
  single 40-coordinate output MLP. Copying only its scalar loss weights does
  not reproduce those mechanisms.

## Next controlled experiment

The source-confirmed reduction mismatch was already corrected in `fa75697`.
Run that explicit objective before attributing all failure to model capacity:
the existing bad checkpoint has never been optimized under it. The controlled
recovery freezes Stage 2 BEV/Agent and trains the map decoder/head with L1 sum
and intermediate supervision. Frozen BEV protects Agent but is a restriction
on map adaptation; it is not asserted to be the final architecture.

If ungated training point errors remain multi-meter and fixed-direction shapes
persist under corrected supervision, the next justified structural experiment
is a map-specific spatial representation/point decoder with dense hard-GT
supervision while preserving the Agent path. Increasing shared resolution or
training all modules without a preservation strategy would confound this test.

No raw labels, map coordinate conventions, query capacity, class taxonomy,
confidence thresholds or architecture were changed based only on these six
images. They remain hypotheses to test rather than numerical guarantees.
