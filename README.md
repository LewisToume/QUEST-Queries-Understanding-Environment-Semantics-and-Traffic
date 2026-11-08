# QUEST: Queries for Understanding Environment Semantics and Traffic

QUEST is a research prototype for **multi-camera perception in autonomous driving**.
It studies whether one trainable scene representation can support four outputs:
3D agent detection, vectorized local maps, semantic segmentation, and metric depth.

The model has been trained and evaluated on an internal OpenScene/nuPlan split.
It is not a production system or a ready-to-run pretrained release. Datasets,
teacher weights, and trained QUEST checkpoints are not distributed here.

## Shared-Scene Model

```text
Current and previous frames (8 calibrated cameras per frame)
  -> frozen DINOv2-S image encoder
  -> shared geometric and temporal feature fusion
  -> shared 64 x 64 bird's-eye-view (BEV) representation and scene-query decoder
  -> task readouts: 3D agents, vector polylines, semantic pixels, camera depth
```

Agent and Map use object and polyline queries in the same decoder. Segmentation
and Depth read from a shared camera-plane feature pyramid with BEV and query
feedback. The scene trunk is trained jointly with all four losses. BEV is an
intermediate representation, not an output task. See the
[architecture and training design](docs/unified_scene_four_task_design.md).

| Output | Training reference |
| --- | --- |
| 3D agents | Navformer pseudo-labels for vehicles and pedestrians |
| Vector map | nuPlan ground-truth centerlines, crossings, and road boundaries |
| Semantic segmentation | SegFormer-B5 Cityscapes pseudo-labels on OpenScene images |
| Metric depth | DVGT-1 OpenScene candidate depth labels |

The teachers are used offline. Teacher agreement for Agent, Segmentation, and
Depth is not accuracy against real ground truth. SegFormer is an out-of-domain
teacher, and the depth labels remain experimental for image-plane supervision.
Map uses nuPlan vector ground truth, not raster-map distillation.

## Evaluation

The joint V1 run used 5,000 training frames and 120 held-out validation frames.
At epoch 12, semantic agreement with the teacher was 42.37% mIoU and depth
agreement was 0.103 AbsRel. Against vector ground truth, centerline precision
and recall were 5.25% and 6.63% under the internal ordered-distance protocol.
These are internal results, not official benchmark scores. Map accuracy and
several regression checks remain below the staged baseline; the joint
checkpoint does not replace it.

- [V1 validation and regression analysis](docs/unified_scene_v1_server_review.md)
- [Feature distillation](docs/unified_feature_kd_v1.md), [GradNorm](docs/unified_gradnorm_v1.md),
  and [Map reference refinement](docs/unified_map_refinement_v1.md) continuations
- [Read-only Map candidate diagnostic](scripts/audit_unified_map_candidates.py)

## Code and Reproduction

- [Shared model](quest/unified_scene.py), [training entry point](scripts/train_unified_scene.py),
  and [experiment configuration](configs/unified_scene_v1.yaml)
- [Offline label preparation](docs/dense_teacher_export.md) and
  [teacher and dataset notes](docs/nuplan_openscene_teacher_selection.md)
- [Staged vector-map baseline](docs/stage3_spatial_map.md) and
  [Segmentation/Depth baseline training](docs/seg_depth_training.md)

Run the preflight on a configured data server to verify local assets and label
provenance before training:

```bash
python scripts/train_unified_scene.py --preflight-only --experimental-unverified-depth
```
