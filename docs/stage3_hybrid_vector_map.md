# Stage 3 hybrid vector-map training

Stage 3 starts from the Stage 2 Agent checkpoint. The production prediction is
`MapDecoder -> MapHead -> [B,N_map,4]` class logits (three map classes and
background) and `[B,N_map,20,2]` normalized local XY points. The existing
`MapHead` sigmoid output is decoded to meters for evaluation. Stage 3 constructs
the model with `C_map=3`; Stage 1/2 model configurations and checkpoints are
unchanged. `N_map` comes from the certified full train/validation capacity audit; excess instances
are an error, never silently truncated.

```
8 cameras -> frozen DINOv2 -> GeometryAwareBEVLift -> shared BEV
                                                   |-> Agent branch -> Agent loss
                                                   |-> MapDecoder/MapHead -> vector GT loss
                                                   `-> MapRasterDistillHead -> raster teacher KD
```

`MapRasterDistillHead` is training-only. It consumes `[B,384,32,32]` BEV features
and produces `[B,K,32,32]` logits. The Navformer/Pansegformer teacher provides
**bounded raster soft mask scores**, not calibrated probabilities or vector instances. Its output is only an auxiliary
KD target; inference needs neither that head nor raster post-processing.

## Coordinate contract

QUEST BEV is the OpenScene LiDAR local XY plane. Its origin is the LiDAR origin;
column index grows with local +X and row index with local +Y. `lidar2global`
maps LiDAR homogeneous points into the nuPlan global frame. Direct nuPlan map
geometry is transformed by its inverse before clipping to the model's metric
ROI. Thus global +X/+Y are **not** assumed to be LiDAR +X/+Y. The global
directions of local +X and +Y are respectively columns 0 and 1 of
`lidar2global[:3,:3]`. A physical forward/left interpretation has not been
verified from the local metadata and is deliberately not assumed. For planar
map geometry, each global XY point is placed at the current LiDAR origin's
global Z before applying the full inverse 3D pose; this is the explicit
reference-height convention. Proper pitch/roll are accepted and validated.

Navformer Pansegformer raster scores live in OpenScene **Ego XY**, while QUEST
BEV and vector GT live in **LiDAR XY**. Each QUEST cell center `[x,y,0,1]`
is transformed by that frame's `lidar2ego` before sampling the teacher grid.
The transform is read directly from metadata or derived as
`inverse(ego2global) @ lidar2global`; when both are available they must agree.
Axis direction and channel semantics still require human audit and cannot
replace this physical frame transform. Export requires an explicit teacher
Ego-frame metric range. The shared alignment utility uses bilinear sampling
with `align_corners=False` and returns a per-cell valid mask; KD excludes
cells outside the teacher range rather than treating padding as background.
The exporter fixes `teacher_coordinate_frame=openscene_ego_xy`; the former
`--teacher-coordinate-frame` argument is removed.
The visual diagnostic displays aligned maps with `origin=lower`, LiDAR origin
at `(0,0)`, and labeled local +X/+Y axes.

## Offline schemas

Vector GT, one `<token>.pt` per original metadata frame:

- `sample_index`, `token`, `scene_token`, `schema_version=4`, `coordinate_frame=openscene_lidar_xy`
- `class_ids: int64[N]`, `points_xy_m: float32[N,20,2]`
- `is_closed: bool[N]`, `length_m: float32[N]`, `xy_range_m`
- `num_points`, `min_length_m`, `map_version`, `map_location`, `map_height_reference`,
  `map_reference_global_z_m`, `vector_semantics_version`
- `geometry_diagnostics.road_area` counts valid and skipped malformed polygons
- `map_cast_audit_version`, `map_cast_diagnostics` record the baseline relation
  fields, null/invalid counts, nearby candidate linkage, and cast warning sites.

Class 0 is LANE/LANE_CONNECTOR baseline centerline; class 1 is CROSSWALK
contour; class 2 is ROADBLOCK/INTERSECTION/CARPARK_AREA boundary. Every
geometry is transformed to local meters, clipped to the ROI, split into
connected pieces, filtered for invalid/short geometry, and resampled by arc
length. For polygons, the **physical boundary is extracted before line
clipping**, so ROI edges are never created as GT. ROADBLOCK, INTERSECTION and
available CARPARK_AREA polygons are combined with `unary_union` before boundary
extraction, removing internal seams. Crosswalk polygons stay separate; clipped
fragments are open polylines, while complete uncut rings stay closed. Centerline
baseline paths retain direct line clipping. Open lines include both ends.
Closed rings use 20 unique points.
Unusable road-area polygons are counted and omitted before union; no automatic
geometry repair is attempted. An empty valid road-area set yields no road
boundary. The exporter prints these counts per frame.

Teacher raster, also one `<token>.pt` per original frame:

- `sample_index`, `token`, `schema_version=3`
- `teacher_map_soft: float32[K,H,W]` in `[0,1]`, `teacher_map_shape`
- `teacher_pc_range`, opaque `teacher_channel_names_or_ids`
- `teacher_checkpoint`, `teacher_config`, `teacher_coordinate_frame=openscene_ego_xy`
- `teacher_alignment_version=quest_lidar_to_navformer_ego_v1`, `teacher_lidar2ego`
- `teacher_score_kind=panseg_mask_score_clamped_0_1`, raw score semantics and transform
- scene-start token and SHA-256 digest of the ordered token/timestamp history

In the inspected Pansegformer `get_bboxes()` implementation, `lane_score` is a
`[K,H,W]` Tensor initialized to zero. Only selected pixels receive their raw
mask score. `score_list` is a `[Q,H,W]` Tensor; its final channel is the raw
drivable score. The export concatenates `lane_score` with `score_list[-1:]`
and applies **only** `clamp(0,1)`. It never applies sigmoid or thresholds, so
unselected lane background stays exactly zero. The bounded scores are usable
as BCE soft targets, but are not asserted to be calibrated probabilities.

Both exporters use the same `metadata["infos"][sample_index:sample_index+num_frames]`
convention as the existing Navformer Agent exporter. Stage 3 validates token,
raw metadata index, ROI, schema, channel list, metric range and checkpoint
identity when joining labels. Missing teacher data is an error unless
`missing_teacher: skip` is explicitly configured. Agent pseudo labels retain
their existing token validation. Older Agent files do not store a raw
`sample_index`; for those, unique metadata tokens establish the index join.
If an Agent file does include `sample_index`, Stage 3 checks it explicitly.
The new scripts default to the same metadata path as `configs/stage1.yaml`;
when overriding `--metadata`, point both exporters and the audit to that same
file. An older Agent export made from a different metadata ordering will fail
the token join instead of training against the wrong frame.

Exports are resumable. An existing vector file is schema/index/token/ROI and
export-provenance validated and skipped if valid; an invalid file is an error unless
`--overwrite` is explicit. Teacher export builds each target scene's timeline
from its first frame. On a partial resume it **still runs inference** through
warm-up and already-exported frames to preserve tracking/BEV memory, but only
writes missing targets. Existing teacher files are checked for schema,
index/token, score semantics, checkpoint/config identity and scene-history
digest before use. Old schema files need explicit regeneration with
`--overwrite`; starting mid-scene does not silently create a new temporal
initialization. Schema 2 Teacher labels asserted the wrong LiDAR frame and
must be regenerated, then re-audited before training.
The projected-city, planar-height and baseline-relation audit contract requires
vector schema 4; old vector GT files
must be regenerated. Training preflight rejects mixed export provenance and
the Stage 3 checkpoint records it for held-out evaluation. Re-run the map
teacher audit after regenerating GT; a verified audit made against the old
geometry cannot authorize KD with the new vector records.

## Audit gate and known external dependencies

The map teacher must have a real `seg_head` in its config **and** checkpoint.
The exporter does not substitute an Agent-only checkpoint. Channel IDs such as
`lane_score_0` are intentionally opaque. The audit reports bounded score
activity, exact and one-cell-dilated IoU, and correlation against
temporary vector-GT masks. Its JSON starts with `verified=false`, all support
flags false and no semantic mappings. Use
`scripts/visualize_navformer_map_audit.py` **before training** to inspect GT
vectors, three rasterized GT classes and all aligned teacher channels without
a Stage 3 checkpoint. A human must inspect orientation, imagery and class
evidence, then set `verified=true`, map each
supported channel to `centerline`, `ped_crossing`, `road_boundary` or
`drivable_aux`, and enable only justified support flags. Unsupported channels
do not contribute to KD.

The nuPlan Map API and Navformer environment are server-side dependencies.
The exporter requires `--map-root`/`--map-version` (or the corresponding
`NUPLAN_MAPS_ROOT`/`NUPLAN_MAP_VERSION` variables). Local code has not verified
the server's map database version, `CARPARK_AREA` layer availability,
server checkpoint's Pansegformer channel layout, or map coordinate convention. Those must be
checked on the server before declaring Stage 3 runnable. No fake GT or
teacher data is generated when they are unavailable.

The exporter resolves missing `map_location` by projecting actual GPKG layer
extents into each city's `projectedCoordSystem` and requiring exactly one
global XY match, with consistent city per `scene_token`. The optional
baseline/lane source row counts are checked against API-loaded layers on every
city. `--check-map-cast` also checks road/crosswalk layers. The nuPlan utility
casts whole baseline association columns to int, including nulls in the
non-applicable lane/connector field. Such a warning is recorded, not blindly
fatal: the exporter verifies that every nearby candidate has one baseline and
every ROI-intersecting baseline has one nearby parent. Invalid non-null IDs,
unlinked/ambiguous ROI baselines, missing geometry or unknown warning sources
stop export. This does not replace visual inspection on the server.

## Loss and checkpoint

`L = lambda_map_gt * L_vector + lambda_map_kd * L_soft_raster +
lambda_agent * L_stage2_agent`. Initial weights are `1.0, 1.0, 0.5` in
`configs/stage3_map.yaml`. Vector queries use class/point Hungarian cost;
open polylines compare forward/reverse point order, closed contours compare
all cyclic and reverse-cyclic orders. Matched point loss uses the selected
ordering; a small neighboring-segment direction loss includes the closing
segment for closed contours. Raster KD is BCE-with-logits against **bounded
soft mask scores**, restricted to audited channels and teacher-valid spatial
cells,
with configurable weights. Stage 2 Agent loss remains unchanged.

AdamW groups: shared BEV `1e-5`, new Map decoder/head/raster head `1e-4`,
Agent proposal/decoder/head `3e-5`. DINO and ego MLP stay frozen;
`use_ego_state=False` is explicit. Gradients are clipped at norm 1.0.

The Stage 3 checkpoint records seven model submodule state dicts, the raster
head, optimizer, epoch, effective config, map classes, teacher channel metadata
and support mask, both schema versions, Agent support mask and
`bev_use_ego_state=False`. Existing Stage 1/2 checkpoint formats are unchanged.
The Stage 2 Agent evaluator accepts the shared BEV and Agent modules from a
Stage 3 checkpoint for regression comparison.

## Server workflow

The formal split is raw metadata indices **0-4999 for training** and
**5000-5099 for held-out evaluation**. The preferred initialization remains
`checkpoints/quest_stage2_agent_5000_e5.pt`.

1. Export the requested raw-index train and evaluation ranges with
   `scripts/export_nuplan_vector_map.py` and
   `scripts/export_navformer_map_soft.py`. Specify the real nuPlan map root,
   teacher config/checkpoint and metric range. Request both 0-4999 and
   5000-5099; the exporters accept repeated runs with validated existing files.
2. Inspect `scripts/inspect_nuplan_vector_map.py` and run
   `scripts/audit_vector_gt_capacity.py` over all 5000 train and 100 validation
   frames. It reports per-class and total P50/P95/P99/max and counts above
   50/64/100/128. Only complete valid split coverage certifies a query count
   for training/evaluation; no GT is truncated.
3. Run `scripts/audit_navformer_map_teacher.py` with candidate axis settings,
   review its per-channel JSON and run `scripts/visualize_navformer_map_audit.py`
   before any training. Explicitly verify only supported channels.
4. Train with `python scripts/train_stage3_map.py` and evaluate with
   `python scripts/evaluate_stage3_map.py` on held-out frames. Run
   `scripts/visualize_stage3_map.py` to inspect the orientation against
   the vector prediction. Run
   `python scripts/evaluate_stage2_agent.py --checkpoint checkpoints/quest_stage3_map.pt`
   to compare Agent quality with the Stage 2 checkpoint.

No Python, tests, training, evaluation or visualization were executed while
authoring this implementation locally.
