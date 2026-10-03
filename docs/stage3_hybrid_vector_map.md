# Stage 3 hybrid vector-map training

Stage 3 starts from the Stage 2 Agent checkpoint. The production prediction is
`MapDecoder -> MapHead -> [B,N_map,4]` class logits (three map classes and
background) and `[B,N_map,20,2]` normalized local XY points. The existing
`MapHead` sigmoid output is decoded to meters for evaluation. Stage 3 constructs
the model with `C_map=3`; Stage 1/2 model configurations and checkpoints are
unchanged. `N_map` is configurable and all GT vectors must fit; excess instances
are an error, never silently truncated.

```
8 cameras -> frozen DINOv2 -> GeometryAwareBEVLift -> shared BEV
                                                   |-> Agent branch -> Agent loss
                                                   |-> MapDecoder/MapHead -> vector GT loss
                                                   `-> MapRasterDistillHead -> raster teacher KD
```

`MapRasterDistillHead` is training-only. It consumes `[B,384,32,32]` BEV features
and produces `[B,K,32,32]` logits. The Navformer/Pansegformer teacher produces
**raster probabilities**, not vector instances. Its output is only an auxiliary
KD target; inference needs neither that head nor raster post-processing.

## Coordinate contract

QUEST BEV is the OpenScene LiDAR local XY plane. Its origin is the LiDAR origin;
column index grows with local +X and row index with local +Y. `lidar2global`
maps LiDAR homogeneous points into the nuPlan global frame. Direct nuPlan map
geometry is transformed by its inverse before clipping to the model's metric
ROI. Thus global +X/+Y are **not** assumed to be LiDAR +X/+Y. The global
directions of local +X and +Y are respectively columns 0 and 1 of
`lidar2global[:3,:3]`. A physical forward/left interpretation has not been
verified from the local metadata and is deliberately not assumed. Tilted
LiDAR poses fail closed until map height is handled in 3D.

Navformer raster axes, sign, transpose and channel meanings are not inferred
from tensor shapes. Export requires an explicitly confirmed teacher metric
range and coordinate frame. Audit requires row axis and row/column direction.
The alignment utility samples each QUEST cell's metric center in the teacher
grid using bilinear interpolation, with `align_corners=False`; it does not
resize the entire teacher image. The visual diagnostic displays the audited
map with `origin=lower`, ego at `(0,0)` and labeled +X/+Y axes.

## Offline schemas

Vector GT, one `<token>.pt` per original metadata frame:

- `sample_index`, `token`, `schema_version=1`, `coordinate_frame=openscene_lidar_xy`
- `class_ids: int64[N]`, `points_xy_m: float32[N,20,2]`
- `is_closed: bool[N]`, `length_m: float32[N]`, `xy_range_m`

Class 0 is LANE/LANE_CONNECTOR baseline centerline; class 1 is CROSSWALK
contour; class 2 is ROADBLOCK/INTERSECTION/CARPARK_AREA boundary. Every
geometry is transformed to local meters, clipped to the ROI, split into
connected pieces, filtered for invalid/short geometry, and resampled by arc
length. Open lines include both ends. Closed rings use 20 unique points.

Teacher raster, also one `<token>.pt` per original frame:

- `sample_index`, `token`, `schema_version=1`
- `teacher_map_soft: float32[K,H,W]` in `[0,1]`, `teacher_map_shape`
- `teacher_pc_range`, opaque `teacher_channel_names_or_ids`
- `teacher_checkpoint`, `teacher_coordinate_frame`

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

## Audit gate and known external dependencies

The map teacher must have a real `seg_head` in its config **and** checkpoint.
The exporter does not substitute an Agent-only checkpoint. It also requires
explicit `--input-kind probabilities|logits`; logits are converted with sigmoid
only when requested. Channel IDs such as `lane_score_0` are intentionally
opaque. The audit reports probability activity, IoU and correlation against
temporary vector-GT masks. Its JSON starts with `verified=false`, all support
flags false and no semantic mappings. A human must inspect the orientation,
visual diagnostic and class evidence, then set `verified=true`, map each
supported channel to `centerline`, `ped_crossing`, `road_boundary` or
`drivable_aux`, and enable only justified support flags. Unsupported channels
do not contribute to KD.

The nuPlan Map API and Navformer environment are server-side dependencies.
The exporter requires `--map-root`/`--map-version` (or the corresponding
`NUPLAN_MAPS_ROOT`/`NUPLAN_MAP_VERSION` variables). Local code has not verified
the server's map database version, `CARPARK_AREA` layer availability,
Pansegformer channel layout, or map coordinate convention. Those must be
checked on the server before declaring Stage 3 runnable. No fake GT or
teacher data is generated when they are unavailable.

## Loss and checkpoint

`L = lambda_map_gt * L_vector + lambda_map_kd * L_soft_raster +
lambda_agent * L_stage2_agent`. Initial weights are `1.0, 1.0, 0.5` in
`configs/stage3_map.yaml`. Vector queries use class/point Hungarian cost;
open polylines compare forward/reverse point order, closed contours compare
all cyclic and reverse-cyclic orders. Matched point loss uses the selected
ordering; a small neighboring-segment direction loss is optional. Raster KD
is BCE-with-logits against **probabilities**, restricted to audited channels
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

1. Export the requested raw-index train and evaluation ranges with
   `scripts/export_nuplan_vector_map.py` and
   `scripts/export_navformer_map_soft.py`. Specify the real nuPlan map root,
   teacher config/checkpoint, metric range and teacher output kind.
2. Inspect `scripts/inspect_nuplan_vector_map.py`. Increase `N_map` if any frame
   exceeds query capacity; do not truncate GT.
3. Run `scripts/audit_navformer_map_teacher.py` with candidate axis settings,
   review its per-channel JSON, and explicitly verify only supported channels.
4. Train with `python scripts/train_stage3_map.py` and evaluate with
   `python scripts/evaluate_stage3_map.py` on held-out frames. Run
   `scripts/visualize_stage3_map.py` to inspect the orientation against
   the vector prediction. Run
   `python scripts/evaluate_stage2_agent.py --checkpoint checkpoints/quest_stage3_map.pt`
   to compare Agent quality with the Stage 2 checkpoint.

No Python, tests, training, evaluation or visualization were executed while
authoring this implementation locally.
