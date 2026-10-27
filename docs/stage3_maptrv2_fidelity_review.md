# Stage 3: MapTRv2 Fidelity Re-audit

## Scope And Evidence

Static review of QUEST at `6e25b03`, `spatial_map_v1`, against the MapTRv2
paper and the official `maptrv2` branch. No Python, tests, inference or training
was executed. This report does not modify architecture, GT, splits, evaluation
thresholds or checkpoints. The current implementation is a MapTR-inspired
experiment, NOT a faithful reproduction or a validated replacement.

User-reported full 120-frame validation at epoch 12, ordered distance <=1.5 m:
- centerline: 590/7446 GT matched, precision 0.071446, recall 0.079237;
- ped_crossing: 0/411, despite 171 foreground predictions of this class;
- road_boundary: 20/1068, precision 0.020492, recall 0.018727.

These results are not acceptable map perception quality. Nonzero gradients and
decreasing training loss do not establish geometrical or held-out correctness.

Primary references, read together rather than inferring behavior from names:
- [Paper, Sections 4-6 and ablations](https://arxiv.org/html/2308.05736v2)
- [R50 24-epoch config](https://github.com/hustvl/MapTR/blob/maptrv2/projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py)
- [Decoder axes/ordering](https://github.com/hustvl/MapTR/blob/maptrv2/projects/mmdet3d_plugin/maptr/modules/decoder.py)
- [Head, targets and loss normalization](https://github.com/hustvl/MapTR/blob/maptrv2/projects/mmdet3d_plugin/maptr/dense_heads/maptrv2_head.py)
- [Ordered point cost and loss](https://github.com/hustvl/MapTR/blob/maptrv2/projects/mmdet3d_plugin/maptr/losses/map_loss.py)

## Confirmed Non-equivalences

### Image To BEV Is A Different Geometry Model

The referenced official config uses LSS/BEVPoolv2: image features and a learned
depth distribution are lifted along camera rays. Its image backbone is partly
fine-tuned, not entirely frozen. It also enables depth supervision.

QUEST `quest/geometry.py::GeometryAwareBEVLift` instead projects each fixed BEV
XY location at only three LiDAR Z anchors (-1,0,1) into images, then weights the
sampled features. `quest/spatial_map.py::MapSpatialBranch` adds window attention
and frozen Agent context, but still uses this original lift. This is not LSS,
not BEVPoolv2, and not the learned spatial cross-attention of BEVFormer either.

This is a major design departure and a high-priority causal hypothesis, not
proof that fixed-anchor lifting is inherently incapable of map learning.
No measurement yet identifies the correct ground height or distortion model
for all frames. Camera pose/height errors affect where evidence is sampled.
Map GT has no measured elevation; it must not be naively projected at global Z=0.

### Physical BEV Resolution Is Much Coarser

Official config: 200x100 cells for a 60x30 m region, 0.3 m per cell.
QUEST Map: 64x64 cells over 100x100 m, 1.5625 m per cell.
The linear grid spacing is about 5.2 times larger. These are different tasks
and physical sampling budgets, not just different tensor dimensions.

Continuous point refinement CAN achieve sub-cell accuracy, as the user's
single-frame result demonstrates; 1.5625 m is not a hard coordinate-error floor.
Nevertheless the sampled spatial evidence and dense auxiliary targets are coarse.
Upsampling a feature grid alone does not supply missing visual evidence.

QUEST's dense-target radius is `width/2 + cell_diagonal/2`. At the current grid
and 1.5 m configured width, this represents an approximately 3.71 m-wide band
around a line's cell-center distance test. Adjacent centerlines can merge in
the auxiliary target. This is confirmed arithmetic, but its contribution to
the failed validation has not been experimentally isolated.

### Decoder Self-attention Lost Point-specific Inter-instance Information

Official decoupled attention retains all point tokens. In batch-first notation:
1. `[B*P,Q,C]`: inter-instance attention independently for each point slot;
2. `[B*Q,P,C]`: intra-instance attention along each vector;
3. BEV deformable attention and FFN, with the official residual/norm ordering.

QUEST `PointMapDecoderLayer.forward` instead performs intra-instance attention,
then `norms[1](value).mean(2)`, attention on `[B,Q,C]`, and broadcasts the same
instance update to all P points. This irreversible averaging is not an equivalent
implementation of the paper's decoupling. QUEST also uses pre-norm ordering and
a different reference-position encoding/initialization. The small circular
initial point offsets are a custom prior, not an official requirement.

Pooling points for a final INSTANCE CLASSIFICATION head is a separate matter;
it should not be confused with discarding point slots inside self-attention.

### Classification And Positive Assignment Are Custom

Official classification uses independent foreground sigmoid Focal scores,
an all-negative target for background, its matching cost, and explicit positive/
negative normalization. The training-only one-to-many branch increases positive
assignments and is attention-isolated from one-to-one deployment queries.

QUEST uses four-way softmax Focal with a background channel, class weights,
and background-subtracted matching cost. It lacks one-to-many queries. Those
choices may be legitimate experiments but are not a faithful baseline. Copying
official coefficients onto this different probability/normalization contract
would not restore equivalence.

An important non-bug: for a square 100 m ROI and 20 points, official-style
`5 * sum(normalized |dx|+|dy|)` equals QUEST's unweighted
`mean_points(|dx_m|+|dy_m|)`. Thus "meters versus normalized coordinates" or
"weight 1 versus 5" alone does NOT prove our regression is underweighted.
Class weights, classification, direction reduction and auxiliary-layer weighting
still differ and require a full equation-level comparison.

### Supervision, Targets And Evaluation Are Not The Same Experiment

The paper's reference setting uses divider/crossing/boundary, a smaller ROI,
Chamfer-based AP, and a substantially different training data/batch schedule.
QUEST uses nuPlan centerlines (including lane connectors), crossing contours,
merged road boundaries, and fixed-confidence ordered-distance precision/recall.
The main paper's lane-divider result is not a centerline benchmark for QUEST.

QUEST retains a real-vector BEV raster auxiliary loss, but has no image-plane
map segmentation or depth auxiliary loss, and no one-to-many supervision.
These omissions matter, but the paper's ablations still show strong performance
without dense auxiliaries. They are not sufficient evidence to explain an almost
failed model. Do not claim adding a depth teacher by itself will fix the map.

Closed cyclic/reverse and open reverse equivalence are already implemented in
QUEST; it would be wrong to diagnose their complete absence. Entire-instance
extent, line fragmentation, visibility, class identity and calibration remain
distinct possible failure sources. No GT is to be dropped to improve a score.

## What Is NOT Established

- There is no evidence that simply adding layers solves this: QUEST already has
  six map decoder layers, matching the reference count, and a larger hidden size.
- Memorizing one frame to 0.311 m proves optimization capacity on that frame,
  not correct image/GT alignment, visual dependence or generalization.
- A tensor-level GT oracle passing does not validate its alignment to real JPEGs.
- The new checkpoint's train/validation visual-dependency report is not available.
- Distortion coefficients alone, or an undistortion preview, do not certify image
  preprocessing. Neither "distortion definitely explains everything" nor
  "calibration is fine because shapes pass" is justified.
- Matching-distance P/R is not AP. Switching to Chamfer may expose a point-order/
  instance-extent issue; it must not be sold as an official mAP reproduction.

## Recovery Requirements Before Another Full Run

1. Keep the Agent/shared BEV checkpoint and current map checkpoint immutable.
   Use the exact existing manifest, tokens and GT provenance for comparisons.
2. Establish the image-to-label geometry and output dependence first. Use the
   existing six-frame diagnostic on the NEW checkpoint, including zero/other-frame
   backbone features. For physical camera calibration, project existing real
   3D Agent GT with pinhole versus documented distorted-camera projection. Do
   not use a guessed map altitude or Agent box centers as depth targets.
3. Replace pooled inter-instance attention with the actual point-preserving
   axes, query/reference logic and residual/norm ordering in a NEW map version.
   Do not silently reinterpret spatial_map_v1 weights as the corrected architecture.
4. For an architecture-faithful map branch, evaluate a learned depth/ray lift and
   finer metric BEV, not another width/depth increase around the same lift. A
   200x200 Map-only grid over the unchanged ROI gives 0.5 m cells; it is a candidate,
   not an already validated GPU budget or an exact 0.3 m reproduction. Profile
   peak memory and chunk camera/depth pooling where necessary. Keep DINO frozen
   if that remains the QUEST constraint and explicitly disclose the departure.
5. Re-establish the reference classifier, matching normalization, ordered L1 and
   edge-direction objectives as one coherent version. Preserve point-order
   equivalences. Introduce one-to-many only with capacity preflight, isolation
   masks and a declared compute budget; don't truncate GT or blindly scale the
   official 50-query auxiliary count to QUEST's audited 224 queries.
6. Never fabricate LiDAR/depth/PV GT to complete the diagram. Missing auxiliary
   supervision stays explicitly disabled. A separately trained frozen-DINO
   DepthHead does not send gradients into the existing BEV branch by itself.
7. Add epoch-level held-out evaluation and retain the best validation checkpoint
   separately. Record ordered and Chamfer diagnostics without changing the
   formal thresholds or presenting a metric switch as improved model quality.

This is a substantial map-branch correction plan, not a request for the user to
run another unchanged 5,000-frame training. Reproducing a published benchmark
would additionally require the same data, classes, ROI and AP protocol; the
QUEST adaptation must not be described as that reproduction.
