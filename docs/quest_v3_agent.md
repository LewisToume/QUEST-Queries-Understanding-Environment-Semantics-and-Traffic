# QUEST V3 Agent

QUEST V3 keeps the frozen DINOv2-S, calibrated 32x32 metric BEV lift, BEV encoder,
Map decoder, and dense per-camera heads from V2. The Agent path is rebuilt to make
every detection query depend on the current image-derived BEV.

The Agent proposal head predicts objectness and a bounded XY offset at each BEV
cell. The highest-scoring 100 cells define spatial references. Query content is
initialized only from bilinearly sampled local BEV features plus an MLP encoding
of the reference XYZ. The Agent decoder has no learned query embedding or learned
reference identity and exposes all four decoder layers for auxiliary supervision.
One shared Agent head predicts class, reference-relative center, size, normalized
yaw sine/cosine, and metric velocity at every layer.

Agent supervision uses the canonical metric schema:

```text
labels             [N]
boxes_metric       [N,7]  x,y,z,dx,dy,dz,yaw
velocity_mps       [N,3]
scores             [N]
class_support_mask [4]
valid_mask         [N]
```

Navformer offline labels support vehicle and pedestrian only. Unsupported QUEST
classes are excluded from the classification softmax instead of being converted
to background evidence. Teacher boxes must also be observable in at least one of
the eight current camera views. Stage2 defaults to `teacher_only`; `hard_gt_only`
and an explicit hard-GT-first `hybrid` merge are available.

Epoch one is proposal warmup by default. Later epochs optimize proposal
objectness/offset plus auxiliary Agent classification, center, size, yaw, and
velocity losses. Checkpoints must contain `architecture_version: 3`; V1, V2, and
unversioned checkpoints are rejected.
