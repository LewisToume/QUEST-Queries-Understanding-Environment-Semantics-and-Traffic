# Teacher Encoders, Decoders And QUEST Integration

Review date: 2026-10-31. Local code baseline: `937485f` on
`feature/multiview-quest`. Primary author sources and the actual QUEST code were
read. No Python, tests, model execution, export or training was run by Codex.
Server evidence below is USER-REPORTED, not an independently inspected server.
This document proposes a versioned architecture; it does NOT implement it.

## 1. Scope And Actual Teacher Status

Four requested outputs do not mean four ready, domain-qualified neural teachers.
BEV is a shared representation, not an extra output teacher.

| Project/source | Role now | Domain and evidence boundary |
| --- | --- | --- |
| Navformer / UniAD track branch | Active Agent teacher | Existing nuPlan checkpoint; user-reported inference, export and Student training; QUEST currently learns vehicle/pedestrian, not all teacher classes or tracking identities |
| Navformer / Pansegformer branch | Map raster teacher, DISABLED | User found zero lane channels and problematic mask scales; not a camera semantic-Seg teacher; drivable quality is also not approved |
| nuPlan HD Map | Active hard Vector Map supervision | Not a neural teacher; preserve existing vector schema, physical boundaries and split/capacity audit |
| MapTRv2 | Vector-map architecture/loss reference | The inspected released config is nuScenes; not an approved nuPlan/OpenScene checkpoint or required online teacher |
| DVGT-1 | Active offline Depth/geometry teacher, experimental adaptation | Official OpenScene training/evaluation evidence; user-reported strict native inference, completed labels and five-epoch Student experiment; actual LiDAR/geometry accuracy still unverified |
| SegFormer-B5 Cityscapes | Historical camera-Seg control and architecture reference | Real server experiment, but Cityscapes checkpoint does NOT satisfy the formal domain requirement |
| Metric3Dv2 ViT-L | Historical Depth control and geometry/decoder reference | Real server experiment; checkpoint-specific nuPlan/OpenScene qualification not established |
| Depth Anything V2 / DPT, ViT-Adapter | Supporting architecture references | Not newly selected domain-qualified teachers; no assets or environments are requested here |
| ViGT | Previously researched geometry backup | Not integrated or runtime-verified; not a camera-Seg model |

The removed StreamPETR integration and future OCC/Flow/World Model routes are
not reinstated by this review. A formal same-domain camera-Seg teacher remains
unresolved. An available unqualified checkpoint is not relabeled as qualified.

## 2. The Three Different Encoder Jobs

- Image encoder: RGB -> image features, such as ResNet, MiT or DINO.
- Neck / image pyramid: expose/fuse features at different spatial scales.
- Geometry encoder: combine camera views and optionally real temporal context
  into geometrically useful image features or metric BEV features.
- Task decoder: turn features into objects, vector points, semantic pixels or
  depth. Query tokens represent requested instances/points, not class identities.

Replacing all of these with one generic Transformer plus shallow output layers
is not a faithful integration. Sharing one image encoder does not require using
the same decoder or the same spatial coordinate system for all four tasks.

## 3. Native Projects Compared

### 3.1 Navformer Agent: calibrated BEV and persistent object queries

The selected [official track+map config](https://github.com/OpenDriveLab/Navformer/blob/main/configs/navformer/track_map_nuplan_r50_navtrain.py)
builds UniAD with ResNet-50, four FPN outputs of width 256, and a 200x200 BEV.
Its six-layer BEVFormer encoder combines temporal self-attention and calibrated
spatial cross-attention. The six-layer object decoder uses deformable BEV
sampling and box-reference refinement. It has 900 query slots, query interaction
and a track memory bank. Classification uses sigmoid focal loss, boxes use L1,
with Hungarian assignment in the tracking matcher.

Mechanisms worth retaining: multi-scale image access, spatially located queries,
iterative box-reference updates and layer supervision. Persistent track memory
requires real sequential Student inputs/state and suitable supervision. Merely
exporting current boxes does not transfer tracking ability.

### 3.2 Navformer Map: BEV masks, not final vectors or image Seg

The same config attaches a Pansegformer head with a deformable Transformer and
instance/stuff mask decoders. Its outputs are BEV map rasters. The
[native head](https://github.com/OpenDriveLab/Navformer/blob/main/mmdet3d_plugin/uniad/dense_heads/panseg_head.py)
uses mask-based decoding/selection; this is a different output representation
from QUEST polylines and camera-aligned 19-class semantic masks.

The user's observed unreliable channels remain a quality blocker. No flips,
255x scaling, contour extraction or reuse of its name makes this a verified
vector/Semantic-Seg teacher. A raster auxiliary head is optional and OFF here.

### 3.3 MapTRv2: depth-aware lift and instance-plus-point decoding

The [R50 config](https://github.com/hustvl/MapTR/blob/maptrv2/projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py)
uses a ResNet/FPN image frontend, a learned-depth LSS lift, width 256 and a
200x100 BEV over a 30x60m ROI. Its six-layer vector decoder predicts 20 points
per instance, with 50 one-to-one queries and an auxiliary one-to-many branch.
The published training recipe includes depth and PV/BEV foreground auxiliaries,
focal classification, ordered-point L1 and direction supervision.

The [decoder source](https://github.com/hustvl/MapTR/blob/maptrv2/projects/mmdet3d_plugin/maptr/modules/decoder.py)
keeps individual point features in both inter-instance and intra-instance
attention, then samples BEV at point references and refines those references
between layers. Averaging each instance's 20 points before all inter-instance
communication is NOT the same operation.

Use these mechanisms with existing nuPlan Vector GT, not a blindly transferred
nuScenes checkpoint. The default divider class is not QUEST centerline. Query
capacity must remain the audited value (224 in the user's current run), not the
reference model's 50. The inspected default config and a centerline variant must
not be confused. Preserve QUEST's documented open/closed equivalences; do not
silently redefine centerlines as directed or truncate GT to fit slots.

### 3.4 SegFormer-B5: a strong hierarchy behind a light decoder

[MiT-B5](https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py)
has hierarchical image stages, widths 64/128/320/512 and block depths 3/6/40/3.
These operate at strides 4/8/16/32. The
[B5 configuration](https://github.com/NVlabs/SegFormer/blob/master/local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py)
sets four-scale fusion width 768 and 19 output classes. The
[decoder](https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py)
projects each scale, resizes, concatenates and fuses them before classification.

The small decoder is supported by a substantial hierarchical encoder. Copying
only its last classifier onto coarse DINO tokens is not equivalent. A DINO-based
Student needs explicit multiscale reassembly and/or a learned RGB spatial-detail
adapter; native MiT features and interpolated ViT maps are not interchangeable.

This informs architecture, NOT approval of the Cityscapes checkpoint as an
OpenScene semantic teacher. Keep its experiments clearly labeled as controls.

### 3.5 DVGT-1: multiview geometry before dense point decoding

The pinned [native model](https://github.com/wzzheng/DVGT/blob/51cf3f6d11fdff8bc7e2bbe1a88f71665ccb2236/dvgt/models/architectures/dvgt1.py)
uses DINOv3 ViT-L/16 (width 1024), 24 geometry blocks and point/ego-pose heads.
Each geometry block has within-image, cross-view and cross-frame attention;
24 such blocks are not equivalent to 24 plain self-attention layers. The
[DPT head](https://github.com/wzzheng/DVGT/blob/51cf3f6d11fdff8bc7e2bbe1a88f71665ccb2236/dvgt/models/heads/dpt_head.py)
reads four geometry feature levels, with learned reassembly and residual fusion,
and predicts XYZ/confidence rather than a single positive scalar.

The [paper](https://arxiv.org/html/2512.16919v2) describes roughly 1.7B parameters,
joint metric point-map/ego-pose supervision, mixed driving datasets and extensive
training. Its geometry inference is calibration-prior-free; adding calibrated
sampling in QUEST is an explicit adaptation, not a claim of exact replication.

The current offline teacher uses two actual frames and may use the next frame
at scene starts. Current QUEST is single-frame. This is legitimate privileged
teacher information, but not an information-matched reproduction. Camera-Z
Student-to-teacher agreement is not the paper's ego-radius-to-reference metric.

### 3.6 Metric3Dv2: camera normalization and iterative dense refinement

The [selected ViT-L config](https://github.com/YvanYin/Metric3D/blob/main/mono/configs/HourglassDecoder/vit.raft5.large.py)
uses canonical focal 1000, 616x1064 input and eight depth/normal refinement
iterations. Its [base model](https://github.com/YvanYin/Metric3D/blob/main/mono/configs/_base_/models/encoder_decoder/dino_vit_large_reg.dpt_raft.py)
uses four 1024-channel ViT features, DPT reconstruction and recurrent decoding.
The [paper](https://arxiv.org/html/2404.15506v2) makes camera normalization and
joint geometry refinement central, not an arbitrary final-layer activation.

Retain the lesson that camera conditioning and iterative dense refinement matter.
Do not combine its focal normalization with DVGT's coordinate scaling or claim
that the current intrinsic MLP is the same canonical-camera transform. No normal
GT or recurrence target is fabricated. Metric3D stays a historical control.

## 4. What QUEST Actually Has

Read `quest/model.py`, `backbone.py`, `geometry.py`, `queries.py`,
`spatial_map.py`, `dense_perception.py`, and the actual stage configurations.

| Component | Existing implementation | Important difference |
| --- | --- | --- |
| Image encoder | One frozen DINOv2-S/14; four width-384 block outputs on 18x32 grids | Same-resolution semantic depths, not MiT's native hierarchy |
| Agent geometry | 32x32 metric lift; three height anchors; four BEV encoder layers | Fixed calibrated samples, not BEVFormer spatial/temporal attention |
| Agent decoder | Image-conditioned Top-100 proposals, four decoder layers, per-layer predictions | Local features initialize queries once; heads reuse the original references rather than iteratively updating them; no track memory |
| Map geometry | Independent 64x64 lift, coarse Agent-BEV context, four shifted-window layers, three BEV levels | Stronger than the old shared decoder, but not depth-guided LSS or the same physical sampling density as the reference |
| Map decoder | Six layers, per-instance/per-point embeddings, multi-scale point sampling, iterative point updates | Inter-instance attention mean-pools points and broadcasts back, unlike point-preserving MapTRv2 attention; no one-to-many branch |
| Seg | Shared four-level interpolated pyramid, one conv block and classifier | No native high-resolution hierarchy or faithful SegFormer fusion |
| Depth | Same pyramid, intrinsic modulation, one conv block and scalar output | No cross-view/time fusion, DPT residual reconstruction, point decoder or teacher-feature KD |

Dense training calls `forward_dense(images, intrinsics)`. Cameras are independent
batch elements inside its neck; neither the BEV nor sensor2lidar enters this
Depth prediction. Eight simultaneous outputs do NOT imply eight-view geometric
fusion. The dense neck is not currently used to improve Agent/Map features.

For channels=128 the analytical trainable counts are 789,504 for the neck,
181,633 for Depth conditioning/head, and 150,291 for Seg. Depth-only training
therefore has 971,137 trainable parameters; joint Seg/Depth has 1,121,428. These
are not whole-Student counts and must not be compared with full teacher counts
as if both measured the same parameter scope. No target count proves quality.

`configs/model.yaml` includes legacy defaults. Effective Stage 3/Dense
configurations/checkpoints override Map classes/query capacity, Seg native19
and rectangular dense output sizes. Inspect the effective profile, not just the
default N_map=50, C_seg=6 or 64x64 output fields.

## 5. Evidence From The Current DVGT Student Experiment

The user reports five epochs over 5,000 frames and full 120-frame held-out
evaluation. Agent/Map fingerprint remained unchanged. Metrics relative to DVGT:
AbsRel=0.1238268, RMSE=6.0619m, delta1=0.8278686. At 50-80m the signed bias is
-9.4681m; near distances have small positive bias. Global geometric-mean
Student/teacher ratio is 0.97947. This does not justify a global scale multiplier.

`depth_relative` in the training log is SmoothL1 of relative residuals, NOT
AbsRel. In particular, 0.009 does not mean 0.9% training AbsRel. Compare the
same metrics before diagnosing overfitting or insufficient epochs. Only three
held-out scenes and unverified geometry do not establish dataset-wide accuracy.

These results establish a functioning imitation baseline, not architectural
fidelity. Missing mechanisms are verified code differences; their individual
contribution to errors still requires controlled experiments, not assertions.

## 6. Proposed Integration: Share The Trunk, Preserve Specialist Decoders

This is a design proposal, NOT the current forward path:

```text
8 RGB views + consistent pixel geometry / K / extrinsics
                     |
           ONE shared image encoder
        (first controlled version: frozen DINOv2-S)
                     |
       learned multilevel image representation
       + optional RGB spatial-detail adapter
                     |
        +------------+------------------------------+
        |                                           |
  semantic task adapter                  multiview geometry encoder
        |                                local + cross-view interaction
  SegFormer-style fusion                 calibrated rays / visibility
        |                                           |
  per-camera semantic masks                 +-------+-------+
                                            |               |
                                    DPT/refinement    metric BEV pyramid
                                            |               |
                                  per-camera Depth    +------+------+
                                            |         |             |
                                            +----> Agent          MapTR-style
                                    depth-guided lift decoder     point decoder
                                                       |             |
                                                  boxes/velocity   vectors
```

The arrow from Depth to BEV denotes a supervised differentiable lift, not a
detached depth export, hard argmax or blind multiplication of feature maps.
Depth distribution/ray evidence and visibility/occlusion masks must be defined.
Seg and Depth remain image-plane decoders; do not force fine pixel boundaries
through a coarse BEV bottleneck. Semantic context can later be fused through
task adapters without making all tasks depend on unqualified Seg predictions.

### Concrete mechanism choices

1. Keep one image encoder. DINO-S is not categorically incapable: the
   [Depth Anything V2 source](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py)
   already pairs DINOv2 with learned DPT reassembly/residual fusion. Use that
   as a backbone-compatible decoder reference, not as a new domain teacher.
2. Build explicit multiscale features. For real spatial detail, investigate a
   [ViT-Adapter-style RGB prior and feature interaction](https://github.com/czczup/ViT-Adapter/blob/main/segmentation/mmseg_custom/models/backbones/vit_adapter.py).
   Full token injection changes the features seen by frozen Agent/Map weights;
   it is not an automatic no-regression change. At fixed 252x448 input, no
   decoder can guarantee the detail of a much higher-resolution teacher.
3. Add real cross-view geometry interaction. Sparse/calibrated attention is an
   integration choice; do not flatten all image tokens into unbounded global
   attention or name a fixed camera embedding a geometry encoder.
4. Retain independent task decoders. Agent needs reference refinement and
   object-query interaction; Map needs point-preserving attention and vector
   geometry; Seg needs multiscale fusion; Depth needs dense reconstruction and
   metric refinement. Copying teacher weights across these different spaces
   is not supported by matching channel dimensions alone.
5. Use a BEV pyramid rather than forcing everything to 32x32. Existing 100m-wide
   grids have 3.125m cells at 32 and 1.5625m cells at 64; 128 gives 0.78125m.
   Those are feature sampling pitches, not hard floors on continuous point
   regression accuracy. Choose fine resolution by physical accuracy and server
   memory measurements, with window/deformable attention, not a parameter quota.
6. Keep single-frame inputs initially. DVGT temporal attention and Navformer
   track memory require an explicit later causal state/sequence contract. Never
   invent past frames, use future frames at Student deployment, or claim that
   current box labels transfer track identities.

### Geometry must be reconciled before coupling branches

Under the current `raw_opencv` hypothesis, Depth labels are restored to original
distorted JPEG pixels. `GeometryAwareBEVLift` currently projects with a pinhole
K and does not accept distortion. A joint geometry encoder must use consistent
distortion-aware rays/projections or an explicitly rectified image/calibration
contract. Do not silently change existing inputs or treat this hypothesis as
verified calibration. This is a potential systematic issue beyond network depth.

## 7. Supervision And Cached Data Limits

| Branch | Current reliable source | Required loss/alignment behavior |
| --- | --- | --- |
| Agent | Navformer metric boxes/velocity for vehicle/pedestrian | Existing supported-class Hungarian supervision; retain normalization and unsupported-class inference/evaluation masks; tracking needs new targets/state |
| Map | Full nuPlan vector GT | Physical-coordinate ordered-point matching, class loss, direction and intermediate supervision; open reverse/closed cyclic-reverse equivalences and no truncation |
| Seg | No formal same-domain teacher yet; old Cityscapes controls | CE on supported valid pixels; soft-label KD only if real aligned scores are exported; no fake labels for unavailable classes |
| Depth | DVGT camera-Z candidates | Scale-sensitive metric supervision, valid masks and gradients; report near/far bins and consistent train/validation metrics; optional 3D consistency only under a documented ray/pose contract |

Current cached Agent labels omit track identities/features, Seg files primarily
contain class IDs, and DVGT files store camera-Z targets rather than full latent
features. These are output-level supervision. Feature or relation distillation
is NOT already implemented by having four heads. It needs actual exported
teacher features or approved teacher access, plus coordinate/channel/token
adapters and a separately budgeted export. No 5,000-frame re-export is requested
by this design review. Existing depth can support derived geometric targets,
but those remain derived pseudo labels, not independent LiDAR GT.

Teacher loss names are not universal. Do not blindly replace Agent's supported
4-class-plus-background objective with a teacher's 7-class sigmoid classifier,
or copy relative-depth losses that remove metric scale. Teacher BEV channels
and task taxonomy are not aligned just because tensor dimensions agree.

## 8. Migration And Acceptance

This is not a proposal to deploy four full specialist networks side by side.
It is one shared image trunk with task-specific adapters/decoders and an explicit
shared geometry pathway. Use a NEW architecture/profile/checkpoint contract for
structural changes; keep all current V3 checkpoints readable and immutable.

- First preserve the independent trained baselines. Seg and Depth checkpoints
  have differently trained shared necks: do not average those weights, keep one
  arbitrary neck, or copy only heads and claim preservation. A compatibility
  model may share DINO while retaining both trained necks as task adapters;
  that is an initial combined baseline, not the final shared geometry design.
- Then replace the simplified mechanisms in a controlled version: DPT plus
  multiview geometry, point-preserving Map decoding, and Agent reference
  refinement. Reuse source-traceable patterns, not only their class names.
- Protect Agent/Map during initial dense changes through actual branch isolation
  or the old frozen feature route. `requires_grad=False` does not preserve
  predictions when its input activations change. Joint updates come only with
  all-task validation against the saved baselines.
- Seg architecture can be prepared, but formal same-domain Seg training remains
  gated on a qualifying checkpoint or independently validated real annotations.
  Do not silently relax the user's dataset requirement to declare completion.
- Maintain token/raw-index/split hashes, ordered cameras, complete vector GT,
  metric boxes, teacher provenance, teacher-map KD=OFF and use_ego_state=False.
- Measure per-task quality and shared-module gradients. Decoder layer count,
  model size, finite loss and low training loss are not acceptance criteria.
  Track parameter counts by component; memory/runtime are server-only unknowns.

Priority is mechanism fidelity and a genuinely shared geometric representation,
not endlessly adding epochs to the current independent shallow branches.
Full temporal DVGT/UniAD reproduction is not promised from single-frame inputs.
