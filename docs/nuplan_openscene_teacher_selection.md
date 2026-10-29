# nuPlan / OpenScene Teacher Selection

Research audit: 2026-10-29. Primary papers, author repositories and model-file
listings were inspected. No weights were downloaded; no Python, tests,
inference, export, installation or training was executed.

## Admission Rule

A formal specialist needs all of the following:

1. Checkpoint-specific evidence of nuPlan/OpenScene task training or task
   validation, not merely a compatible image loader or a planning score.
2. A published task checkpoint and its matching architecture/preprocessing.
3. Outputs matching the QUEST task, or an explicit, auditable conversion.
4. Actual server checkpoint/inference/quality checks before label export.

Mixed-dataset training INCLUDING nuPlan/OpenScene qualifies for domain evidence;
training exclusively on those datasets has not been required. Domain evidence
does not establish quality on our cameras or independence from our validation
scenes. These are separate checks.

## Decision

| Task | Selection | Domain/asset evidence | Actual readiness |
| --- | --- | --- | --- |
| Agent | Retain the existing Navformer nuPlan track checkpoint | Existing checkpoint/config and user-reported real inference/export/training | Retain vehicle/pedestrian support; no replacement |
| Vector Map | Retain direct nuPlan vector GT | Existing real-map source and audited label pipeline | GT supervision, NOT a fourth neural teacher |
| Depth / geometry | DVGT-1, first integration candidate | Author paper includes OpenScene training/evaluation; author repo links an actual public checkpoint | Domain-qualified candidate; NOT integrated or server-tested |
| Camera Semantic Segmentation | No qualifying checkpoint selected | No reusable, real-nuPlan/OpenScene camera-semantic checkpoint verified in this search | BLOCKED, not silently replaced by a Cityscapes model |

The project does NOT currently have four ready, domain-qualified neural
teachers. Preserve the old dense labels/checkpoints as out-of-domain experiment
controls; do not delete them or describe their agreement scores as real GT
accuracy. No training configuration is switched by this document.

## Depth: DVGT-1

Primary sources:

- [Author repository](https://github.com/wzzheng/DVGT).
- [Paper](https://arxiv.org/html/2512.16919v2).
- [Public checkpoint files](https://huggingface.co/RainyNight/DVGT-1/tree/main):
  `dvgt1.pt`, listed size 6.93 GB. File contents/load are not checked here.
- [Training data configuration](https://github.com/wzzheng/DVGT/blob/main/configs/_base_/default_dataset.yaml)
  explicitly includes `openscene_total_train.parquet`.
- [Evaluation configuration](https://github.com/wzzheng/DVGT/blob/main/configs/eval/dvgt1/geometry.yaml)
  includes `openscene_test.parquet`.

The paper trains on driving data including OpenScene. Its OpenScene ray-depth
AbsRel is 0.049 and delta1 is 0.971. Those are AUTHOR results against their
reference, not QUEST results or camera-Z accuracy. The paper defines this ray
depth as distance to the current ego center; it is not our camera optical-axis
Z. Do not compare it numerically to Metric3D pseudo-agreement.

[Native architecture](https://github.com/wzzheng/DVGT/blob/main/dvgt/models/architectures/dvgt1.py):
DINOv3 ViT-L/16, 24 geometry blocks, intra-view/cross-view/cross-frame attention,
DPT point-map head and ego-pose head. The author model table lists about 1.7B
parameters. Input is `[B,T,V,3,H,W]`; native results include `points`,
`points_conf`, `absolute_ego_pose_enc`. Points are expressed in the first-frame
ego reference, not directly as eight camera depth maps.

Integration requirements, not yet implemented:

- Pin source revision, weight hash, dataset provenance and inference contract.
  Use the official inference wrapper and strict full task-weight loading, not
  the VGGT initialization path which intentionally omits task decoders.
- Native configs use `gt_scale_factor=0.1`; the
  [postprocessor](https://github.com/wzzheng/DVGT/blob/main/dvgt/models/architectures/dvgt1.py)
  divides points/pose translations by that factor for metric evaluation.
  Audit the released checkpoint's scaling before exporting meters; raw output
  must not be blindly assumed to be meters.
- Read a short REAL same-scene sequence via metadata. No fabricated/repeated
  history. Keep ordered camera identities and token/timestamp joins. The
  [demo loader](https://github.com/wzzheng/DVGT/blob/main/dvgt/utils/load_fn.py)
  sorts views alphabetically; an adapter must explicitly track that permutation
  and restore QUEST order without changing Student order.
- Reproduce native resize/crop/pad and retain the pixel-coordinate transform.
  Do not feed the Student's normalized 252x448 tensor into the teacher.
- Resolve first-ego -> current-ego -> current-camera coordinates using audited
  transform conventions; decode positive camera Z, not norm(XYZ) or ego Z.
  Keep native depth/pixel validity and actual image-geometry evidence.
- Predicted confidence is not a calibrated probability or a semantic sky mask.
  Reject unknown conventions rather than producing apparent valid targets.
- Official tested stack is Python 3.11 / torch 2.8 / CUDA 12.8, not evidence
  our existing environments work. A 3090 inference memory fit is unverified;
  short sequences/chunked native decoding must be tested on the server before
  a bulk run. Do not modify QUEST or Navformer environments automatically.
- Compare with synchronized nuPlan/OpenScene LiDAR where available. Audit
  teacher-training overlap with the QUEST split; Student-held-out scenes are
  not automatically teacher-held-out scenes.

## Backup Geometry Candidates

### ViGT

The [author paper](https://arxiv.org/html/2602.05573v1) trains with nuPlan in a
multi-driving-dataset mixture and evaluates point geometry against LiDAR on
nuPlan. It estimates a continuous occupancy field, from which depth can be
rendered; this is not semantic camera segmentation. The
[author README](https://github.com/whesense/ViGT) publishes a checkpoint link
and a checkpoint-specific train-dataset table including nuPlan. Its HF model
page/file contents could not be inspected with the web tool in this audit.

Status: domain-supported backup, checkpoint retrieval/load and calibrated
camera-ray rendering NOT verified. DVGT-1 has a more directly inspectable public
weight listing and pixel-aligned point outputs, so it is the first candidate.
The ViGT paper's nuPlan LiDAR-ray protocol and DVGT's ego-radius protocol are
different; their numbers must not be ranked as if they shared one evaluation.

### DVGT-FM

[Public repository](https://github.com/Public-BOTs/DVGT-FM) and
[weight listing](https://huggingface.co/X-ActM/DVGT-FM/tree/main) claim geometry
pretraining including OpenScene. Its README still leaves its own manuscript
citation pending. Prefer the original paper-linked DVGT-1 release for now.
The downstream MapTR checkpoint does not establish nuPlan-trained VECTOR map
supervision: geometry-backbone pretraining domain is not downstream-head domain.

## Segmentation: Remaining Evidence Gap

The required output is per-camera semantic masks with a defined class taxonomy,
not BEV road segmentation, class-agnostic object masks, occupancy semantics or
scene-specific 3D reconstruction. Never rename one of these as camera Seg.

Search outcome is LIMITED: no qualifying public reusable checkpoint was
verified. This is not a proof that none exists. Relevant inspected alternatives:

- [DriveStudio nuPlan instructions](https://github.com/ziyc/drivestudio/blob/main/docs/Nuplan.md)
  explicitly use the Cityscapes SegFormer B5 weights for sky/dynamic masks.
  Processing nuPlan with this model does not make it nuPlan-trained.
- [Learning to Drive is a Free Gift](https://openreview.net/pdf/b2dbc0b49c9892b3a116707adebc12ec3b4bf695.pdf)
  explicitly uses a Cityscapes SegFormer teacher. NAVSIM planning transfer is
  not a native nuPlan camera-Seg teacher checkpoint certification.
- [NAVSIM TransFuser source](https://github.com/autonomousvision/navsim/blob/main/navsim/agents/transfuser/transfuser_model.py)
  returns `bev_semantic_map`, not eight RGB-aligned semantic maps.
- [Guido](https://github.com/tancredelg/guido) advertises auxiliary Seg/Depth,
  but its README identifies the Phase 1/2 data as SYNTHETIC nuPlan samples and
  does not establish a downloadable real-nuPlan semantic teacher checkpoint.
  Do not mistake the initial nuPlan wording for real sensor-label provenance.
- `pengsongyou/openscene` is the unrelated open-vocabulary 3D project, not the
  OpenDriveLab driving dataset. Its name does not meet our domain requirement.

Do not request more of our own pseudo labels as if they were independent
semantic ground truth. If no author-supplied domain checkpoint becomes available,
a strict same-domain Seg teacher requires real nuPlan/OpenScene pixel labels
(with class IDs and scene-disjoint validation) and an explicit adaptation
experiment, or a separately documented in-domain evaluation of a generic model.
Neither has been completed. Do not start such training without user approval.

## Map Candidate Boundaries

Keep direct nuPlan vector GT and leave the unreliable Navformer raster KD OFF.
The [ReSMap author release](https://github.com/dehsurc/ReSMap) publishes a
nuScenes checkpoint; it does not qualify through that checkpoint.
A separate [NAVSIM output cache](https://huggingface.co/datasets/rudals/resmap-navsim-teacher-kd)
reports an adapted camera+satellite teacher, front-only ROI 0..32m by +/-32m
with four classes. This is NOT the author nuScenes checkpoint, not verified
reproducible nuPlan model weights, and not our full +/-50m surrounding taxonomy.
Do not auto-download/use the cache or reinterpret its labels as our GT.

## Next Server Evidence Needed

1. Depth: a short same-scene eight-view sequence with current metadata and
   calibration, followed by strict DVGT-1 native inference (future work).
2. Metric validation: matching real LiDAR paths/tokens/timestamps and calibration
   for a representative held-out subset. Unrelated PCDs and mmdet test clouds
   cannot certify OpenScene depth.
3. Seg: independently sourced real pixel annotations or an author's domain
   checkpoint/config/training and task-evaluation provenance. Existing
   teacher-agreement JSON/PNGs are useful diagnostics, not this evidence.

No new Student architecture, training stage, pseudo-label schema, checkpoint or
split is introduced by this audit. Agent and Map baselines remain untouched.
