# QUEST: Four-Teacher Student Design Contract

Current project-wide encoder/decoder comparison and integration proposal:
[2026-10-31 audit](teacher_encoder_decoder_comparison.md). It incorporates the
user-reported DVGT Student experiment; historical readiness notes below must not
override the newer evidence or be mistaken for an implemented replacement.

For the code-versus-teacher mechanism audit and parameter breakdown, see
[Seg/Depth fidelity review](seg_depth_teacher_fidelity_review.md). The active
dense profile is a lightweight baseline, not a faithful teacher reconstruction.

## Project Requirement

The user reaffirmed this requirement on 2026-10-27: ONE deployed QUEST student
must jointly provide Agent, Vector Map, per-camera Semantic Segmentation and
metric Depth. Preserve the useful mechanisms of the four specialists, not just
four output projections on a generic latent. BEV is their shared spatial
representation, not a fifth final task or teacher. Real GT may replace or
supplement teacher supervision when available.

Formal domain requirement (2026-10-29): task checkpoints must have nuPlan or
OpenScene training/validation evidence. See the superseding
[teacher selection audit](nuplan_openscene_teacher_selection.md).
Retain Agent=Navformer and Map=nuPlan Vector GT. SegFormer Cityscapes and
Metric3Dv2 are OUT-OF-DOMAIN / UNQUALIFIED-DOMAIN experiment controls, not formal
teachers satisfying this requirement. DVGT-1 is the new domain-supported Depth
teacher: [native adapter and user-reported server experiment](dvgt_depth_teacher.md).
A same-domain camera-Seg checkpoint is unresolved.
Navformer raster KD remains disabled. Four qualified neural teachers are not
ready; real map GT is not a neural teacher.

This is a design/readiness audit, NOT an implemented replacement model. No
model, training, labels, environment or checkpoint is changed by this document.
Keep eight ordered cameras F0,L0,R0,L1,R1,L2,R2,B0 (all prefixed CAM_), frozen
DINOv2-S and 252x448 student input unless explicitly versioning an experiment.

## Historical Experimental Route

The table below records the earlier route, NOT current formal approval. Its
implementation/runtime notes are historical; the current status is in
[Teacher Integration Status](teacher_integration_status.md).

| Task | Specialist | Evidence available | Still missing |
| --- | --- | --- | --- |
| Agent | Existing Navformer UniAD track+map checkpoint, Agent only | User-reported native inference, label export and Stage 2 training | Regression checks after shared-front-end changes; not full tracking distillation |
| Map | nuPlan Vector GT; Navformer raster KD remains OFF | Real vector GT export/audit/training reported by user | Map student quality; no MapTR checkpoint prerequisite |
| Seg | SegFormer-B5 Cityscapes, official NVIDIA release | Native offline exporter implemented; not executed locally | Server assets/runtime, eight-view quality and taxonomy review; no student trainer yet |
| Depth | Existing Metric3Dv2 ViT-L choice | Adaptation/diagnostic code; user reports downloads | Actual strict load/inference and image-geometry approval; no student training |

This experimental route used three neural teachers plus true vector GT; it did
not establish three domain-qualified teachers. MapTRv2 is an architecture/loss
reference, NOT the operational Map teacher. Seg weights are Cityscapes; mixed
generic Metric3Dv2 training is not verified nuPlan checkpoint provenance.
Student/teacher agreement does not establish teacher accuracy. Local files do
not establish server state.
Navformer map masks remain disabled: observed zero lane channels are not GT.
Direct nuPlan vector GT remains the primary map supervision.

## Teacher Mechanisms To Preserve

### Agent: Navformer UniAD

The [native config](https://github.com/OpenDriveLab/Navformer/blob/main/configs/navformer/track_map_nuplan_r50_navtrain.py)
uses R50/FPN, 200x200 BEV, six BEVFormer encoder layers with spatial/temporal
attention, and six reference-refining detection decoder layers. Track queries
carry memory. Its classification is sigmoid focal, regression L1, assignment
Hungarian. QUEST should retain image-conditioned object queries, local spatial
sampling, reference refinement and layer supervision.

Existing Agent V3 predicts current-frame boxes/velocity; it is not a reproduction
of track memory. Track IDs/temporal features cannot be claimed from current box
files. Keep vehicle/pedestrian support [true,true,false,false], visible targets,
and the working Agent loss. It need not adopt the Map classifier to share BEV.

### Map: MapTRv2

The [paper](https://arxiv.org/html/2308.05736v2) and
[decoder source](https://github.com/hustvl/MapTR/blob/maptrv2/projects/mmdet3d_plugin/maptr/modules/decoder.py)
use depth-aware image-to-BEV lift, instance-plus-point queries, point-preserving
inter/intra-instance attention, deformable sampling and iterative point
refinement. Preserve classification/matching normalization, ordered point L1,
direction and layer losses, order equivalences, and attention-isolated one-to-
many training when budgeted. Existing point-averaging attention is not equivalent;
see [the detailed fidelity audit](stage3_maptrv2_fidelity_review.md).

Do not download/require a nuScenes MapTRv2 checkpoint to proceed. Its semantics
and ROI differ from our real nuPlan labels; even a centerline-enabled variant
would not establish same-dataset teacher coverage. Train the vector decoder
against the existing nuPlan GT and audited query capacity. No GT truncation.
Only revisit Navformer raster-to-vector KD after its output and alignment pass
a new human-reviewed audit; do not treat unavailable/zero raster channels as GT.

### Seg: SegFormer-B5 Cityscapes

[SegFormer](https://arxiv.org/html/2105.15203v3) combines a hierarchical MiT
encoder and an MLP decoder fusing four scales. The light decoder depends on that
multi-scale encoder. A single last-layer DINO feature plus a few convolutions is
not equivalent. The [B5 config](https://github.com/NVlabs/SegFormer/blob/master/local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py)
uses 19 classes and CE; its [test pipeline](https://github.com/NVlabs/SegFormer/blob/master/local_configs/_base_/datasets/cityscapes_1024x1024_repeat.py)
preserves aspect ratio, with sliding-window inference defined by the B5 config.

Weights: [nvidia/segformer-b5-finetuned-cityscapes-1024-1024](https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024).
This is a loading candidate without requiring legacy training dependencies, not
proof its default processor reproduces the original sliding-window protocol.
Declare/audit preprocessing. The active dense profile preserves native19 IDs;
`C_seg=6` is only the legacy default. Never feed native IDs into that legacy
head or silently merge unknown classes into background.

### Depth: Metric3Dv2 ViT-L

[Metric3Dv2](https://arxiv.org/html/2404.15506v2) uses canonical camera focal
normalization, DPT-style dense decoding and recurrent depth/normal refinement.
The [selected config](https://github.com/YvanYin/Metric3D/blob/main/mono/configs/HourglassDecoder/vit.raft5.large.py)
uses canonical focal 1000 and 616x1064 input. Reuse our existing strict adapter
and [official ViT-L weights](https://huggingface.co/JUGGHM/Metric3D).

The student needs intrinsic-conditioned metric geometry and multi-level dense
decoding, not just an unconditioned positive scalar head. Camera-Z meters, valid
metric/log-depth residuals and edge/gradient supervision are proposed student
objectives, not a claim to reproduce all teacher losses. Scale-invariant loss
alone cannot establish metric scale. Normal refinement is a mechanism reference,
not a fifth requested output. Image geometry must still be verified; distortion
coefficients or an undistortion preview cannot approve calibration.

## Proposed Unified Architecture

```text
8 RGB views + calibration
  -> ONE frozen DINOv2-S, selected intermediate block outputs
  -> shared TRAINABLE image neck / dense feature pyramid
       |-> multi-level semantic decoder -> per-camera Segmentation
       |-> intrinsic-conditioned depth/refinement -> per-camera metric Depth
       |                              |
       |      features + learned depth distribution + calibrated camera rays
       |                              v
       +-------------------> shared fine-metric BEV pyramid
                                |-> object proposal/query decoder -> Agent
                                +-> point-preserving MapTRv2 decoder -> Vector Map
```

This is the proposed destination, NOT current `QUESTModel.forward`.
`quest/dense_perception.py` implements the shared image-pyramid/dense-head
component. `QUESTModel` now connects its four-layer DINO features to the actual
Seg/Depth outputs under a separately versioned `dense_frontend` profile, with
offline target joins, losses, optimizer, checkpoint and held-out evaluation.
See [training instructions](seg_depth_training.md). Initially the existing
Agent/Map receive unchanged final DINO features and stay frozen, not the neck's
BEV residual. Thus all four outputs coexist in ONE model, but this is not yet
joint four-task optimization or a depth-guided BEV lift. Those remain separate
experiments requiring regression checks and verified geometry.
DINO
intermediate blocks have different semantic depths at the SAME 18x32 resolution;
they are not a native MiT pyramid. A trainable neck constructs scales, but
upsampling cannot restore image detail lost at 252x448/patch14. Record this
capacity limit rather than promising teacher accuracy from a small fixed input.
The new dense profile outputs rectangular 252x448; legacy defaults stay 64x64
only to keep old checkpoints/training entry points intact.

Choose BEV resolution from metric precision and measured memory, not a fixed
32x32 rule or target parameter count. Fine BEV and a lower-resolution Agent
view can coexist. Use sparse/window attention, not full quadratic self-attention
over 200x200 tokens. A new depth/ray lift with learned refinement requires a
versioned implementation and memory validation; this audit enables neither.

Depth used for lifting MUST receive depth supervision and Agent/Map gradients
through a differentiable path. A disconnected output head, hard argmax, detached
depth or disconnected neck does not supply this. Seg/Depth gradients should
reach the shared trainable neck; DINO remains frozen. Per-task adapters remain
available to reduce conflicting updates. BEV must not become the forced
bottleneck for fine image-plane boundaries.

Do not merge incompatible teacher parameter tensors or deploy four full teacher
encoders. Distill predictions, aligned features or relations through student
adapters. ResNet256, MiT and ViT-L tensors are not interchangeable with
DINO-S384. Teacher BEV feature matching additionally requires frame/ROI/axes,
channel projection and valid-spatial masks, not just equal tensor shapes.

## Supervision And Acceptance Contract

- Agent: existing metric canonical pseudo targets, visibility and support mask;
  preserve explicit hard-GT/hybrid options and the immutable Stage 2 baseline.
- Map: full schema-5 nuPlan GT, permutations and Hungarian matching. Existing
  Navformer raster KD stays disabled; MapTRv2 remains a structural reference.
- Seg: mapped probabilities for KL and/or reviewed class IDs for CE; mandatory
  pixel/class-support masks and ignore IDs. Six-class semantics not assumed.
- Depth: valid camera-Z teacher/real depth and correct intrinsics/pixel mapping;
  invalid pixels are unavailable, not zero-distance background.
- BEV: direct geometry/task gradients; optional aligned teacher feature loss is
  not enabled by tensor shape equality or by this document.
- Four tasks share raw source index, token, ordered cameras, split hash,
  augmentation transforms and provenance. Never train on held-out labels.
- Normalize losses on valid targets; log per-task gradient RMS on shared neck
  and BEV. Loss magnitude alone does not measure useful shared learning.
- Validate ALL four outputs and Agent regression jointly. New neck/lift/head
  contracts require checkpoint versioning and explicit warm-start reports, not
  unexplained `strict=False` missing keys.
- Before another full run: eight-view teacher/geometry evidence, tiny per-task
  fit, real/zero/shuffled dependency, then held-out multi-task metrics. One-frame
  memorization does not establish usable generalization.

[LFG](https://lfg-ai.github.io/) supports the principle of unified outputs under
specialist supervision, not arbitrary weight merging. Its temporal reconstruction
tasks differ from QUEST object/vector-map outputs. QUEST is not its reproduction.

## Next Server Evidence: No GPU

```bash
python scripts/inspect_multitask_teachers.py
```

Standard-library-only inventory: explicit paths, no image scan, model import,
weight loading, download or writes. Missing files yield exit code 2.
`FILES_PRESENT` means only nonempty files, NOT runtime/integrity/quality. All
runtimes remain `NOT_CHECKED` and all-teacher readiness remains unverified.
Map inventory checks the split manifest, capacity audit and vector-label directory,
not MapTRv2 weights or unverified raster labels. Full GT file coverage is checked
by existing Stage 3 preflight, not by directory presence in this inventory.
Use path overrides for existing server teacher locations; no automatic download occurs.

Follow-up evidence needed: exact config/weight pairs, strict load reports,
eight-camera outputs/overlays, image geometry approval and Seg taxonomy. Map does
not wait for another neural teacher. See [Dense Teacher Export](dense_teacher_export.md)
for the next actual export command. The user runs server commands. No local Python or tests
were executed, and no four-task training or completed integration is claimed.
