# LFG and Multi-Task Literature: Implications for QUEST

Status: source review and experiment specification only. No new model was trained,
no existing checkpoint changed, and no proposed improvement is a measured result.
This supersedes the idea that simply hand-adjusting static loss weights is a
sufficient next fix. The shared-scene V1 result remains a failed baseline gate.

## Verified Primary Sources

### LFG

- [Paper, architecture/training and Appendix A](https://arxiv.org/html/2602.22091v1)
- [Authors' project and ablations](https://lfg-ai.github.io/)
- [Authors' released repository](https://github.com/Applied-Intuition-Open-Source/LFG)

LFG starts from pretrained Pi3, initializes semantic/motion decoders from its
point decoder, and uses staged training. Appendix A specifies frozen DINOv2 and
geometry decoders. Its semantic teacher is Cityscapes SegFormer; supervision is
soft and class-weighted. It predicts current/future outputs from observed video.
Its roughly 1.45B-parameter, large-scale OpenDV pretraining and seven-class
semantic evaluation are not comparable to QUEST's 19-class OpenScene teacher
agreement. The project ablations do not show every added component improving
every metric or data regime. LFG is a design precedent, not a guarantee of positive
transfer. In particular its motion masks are not Agent 3D detection and its point
maps are not vector HD maps. Do not describe QUEST as an LFG reproduction.

Application to QUEST (our inference): preserve useful geometry while transferring
semantic structure into shared features; use a staged introduction of new
supervision rather than simultaneously replacing trained readouts and assuming
all learned capabilities survive. This does not require changing DINOv2 to V3.

### Online Knowledge Distillation for Multi-Task Learning, WACV 2023

- [Official paper page](https://openaccess.thecvf.com/content/WACV2023/html/Jacob_Online_Knowledge_Distillation_for_Multi-Task_Learning_WACV_2023_paper.html)
- [Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Jacob_Online_Knowledge_Distillation_for_Multi-Task_Learning_WACV_2023_paper.pdf)

The paper explicitly addresses multi-task performance falling below single-task
networks. Its method jointly trains single-task guides and the multi-task student,
using adaptive feature distillation and online task weighting. At inference only
the multi-task model remains. For QUEST, using already-trained frozen guides or
cached features would be an adaptation, not the original online method. Training
teachers/adapters do not imply shipping four models in the inference graph.

### Knowledge Distillation for Multi-Task Learning

- [Original paper](https://arxiv.org/abs/2007.06889)
- [Authors' implementation](https://github.com/VICO-UoE/KD4MTL)

This is another relevant teacher-to-shared-student direction. Use it as a reference
for transferring task knowledge, not evidence that adding hard output labels alone
is sufficient or that all teacher feature spaces are directly interchangeable.

### GradNorm and PCGrad Solve Different Problems

- [GradNorm original paper](https://proceedings.mlr.press/v80/chen18a.html)
- [GradNorm algorithm PDF](https://proceedings.mlr.press/v80/chen18a/chen18a.pdf)
- [PCGrad original paper](https://arxiv.org/abs/2001.06782)
- [PCGrad authors' code](https://github.com/tianheyu927/PCGrad)

GradNorm adjusts task weights using gradient magnitudes on shared weights and
relative training rates. PCGrad projects conflicting gradient directions. Neither
is equivalent to comparing scalar losses, neither restores removed decoder
capacity, and neither guarantees all metrics improve. GradNorm need not reduce
Depth's weight: its outcome depends on actual gradients and relative learning
rates. A new continuation phase must explicitly define its initial task losses,
rather than silently reuse incompatible initial losses from a different phase.

## What the Actual V1 History Shows

All 12 epochs fail the all-task gate. Seg plateaus near 0.424 versus 0.4596 in the
old selected model. Depth stays near its selected baseline. Agent macro-F1 is
similar only because improved pedestrians offset worse vehicles. Centerline Map
F1 remains below the old model, while the old Map baseline itself is poor.

Last mean normalized task losses from the checkpoint:

| Task | Value | Share of summed scalar loss |
| --- | --- | --- |
| Agent | 0.194653 | about 13.5% |
| Map | 0.220165 | about 15.3% |
| Seg | 0.057192 | about 4.0% |
| Depth | 0.967417 | about 67.2% |

This is a loss-balance warning, NOT proof that Depth dominates shared gradients.
The logged gradient audit is a sampled probe, not a full gradient-conflict history.

Structural confounders must also be acknowledged: V1 replaced the trained Seg
decoder with a 1x1 readout, changed the fine-BEV input feature distribution, reduced
Map query layers, and removed per-layer reference-point refinement. These changes
mean its regression cannot be attributed exclusively to negative transfer or
insufficient training. Fixing these is not the same experiment as loss balancing.

## Concrete Experiments, Keeping One Deployable Model

Each experiment needs the SAME V1 epoch-12 start, training frames, seeds, schedule
and budget as its continuation control. New checkpoint names are mandatory; old
baselines remain immutable. A fixed six-epoch continuation is a development budget,
not a promise that six epochs will suffice. Do not start another unmodified run.

1. **Gradient-based task balancing**: add actual GradNorm on a declared shared
   weight tensor. Save initial train-only task losses, task weights, their optimizer
   state, gradient norms and learning-rate convention. Compare with a fixed-weight
   control under the same continuation optimizer/schedule. Test PCGrad separately,
   not at the same time, so direction and magnitude effects are identifiable.

2. **Knowledge-preserving shared feature distillation**: keep one scene trunk and
   one shared dense decoder. During training only, align shared multiscale features
   through small dimension adapters to semantic/geometric guide features. Use
   teacher-valid spatial masks and explicit layer/channel/pixel alignment. Start
   with Seg supervision plus a geometry-retention constraint, not four arbitrary
   feature tensors concatenated together. Teacher forward passes and adapters
   must be absent from the deployed checkpoint/inference graph. A frozen old QUEST
   branch is a baseline guide, not the original specialist teacher; label it clearly.

3. **Restore Map reference refinement**: within the SAME shared query decoder,
   use each layer's predicted points as next-layer sampling references, as the old
   Map decoder did. Keep class/order matching unchanged. This needs no private
   Map encoder or independent Transformer stack. Compare refinement on/off without
   simultaneously changing query depth or supervision weights.

4. **Shared high-resolution refinement, if needed**: preserve image-detail skips
   and multiscale fusion in the common dense decoder, rather than reintroducing a
   large Seg-only decoder. Evaluate edge/rare-class changes and Depth separately.
   Use the pre-integration semantic branch as an offline quality reference, not
   an additional deployed model. Do not assume a 1x1 classifier is intrinsically
   insufficient: the question is whether its shared input features retain semantic
   information and spatial detail.

5. **Temporal curriculum later**: follow the spirit of observed/future supervision
   only after current-frame tasks recover. Student inputs must remain causal;
   future frames, if used, are training targets, never observations. Additional
   future-target manifests, validity and motion/visibility handling are required.
   LFG's future weighting factor must not be copied without this target pipeline.

Priority: gradient-balanced continuation and shared semantic feature transfer are
the first two isolated experiments; reference refinement is the targeted Map
experiment. Do not combine all switches into an uninterpretable new V2 result.

## What Data Are Actually Needed

Existing four-task labels and original teacher bundles suffice to begin the
balancing/refinement experiments. No new LiDAR is required for those tests.

For feature/soft-target distillation, current Seg export code CAN optionally store
19-channel probabilities, but existing server files must be inspected before
assuming they contain them. Default hard labels plus top-1 confidence cannot
reconstruct the full teacher distribution. Teacher intermediate features are not
currently a validated cache. Add an explicit, versioned exporter/cache with token,
teacher hash, layer names, grid and preprocessing metadata. Use fixed low-resolution
feature grids to bound cache size, and audit raw/undistorted pixel alignment.

Keep all 19 semantic classes. Do not merge rare classes into LFG's coarser labels
to inflate mIoU. For research-level generalization, add independent held-out scenes
and real-GT checks separately; this does not block the controlled experiments above.

## Acceptance

Retain per-class Agent/Map precision and recall, Seg 19-class IoUs and Depth
distance bins, not just total loss or averaged F1. Existing no-regression gates
stay unchanged. A passing comparison against a poor Map baseline still does not
qualify the vector Map task. Only measured, matched single-task/joint ablations
support a claim of positive transfer; no such claim is established by this review.
