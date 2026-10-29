# Seg / Depth Teacher Fidelity Review

## Verdict

Primary-paper/source audit, 2026-10-29. No Python, tests, model execution,
export, training or evaluation was run locally. No architecture, checkpoint,
labels, environment or Agent/Map behavior is changed by this review.

`shared_image_pyramid_v1` is an output-pseudo-supervised lightweight baseline,
NOT a faithful SegFormer-B5 or Metric3Dv2 reconstruction. Important teacher
mechanisms were simplified away. Names such as multi-scale or intrinsic
conditioning do not establish architectural equivalence. The class-weight
experiment establishes a modest benefit, not sufficient architectural fidelity.

## Current Code And Effective Capacity

Sources: `quest/backbone.py`, `quest/dense_perception.py`,
`quest/seg_depth_losses.py`, `configs/seg_teacher.yaml`.

- Frozen DINOv2-S/14 blocks 3/6/9/12: four [B,8,384,18,32] maps at 252x448.
- Four lateral 1x1 projections to128, bilinear rescaling, top-down addition,
  and one 3x3/GN/GELU refinement per level.
- Seg decoder: one 3x3/GN/GELU block plus 1x1 classifier and output resize.
- Depth decoder: intrinsic MLP feature modulation, one 3x3/GN/GELU block,
  1x1 output, softplus and output resize. No canonical-space target transform,
  recurrent depth/normal refinement or normal supervision in the Student.
- Default Seg labels are argmax native19 IDs, not soft logits. Depth uses the
  final camera-Z field. Teacher features/normals are not distilled.
- Agent/Map consume unchanged final DINO features, not the dense neck. Four
  outputs coexist; this is not joint four-task representation optimization.

Analytical layer counts, matching the user's server log (not a local model run):

| Active component | Trainable parameters |
| --- | ---: |
| Lateral/refinement neck | 789,504 |
| Seg decoder | 150,291 |
| Depth conditioning/decoder | 181,633 |
| Total | 1,121,428 |

The whole-model 70.77M includes frozen spatial branches unused by dense training.
It is not evidence of 70M effective dense capacity. Neither a fixed parameter
target nor layer count alone proves sufficiency; compare a stronger sourced head.

## Teacher Differences

### SegFormer-B5

The [MiT source](https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py)
uses widths [64,128,320,512], depths [3,6,40,3], strides4/8/16/32. These are
hierarchical encoder stages, not four depths of a single-grid ViT.
The [B5 config](https://github.com/NVlabs/SegFormer/blob/master/local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py)
sets fusion width768, 19 classes, CE and 1024x1024 sliding windows, stride768.
The [head](https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py)
projects, resizes, concatenates and fuses the four stages. Its light decoder
depends on a strong hierarchical encoder, not just a low decoder layer count.

The [paper](https://arxiv.org/html/2105.15203v3) reports 84.7M parameters and
84.0 Cityscapes validation mIoU. This is NOT OpenScene teacher accuracy and NOT
our Student-versus-teacher metric. Current QUEST lacks the MiT hierarchy,
768-wide fusion and teacher-resolution visual observations.

### Metric3Dv2 ViT-L

The [selected inference config](https://github.com/YvanYin/Metric3D/blob/main/mono/configs/HourglassDecoder/vit.raft5.large.py)
uses input616x1064, focal1000 and RAFTDepthNormalDPT5 with8 iterations.
The [base model config](https://github.com/YvanYin/Metric3D/blob/main/mono/configs/_base_/models/encoder_decoder/dino_vit_large_reg.dpt_raft.py)
specifies four1024-channel ViT-L features and multi-level DPT/GRU decoding.
Complete teacher parameter count has not been measured here; backbone count or
checkpoint byte size must not be presented as its exact full-model count.

The [paper](https://arxiv.org/html/2404.15506v2) uses canonical camera transforms,
depth/normal joint refinement, local random-proposal normalization and geometric
supervision. Current intrinsic feature modulation is NOT the same transform.
Current log/relative SmoothL1 and log-gradient losses are engineering choices,
not its reproduced objective. Student576 patch positions/camera versus
teacher3344 also changes visual sampling; this is a candidate limitation, not
a measured unique cause of the error.

The released [training example](https://github.com/YvanYin/Metric3D/blob/main/training/mono/configs/RAFTDecoder/vit.raft5.giant2.nyu.py)
also has sequence, normal and consistency losses. That giant/NYU config is not
proof of the exact selected large checkpoint's full training history. Do not
fabricate normals or verified geometry to enable losses requiring absent data.

## Follow-Up Evidence

- [ViT-Adapter](https://arxiv.org/html/2205.08534v3), which cites SegFormer,
  adds an RGB spatial-prior module and feature injection/extraction for plain
  ViT dense prediction. [Official implementation](https://github.com/czczup/ViT-Adapter/blob/main/segmentation/mmseg_custom/models/backbones/vit_adapter.py).
  A single RGB convolution is not equivalent. Full injection changes token
  activations even with frozen weights: Agent/Map preservation needs explicit
  branch design and regression checks, not just requires_grad=False.
- [Official DINOv2 DPT](https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/depth/models/decode_heads/dpt_head.py)
  uses learned reassembly and residual fusion. This is a direct backbone-
  compatible reference. Same-grid ViT inputs are not inherently invalid;
  DPT reconstructs multiple scales from them. Bilinear resize alone supplies
  no new observations, but patch size does not prove small objects unrecoverable.
- [Depth Anything V2](https://arxiv.org/html/2406.09414v2) cites Metric3Dv2
  and demonstrates large-teacher-to-small-Student transfer with DINOv2/DPT.
  Its [DPT implementation](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py)
  contains learned resize and residual fusion. Released
  [Small has24.8M total parameters](https://github.com/DepthAnything/Depth-Anything-V2#pre-trained-models):
  small backbones are not categorically inadequate. Its data/noise, gradient and
  resolution studies motivate controls, not guaranteed QUEST gains. Relative
  inverse-depth training cannot be copied blindly to metric camera-Z output.

## Next Controls, Not Yet Implemented

1. Compare source-traceable DPT reassembly/residual fusion using the same frozen
   DINO, split, labels and resolution. Compare an explicit SegFormer-style
   fusion head; document adaptations rather than calling them exact copies.
2. Check teacher predictions with matched low-resolution input and inspect
   representative rare-class regions against RGB. Teacher agreement cannot
   determine whether teacher or Student is correct without actual GT.
3. Test higher-resolution dense inputs only as a versioned memory-measured
   ablation. Do not silently alter the Agent/Map image geometry or features.
   A full ViT-Adapter experiment requires a separate preservation design.
4. For metric-loss fidelity, reproduce verified Metric3D mechanisms while
   retaining scale-sensitive supervision and valid masks. Normal/refinement
   supervision needs additional audited targets; no fake normal GT.
5. Preserve both existing checkpoints and fixed evaluation. Compare rare-class
   precision/recall, common-class regression and Depth jointly. No promise of
   teacher parity follows from implementing a module or increasing parameters.

The exact teacher accuracy on OpenScene, useful input resolution, sufficient
capacity and compression ratio remain experimental questions. No new Student
architecture or training is certified ready by this document.
