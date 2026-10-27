# Active Dense Teacher Implementation

## Fixed Supervision Routes

- Agent: existing Navformer, vehicle/pedestrian, existing thresholds and labels.
- Map: existing real nuPlan Vector GT; no MapTRv2 weights required. Navformer
  raster-to-vector KD remains OFF. No label, capacity, split or Map loss changed.
- Seg: SegFormer-B5 Cityscapes19, original eight JPEGs, offline export.
- Depth: existing Metric3Dv2 ViT-L adapter; image geometry remains an explicit
  requirement for formal metric labels, not for image-plane Seg predictions.

## User-Run Commands

Activate `quest_env` and run from `/home/user/DataDisk/QUEST_WORK/QUEST`.
No commands below were executed locally by Codex.

Download missing Seg/Depth files, resuming `.part` downloads and preserving
existing final files. No installation, conda changes, GPU work or MapTR downloads:

```bash
bash scripts/prepare_multitask_assets.sh
```

It downloads the official NVIDIA SegFormer config/processor/weights and delegates
Metric3D assets to the existing preparation script. File presence is not a model
load certificate. Do not run concurrent downloads to the same `.part` files.
If assets already exist elsewhere, use CLI path overrides rather than moving or
overwriting them. An empty final file fails clearly and is preserved for inspection.

Run one real eight-camera semantic diagnostic on a user-selected idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/export_segformer_pseudo.py --sample-index 0 --num-frames 1 --diagnostic-only
```

This loads the LOCAL B5 weights strictly, performs real inference and writes
PNG/JSON previews, NOT `.pt` labels. No QUEST/DINO model is imported or trained.
Expected preview:
`data/diagnostics/segformer_cityscapes19_v1/1a11259978a256cd.png`.
Reuse of a nonempty output location fails; choose `--audit-dir` to run again.
The CPU/checkpoint-file preflight flag is `--preflight-only`; it does not certify
teacher import/weight contents/CUDA inference.

After examining native output, candidate raw pseudo-label export is available:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/export_segformer_pseudo.py --split-manifest data/stage3_split.json --split train --sample-index 0 --num-frames 100
```

Under a manifest, sample-index/count slice the manifest rows; each file still
stores the ORIGINAL metadata source index and token. The shared selection helper
is also reused by the depth exporter, preserving its old import interface.
No scene sorting or target-token change is introduced.

## Semantic Contract

Sources: official [B5 configuration](https://github.com/NVlabs/SegFormer/blob/master/local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py),
[resize pipeline](https://github.com/NVlabs/SegFormer/blob/master/local_configs/_base_/datasets/cityscapes_1024x1024_repeat.py),
[sliding-window implementation](https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/segmentors/encoder_decoder.py),
and [converted NVIDIA weights](https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024).

The teacher reproduces the official B5 test resize/slide mechanism through the
HF-converted model: keep-ratio long/short limits 2048/1024, RGB ImageNet mean/std,
1024x1024 crops, stride 768, overlap-averaged logits, original-pixel rescale,
then the student's rectangular 252x448 grid. OpenCV linear resize is used;
there is NO undistortion and no square image warp. The preprocessing adapter
version and its SHA256 are recorded. Numerical parity with the original legacy
runtime remains a server-only assumption, not an asserted reproduction result.

Candidate `.pt` record:

```text
schema_version = 1
token, source_index, scene_token, timestamp
camera_names = canonical OpenScene 8-camera order
class_names = all 19 Cityscapes class names
label_space = cityscapes19
pixel_alignment = original_openscene_jpeg_resized_v1
metadata_sha256, split_sha256, source_image_hw, output_hw
adaptation_verified = false
provenance = checkpoint/config/adapter hashes, versions, preprocessing
seg.labels       uint8 [8,252,448]
seg.confidence   fp16  [8,252,448] (not calibrated)
seg.valid_mask   bool  [8,252,448]
```

Optional `--save-probabilities` stores `[8,19,252,448]` fp16 soft targets
(about 33 MiB/frame, roughly 160 GiB for 5,000 frames). Default compact labels
are suited for later CE-style pseudo supervision, NOT lossless soft-logit KD.
No confidence threshold is silently introduced, no teacher class is remapped,
and confidence is not claimed calibrated. Validity is not a sky/depth mask.
Do not load these native IDs into the old six-class Student head. Formal use
needs the reviewed native-19 output contract or an explicit canonical mapping.
`adaptation_verified=false` must not be mistaken for human approval.

JSON logs and eight-camera color previews make raw output auditable. Existing
files are never overwritten; final PT files publish atomically. Nonfinite native
logits, class/config mismatches or missing teacher parameters fail rather than
producing random targets. Labels do not alter original JPEGs or metadata.

## Shared Student Component: Implemented But Not Activated

`quest/dense_perception.py::SharedDensePerception` fuses four normalized DINO
intermediate layers (same patch grid) into constructed scales and exposes:

```text
seg_logits         [B,8,19,252,448]
depth              [B,8,1,252,448], positive camera-Z candidate
image_feature_maps [B,8,384,18,32], differentiable spatial-branch input
image_pyramid      four constructed, trainable image scales
```

Depth receives normalized focal/principal-point features, never absolute ego
position. Seg and depth update the shared neck; spatial tasks can backpropagate
through its nonzero-gated BEV feature adapter. Shape/gradient test SOURCES are
included but were NOT run. No depth accuracy is established by positive values.

This is not yet wired into `QUESTModel`, legacy Stage 2/3 optimizers or
checkpoint schemas. Their trained behavior remains unchanged. A separately
versioned joint integration must connect it to the actual lift/Agent/Map, add
target loading/losses/optimizer/checkpoint handling, and validate Agent regression.
The component alone is NOT the final depth-guided BEV architecture or a complete
four-task training chain. It does not pretend that a resampled 18x32 grid restores
missing high-frequency image evidence.

No local Python, unit tests, forward/backward, exporters or training were run.
