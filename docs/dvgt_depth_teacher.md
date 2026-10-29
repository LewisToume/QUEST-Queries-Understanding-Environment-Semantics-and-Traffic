# DVGT-1 OpenScene Depth Teacher

Implementation added 2026-10-29. Static review ONLY: no Python/tests, model
execution, data export, installation or training was run by Codex. The user
executes the commands below on the server. No Student architecture/loss or
Agent/Map weights are changed. Old Metric3D/Seg experiments are preserved.

## Source And Weight Contract

- [Author code](https://github.com/wzzheng/DVGT), pinned commit
  `51cf3f6d11fdff8bc7e2bbe1a88f71665ccb2236`.
- [Author checkpoint](https://huggingface.co/RainyNight/DVGT-1/tree/main), revision
  `4918b6ed3547a00fd217d602fabb84ada180dba4`, file `dvgt1.pt`.
- HF API listed 6,929,346,337 bytes and SHA256
  `8e1ab60f901cffc3316fa0a09f24ddcf6b90d0221cf7bfea64664c6f83cc1e22`.
  These were checked as remote metadata, NOT by downloading/loading the file.
- DINOv3 code revision `6876159a11b4df116f30f667f8c9888617df0751`.
  No separate gated DINOv3 weight is needed: code-only construction is followed
  by STRICT full DVGT checkpoint loading. Missing/unexpected/shape-mismatched
  keys are fatal; no Metric3D-style omission exception is applied.
- Teacher training includes OpenScene and its task evaluation configuration
  uses OpenScene. This qualifies its dataset provenance, not our actual depth
  accuracy. Teacher-training overlap with the QUEST validation scenes is UNKNOWN.

## Native Geometry And Pixel Alignment

The pinned source is the specification, not generic monocular-depth rules:

1. Read original JPEGs/calibration through the existing OpenScene path resolver.
   Require explicit `raw_opencv` or `rectified_pinhole` hypothesis. A hypothesis
   or undistortion preview is NOT calibration approval.
2. Call official `process_image_sequentially` and `resize_to_aspect_ratio`, then
   official principal-point crop / deterministic resize / strict crop helpers.
   Final native input is `[1,T,8,3,288,512]`, float RGB in [0,1]. Normalization
   happens once INSIDE the native backbone. No QUEST normalized tensor is used.
3. Native order is F0,B0,L0,L1,L2,R0,R1,R2. The adapter tracks identities and
   restores QUEST output order F0,L0,R0,L1,R1,L2,R2,B0. Student order is unchanged.
4. Native `points` are in first-ego RDF (right/down/front), not nuPlan FLU
   (front/left/up). Use the official `T_flu_rdf` matrix from
   `tools/data_process/dataloaders/base_dataset.py`.
5. The pinned training config multiplies geometric targets by 0.1. Divide raw
   points by 0.1 ONCE before metric camera transformation; do not median-align
   targets, reinterpret radial range as Z, or rescale by Metric3D focal rules.
6. Explicitly compute:

```text
T_camera_from_first_ego_rdf =
 inverse(T_camera_to_lidar_current)
 @ inverse(T_lidar_to_ego_current)
 @ inverse(T_ego_current_to_global)
 @ T_ego_first_to_global
 @ T_flu_from_rdf
```

Keep full pitch/roll and use double-precision transform composition. Reuse the
existing `resolve_lidar2ego`: explicit and derived transforms must agree. The
pose source is METADATA, as in native point evaluation, not guessed camera
poses. Native predicted pose encodings are retained in the diagnostic report
with their scaled RDF definition, not silently used as meters.

7. Convert points to positive camera Z. Use original JPEG pixel rays, distortion
   and the final native K to sample the predicted depth at Student pixel centers.
   Never stretch a cropped prediction over the whole JPEG. Out-of-image and
   partially invalid interpolation support are excluded, not fake negatives.
8. Validity requires finite XYZ/confidence, positive/range-valid Z and an
   explicit configurable reprojection-consistency threshold. Confidence is
   NOT calibrated probability; these masks do NOT identify sky/reflections.

Reports include per-camera valid fraction, depth median and native reprojection
errors, native K, camera transforms, sequence tokens/indices/timestamps and
weight/source/config/adapter hashes. Reprojection consistency is not a real
LiDAR accuracy measurement. Camera timestamp offsets remain uncorrected.

## Temporal Selection And Output Safety

`--sample-index/--num-frames` select TARGETS using the existing raw-index or
manifest-position convention. Context is built with one metadata scan over
the target scenes, sorted by timestamp. Target token/index selection is never
changed by sorting. Default two real frames, <=1.1s adjacent gaps, timestamps
in microseconds. No repeated/fake history and no cross-scene/split context.

Use preceding context when available. A scene's first target needs explicit
`--allow-future-context`; that permits real future frames for the OFFLINE teacher
only. Actual future usage is recorded per target. Missing/discontinuous context
fails instead of generating labels. Identical windows are inferred once.

Default inference is diagnostic-only: at most three TARGET frames, PNG/JSON,
no `.pt`. `--export-candidate-labels` is explicit and writes only requested
tokens under `data/soft_labels_depth_dvgt1_v1_candidate`. No overwrite/resume is
implicit. Failure leaves no published label for the failing frame. A frame with
any all-invalid camera is rejected. Existing Metric3D outputs are not reused.

Depth record schema 2 and alignment `dvgt1_first_ego_rdf_to_camera_z_original_pixels_v1`
are distinct from old Metric3D schema 1. The existing dense Dataset accepts both
explicit contracts but rejects mixing them or changing preprocessing within a
run. Old checkpoints remain readable; old-label resumes cannot swap to DVGT.
DVGT candidates still require `--experimental-unverified-depth` for training.

## Server Commands

### 1. Prepare Assets (Network / Disk Only)

This downloads about 6.93 GB plus source code; installs NO dependencies and does
not start a model. Git source revisions and weight checksum are pinned. Existing
weights are verified rather than overwritten; partial downloads resume. Existing
source with tracked modifications or a different origin is refused.

```bash
cd /home/user/DataDisk/QUEST_WORK/QUEST && git pull --ff-only origin feature/multiview-quest && bash scripts/prepare_dvgt_assets.sh
```

### 2. CPU Preflight (Read-Only)

Try the existing QUEST environment; do not install native requirements into it.
This inspects metadata/JPEG headers/transforms/assets and hashes weights, but
does not construct a model or require GPU. Missing imports are reported. Official
DVGT tested Python3.11/torch2.8/CUDA12.8; our Python3.10/torch2.7 runtime has NOT
been verified. If dependencies conflict, use a separately authorized teacher
environment; never modify Navformer or QUEST automatically.

```bash
conda activate quest_env && cd /home/user/DataDisk/QUEST_WORK/QUEST && python scripts/export_dvgt_depth.py --sample-index 0 --num-frames 1 --allow-future-context --image-geometry raw_opencv --preflight-only
```

Expected: `DVGT_PREFLIGHT=PASS` after assets/dependencies are present; this is
NOT inference/accuracy approval. No weights/data are downloaded by the script.

### 3. One-Target GPU Diagnostic (After Preflight)

Runs two REAL same-scene frames x eight cameras and saves PNG/JSON only. Check
GPU availability before choosing GPU0. It does not train, save `.pt` labels or
overwrite old artifacts. Actual memory is measured; no 3090 fit is promised.

```bash
conda activate quest_env && cd /home/user/DataDisk/QUEST_WORK/QUEST && CUDA_VISIBLE_DEVICES=0 python -u scripts/export_dvgt_depth.py --sample-index 0 --num-frames 1 --allow-future-context --image-geometry raw_opencv --diagnostic-only
```

Return the missing/unexpected-key summary, native output shapes, eight camera
statistics, peak memory and `data/diagnostics/depth_dvgt1_v1/diagnostic/<token>.json`
plus PNG. If all pixels fail consistency, inspect axes/scale/calibration first;
do not simply relax the threshold to manufacture valid targets.

### 4. Test Sources (CPU, User Executes)

```bash
conda activate quest_env && cd /home/user/DataDisk/QUEST_WORK/QUEST && python -m unittest discover -s tests -p 'test_dvgt_depth_adapter.py' -v
```

### 5. Candidate Export / Student Training (NOT Yet Approved)

After real diagnostic and geometry review, use the SAME manifest for each split,
explicit `--export-candidate-labels` and `--allow-future-context`. Review output
quality before a 5,000-frame rerun. `configs/seg_depth_dvgt.yaml` is a separate
Depth-only experimental training config; no Seg masks are implicitly consumed.
The existing Student head/loss is preserved, not claimed as a DVGT reproduction.
This isolated config starts new dense heads from the spatial checkpoint; it
does NOT transplant the previous trained Seg head into the new Depth experiment.
The old four-output checkpoint is preserved, not overwritten. Consolidating
trained dense branches needs a separate audited initialization step later.

The eventual CPU label/training preflight command, after both split exports:

```bash
conda activate quest_env && cd /home/user/DataDisk/QUEST_WORK/QUEST && python scripts/train_seg_depth.py --config configs/seg_depth_dvgt.yaml --experimental-unverified-depth --preflight-only
```

Do not start formal training from this document before diagnostic review. Real
metric certification still needs correctly matched nuPlan/OpenScene LiDAR. A
same-domain Seg teacher remains unresolved; it is not replaced by BEV semantic
segmentation or a renamed Cityscapes checkpoint.
