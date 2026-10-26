# Depth Teacher Adaptation

## Status

Metric3Dv2 ViT-L is the selected candidate, not a validated OpenScene teacher yet.
This change adds inspection, native inference, diagnostic previews and offline
label export only. It does not add or start depth training. No Python or tests
were run on the development machine. Existing Agent/Map/Seg models, losses,
checkpoints, labels and training configurations are unchanged.

References:
- [Official inference/weights](https://github.com/YvanYin/Metric3D/blob/main/hubconf.py)
- [Metric3Dv2 paper](https://arxiv.org/html/2404.15506v2): driving training data and
  nuScenes surround-view reconstruction support this choice, but are not proof
  of OpenScene accuracy.
- [OpenScene data](https://github.com/OpenDriveLab/OpenScene/blob/main/docs/getting_started.md):
  v1.0 image/occupancy data should not be assumed to contain v1.1 raw LiDAR.

## Server Preflight

From the QUEST root in `quest_env`, run this CPU-only, read-only command first:

```bash
python scripts/export_metric3d_depth.py --indices 0 2500 4999 --preflight-only
```

It reads metadata once and checks only the selected eight-camera JPEG headers,
intrinsics, sensor2lidar rigid transforms, distortion, teacher paths and installed
package metadata. It does not import the teacher, load its checkpoint or create
files. First-frame calibration plus one line per frame is printed. Missing
weights/repository/dependencies and unverified image geometry are reported as
BLOCKER, exit code 2, not disguised as an inference failure. No installation or
network download is attempted. Existing training can keep running.

Defaults in `configs/depth_teacher.yaml` refer to local assets:
- `third_party/Metric3D` (official repository)
- `weights/teachers/metric3d/metric_depth_vit_large_800k.pth` (official ViT-L weights)

Use `--repo` and `--checkpoint` if already stored elsewhere. Do not install the
teacher into a running QUEST/Navformer environment without reviewing dependency
compatibility. Preflight package availability is not a successful import check.

## Asset And Environment Preparation

User-run Linux command, downloads official source/checkpoint only (no installation
or GPU usage, no overwriting existing assets):

```bash
bash scripts/prepare_metric3d_assets.sh
```

The native import chain needs **mmcv 1.x** even though hubconf itself has an
mmengine fallback: [mono/utils/comm.py](https://github.com/YvanYin/Metric3D/blob/main/mono/utils/comm.py)
unconditionally imports `mmcv.utils.collect_env`. mmengine alone is insufficient.
Use pure-Python `mmcv==1.7.2`, not mmcv-full/CUDA ops. xformers is optional in the
native ViT backbone; start with its built-in Torch attention fallback rather than
replacing the working Torch/CUDA installation to install an old xformers binary.

The user's reported quest_env has Torch 2.7.0+cu128 and NumPy 2.2.6. The upstream
legacy requirements use NumPy 1.x. The following optional USER-EXECUTED setup makes
a new teacher venv outside the repository, inherits Torch/torchvision read-only,
and installs isolated inference dependencies. It does NOT change quest_env or
navformer_env. It is a proposed compatibility profile, NOT a tested official
Torch-2.7 environment. Do not apply the upstream full requirements to quest_env.

```bash
conda activate quest_env && python -m venv --system-site-packages "$HOME/.venvs/quest_depth_teacher" && "$HOME/.venvs/quest_depth_teacher/bin/python" -m pip install -c envs/metric3d_quest_torch_constraints.txt -r envs/metric3d_inference.txt
```

Run teacher commands using `$HOME/.venvs/quest_depth_teacher/bin/python` explicitly.
The constraint file matches the reported Torch/torchvision versions so pip cannot
silently upgrade them. Review any installation error instead of changing the
parent environment. None of this setup was run by Codex.

Meanwhile, generate the geometry comparison on CPU in the existing quest_env:

```bash
python scripts/export_metric3d_depth.py --indices 0 --geometry-preview-only
```

This writes only `data/diagnostics/depth_metric3d_v1/geometry/<token>.png`, showing
all eight original JPEGs next to the metadata OpenCV-undistortion HYPOTHESIS.
No teacher weights or mmcv are needed. No image-geometry gate is bypassed: status
remains UNVERIFIED, no depth .pt is generated, and neither metadata nor original
images are modified. Compare building/pole straightness and corroborate the result
with image provenance; visual appearance alone is not proof. Previews are not
overwritten. Do not run this mode on all 5,000 frames.

## Geometry Gate

`image_geometry=unverified` is intentionally the default. The metadata having
distortion coefficients does not establish whether the JPEG is already rectified.
Confirm the sensor/image preparation source before changing this value:
- `raw_opencv`: raw pixels, with the documented OpenCV coefficient ordering
  (4/5/8/12/14 coefficients). Fisheye coefficients cannot be passed as OpenCV
  pinhole distortion merely because their length happens to fit.
- `rectified_pinhole`: verified rectified image and its corresponding intrinsic;
  nonzero stored distortion is rejected rather than silently reusing raw K.

Export additionally requires `--geometry-evidence` (or its config value) to
record the source of this declaration. This text is human evidence, not an
automated proof. Never supply an invented declaration to bypass the gate.

For raw_opencv, rectify using R=I, K'=K. Run the native teacher on that image,
remove its padding, restore original resolution and metric focal scale, then
sample its camera-Z depth back at the original distorted JPEG pixels. Thus the
saved depth and the existing Student RGB have the same pixel layout. Invalid
remap/resize support is masked; values outside [0.1,200] meters are invalidated,
not clamped into fake far-depth supervision. No per-image scale alignment is used.

The current BEV lift uses pinhole projection without distortion. This is explicitly
reported, not changed mid-training. If the JPEGs are raw and distortion matters,
BEV alignment needs a separate controlled audit before using depth for BEV fusion.

## Native Export After Geometry/Assets Are Confirmed

For example, ONLY if the source confirms raw OpenCV geometry, set the verified
geometry and evidence in `configs/depth_teacher.yaml`, then run on an available GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/export_metric3d_depth.py --indices 0 2500 4999
```

The same frozen teacher instance handles all eight cameras sequentially, using
original RGB JPEGs (not QUEST normalized tensors), aspect-preserving 616x1064
resize/pad/normalization and `fx_resized / 1000` metric conversion as in the
official ViT hub example. Local construction uses `pretrain=False` and a strict
local checkpoint load; missing/unexpected keys or shape mismatches abort export.
The script does not download weights. It does not load QUEST or DINO Student.

Output `data/soft_labels_depth_metric3d_v1/<token>.pt` contains:
- schema/alignment version, ORIGINAL source_index, token, scene, timestamp;
- metadata hash, optional split hash, camera order, source image paths/shapes;
- model/repository/checkpoint provenance, per-camera resize/metric scale;
- `depth.depth_m`, `depth.valid_mask`: `[8,1,252,448]`, float32/bool;
- units `m`, definition `camera_z`, scaled intrinsics, unchanged sensor2lidar,
  distortion and image geometry. Camera Z is not Euclidean radial distance.
- `confidence=None`: native confidence semantics are not assumed to be a
  calibrated probability, and geometric validity does not detect sky/reflections.
- `adaptation_verified=False`: not yet approved for formal training.

With `--split-manifest data/stage3_split.json --split train|validation`,
`--sample-index` is a position in that manifest list; saved source_index is still
the original metadata index, and every entry is validated against the manifest.
Without a manifest, sample-index directly denotes the original metadata index.
Explicit `--indices` cannot be combined with a manifest. There is no temporal
warm-up: this teacher predicts single images; QUEST remains eight-camera.

The audit directory contains `<token>.png` (8-view RGB/depth, one fixed log color
scale for every view/frame) and `<token>.json` (per-camera statistics and pairwise
reprojection diagnostics). Nothing existing is overwritten; choose fresh output
and audit directories for reruns. Errors stop export; no partial-camera .pt is
saved. A report/preview can remain after a later file-write failure.

Cross-view comparisons backproject camera-Z through the matching ray model,
sensor2lidar, target lidar2camera and target projection. They compare observed
target depth to reprojected target Z. Occlusions, timestamp offsets, dynamics,
pixel rounding and teacher errors all contribute; this is NOT GT accuracy, not
a synchronization certificate and not an automatic acceptance gate. Empty
overlap is unavailable, never a zero error. Review road/car boundaries and near/
far structure on several scenes; independent synchronized LiDAR is needed for
objective metric-depth accuracy. No Agent box-center or virtual centerline is
used as a depth target.

## Multi-Task Boundary

First train a separate depth decoder on frozen DINO features after approval.
It should receive camera/ray conditioning for metric prediction across focal
lengths. The current unconditioned 64x64 DepthHead is not promoted to a verified
multi-camera metric decoder by this export. Reuse the existing train/validation
manifest; evaluate teacher agreement separately from sensor-ground-truth accuracy.
Shared-BEV improvement is a future explicit integration, not a consequence of
training only the depth head. Do not constrain Agent box centers or invisible
nuPlan centerlines to equal visible-surface depth. Segmentation taxonomy remains
independent; an untrained segmentation prediction must not become a depth mask.

Server-only optional CPU tests (sources added, NOT executed locally):

```bash
python -m unittest discover -s tests -p test_metric3d_depth_adapter.py -q
```
