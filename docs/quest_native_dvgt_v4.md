# Full Native DVGT Geometry In QUEST

## Decision And Scope

The user explicitly requests approaching the teacher's full architecture, not
another small randomly initialized approximation. The new version imports the
pinned native DVGT-1 component and inherits its ENTIRE strict task checkpoint.
There is no new DINO download: the full DVGT checkpoint includes DINOv3 weights.
This is a large pretrained component transplant, NOT a new distillation result.

Motivation from the user's V3 report: validation teacher AbsRel 0.10146475,
history-disabled same-weight AbsRel 0.10190397, and 50-80m bias -6.75268m.
These are teacher-pseudo agreement metrics, not official true-GT teacher scores.
They motivate testing a full pretrained component rather than assuming that
more epochs of the small branch will close the difference.

| Component | Previous temporal V3 | Native V4 integration |
| --- | --- | --- |
| Depth image encoder | Frozen DINOv2-S/14, width 384 | Native DINOv3 ViT-L/16, width 1024 |
| Geometry | Four calibrated spatial blocks plus four small history adapters | All 24 native rounds, each with intra-view, cross-view and cross-frame attention (72 blocks) |
| Dense decoder | Reimplemented scalar camera-Z DPT | Native multilevel DPT XYZ/confidence head, indices 4/11/17/23 |
| Pose | Metadata transforms, no predicted pose head | Native ego-pose head retained; metadata used only for output coordinate conversion |
| Weights | Small geometry branch learned from offline depth | Every native backbone/geometry/point/pose weight strictly inherited |
| Size | User-reported total 94,897,717 / trainable 25,302,609 | About 1.7B native parameters plus preserved Agent/Map; script prints actual scope/counts |
| Learning status | Five-epoch trained experiment | Initialized pretrained baseline; no new training claimed |

Primary sources, pinned to the actual teacher export revision:
- [Architecture](https://github.com/wzzheng/DVGT/blob/51cf3f6d11fdff8bc7e2bbe1a88f71665ccb2236/dvgt/models/architectures/dvgt1.py)
- [Geometry aggregator](https://github.com/wzzheng/DVGT/blob/51cf3f6d11fdff8bc7e2bbe1a88f71665ccb2236/dvgt/models/backbones/dvgt1_aggregator.py)
- [DPT](https://github.com/wzzheng/DVGT/blob/51cf3f6d11fdff8bc7e2bbe1a88f71665ccb2236/dvgt/models/heads/dpt_head.py)

## One QUEST, Transitional Dual Encoders

`QUESTNativeDVGT` owns `spatial` and `geometry_core` as registered submodules.
It produces Agent, vector Map, depth, native point maps, confidence and ego pose
from one original JPEG packet. The old small Depth branch is REMOVED, not run
alongside the new native branch. The native model is not an external inference
service or four independently run teachers.

It is nevertheless NOT yet a single shared image encoder. The preserved
Agent/Map route still uses its DINOv2 features. Silently replacing those with
DINOv3 features through a random 1024-to-384 projection would not preserve
trained task behavior. A shared DINOv3-to-BEV task-adapter migration needs its
own supervision and Agent/Map nonregression evaluation; it is not implemented
or declared complete here. Formal same-domain camera Seg is still unavailable.

## Input And Geometry Contracts

- Same-split, same-scene strictly earlier timestamps choose previous/current.
- No future inputs, cross-split context or repeated-current fake history.
- At scene starts or gaps the NATIVE model receives T=1, not a fake T=2.
  This differs from the offline teacher's permitted future-context scene start;
  those predictions cannot be expected to match offline labels exactly.
- Both routes start from the same original JPEGs. Spatial preprocessing stays
  normalized 252x448; native preprocessing replays official undistortion/crop/
  resize to RGB [0,1] at 288x512. Each keeps its own exact calibration.
- Native camera order is explicitly permuted to QUEST order for camera-Z.
- Native points stay in scaled first-ego RDF coordinates. Camera conversion
  uses the existing audited first-ego-to-camera transform and scale 0.1.
- Depth restoration is torch-differentiable and exports an explicit validity
  mask. Invalid pixels are zero, NOT filled supervision. The old dense loss
  rejects zeros globally and must NOT be used unmodified on these outputs.
- Confidence is not a calibrated probability. Calibration/true depth accuracy
  remains UNVERIFIED. Native predicted pose is retained, but metadata is still
  required to express points in the existing camera-Z evaluation convention.

## Server Checks

No local Python, tests, GPU forward, export or training is run by Codex.
The server entry point refuses changed native source/checkpoint hashes,
missing dependencies, incompatible spatial sources and output overwrites.
Default spatial source is V2, avoiding the regressed V3 temporal Agent adapter.

```bash
conda activate quest_env
cd /home/user/DataDisk/QUEST_WORK/QUEST
python scripts/check_quest_native_dvgt.py --preflight-only
OMP_NUM_THREADS=2 python -m unittest discover -s tests -p 'test_native_dvgt_quest.py' -q
CUDA_VISIBLE_DEVICES=3 python -u scripts/check_quest_native_dvgt.py --save-initialized
```

The combined runtime needs both native DVGT dependencies and the existing
QUEST transformers runtime. Preflight reports missing native dependencies;
do not replace torch/CUDA or upgrade the server environment wholesale.

Runtime checks compare actual native XYZ/confidence/pose tensors before/after
integration, and actual Agent/Map predictions against their preserved source.
They reuse ONE resident native component sequentially, not two GPU copies.
This tests wiring equivalence, not independent teacher quality or generalization.
The CPU fixtures test ordering, scales, masks, gradients, cold starts and state
round trips; they are not evidence that real 1.7B native inference succeeded.

The initialized checkpoint contains ALL geometry and spatial model weights in
one file (roughly 7GB; exact count printed). It is a NEW profile and is not
accepted by `train_seg_depth.py` / `evaluate_seg_depth.py` as legacy V2/V3.
To reload, use `load_native_quest(path, repo, device)`: it constructs native code
without downloading/loading the original teacher checkpoint, then restores ALL
saved weights strictly. Pinned native source and local DINOv2 construction
assets remain required. The original V2/task/DVGT checkpoint files are NOT needed
by this loader. Keep original baseline files for comparisons regardless.

## Training And Evidence Boundary

Do NOT immediately train 5,000 samples against the very same frozen teacher
outputs and call the near-zero error a new learning success. With identical
preprocessing/context, the copied pretrained component should already produce
the native outputs. First establish that baseline and memory use.

Full 1.7B Adam training is not the prior 1.8GB experiment: FP32 parameters,
gradients and two optimizer moments alone are roughly 27GB before activations.
A single RTX3090 cannot be assumed sufficient. Subsequent adaptation should
use measured selective unfreezing/low-rank updates or distributed sharding,
preserving the full native architecture. No training-memory PASS is claimed.

The immediate next data for real quality improvement is synchronized LiDAR or
audited metric point-map/pose targets, and more independent held-out scenes.
Offline scalar depth alone is not the teacher's full geometry training recipe.
No pose/XYZ GT, tracking IDs or camera Seg labels are fabricated.
