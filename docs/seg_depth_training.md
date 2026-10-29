# Seg / Depth Student Training

## Implemented Scope

ONE QUEST V3 model reuses frozen DINOv2-S for all four outputs. The new
`shared_image_pyramid_v1` profile reads normalized blocks 3/6/9/12, constructs
four scales, and trains a shared 128-channel image neck with separate Seg and
intrinsic-conditioned Depth decoders. These are actual model outputs, not
disconnected teacher adapters:

```text
images          [B,8,3,252,448]
DINO maps x4    [B,8,384,18,32]
seg_logits      [B,8,19,252,448]  native Cityscapes19
depth           [B,8,1,252,448]   positive camera Z in meters
agent_boxes     [B,100,8]        preserved Stage 3 Agent branch
map_points      [B,N_map,20,2]   preserved Stage 3 Vector Map branch
```

The existing spatial branch still sees unmodified final DINO features.
Its DINO/BEV/Agent/Map weights and buffers are fingerprinted and must remain
unchanged. Its frozen modules run in eval mode. No Agent/Map label re-export is
needed. The unused dense BEV adapter remains frozen. This phase is NOT joint
four-task finetuning, nor depth-guided geometry; those require separate evidence.
No Student code, tests, model forward or training was executed locally.

## Supervision And Loss

`SegDepthDataset` reuses OpenScene image/calibration reading and manifest raw
indices. Each label must match token/index/scene, canonical camera order,
metadata SHA256 and split SHA256. Depth calibration is checked against metadata.
Teacher/preprocessing signatures cannot mix between train/validation. Full label
file hashes are audited and checked again on read. Missing labels fail by
default; optional `missing_labels: mask` keeps absent targets as None and skips
their losses, never fabricating dummy GT. Depth also needs aligned Seg labels
for sky exclusion. A task with no usable target pixels fails preflight.

- Seg: cross-entropy on valid native19 pseudo labels. Confidence is NOT treated
  as calibrated correctness; this is pseudo supervision, not lossless soft KD.
- Depth: valid-pixel log SmoothL1 + 0.5 relative SmoothL1 + 0.5 log-gradient L1.
  Keep metric scale; never median-align prediction to teacher. Only pairs of
  valid neighboring pixels contribute edge loss. Default range is 0.1-80m;
  teacher Seg sky ID10 and invalid pixels are excluded. Export additionally
  removes the teacher's native canonical 200m saturation BEFORE focal rescaling.
- Total = Seg + Depth; weights/range/learning rate are in `configs/seg_depth.yaml`.
- Held-out metrics are **teacher agreement**, NOT measured real GT accuracy:
  Seg pixel accuracy/mIoU, Depth AbsRel/RMSE/delta1. Original train/validation
  manifest remains unchanged and disjoint.

## Geometry Policy

Preview appearance does not verify raw-vs-rectified image provenance or metric
depth accuracy. Formal default training requires `verified_geometry` labels
with explicit source/pipeline evidence at export. Existing diagnostic PNG/JSON
files are NOT training labels.

For an explicitly experimental image-plane run, the exporter provides
`--export-candidate-labels` with an explicit image-geometry hypothesis. It writes
to a separate candidate directory and marks `geometry_verified=false` and
`training_label_scope=image_plane_candidate`. Training requires the separate
`--experimental-unverified-depth` flag and uses a separate checkpoint. This
does not authorize using uncertain depth to alter BEV/Agent/Map geometry.

## Server: First One-Frame Check

Run in `quest_env`, from `/home/user/DataDisk/QUEST_WORK/QUEST`, after syncing
the branch. Choose an idle GPU; the examples use physical GPU0. The second
command uses the existing isolated depth Python, NOT a package install.
Exports create new files and refuse existing outputs; choose another audit
directory on rerun. Do not run concurrent exports to the same target directory.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/export_segformer_pseudo.py --split-manifest data/stage3_split.json --split train --num-frames 1 --audit-dir data/diagnostics/seg_dense_training_v1
```

```bash
CUDA_VISIBLE_DEVICES=0 "$HOME/.venvs/quest_depth_teacher/bin/python" scripts/export_metric3d_depth.py --split-manifest data/stage3_split.json --split train --num-frames 1 --image-geometry raw_opencv --export-candidate-labels
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_seg_depth.py --num-samples 1 --smoke-steps 1 --experimental-unverified-depth
```

The last command loads the existing spatial checkpoint, checks real losses,
gradients and a parameter update, then prints the four-task output shapes.
Expected: `SEG_DEPTH_SMOKE=PASS`, frozen Agent/Map unchanged, checkpoint save OFF.
This is one real optimization step, NOT a claim of convergence or geometry approval.

## Server: Labels And Longer Training

For 5000 train / 120 held-out frames, run each exporter for both manifest splits.
After the one-frame check, start remaining train exports at `--sample-index 1
--num-frames 4999` (not zero, because output overwrite is refused). Validation
uses `--split validation --sample-index 0 --num-frames 120`. Use the same adapter
revision, geometry hypothesis and teacher checkpoint across all files.
Exports can be expensive: 8 real teacher inferences per frame and tens of GB of
labels/previews. Check free disk/GPU availability first. Do not launch training
until ALL selected train/validation labels are available.

```bash
python scripts/train_seg_depth.py --preflight-only --experimental-unverified-depth
```

This audits all 5000+120 labels, source checkpoint metadata and certified Map
query count, WITHOUT constructing a Student or performing forward. Actual strict
component weight loading happens at training/smoke startup.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_seg_depth.py --experimental-unverified-depth
```

Default: 5 epochs, batch1, AdamW/cosine, new
`checkpoints/quest_seg_depth_experimental_v1.pt`. The formal verified mode omits
the experimental flag and uses `checkpoints/quest_seg_depth_v1.pt`. Formal
Depth export additionally requires `--geometry-evidence` and approved geometry.
To train only Seg while Depth geometry remains unresolved, use `--tasks seg`;
Depth output is then explicitly untrained in checkpoint task metadata.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_seg_depth.py --checkpoint checkpoints/quest_seg_depth_experimental_v1.pt
```

Evaluation now also saves `data/diagnostics/seg_depth_review_v1/report.json` and
up to three validation-frame previews (prefer one per scene). Seg shows RGB,
teacher, Student, disagreement; Depth shows RGB, teacher, Student, relative
error. Depth panels use a common evaluation mask and fixed log color scale,
never independent per-image scaling. Black areas are excluded, not correct
predictions. No geometric projection, teacher rerun or label change is performed.
Use `--output-dir <new-empty-directory>` for a repeat or `--preview-frames 0`
to skip PNGs. Existing reports/previews are not overwritten.

Console output stays bounded: existing summary, 19-class IoU/precision/recall,
depth camera-Z bins and per-scene scores. JSON additionally contains the full
teacher-row/Student-column confusion matrix, per-camera scores, signed depth
bias, scale ratio, pixel-mask counts, original frame indices/tokens and hashes.
Bins use teacher Z: [0.1,10), [10,20), [20,30), [30,40), [40,50), [50,80] by
default. Empty bins/classes have null metrics, not zero-error or perfect scores.
The original global metric definitions and 0.1-80m/sky exclusion are unchanged.

User-reported first held-out result (120 frames, 3 scenes): Seg teacher pixel
accuracy 0.901887, mIoU 0.419609; Depth teacher AbsRel 0.153423, RMSE 7.082613m,
delta1 0.779979. These establish teacher agreement only. No structural/loss
change is justified by these aggregate values alone. The new diagnostic code
and tests have NOT been run locally.

No additional dataset is needed for this diagnostic. To measure true depth
accuracy later, provide a small synchronized LiDAR subset with timestamps,
format/units and LiDAR-to-camera calibration, plus verified JPEG geometry.
Sparse LiDAR must use visibility/z-buffer handling; object box centers, OCC
centers and map lines are not surface-depth GT. True semantic accuracy needs
human-labeled pixels (including ignore IDs), rather than another teacher score.

Dense checkpoints include full model + optimizer/scheduler/epoch, model/profile
config, trained tasks and Agent support mask, label/manifest hashes, source
checkpoint hash, geometry policy and frozen-spatial checksum. Existing V3
Stage2/3 checkpoints are used only as explicit spatial warm starts; automatic
legacy-to-dense loading is refused. Use `--resume <dense.pt>` without overrides
to continue exactly that run. Resume rejects changed labels/split/source/config.
Legacy Agent/Map evaluators still consume their original checkpoint formats;
use the saved `model_config` + `load_dense_model` when extending their dense
checkpoint support. A full dense smoke verifies all four outputs coexist.

## Server-Only Tests

Test sources cover four-task forward, single DINO extraction, frozen spatial
weights, dense gradients/updates, checkpoint profile/roundtrip, teacher-agreement
metrics, sky/range/invalid masks, non-contiguous joins and changed label hashes.
They have NOT been executed by Codex:

```bash
python -m unittest discover -s tests -p 'test_seg_depth_training.py' -v
```

Also run existing `test_shared_dense_perception.py`, `test_metric3d_depth_adapter.py`
and `test_cuda_runtime.py` in their appropriate environments. Test success is
not a substitute for the real one-frame GPU check above.

New diagnostic test sources (not executed locally):

```bash
python -m unittest discover -s tests -p 'test_seg_depth_diagnostics.py' -v
```

## Bounded Seg Class-Balance Fine-Tuning

The user-reported training export contains hundreds of millions of road/building
pixels but fewer than a million rider/train/motorcycle pixels. Held-out teacher
agreement confirms near-zero recall for several rare classes. This motivates a
controlled loss experiment, not a new taxonomy or a claim that all rare teacher
labels are correct. Do not substitute unrelated nuScenes test or RoboSense PCDs
for synchronized OpenScene depth GT.

`configs/seg_balance.yaml` defines an opt-in recipe:

```text
w_c = clip(sqrt(median(positive training class counts) / max(count_c, 1)), 0.5, 5.0)
Seg CE = sum(w_target * pixel_CE) / sum(w_target) over valid pixels
```

The ratio between largest/smallest pixel weights is capped at 10, rather than
amplifying a rare/noisy class by thousands. The formula and limits are an
engineering starting point, not a proven optimum. Weighted mean normalization
follows [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
No focal/Dice loss, oversampling, image/camera changes or class remapping is added
in this first experiment. Depth loss and geometry policy remain unchanged;
the shared dense neck can still affect Depth, so compare its held-out metrics.
DINO/Agent/Map remain frozen and fingerprint-protected.

Frequencies are recomputed from ALL selected TRAIN `.pt` labels during existing
preflight, including frame0. The 4999-frame export log is not hardcoded. No
validation statistics determine the weights. Counts, class order, weights and
label hashes are saved with the training config/audit. Ordinary old runs keep
their old loss and audit schema; balanced resumes require identical weights.
Frames with no positive teacher example of a class cannot learn that class
merely by changing weights; frequency/qualitative audits remain necessary.

Start from the trained dense model, with a NEW optimizer and output checkpoint:

```bash
python scripts/train_seg_depth.py --finetune-from checkpoints/quest_seg_depth_experimental_v1.pt --seg-class-balance --checkpoint checkpoints/quest_seg_depth_balanced_v1.pt --preflight-only
```

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_seg_depth.py --finetune-from checkpoints/quest_seg_depth_experimental_v1.pt --seg-class-balance --checkpoint checkpoints/quest_seg_depth_balanced_v1.pt
```

Default fine-tune is 3 additional epochs at lr3e-5 (not restarting Seg/Depth from
random weights). Source tasks, full train/validation lists, metadata/label/teacher
hashes, spatial initialization and architecture must match. Source geometry
policy is inherited and stays experimental; do not add the experimental flag
again. `--epochs`/`--lr` can override the recipe; changing tasks or frame lists
is refused. `--smoke-steps 1` is available for a no-save check after the same
full training-label audit. Original checkpoints and teacher labels are never
overwritten. `--resume` retains exact saved config/optimizer semantics instead
of applying a new recipe.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_seg_depth.py --checkpoint checkpoints/quest_seg_depth_balanced_v1.pt --output-dir data/diagnostics/seg_depth_balanced_review_v1
```

Compare per-class IoU/recall/precision AND common-class/Depth regression against
the existing baseline. Improved recall alone with excessive false positives is
not success. Validation definitions, masks and taxonomy do not change; weighted
training losses are not directly comparable to the old unweighted total loss.
Do not keep tuning against these same 3 scenes as if they were a final test set.

Test sources are added, NOT executed locally:

```bash
python -m unittest discover -s tests -p 'test_seg_class_balance.py' -v
```
