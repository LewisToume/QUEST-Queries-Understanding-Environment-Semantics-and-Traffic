# SegFormer Segmentation On Frozen QUEST Depth V3

This implements the user's approved nuPlan-use Seg teacher decision.
Teacher: existing `weights/teachers/segformer_cityscapes_b5`, existing audited
Cityscapes19 labels in `data/soft_labels_segformer_cityscapes19_v1`.
No teacher inference, download, environment installation or label regeneration
is part of this run. The candidate teacher is Cityscapes-trained; its nuPlan
use evidence does not establish independent 19-class nuPlan GT accuracy.

## Model

One `SegOnFrozenV3` owns the complete trained V3 QUEST and one new Seg branch.
There is ONE DINOv2 backbone, not a second MiT/SegFormer image encoder. The new
branch adapts four DINO block features to the actual teacher decoder channel
widths (64/128/320/512), learns reassembly scales, adds a small RGB detail
adapter, and uses the native Transformers SegFormer B5 decode head (width 768).
Every teacher decode-head weight/buffer is loaded strictly; missing keys fail.
Adapter-generated scales are not the teacher's native MiT image hierarchy,
and teacher decoder initialization does not guarantee immediate good predictions.

V3 Depth, Agent, Map and DINO parameters AND buffers are frozen. `train()`
always keeps the entire base in eval mode. Seg training uses only current-image
features; inference also uses true same-scene causal history for frozen Depth.
Agent intentionally uses the original current-only route, NOT the V3 temporal
Agent adapter which regressed on pedestrians. Existing adapter weights remain
stored/frozen but unused; no tracking-ID capability is claimed.

Source V3 joint-profile precision is preserved: DINO/spatial FP32, Depth bf16,
without dropout during base inference. Every epoch hashes ALL base state,
including dense Depth; this is stricter than the old spatial-only fingerprint.
Each epoch also compares Depth/Agent/Map outputs on a fixed probe exactly.
These checks test preservation, not true-GT task quality.

## Training And Data

New entry point `scripts/train_seg_on_v3.py`, config
`configs/seg_on_depth_v3_12ep.yaml`. It audits 5000 train and 120 validation
tokens, label hashes, class counts and same-split causal context before GPU use.
Temporal dataset now explicitly accepts Seg-only targets as well as Depth-only;
Seg training never requires or fabricates depth GT. Metadata/splits must match V3.

Only Seg is optimized for 12 epochs with AdamW/cosine scheduling. Loss is
validity-masked, training-frequency-balanced CE, bounded weights 0.5..5,
normalized by the sum of target weights. No teacher confidence calibration,
independent GT or class-merging claims are made. Validation reports pixel
accuracy and mIoU relative to the teacher. `--evaluate` additionally prints
per-class IoU/recall to expose minority-class errors.

Main output: `checkpoints/quest_seg_on_depth_v3_v1.pt`.
Best validation mIoU: `checkpoints/quest_seg_on_depth_v3_v1_best.pt`.
Each includes ALL four-task weights and a reconstructible base profile. Reload
via `load_seg_on_v3`, not the old dense-only loader. Existing weights are kept.

## One Command On The Server

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/train_seg_on_v3.py --run-all
```

`--run-all` executes isolated CPU tests, a 2-frame/4-step no-save GPU smoke,
then full 5000-frame/12-epoch training, stopping on any nonzero subprocess exit.
Tests/smoke do NOT establish final segmentation quality. Missing labels/assets
stop with their paths; they are not silently regenerated. Latest and best
outputs are separate. Use `--resume path` explicitly to resume a partial run;
pre-existing outputs are never silently treated as a new run.

Evaluation: `python scripts/train_seg_on_v3.py --evaluate checkpoints/quest_seg_on_depth_v3_v1_best.pt`.
This reports Seg pseudo-label agreement; frozen Depth's prior validation is not
recomputed each epoch. Four-task output shapes and preservation are checked by
CPU tests and the GPU probe. No local Python/model/test execution was run by Codex.
