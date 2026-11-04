# SegFormer Teacher Approval For nuPlan Use

Source check and user approval: 2026-11-04. The user now accepts a teacher
with public usable weights and evidence of application to nuPlan; training
specifically on nuPlan/OpenScene is NOT a requirement for this Seg experiment.
This supersedes the earlier same-domain-only Seg admission restriction.

## Verified Primary Sources

1. [DriveStudio nuPlan preparation](https://github.com/ziyc/drivestudio/blob/main/docs/Nuplan.md),
   section 4, explicitly requests `segformer.b5.1024x1024.city.160k.pth` and
   runs mask extraction on processed nuPlan images.
2. [DriveStudio mask extraction code](https://github.com/ziyc/drivestudio/blob/main/datasets/tools/extract_masks.py)
   runs SegFormer Cityscapes19 and extracts sky and optionally dynamic masks.
   This is concrete deployment evidence, NOT a full 19-class nuPlan accuracy
   benchmark or evidence of nuPlan task training.
3. [NVIDIA-hosted Transformers model](https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024)
   and its [weight listing](https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024/tree/main)
   publish the B5 Cityscapes config, preprocessing config and approximately
   339MB PyTorch checkpoint. The model card describes Cityscapes fine-tuning;
   it states that the model card was written by the Hugging Face team.
4. [Original implementation](https://github.com/NVlabs/SegFormer) publishes
   the original model/config/checkpoint release used by DriveStudio.

Our HF checkpoint is the corresponding B5 Cityscapes model family/implementation,
not the same binary file as the original MMSeg checkpoint. Byte-for-byte or
numerical equivalence between those formats was NOT established by this audit.
Keep our exact checkpoint hash and preprocessing in exported-label provenance.

## Selected Existing Assets

- Teacher ID: `nvidia/segformer-b5-finetuned-cityscapes-1024-1024`.
- Config: `configs/seg_teacher.yaml`, unchanged.
- Local server bundle: `weights/teachers/segformer_cityscapes_b5`.
- Existing exporter: `scripts/export_segformer_pseudo.py`.
- Existing label directory: `data/soft_labels_segformer_cityscapes19_v1`.
- Label space: canonical 19 Cityscapes classes, not silently merged classes.

The user previously reported strict teacher loading with no missing/unexpected
weights, eight-camera inference, completed train/validation label exports and
Student experiments. That is prior USER-REPORTED runtime evidence, not a new
local/server run performed during this source check. Do not re-download or
regenerate all labels just because the admission criterion changed.

## Boundaries And Next Work

SegFormer-B5 is SELECTED FOR THE nuPlan SEG EXPERIMENT under the revised rule.
Do not relabel it as nuPlan-trained or assign its Cityscapes benchmark accuracy
to nuPlan. No independent nuPlan semantic GT score is established.

Keep Depth V3 as the selected depth baseline. The old Seg Student experiment
is not automatically part of that V3 checkpoint. A new Seg decoder/integration
must preserve Depth/Agent/Map and use existing token/split/pixel alignment checks.
No runtime environment, teacher weights, pseudo labels, model architecture or
training configuration is changed by this confirmation document.
