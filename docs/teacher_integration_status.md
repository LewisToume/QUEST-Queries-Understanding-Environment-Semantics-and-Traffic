# Teacher Integration Status

Design contract: [Four-Teacher Student](four_teacher_student_design.md).
BEV is the shared spatial representation, not a fifth teacher task.
The active route is THREE neural teachers plus real vector-map GT, NOT four
required neural checkpoints. Four-task training is not yet validated.

| Teacher | Task | Evidence | Remaining gate |
| --- | --- | --- | --- |
| Navformer UniAD Agent branch | Agent | User-reported real inference/export and Stage 2 training | Agent regression after shared-front-end changes |
| nuPlan HD map | Vector Map | User-reported real vector GT export/audit/training | Keep schema/split/capacity validation; no neural Map teacher is required |
| Navformer raster map branch | Optional Map KD | User-reported unreliable lane masks | DISABLED; cross-representation KD must not resume without a new audit |
| SegFormer-B5 Cityscapes | Semantic Segmentation | User-reported strict load and real 8-view GPU inference; received PNG/JSON for source index 0 | Native19 Student training connected in code; actual labels/export/training still server-only; one frame is not GT accuracy |
| Metric3Dv2 ViT-L | Depth | User-reported real 8-view CUDA inference, shape [8,1,252,448], 3277 MiB; only audited non-inference mask_token omission | Student training connected in code; geometry remains unverified; experimental image-plane scope must be explicit |

QUEST Stage2 reads exported labels by OpenScene sample token. Online teacher
construction is disabled in the student training process.

`configs/stage2_distill.yaml` remains a legacy Agent-only configuration, not
proof of four-teacher readiness: Seg/Depth there are disabled placeholders and
`maptr_output_to_quest()` still rejects unverified taxonomy. MapTRv2 is ONLY a
structure/loss reference here; its nuScenes checkpoints are not prerequisites.
The separate `configs/seg_depth.yaml` enables dense offline training, with
frozen Agent/Map and strict label preflight. Runtime is not yet verified.
Stage 3 trains against nuPlan vector GT, NOT
MapTRv2 predictions. Navformer raster-map KD stays disabled because of the
observed unreliable lane masks.

`scripts/inspect_multitask_teachers.py` inventories explicit asset paths using
only the standard library. It does not certify model load, outputs or quality.

Asset preparation and actual Seg exporter commands are documented in
[Dense Teacher Export](dense_teacher_export.md). No MapTRv2 download is required.

Depth adaptation uses `configs/depth_teacher.yaml` and
`scripts/export_metric3d_depth.py`. It does not enable the legacy Stage2 Depth
task or modify Agent/Map training. See [Depth Teacher Adaptation](depth_teacher_adaptation.md)
for the geometry gate, canonical labels, provenance and server-only validation.
