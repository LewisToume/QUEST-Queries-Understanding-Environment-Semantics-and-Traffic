# Teacher Integration Status

Design contract: [Four-Teacher Student](four_teacher_student_design.md).
BEV is the shared spatial representation, not a fifth teacher task.
The four teachers are NOT all inference-ready.

| Teacher | Task | Evidence | Remaining gate |
| --- | --- | --- | --- |
| Navformer UniAD Agent branch | Agent | User-reported real inference/export and Stage 2 training | Agent regression after shared-front-end changes |
| MapTRv2 | Vector Map | Reference/config; existing local three-class checkpoint | OpenScene KD adapter/inference unverified; current weights lack centerline; centerline variant needs audit |
| SegFormer-B5 Cityscapes | Semantic Segmentation | Official model identified; local legacy source | Server assets/inference and 19-class taxonomy mapping unverified; training still disabled |
| Metric3Dv2 ViT-L | Depth | Adaptation/export code; user reports downloads | Real inference and image geometry unverified; student training not connected |

QUEST Stage2 reads exported labels by OpenScene sample token. Online teacher
construction is disabled in the student training process.

`configs/stage2_distill.yaml` remains a legacy Agent-only configuration, not
proof of four-teacher readiness: Seg/Depth there are disabled placeholders and
`maptr_output_to_quest()` still rejects unverified taxonomy. No training config
is enabled by this status update. Stage 3 trains against nuPlan vector GT, NOT
MapTRv2 predictions. Navformer raster-map KD stays disabled because of the
observed unreliable lane masks.

`scripts/inspect_multitask_teachers.py` inventories explicit asset paths using
only the standard library. It does not certify model load, outputs or quality.

Depth adaptation uses `configs/depth_teacher.yaml` and
`scripts/export_metric3d_depth.py`. It does not enable the legacy Stage2 Depth
task or modify Agent/Map training. See [Depth Teacher Adaptation](depth_teacher_adaptation.md)
for the geometry gate, canonical labels, provenance and server-only validation.
