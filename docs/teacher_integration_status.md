# Teacher Integration Status

Design contract: [Four-Teacher Student](four_teacher_student_design.md).
Domain requirement and selection: [nuPlan/OpenScene audit](nuplan_openscene_teacher_selection.md).
BEV is the shared spatial representation, not a fifth teacher task.
FORMAL teachers must have nuPlan/OpenScene task training/validation evidence.
The old three-teacher experimental route does NOT meet that requirement.
Four ready domain-qualified neural teachers are NOT available in this checkout.

| Source | Task | Evidence | Formal status / remaining gate |
| --- | --- | --- | --- |
| Navformer UniAD Agent branch | Agent | User-reported nuPlan checkpoint inference/export and Stage 2 training | RETAIN; Agent regression after shared-front-end changes |
| nuPlan HD map | Vector Map | User-reported real vector GT export/audit/training | RETAIN AS GT, not a neural teacher; preserve schema/split/capacity checks |
| Navformer raster map branch | Optional Map KD | Same-domain checkpoint, unreliable lane masks observed | DISABLED FOR QUALITY; no silent reactivation |
| DVGT-1 | Depth / geometry candidate | Author OpenScene training/evaluation configs and public dvgt1.pt listing verified online | DOMAIN CANDIDATE ONLY; no download/load/inference/conversion tested |
| ViGT | Backup geometry candidate | Author paper/README includes nuPlan and publishes weight link | No checkpoint-content/load/rendering check; not camera Seg |
| No selected same-domain camera-Seg checkpoint | Semantic Segmentation | Search did not verify a reusable qualifying release | BLOCKED; BEV segmentation does not satisfy camera-Seg |
| SegFormer-B5 Cityscapes | Out-of-domain Seg control | User-reported export, Student training and 120-frame pseudo-agreement | NOT FORMAL: Cityscapes weights, no independent nuPlan semantic GT validation |
| Metric3Dv2 ViT-L | Unqualified-domain Depth control | User-reported export/training and 120-frame pseudo-agreement | NOT FORMAL: no verified nuPlan/OpenScene checkpoint task provenance; geometry experimental |

QUEST Stage2 reads exported labels by OpenScene sample token. Online teacher
construction is disabled in the student training process.

`configs/stage2_distill.yaml` remains a legacy Agent-only configuration, not
proof of four-teacher readiness: Seg/Depth there are disabled placeholders and
`maptr_output_to_quest()` still rejects unverified taxonomy. MapTRv2 is ONLY a
structure/loss reference here; its nuScenes checkpoints are not prerequisites.
The separate `configs/seg_depth.yaml` implements the experimental dense route,
with frozen Agent/Map and strict label preflight. The user reported real server
training and validation. These measure imitation of OUT-OF-DOMAIN pseudo labels,
not same-domain teacher qualification or real-GT accuracy. Existing experimental
labels/checkpoints are preserved; configurations are not switched by this audit.
Stage 3 trains against nuPlan vector GT, NOT
MapTRv2 predictions. Navformer raster-map KD stays disabled because of the
observed unreliable lane masks.

`scripts/inspect_multitask_teachers.py` inventories explicit legacy asset paths
using only the standard library. It does not certify domain provenance, model
load, outputs or quality, and does not inventory DVGT-1 yet. FILES_PRESENT must
not be interpreted as formal teacher approval.

Asset preparation and actual Seg exporter commands are documented in
[Dense Teacher Export](dense_teacher_export.md). No MapTRv2 download is required.

Depth adaptation uses `configs/depth_teacher.yaml` and
`scripts/export_metric3d_depth.py`. It does not enable the legacy Stage2 Depth
task or modify Agent/Map training. See [Depth Teacher Adaptation](depth_teacher_adaptation.md)
for the geometry gate, canonical labels, provenance and server-only validation.
