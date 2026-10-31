# Teacher Integration Status

Design contract: [Four-Teacher Student](four_teacher_student_design.md).
Current encoder/decoder audit and proposed integration:
[Teacher/QUEST comparison, 2026-10-31](teacher_encoder_decoder_comparison.md).
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
| DVGT-1 | Offline Depth / geometry teacher; experimental adaptation | User-reported strict native GPU inference, label export and five-epoch 5,000-frame Student training; full 120-frame evaluation | RUNTIME REPORTED PASS; Student/teacher AbsRel=0.1238268, RMSE=6.0619m, delta1=0.8278686; real LiDAR accuracy / geometry NOT verified |
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
The DVGT Depth-only experiment is separate from the historical Metric3D/Seg
control. It freezes Agent/Map and does NOT automatically merge the trained Seg
branch. Current data signatures permit the reviewed safe-export resume update
without weakening geometry/config/split checks.
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

The new same-domain route uses `configs/depth_teacher_dvgt.yaml`,
`scripts/export_dvgt_depth.py` and schema 2, not the old Metric3D schema 1.
[DVGT server guide](dvgt_depth_teacher.md) provides asset preparation, CPU
preflight and one-target GPU diagnostics. Default inference writes NO training
labels. A separate Depth-only config `configs/seg_depth_dvgt.yaml` avoids
requiring the unqualified-domain Seg labels. As of 2026-10-31 the user has
reported real inference, export, training and validation for this route; these
are not local Codex executions or independent geometry/GT certification.
