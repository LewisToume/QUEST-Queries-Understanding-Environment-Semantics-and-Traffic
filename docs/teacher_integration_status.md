# Teacher Integration Status

| Teacher | Task | Camera count | Stage2 execution |
| --- | --- | ---: | --- |
| Navformer | Agent | 8 | separate offline export |
| MapTRv2 | Vector Map | configured externally | separate offline export |
| external_offline | Semantic Segmentation | unspecified | placeholder only |
| Metric3Dv2 ViT-L | Depth | same frozen teacher on 8 views | adaptation/export code; server inference and image geometry unverified |

QUEST Stage2 reads exported labels by OpenScene sample token. Online teacher
construction is disabled in the student training process.

Depth adaptation uses `configs/depth_teacher.yaml` and
`scripts/export_metric3d_depth.py`. It does not enable the legacy Stage2 Depth
task or modify Agent/Map training. See [Depth Teacher Adaptation](depth_teacher_adaptation.md)
for the geometry gate, canonical labels, provenance and server-only validation.
