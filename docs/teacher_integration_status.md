# Teacher Integration Status

| Teacher | Task | Camera count | Stage2 execution |
| --- | --- | ---: | --- |
| Navformer | Agent | 8 | separate offline export |
| MapTRv2 | Vector Map | configured externally | separate offline export |
| external_offline | Semantic Segmentation | unspecified | placeholder only |
| external_offline | Depth | unspecified | placeholder only |

QUEST Stage2 reads exported labels by OpenScene sample token. Online teacher
construction is disabled in the student training process.
