# Map Taxonomy Mapping

Status: NOT_VERIFIED

```json
{
  "status": "NOT_VERIFIED",
  "map_kd": "disabled",
  "maptrv2_config": "C:\\Users\\31722\\Desktop\\Research\\QUEST\\third_party\\MapTR\\projects\\configs\\maptrv2\\maptrv2_nusc_r50_24ep.py",
  "maptrv2_classes": [
    "divider",
    "ped_crossing",
    "boundary"
  ],
  "openscene_metadata": {
    "map_location": "us-nv-las-vegas-strip",
    "roadblock_ids": [
      "65541",
      "60026",
      "47148",
      "66306",
      "47186",
      "67011",
      "65580",
      "66900",
      "65522",
      "60121",
      "48881",
      "60109",
      "47004",
      "66405",
      "65419",
      "66969",
      "65481",
      "60208",
      "48680",
      "66191"
    ],
    "ego_pose_shape": [
      4,
      4
    ]
  },
  "nuplan_map_api_available": false,
  "minimal_common_taxonomy": {},
  "reason": "Current extracted OpenScene sample exposes map_location, roadblock_ids, and ego_pose, but no local nuPlan Map API/vector-map extraction is available in this environment. MapTRv2 nuScenes classes are divider, ped_crossing, boundary; OpenScene/nuPlan layer equivalence is not verified, so Map KD remains disabled."
}
```
