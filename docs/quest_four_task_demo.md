# Combined QUEST Four-Task Preview

The trained `quest_seg_on_depth_v3_v1_best.pt` contains one DINOv2-S backbone
and four task branches, not four online teacher models. Total stored parameters
are 100,969,608 in the current server run. Training was staged, not joint
four-task optimization. The latest stage trained only the 6,071,891 Seg branch
parameters; all base parameters and fixed-probe predictions stayed unchanged.

Current routing:

```text
current 8 RGB views -> shared frozen DINOv2-S features
  -> geometry lift / BEV -> original Agent queries / box head
  -> spatial Map path / queries -> vector classes and 20 XY points
  -> multilevel feature adapters / native SegFormer decoder -> Cityscapes19
  -> calibrated spatial geometry / causal temporal fusion / DPT -> camera-Z
previous 8 RGB views -> same DINOv2-S weights -> Depth temporal fusion only
```

Map also has its trained task-specific spatial processing. This diagram does
not imply all branches consume the same BEV tensor. In particular Seg/Depth use
camera-plane features, not only BEV. Actual strict-past frames are selected from
same-split/same-scene timestamps, with a cold-start/gap mask and ego transforms.
Agent intentionally uses the original current-only route: the temporal Agent
adapter regressed pedestrian agreement and is not used in this selected model.

## Server Command

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/visualize_quest_four_tasks.py
```

Default: three validation scenes, prefer frames with real history. One strict
combined checkpoint load; one combined forward per target. No online teachers,
training, checkpoint writes, or source-label changes. The existing validation
Seg labels are audited for split/provenance consistency, not shown as GT.
Output is a new timestamped `data/diagnostics/quest_four_tasks_*` directory.
Each figure has Agent and vector Map in LiDAR XY above eight RGB/Seg/Depth rows.
Depth uses a fixed log color range of 0.1-80m with invalid values black; predicted
depth is not masked by teacher validity or teacher sky masks. Saturation above
80m is visualization clipping, not a claim of correct far-depth prediction.
`report.json` records checkpoint hash, indices, true history, routes, thresholds,
all four output shapes and kept Agent/Map counts. Counts are not precision/recall.

Override targets with repeatable raw validation indices, e.g.
`--sample-index 15166 --sample-index 20000 --sample-index 34863` (only if present
in this manifest). Change the score filters explicitly with `--agent-threshold`
and `--map-threshold`; default is 0.25 for both. These filters do not establish
official evaluation thresholds or improve model quality.

Server synthetic checks before visualization:

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -p 'test_four_task_visualization.py' -q
```

This preview demonstrates integrated inference, not four-task qualification.
Seg and Depth metrics use offline pseudo-label references, not real GT. Map's
previous validation was weak and still needs improvement. Agent is limited to
vehicle/pedestrian supported pseudo labels, without validated tracking IDs.
Depth camera geometry remains experimental. Do not claim all four tasks meet
teacher performance or that BEV is a fifth independently validated task.
