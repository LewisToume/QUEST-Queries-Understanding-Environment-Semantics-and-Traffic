# Selective Temporal Integration

2026-11-01. Implementation/static inspection only. Codex did NOT run Python,
unit tests, teacher inference, training, or server commands. Runtime, memory and
quality below require real user-run results; no PASS is claimed in advance.

## Teacher Input Audit

| Task/source | Actual temporal contract | QUEST decision |
| --- | --- | --- |
| Active Navformer/UniAD Agent | Official selected config has queue_length=3 and video_test_mode=True; sequential inference carries BEV/tracking state | Add opt-in historical BEV fusion to Agent |
| Active DVGT-1 Depth | Current real export uses two frames, eight cameras; offline future context was allowed at scene starts | Add current plus real previous frame to Depth; Student is strictly causal |
| Historical Navformer raster Map | Shares the temporal track/BEV route | Still disabled; do NOT reactivate unreliable map KD |
| Current vector Map | Real nuPlan vector GT; MapTRv2 is a structure reference, not a qualified nuPlan neural teacher | Preserve current-frame Map route |
| Selected MapTRv2 centerline reference | queue_length=1, video_test_mode=False | No forced temporal Map conversion |
| Historical SegFormer-B5 Cityscapes control | Each camera image is inferred independently, no temporal memory | No temporal Seg conversion; same-domain teacher remains unresolved |
| Historical Metric3Dv2 control | Single-image inference | Not active; no temporal conversion |

Sources: [Navformer selected config](https://github.com/OpenDriveLab/Navformer/blob/main/configs/navformer/track_map_nuplan_r50_navtrain.py),
[MapTRv2 selected reference config](https://raw.githubusercontent.com/hustvl/MapTR/maptrv2/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_w_centerline.py),
[DVGT aggregation](https://github.com/wzzheng/DVGT/blob/51cf3f6d11fdff8bc7e2bbe1a88f71665ccb2236/dvgt/models/backbones/dvgt1_aggregator.py).
Local export evidence: `export_navformer_pseudo.py` maintains same-scene state
and resets at boundaries/gaps; `export_segformer_pseudo.py` loops over independent
camera images; `export_dvgt_depth.py` stacks real time and view dimensions.
Old Agent records do not store full per-target context identities: current code
does NOT retrospectively prove which historical states generated every old file.

## Implemented Contract

The opt-in `agent_depth_temporal_v3.yaml` runs ONE QUEST with current and previous
eight-view RGB, one frozen DINO parameter set and shared history selection:

- Depth: V2 within/cross-view geometry plus four gated cross-frame attention
  stages, ego-motion-compensated raw-camera ray sampling, DPT current-depth output.
- Agent: original current/past BEV encoders, full relative-pose compensation of
  current ground-plane anchors into historical BEV, local 3x3 temporal attention,
  zero-initialized residual gates, existing proposal and Agent decoders.
- Map: original CURRENT features/BEV only, never the Agent-fused BEV. Its weights
  and input route are preserved. Seg is not emitted as an untrained fake output.
- This first common window is two frames. It is NOT a reproduction of Navformer's
  three-frame training queue or persistent track-query memory. Existing canonical
  vehicle/pedestrian boxes/velocity do not contain track IDs; no ID supervision
  or full tracking capability is claimed.

History is selected by timestamp within the exact same manifest split and scene.
No future frame, cross-scene pair, validation-to-train context, or duplicated
current image is accepted as history. Missing first-frame/gap context uses shape
buffers with a false mask and zero time interval. Gaps above 1.1s are masked.
Transform is inverse(current lidar2global) @ previous lidar2global, computed in
float64 before float32 use; all pose axes are retained. Relative motion only,
not absolute global position, enters the networks. Dynamic-object motion and
true occlusion are not supplied by ego compensation: attention can reject
history, but projectability is not a correctness/visibility guarantee.

The joint profile keeps the original Agent/Map image/BEV/inference route in fp32;
the dense decoder uses its explicit bf16 profile. Current/past image features are
reused across branches. This differs from the earlier Depth-only profile's bf16
backbone execution; initial validation is logged before joint fine-tuning, so do
not attribute all differences from the old 0.10214 result solely to history.
The old pinhole BEV lift remains unchanged; raw-lens accuracy is still unverified.

## Supervision And Comparison

Warm-start from `quest_depth_geometry_v2.pt`, not random new Depth weights.
Only explicitly added temporal state keys may be missing. Base profiles, spatial
initialization, full split, loss recipe, teacher signatures and complete Depth
label hashes must match. Agent records have separate token/index/support/hash
checks, retained through resume/evaluation. No teacher re-export is needed.

Depth loss is unchanged. Joint Agent uses the existing Stage 2 confidence-weighted
proposal/class/box/velocity objective, with loss weight 0.5. The original DINO,
BEV, Map and Agent parameters stay frozen; Depth and new Agent adapters train.
FREEZING AGENT WEIGHTS DOES NOT GUARANTEE UNCHANGED AGENT PREDICTIONS: their inputs
now include history. Validate vehicle/pedestrian precision/recall and matched
center/size/yaw/velocity errors against the history-disabled original route.

Each epoch saves validation history and reports both history-enabled and
history-disabled Depth/Agent metrics on the same 120 frames and masks. This
ablation measures dependence on history, not an independently trained control.
Also provided: `depth_temporal_v3.yaml` (Depth only) and
`depth_geometry_v2_continue.yaml` (single-frame Depth continuation, same source,
five extra epochs, LR=3e-5 and target labels). Those two Depth-only configs are
the closer controlled comparison for extra training versus temporal Depth.

Dense checkpoint version 3/profile is explicit; legacy versions 1/2 remain
readable. Joint checkpoints separately declare trained_temporal_agent. Output
and baseline filenames cannot coincide. All default/old configs remain opt-in
and unchanged. A new experimental checkpoint is not automatic acceptance.

## User-Run Server Commands

Use `quest_env`; never the teacher environment. GPU3 must actually be available.

```bash
conda activate quest_env && cd /home/user/DataDisk/QUEST_WORK/QUEST && git pull --ff-only origin feature/multiview-quest && OMP_NUM_THREADS=2 python -m unittest discover -s tests -p 'test_temporal*.py' -q && CUDA_VISIBLE_DEVICES=3 python -u scripts/train_seg_depth.py --config configs/agent_depth_temporal_v3.yaml --num-samples 2 --smoke-steps 4 --experimental-unverified-depth
```

Smoke checks two selected target samples, real history, zero-gate identity,
finite gradients and parameter updates, frozen base checksum, and full output
shapes. It NEVER saves a checkpoint. Expected evidence includes nonzero temporal
Depth/Agent gradients on real-history steps and measured memory, not just a
finite total loss. CPU tests use small, explicitly synthetic unit fixtures.

Full preflight only, optional (the training command also audits all labels):

```bash
python scripts/train_seg_depth.py --config configs/agent_depth_temporal_v3.yaml --experimental-unverified-depth --preflight-only
```

After server tests/smoke succeed, 5000 targets, five EXTRA fine-tuning epochs:

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/train_seg_depth.py --config configs/agent_depth_temporal_v3.yaml --experimental-unverified-depth
```

New output: `checkpoints/quest_agent_depth_temporal_v3.pt`. Concise evaluation:

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/evaluate_seg_depth.py --checkpoint checkpoints/quest_agent_depth_temporal_v3.pt --output-dir data/diagnostics/agent_depth_temporal_v3_review --preview-frames 0 --log-interval 120
```

No independent real LiDAR accuracy claim; only teacher agreement on three held-out
scenes. Temporal gains, Agent nonregression and 3090 memory fit remain unverified.
