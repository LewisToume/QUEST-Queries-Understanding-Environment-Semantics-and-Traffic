# StreamPETR on OpenScene

This runner uses the original six OpenScene JPEGs in the order
`CAM_F0 CAM_R0 CAM_R2 CAM_B0 CAM_L2 CAM_L0`. It passes the local
StreamPETR checkpoint and the repository's original test pipeline to the
native `Petr3D` model. It does not use QUEST's 224x224 Student tensors or
perform QUEST class mapping.

Target server runtime (provided, not verified locally):

```text
Ubuntu/Linux; Python 3.8
torch 1.9.0+cu111; torchvision 0.10.0+cu111
mmcv-full 1.6.0; mmdet 2.28.2; mmseg 0.30.0
mmdet3d 1.0.0rc6; nuscenes-devkit 1.1.10
4 x RTX 3090 24GB
```

From the QUEST repository root, in that Teacher environment:

```bash
python scripts/run_streampetr_openscene.py
```

The runner checks both known metadata and checkpoint archive layouts. If the
server differs, pass `--metadata`, `--camera-root`, `--checkpoint`, and
`--stream-petr-root` explicitly. If the StreamPETR snapshot has no sibling
`mmdetection3d` source tree, provide the source checkout with
`--mmdet3d-config-root /path/to/mmdetection3d`; the runner also searches
the installed editable mmdet3d source for its two base configs.

`configs/stage1.yaml` uses the server's
`data/openscene/meta_datas/meta_data_mini.pkl` path. The local desktop
archive is nested one level deeper; the standalone StreamPETR runner detects
either location without modifying Student code.

OpenScene mini camera metadata may not provide separate image timestamps.
In that case the runner uses the frame timestamp for all six views. The
first sample is passed as `prev_exists=False`; the native model also resets
memory when its `scene_token` changes.

Local verification covers Python 3.8 syntax, real OpenScene JPEG path
resolution, and camera/ego geometry. The local QUEST environment has no
OpenMMLab runtime, so native pipeline, checkpoint/model build, and CUDA
inference must be confirmed on the server.
