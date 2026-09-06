# Teacher Legacy Environment

Target environment:

```text
conda env name: quest_teacher_legacy
Python 3.8
PyTorch 1.10.1
torchvision 0.11.2
torchaudio 0.10.1
CUDA 11.1
mmcv-full 1.4.0
mmdet 2.14.0
mmsegmentation 0.14.1
mmdet3d 0.17.1 from source
```

The QUEST `.venv` is not used for legacy teacher installation. It remains the
Student/runtime environment.

## Windows Status

The local machine has conda at:

```text
D:\Users\31722\anaconda3\Scripts\conda.exe
```

Existing environments before this task:

```text
base
flashocc_env
pytorch-gpu
streampetr_env
vggt
```

Attempted environment creation on 2026-09-07 with:

```powershell
conda env create -f envs\teacher_legacy.yml
```

Conda created the environment directory, but pip dependency installation failed
while downloading the requested Windows CUDA wheel:

```text
ERROR: Wheel 'torch' located at ...\torch-1.10.1+cu111-cp38-cp38-win_amd64.whl is invalid.
CondaEnvException: Pip failed
```

Actual partial environment state:

```text
conda env name: quest_teacher_legacy
python: 3.8.20
torch: MISSING
torchvision: MISSING
torchaudio: MISSING
mmcv: MISSING
mmdet: MISSING
mmseg: MISSING
mmdet3d: MISSING
```

The required legacy stack is fragile on native Windows because `mmcv-full`
1.4.0 and `mmdet3d` 0.17.1 require old compiled CUDA/C++ extensions. If native
Windows installation fails, use this repository's `envs/teacher_legacy.yml` on
WSL2/Linux or a CUDA 11.1 server.

## Required mmdet3d Source Install

```powershell
git clone https://github.com/open-mmlab/mmdetection3d.git third_party\mmdetection3d-v0.17.1
git -C third_party\mmdetection3d-v0.17.1 checkout v0.17.1
conda run -n quest_teacher_legacy python third_party\mmdetection3d-v0.17.1\setup.py install
```

## Version Check

Run after environment creation:

```powershell
conda run -n quest_teacher_legacy python -c "import torch, mmcv, mmdet, mmdet3d; print(torch.__version__); print(mmcv.__version__); print(mmdet.__version__); print(mmdet3d.__version__)"
```

## Current Blocker Policy

Teacher scripts do not fall back to dummy models. A teacher is marked BLOCKED
when its legacy environment, checkpoint, native model build, temporal sequence,
or taxonomy/alignment requirements are not satisfied.
