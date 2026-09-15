from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.dataset import collate_fn
from quest.openscene_dataset import OpenSceneMetadataDataset
from quest.teachers import TeacherUnavailableError, build_teacher, tensor_shapes
from quest.utils import load_yaml_config


def load_one_openscene_batch() -> tuple[dict[str, Any], dict[str, Any]]:
    stage1 = load_yaml_config(PROJECT_ROOT / "configs" / "stage1.yaml")
    dataset_config = dict(stage1["dataset"])
    for key in ("metadata_path", "camera_root"):
        path = Path(dataset_config[key])
        if not path.is_absolute():
            dataset_config[key] = str(PROJECT_ROOT / path)
    dataset = OpenSceneMetadataDataset(max_samples=1, **dataset_config)
    batch = next(
        iter(DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_fn))
    )
    return batch, dataset.infos[0]


def raw_camera_inputs(
    info: dict[str, Any], dataset_root: Path, camera_order: tuple[str, ...]
) -> list[dict[str, str]]:
    from quest.openscene_dataset import resolve_openscene_camera_path

    return [
        {
            "camera": camera,
            "path": str(resolve_openscene_camera_path(info["cams"][camera]["data_path"], dataset_root)),
        }
        for camera in camera_order
    ]


def run_teacher_task(task: str) -> int:
    stage1 = load_yaml_config(PROJECT_ROOT / "configs" / "stage1.yaml")
    stage2 = load_yaml_config(PROJECT_ROOT / "configs" / "stage2_distill.yaml")
    teacher = build_teacher(stage2["teachers"][task], task=task)
    batch, info = load_one_openscene_batch()
    prepared = teacher.prepare_teacher_batch(batch)
    camera_root = Path(stage1["dataset"]["camera_root"])
    if not camera_root.is_absolute():
        camera_root = PROJECT_ROOT / camera_root
    print(f"Teacher task: {task}")
    print(f"Teacher name: {teacher.spec.name}")
    print(f"Sample token: {batch['sample_token'][0]}")
    print(f"Status: {teacher.check_result.status}")
    print(f"Camera mapping: {json.dumps(teacher.check_result.camera_mapping)}")
    print(
        "Camera inputs:",
        json.dumps(raw_camera_inputs(info, camera_root, tuple(prepared["camera_order"]))),
    )
    print(f"Prepared images: {tuple(prepared['images'].shape)}")
    try:
        with torch.no_grad():
            output = teacher(batch)
        print(json.dumps({key: list(shape) for key, shape in tensor_shapes(output).items()}))
        return 0
    except TeacherUnavailableError as error:
        print(f"BLOCKED: {error}")
        return 2
    except Exception as error:
        print(f"FAILED: {type(error).__name__}: {error}")
        print(traceback.format_exc(limit=8))
        return 1
