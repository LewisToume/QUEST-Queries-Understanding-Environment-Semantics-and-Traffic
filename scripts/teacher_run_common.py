from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.dataset import collate_fn
from quest.openscene_dataset import OpenSceneFirstTestDataset
from quest.teachers import TeacherUnavailableError, build_teacher, tensor_shapes
from quest.utils import load_yaml_config


def inspect_raw_camera_inputs(sample_info: dict[str, Any], camera_order: tuple[str, ...]) -> list[dict[str, Any]]:
    sample_dir = PROJECT_ROOT / "data" / "openscene_first_test_100" / sample_info["sample_id"]
    raw_inputs: list[dict[str, Any]] = []
    for camera_name in camera_order:
        image_path = sample_dir / "cams" / f"{camera_name}.jpg"
        if not image_path.exists() and camera_name == "CAM_F0":
            image_path = sample_dir / "cam_f0.jpg"
        if image_path.exists():
            with Image.open(image_path) as image:
                shape = (image.height, image.width, len(image.getbands()))
        else:
            shape = None
        raw_inputs.append(
            {
                "camera": camera_name,
                "path": str(image_path),
                "shape": shape,
            }
        )
    return raw_inputs


def load_one_openscene_batch() -> tuple[dict[str, Any], dict[str, Any]]:
    model_config = load_yaml_config(PROJECT_ROOT / "configs" / "model.yaml")["model"]
    stage1_config = load_yaml_config(PROJECT_ROOT / "configs" / "stage1.yaml")
    dataset_kwargs = dict(stage1_config.get("dataset", {}))
    data_root = dataset_kwargs.pop("root", "data/openscene_first_test_100")
    dataset_kwargs.pop("C_agent", None)
    dataset_kwargs.pop("D_box", None)
    dataset_kwargs.setdefault("camera_names", model_config["camera_names"])
    dataset_kwargs.setdefault("C_map", model_config["C_map"])
    dataset_kwargs.setdefault("P", model_config["P"])
    dataset_kwargs.setdefault("C_occ", model_config["C_occ"])
    dataset_kwargs.setdefault("C_flow", model_config["C_flow"])
    dataset_kwargs.setdefault("occ_size", (model_config["X"], model_config["Y"], model_config["Z"]))
    dataset = OpenSceneFirstTestDataset(root=data_root, **dataset_kwargs)
    dataset.manifest = dataset.manifest[:1]
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)))
    return batch, dataset.manifest[0]


def run_teacher_task(task: str) -> int:
    stage2_config = load_yaml_config(PROJECT_ROOT / "configs" / "stage2_distill.yaml")
    teacher_config = stage2_config["teachers"][task]
    teacher = build_teacher(teacher_config, task=task)
    batch, sample_info = load_one_openscene_batch()
    prepared = teacher.prepare_teacher_batch(batch)
    camera_order = tuple(prepared["camera_order"])
    raw_inputs = inspect_raw_camera_inputs(sample_info, camera_order)

    print("=" * 96)
    print(f"Teacher task       : {task}")
    print(f"Teacher name       : {teacher.spec.name}")
    print(f"Sample token       : {sample_info.get('token')}")
    print(f"Repo               : {teacher.check_result.repo_path}")
    print(f"Config             : {teacher.check_result.config_path}")
    print(f"Checkpoint         : {teacher.check_result.checkpoint_path}")
    print(f"Status             : {teacher.check_result.status}")
    print(f"Camera count       : {teacher.check_result.camera_count}")
    print(f"Camera mapping     : {json.dumps(teacher.check_result.camera_mapping, ensure_ascii=False)}")
    print(f"Raw camera inputs  : {json.dumps(raw_inputs, ensure_ascii=False)}")
    print(f"Student tensor     : {tuple(prepared['images'].shape)} (not valid teacher-native preprocessing)")
    print(f"Prepared intrinsics: {tuple(prepared['intrinsics'].shape)}")
    print(f"Prepared extrinsics: {tuple(prepared['extrinsics'].shape)}")
    print(f"Checkpoint keys    : {teacher.check_result.checkpoint_keys}")
    print(f"Missing deps       : {teacher.check_result.missing_dependencies}")
    print("=" * 96)

    try:
        with torch.no_grad():
            raw_output = teacher(batch)
        shapes = tensor_shapes(raw_output)
        print("Raw output keys/shapes:")
        print(json.dumps({key: list(value) for key, value in shapes.items()}, indent=2))
        return 0
    except TeacherUnavailableError as exc:
        print(f"BLOCKED: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print(traceback.format_exc(limit=8))
        return 1
