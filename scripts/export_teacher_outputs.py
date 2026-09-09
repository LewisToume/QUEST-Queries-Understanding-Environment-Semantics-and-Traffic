from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.dataset import collate_fn
from quest.openscene_dataset import OpenSceneMetadataDataset
from quest.teachers import COORDINATE_CONVENTION, TeacherUnavailableError, build_enabled_teachers, tensor_shapes
from quest.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export real teacher outputs for OpenScene samples.")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_config = load_yaml_config(PROJECT_ROOT / "configs" / "model.yaml")["model"]
    stage1_config = load_yaml_config(PROJECT_ROOT / "configs" / "stage1.yaml")
    stage2_config = load_yaml_config(PROJECT_ROOT / "configs" / "stage2_distill.yaml")

    output_dir = Path(args.output_dir or stage2_config.get("paths", {}).get("soft_labels_dir", "data/soft_labels"))
    dataset_kwargs = dict(stage1_config.get("dataset", {}))
    if args.data_root:
        dataset_kwargs["metadata_path"] = args.data_root
    dataset_kwargs.setdefault("camera_names", model_config["camera_names"])
    dataset_kwargs.setdefault("C_map", model_config["C_map"])
    dataset_kwargs.setdefault("P", model_config["P"])
    dataset_kwargs.setdefault("C_occ", model_config["C_occ"])
    dataset_kwargs.setdefault("C_flow", model_config["C_flow"])
    dataset_kwargs.setdefault("occ_size", (model_config["X"], model_config["Y"], model_config["Z"]))
    dataset_kwargs.pop("C_agent", None)
    dataset_kwargs.pop("D_box", None)
    for path_key in ("metadata_path", "camera_root", "occupancy_root"):
        path = Path(dataset_kwargs[path_key])
        if not path.is_absolute():
            dataset_kwargs[path_key] = str(PROJECT_ROOT / path)
    dataset = OpenSceneMetadataDataset(max_samples=args.num_samples, **dataset_kwargs)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    teachers = build_enabled_teachers(stage2_config.get("teachers", {}))
    if not teachers:
        raise RuntimeError("no enabled teachers in configs/stage2_distill.yaml")

    run_report: list[dict[str, Any]] = []
    print("Export teacher outputs")
    print("=" * 96)
    for sample_index, batch in enumerate(dataloader):
        sample_info = dataset.infos[sample_index]
        sample_token = sample_info.get("token", sample_info.get("sample_id", f"sample_{sample_index:06d}"))
        print(f"sample={sample_index} token={sample_token}")
        for task, teacher in teachers.items():
            task_dir = output_dir / task
            status_payload = {
                "sample_token": sample_token,
                "teacher_name": teacher.spec.name,
                "checkpoint": str(teacher.check_result.checkpoint_path) if teacher.check_result.checkpoint_path else None,
                "class_mapping_version": teacher.spec.extra.get("class_mapping_version", "not_verified"),
                "coordinate_convention": COORDINATE_CONVENTION,
                "status": teacher.check_result.status,
                "check": teacher.check_result.as_dict(),
                "exported_at": datetime.now().isoformat(timespec="seconds"),
            }
            try:
                with torch.no_grad():
                    raw_output = teacher(batch)
                output_path = task_dir / f"{sample_token}.pt"
                torch.save(
                    {
                        **status_payload,
                        "status": "EXPORTED",
                        "raw_output_shapes": tensor_shapes(raw_output),
                        "output_tensors": raw_output,
                    },
                    output_path,
                )
                print(f"  {teacher.spec.name:<12} EXPORTED {output_path}")
                run_report.append({**status_payload, "status": "EXPORTED", "path": str(output_path)})
            except TeacherUnavailableError as exc:
                blocked_path = task_dir / f"{sample_token}.blocked.json"
                payload = {
                    **status_payload,
                    "status": "BLOCKED",
                    "reason": str(exc),
                    "output_tensors": None,
                }
                save_json(blocked_path, payload)
                print(f"  {teacher.spec.name:<12} BLOCKED {exc}")
                run_report.append({**payload, "path": str(blocked_path)})

    report_path = output_dir / "teacher_export_report.json"
    save_json(report_path, {"items": to_jsonable(run_report)},)
    print("=" * 96)
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
