from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.dataset import collate_fn
from quest.openscene_dataset import OpenSceneFirstTestDataset
from quest.teachers import TeacherUnavailableError, build_enabled_teachers
from quest.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate teacher predictions against OpenScene GT when outputs are available.")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--data-root", default=None)
    return parser.parse_args()


def count_agent_gt(batch: dict[str, Any]) -> int:
    return int((batch["agent_gt"]["labels"] >= 0).sum().item())


def count_occ_gt(batch: dict[str, Any]) -> int:
    return int((batch["occ_gt"] > 0).sum().item())


def main() -> None:
    args = parse_args()
    model_config = load_yaml_config(PROJECT_ROOT / "configs" / "model.yaml")["model"]
    stage1_config = load_yaml_config(PROJECT_ROOT / "configs" / "stage1.yaml")
    stage2_config = load_yaml_config(PROJECT_ROOT / "configs" / "stage2_distill.yaml")
    data_root = args.data_root or stage1_config.get("dataset", {}).get("root", "data/openscene_first_test_100")

    dataset_kwargs = dict(stage1_config.get("dataset", {}))
    dataset_kwargs["root"] = data_root
    dataset_kwargs.setdefault("camera_names", model_config["camera_names"])
    dataset_kwargs.setdefault("C_map", model_config["C_map"])
    dataset_kwargs.setdefault("P", model_config["P"])
    dataset_kwargs.setdefault("C_occ", model_config["C_occ"])
    dataset_kwargs.setdefault("C_flow", model_config["C_flow"])
    dataset_kwargs.setdefault("occ_size", (model_config["X"], model_config["Y"], model_config["Z"]))
    dataset_kwargs.pop("C_agent", None)
    dataset_kwargs.pop("D_box", None)
    root = dataset_kwargs.pop("root")
    dataset = OpenSceneFirstTestDataset(root=root, **dataset_kwargs)
    if args.num_samples > 0 and args.num_samples < len(dataset):
        dataset.manifest = dataset.manifest[: args.num_samples]

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    teachers = build_enabled_teachers(stage2_config.get("teachers", {}))

    results: list[dict[str, Any]] = []
    print("Evaluate teachers on OpenScene")
    print("=" * 96)
    for sample_index, batch in enumerate(dataloader):
        sample_info = dataset.manifest[sample_index]
        sample_token = sample_info.get("token", sample_info.get("sample_id", f"sample_{sample_index:06d}"))
        gt_summary = {
            "agent_gt_count": count_agent_gt(batch),
            "occ_gt_occupied_voxels": count_occ_gt(batch),
            "map_gt_available": bool(batch["map_gt"]["valid"].bool().any().item()),
            "flow_gt_available": bool(batch["flow_valid"].bool().any().item()),
        }
        print(f"sample={sample_index} token={sample_token} gt={gt_summary}")
        for task, teacher in teachers.items():
            try:
                with torch.no_grad():
                    output = teacher(batch)
                del output
                item = {
                    "sample_token": sample_token,
                    "task": task,
                    "teacher": teacher.spec.name,
                    "status": "NEEDS_METRIC_IMPLEMENTATION",
                    "gt_summary": gt_summary,
                }
                print(f"  {teacher.spec.name:<12} output available; metric implementation pending")
            except TeacherUnavailableError as exc:
                item = {
                    "sample_token": sample_token,
                    "task": task,
                    "teacher": teacher.spec.name,
                    "status": "BLOCKED",
                    "reason": str(exc),
                    "gt_summary": gt_summary,
                }
                print(f"  {teacher.spec.name:<12} BLOCKED {exc}")
            results.append(item)

    output_path = PROJECT_ROOT / "docs" / "teacher_eval_openscene.json"
    output_path.write_text(json.dumps({"items": results}, indent=2), encoding="utf-8")
    print("=" * 96)
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
