from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.dataset import collate_fn
from quest.openscene_dataset import OpenSceneMetadataDataset
from quest.teachers import TeacherUnavailableError, build_enabled_teachers, tensor_shapes
from quest.utils import load_yaml_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect available Agent/Map teacher outputs")
    parser.add_argument("--num-samples", type=int, default=1)
    args = parser.parse_args()
    stage1 = load_yaml_config(PROJECT_ROOT / "configs" / "stage1.yaml")
    stage2 = load_yaml_config(PROJECT_ROOT / "configs" / "stage2_distill.yaml")
    dataset_config = dict(stage1["dataset"])
    for key in ("metadata_path", "camera_root"):
        path = Path(dataset_config[key])
        if not path.is_absolute():
            dataset_config[key] = str(PROJECT_ROOT / path)
    dataset = OpenSceneMetadataDataset(max_samples=args.num_samples, **dataset_config)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_fn)
    teachers = build_enabled_teachers(stage2["teachers"])
    blocked = False
    for batch in loader:
        token = batch["sample_token"][0]
        print(f"token={token} agent_gt={(batch['agent_gt']['labels'] >= 0).sum().item()}")
        for task, teacher in teachers.items():
            try:
                with torch.no_grad():
                    output = teacher(batch)
                print(f"  {task}: {tensor_shapes(output)}")
            except TeacherUnavailableError as error:
                blocked = True
                print(f"  {task}: BLOCKED ({error})")
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
