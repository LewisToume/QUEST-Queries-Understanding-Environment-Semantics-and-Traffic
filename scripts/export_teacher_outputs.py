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
from quest.teachers import (
    TeacherUnavailableError,
    build_enabled_teachers,
    maptr_output_to_quest,
    stream_petr_output_to_quest,
)
from quest.utils import load_yaml_config


CONVERTERS = {
    "agent": stream_petr_output_to_quest,
    "map": maptr_output_to_quest,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export token-aligned teacher labels")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--output-dir", default="data/soft_labels")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        token = batch["sample_token"][0]
        payload: dict[str, object] = {"token": token}
        for task, teacher in teachers.items():
            if task not in CONVERTERS:
                continue
            try:
                raw = teacher(batch)
                payload[task] = CONVERTERS[task](raw)
            except TeacherUnavailableError as error:
                print(f"{token} {task}: not exported ({error})")
        if len(payload) > 1:
            path = output_dir / f"{token}.pt"
            torch.save(payload, path)
            print(f"saved {path}")
        else:
            print(f"{token}: no verified labels; no file written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
