from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.dataset import collate_fn
from quest.model import QUESTModel, load_quest_v3_checkpoint
from quest.openscene_dataset import OpenSceneMetadataDataset
from quest.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose QUEST V3 input-to-BEV and BEV-to-Agent conditioning"
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def mean_absolute_difference(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.float() - second.float()).abs().mean())


def main() -> int:
    args = parse_args()
    if args.start < 0:
        raise ValueError("start must be non-negative")
    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")["model"]
    stage1 = load_yaml_config(PROJECT_ROOT / "configs/stage1.yaml")
    model_config["local_backbone_dir"] = str(resolve(model_config["local_backbone_dir"]))
    dataset_config = dict(stage1["dataset"])
    dataset_config["metadata_path"] = str(resolve(dataset_config["metadata_path"]))
    dataset_config["camera_root"] = str(resolve(dataset_config["camera_root"]))
    dataset = OpenSceneMetadataDataset(max_samples=args.start + 2, **dataset_config)
    samples = [dataset[args.start], dataset[args.start + 1]]
    loader = DataLoader(samples, batch_size=1, collate_fn=collate_fn, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QUESTModel(**model_config).to(device).eval()
    if args.checkpoint is not None:
        checkpoint = torch.load(resolve(args.checkpoint), map_location=device)
        load_quest_v3_checkpoint(model, checkpoint)

    outputs = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["images"].to(device),
                batch["intrinsics"].to(device),
                batch["extrinsics"].to(device),
                batch["ego_state"].to(device),
            )
            outputs.append(output)
            print(f"token: {batch['sample_token'][0]}")
            print(f"camera_visible_ratio: {output['camera_visible_ratio'][0].cpu().tolist()}")
            print(f"bev_visible_ratio: {float(output['bev_visible_ratio'][0]):.6f}")
            print(f"lifted_bev_std: {float(output['lifted_bev_std'][0]):.6f}")
            print(f"encoded_bev_std: {float(output['encoded_bev_std'][0]):.6f}")
            print(f"proposal_spatial_std: {float(output['proposal_spatial_std'][0]):.6f}")

    first, second = outputs
    print("two_frame_mean_absolute_difference:")
    for key in (
        "bev_features",
        "proposal_objectness_logits",
        "agent_cls_logits",
        "agent_boxes",
    ):
        print(f"  {key}: {mean_absolute_difference(first[key], second[key]):.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
