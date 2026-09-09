from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.utils import load_yaml_config
from quest.openscene_dataset import OpenSceneMetadataDataset


def parse_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    return None


def main() -> None:
    stage1_config = load_yaml_config(PROJECT_ROOT / "configs" / "stage1.yaml")
    stage2_config = load_yaml_config(PROJECT_ROOT / "configs" / "stage2_distill.yaml")
    map_config = PROJECT_ROOT / stage2_config["teachers"]["map"]["config_path"]
    dataset_config = dict(stage1_config["dataset"])
    dataset_config.pop("C_agent", None)
    for path_key in ("metadata_path", "camera_root", "occupancy_root"):
        path = Path(dataset_config[path_key])
        if not path.is_absolute():
            dataset_config[path_key] = str(PROJECT_ROOT / path)
    metadata = OpenSceneMetadataDataset(max_samples=1, **dataset_config).infos[0]

    maptrv2_classes = parse_assignment(map_config, "map_classes")
    report = {
        "status": "NOT_VERIFIED",
        "map_kd": "disabled",
        "maptrv2_config": str(map_config),
        "maptrv2_classes": maptrv2_classes,
        "openscene_metadata": {
            "map_location": metadata.get("map_location"),
            "roadblock_ids": metadata.get("roadblock_ids"),
            "ego_pose_shape": list(getattr(metadata.get("ego2global"), "shape", [])),
        },
        "nuplan_map_api_available": False,
        "minimal_common_taxonomy": {},
        "reason": (
            "Current extracted OpenScene sample exposes map_location, roadblock_ids, and ego_pose, "
            "but no local nuPlan Map API/vector-map extraction is available in this environment. "
            "MapTRv2 nuScenes classes are divider, ped_crossing, boundary; OpenScene/nuPlan layer "
            "equivalence is not verified, so Map KD remains disabled."
        ),
    }

    output_path = PROJECT_ROOT / "docs" / "map_taxonomy_mapping.md"
    lines = [
        "# Map Taxonomy Mapping",
        "",
        "Status: NOT_VERIFIED",
        "",
        "```json",
        json.dumps(report, indent=2),
        "```",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
