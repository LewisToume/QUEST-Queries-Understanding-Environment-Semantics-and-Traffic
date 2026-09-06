from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.utils import load_yaml_config


def parse_simple_assignments(path: Path, names: set[str]) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except Exception:
                    values[target.id] = "<non-literal>"
    return values


def main() -> None:
    model_config = load_yaml_config(PROJECT_ROOT / "configs" / "model.yaml")["model"]
    stage2_config = load_yaml_config(PROJECT_ROOT / "configs" / "stage2_distill.yaml")
    occ_config = PROJECT_ROOT / stage2_config["teachers"]["occ"]["config_path"]
    parsed = parse_simple_assignments(
        occ_config,
        {"point_cloud_range", "occupancy_size", "bev_h_", "bev_w_", "num_cams", "class_names"},
    )

    point_cloud_range = parsed.get("point_cloud_range")
    occupancy_size = parsed.get("occupancy_size")
    expected = {}
    if isinstance(point_cloud_range, list) and isinstance(occupancy_size, list):
        expected = {
            "X": int(round((point_cloud_range[3] - point_cloud_range[0]) / occupancy_size[0])),
            "Y": int(round((point_cloud_range[4] - point_cloud_range[1]) / occupancy_size[1])),
            "Z": int(round((point_cloud_range[5] - point_cloud_range[2]) / occupancy_size[2])),
        }

    report = {
        "occ_config": str(occ_config),
        "axis_order": "QUEST uses [B, C_occ, X, Y, Z]; OpenScene sparse labels use flat voxel index plus semantic id.",
        "x_definition": "X spans point_cloud_range[0] -> point_cloud_range[3]",
        "y_definition": "Y spans point_cloud_range[1] -> point_cloud_range[4]",
        "z_definition": "Z spans point_cloud_range[2] -> point_cloud_range[5]",
        "voxel_origin": point_cloud_range[:3] if isinstance(point_cloud_range, list) else None,
        "point_cloud_range": point_cloud_range,
        "voxel_size": occupancy_size,
        "expected_grid_from_config": expected,
        "semantic_ids": "OpenScene/nuPlan config occupancy_classes=11 in pts_bbox_head; exact ids must be validated against dataset docs before KD.",
        "empty_free_class": "Current QUEST dense loader initializes missing sparse voxels to class 0; verify whether class 0 is free/empty before OCC KD.",
        "quest": {
            "C_occ": model_config["C_occ"],
            "X": model_config["X"],
            "Y": model_config["Y"],
            "Z": model_config["Z"],
        },
        "alignment_status": "STRUCTURE_MATCHES" if expected == {"X": model_config["X"], "Y": model_config["Y"], "Z": model_config["Z"]} else "BLOCKED_ALIGNMENT",
        "note": "No reshape-based conversion is allowed; KD must align by physical voxel centers and semantic mapping.",
    }

    output_path = PROJECT_ROOT / "docs" / "occ_alignment.md"
    lines = [
        "# OCC Alignment",
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
