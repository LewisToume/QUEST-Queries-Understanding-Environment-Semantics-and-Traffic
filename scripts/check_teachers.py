from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quest.teachers import build_all_teachers
from quest.utils import load_yaml_config


REPORT_PATH = PROJECT_ROOT / "docs" / "teacher_integration_status.md"


def mark(value: bool) -> str:
    return "OK" if value else "BLOCKED"


def write_report(results: list[dict]) -> None:
    lines = [
        "# Teacher Integration Status",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Summary",
        "",
        "| Teacher | Repo | Config | Checkpoint | Load | Deps | Inference | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        lines.append(
            "| {name} | {repo} | {config} | {checkpoint} | {load} | {deps} | {inference} | {status} |".format(
                name=item["name"],
                repo=mark(item["repo_ok"]),
                config=mark(item["config_ok"]),
                checkpoint=mark(item["checkpoint_ok"]),
                load=mark(item["checkpoint_load_ok"]),
                deps=mark(item["dependencies_ok"]),
                inference=mark(item["inference_ok"]),
                status=item["status"],
            )
        )
    lines.extend(["", "## Details", ""])
    for item in results:
        lines.extend(
            [
                f"### {item['name']} -> {item['task']}",
                "",
                f"- repo_path: `{item['repo_path']}`",
                f"- config_path: `{item['config_path']}`",
                f"- checkpoint_path: `{item['checkpoint_path']}`",
                f"- missing_dependencies: `{', '.join(item['missing_dependencies']) or 'none'}`",
                f"- checkpoint_keys: `{', '.join(item['checkpoint_keys']) or 'none'}`",
                f"- searched_paths: `{'; '.join(item['searched_paths'])}`",
                f"- errors: `{'; '.join(item['errors']) or 'none'}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Current Assessment",
            "",
            "- StreamPETR has a local checkpoint that can be loaded with `torch.load`, but MMDetection3D dependencies are missing, so real model inference is blocked.",
            "- MapTRv2 repo and config are present, but no MapTR checkpoint was found under `weights/` or `third_party/`.",
            "- OccNet repo and OpenScene Occ baseline config are present, but no OccNet/OpenScene checkpoint was found under `weights/` or `third_party/`.",
            "- ViDAR repo and OpenScene config are present, but no ViDAR checkpoint was found under `weights/` or `third_party/`.",
            "- No random soft labels or dummy teacher outputs are generated.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_yaml_config(PROJECT_ROOT / "configs" / "stage2_distill.yaml")
    teachers = build_all_teachers(config.get("teachers", {}))
    results = [teacher.check_result.as_dict() for teacher in teachers.values()]

    print("Teacher repo/config/checkpoint/dependency check")
    print("=" * 96)
    print(f"{'Teacher':<12} {'repo':<8} {'config':<8} {'checkpoint':<11} {'load':<8} {'deps':<8} {'inference':<10} status")
    for item in results:
        print(
            f"{item['name']:<12} "
            f"{mark(item['repo_ok']):<8} "
            f"{mark(item['config_ok']):<8} "
            f"{mark(item['checkpoint_ok']):<11} "
            f"{mark(item['checkpoint_load_ok']):<8} "
            f"{mark(item['dependencies_ok']):<8} "
            f"{mark(item['inference_ok']):<10} "
            f"{item['status']}"
        )
        if item["errors"]:
            for error in item["errors"]:
                print(f"  - {error}")
    print("=" * 96)
    print("JSON:")
    print(json.dumps(results, indent=2))
    write_report(results)
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
