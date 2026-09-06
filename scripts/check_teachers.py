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
REQUIRED_MODULES = ("torch", "mmcv", "mmdet", "mmdet3d")


def mark(value: bool) -> str:
    return "OK" if value else "BLOCKED"


def write_report(results: list[dict]) -> None:
    env_status = "OK" if all(item["dependencies_ok"] for item in results) else "BLOCKED"
    lines = [
        "# Teacher Integration Status",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Summary",
        "",
        "| Teacher | Task | Environment | Repo | Config | Checkpoint | Camera count | Camera mapping | Build | Real inference | Raw output | Conversion | GT evaluation | Remaining blocker |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        blocker = "; ".join(item["errors"]) or "none"
        lines.append(
            "| {name} | {task} | {env} | {repo} | {config} | {checkpoint} | {camera_count} | {mapping} | {build} | {inference} | {raw} | {conversion} | {gt_eval} | {blocker} |".format(
                name=item["name"],
                task=item["task"],
                env=mark(item["dependencies_ok"]),
                repo=mark(item["repo_ok"]),
                config=mark(item["config_ok"]),
                checkpoint=mark(item["checkpoint_ok"]),
                camera_count=item["camera_count"],
                mapping=json.dumps(item["camera_mapping"], ensure_ascii=False) if item["camera_mapping"] else "OpenScene native",
                build="BLOCKED" if item["status"].startswith("BLOCKED") else "PENDING_NATIVE_BUILD",
                inference=mark(item["inference_ok"]),
                raw="none" if not item["raw_output_shapes"] else json.dumps(item["raw_output_shapes"]),
                conversion="BLOCKED" if not item["converted_output_shapes"] else "OK",
                gt_eval="BLOCKED" if not item["inference_ok"] else "PENDING_METRIC",
                blocker=blocker,
            )
        )
    lines.extend(["", "## Details", ""])
    for item in results:
        lines.extend(
            [
                f"### {item['name']} -> {item['task']}",
                "",
                f"- environment: `{mark(item['dependencies_ok'])}`",
                f"- repo_path: `{item['repo_path']}`",
                f"- config_path: `{item['config_path']}`",
                f"- checkpoint_path: `{item['checkpoint_path']}`",
                f"- camera_count: `{item['camera_count']}`",
                f"- camera_mapping: `{json.dumps(item['camera_mapping'], ensure_ascii=False)}`",
                f"- missing_dependencies: `{', '.join(item['missing_dependencies']) or 'none'}`",
                f"- checkpoint_keys: `{', '.join(item['checkpoint_keys']) or 'none'}`",
                f"- searched_paths: `{'; '.join(item['searched_paths'])}`",
                f"- errors: `{'; '.join(item['errors']) or 'none'}`",
                "",
            ]
        )
    lines.extend(["## Current Assessment", ""])
    for item in results:
        blocker = "; ".join(item["errors"]) or item["status"]
        if item["inference_ok"]:
            lines.append(f"- {item['name']} real inference: OK.")
        else:
            lines.append(f"- {item['name']} real inference: {item['status']} ({blocker}).")
    lines.extend(
        [
            "- Flow remains supervised by OpenScene Flow GT. ViDAR is recorded only as the Future World teacher.",
            "- `quest_teacher_legacy` creation was attempted on Windows; pip failed on `torch-1.10.1+cu111` with an invalid wheel error, leaving torch/mmcv/mmdet/mmdet3d unavailable.",
            "- No random soft labels, fake tensors, or dummy teacher outputs are generated.",
            f"- Aggregate environment status: {env_status}. Required modules: {', '.join(REQUIRED_MODULES)}.",
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
    print(f"{'Teacher':<12} {'task':<13} {'env':<8} {'repo':<8} {'config':<8} {'checkpoint':<11} {'load':<8} {'cams':<5} {'inference':<10} status")
    for item in results:
        print(
            f"{item['name']:<12} "
            f"{item['task']:<13} "
            f"{mark(item['dependencies_ok']):<8} "
            f"{mark(item['repo_ok']):<8} "
            f"{mark(item['config_ok']):<8} "
            f"{mark(item['checkpoint_ok']):<11} "
            f"{mark(item['checkpoint_load_ok']):<8} "
            f"{item['camera_count']:<5} "
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
