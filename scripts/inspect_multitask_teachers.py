"""Inventory active supervision: Navformer Agent, nuPlan Vector GT, Seg and Depth."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAVFORMER_ROOT = Path("/home/user/DataDisk/QUEST_WORK/Navformer")


def nonempty_file(path):
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def inspect_files(task, teacher, required_groups, alignment):
    """Each group is an explicit list of interchangeable files; never scan."""
    missing = [
        [str(path) for path in alternatives]
        for alternatives in required_groups
        if not any(nonempty_file(path) for path in alternatives)
    ]
    return {
        "task": task,
        "teacher": teacher,
        "asset_status": "FILES_PRESENT" if not missing else "ASSETS_MISSING",
        "missing": missing,
        "runtime_status": "NOT_CHECKED",
        "openscene_alignment": alignment,
    }


def inspect_teachers(root, navformer_root, segformer_dir, metric3d_root, metric3d_checkpoint):
    vector_dir = root / "data/vector_map_gt"
    map_report = inspect_files("map", "nuPlan Vector GT (NOT a neural teacher)", (
        (root / "data/stage3_split.json",),
        (root / "data/vector_map_capacity_audit.json",),
    ), "MapTRv2=STRUCTURE_REFERENCE_ONLY; raster_KD=DISABLED; full_GT_coverage=NOT_CHECKED")
    if not vector_dir.is_dir():
        map_report["missing"].append([str(vector_dir) + " (directory)"])
        map_report["asset_status"] = "ASSETS_MISSING"
    reports = [
        inspect_files("agent", "Navformer UniAD track+map", (
            (navformer_root / "mmdet3d_plugin/__init__.py",),
            (navformer_root / "configs/navformer/track_map_nuplan_r50_navtrain.py",),
            (navformer_root / "data/alg_engine/ckpts/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth",),
        ), "USER_REPORTED_AGENT_VALIDATION; current files/runtime not revalidated"),
        map_report,
        inspect_files("seg", "SegFormer-B5 Cityscapes (HF bundle)", (
            (segformer_dir / "config.json",),
            (segformer_dir / "preprocessor_config.json",),
            (segformer_dir / "model.safetensors", segformer_dir / "pytorch_model.bin"),
        ), "UNVERIFIED; native Cityscapes19 export only; not legacy QUEST C_seg=6"),
        inspect_files("depth", "Metric3Dv2 ViT-L", (
            (metric3d_root / "hubconf.py",),
            (metric3d_root / "mono/configs/HourglassDecoder/vit.raft5.large.py",),
            (metric3d_checkpoint,),
        ), "UNVERIFIED; image geometry and real inference evidence required"),
    ]
    partial = metric3d_checkpoint.with_name(metric3d_checkpoint.name + ".part")
    if nonempty_file(partial):
        reports[-1]["partial_download_bytes"] = partial.stat().st_size
    return reports


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--navformer-root", type=Path, default=NAVFORMER_ROOT)
    parser.add_argument("--segformer-dir", type=Path)
    parser.add_argument("--metric3d-root", type=Path)
    parser.add_argument("--metric3d-checkpoint", type=Path)
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout; write no files")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = args.root.resolve()
    reports = inspect_teachers(
        root, args.navformer_root,
        args.segformer_dir or root / "weights/teachers/segformer_cityscapes_b5",
        args.metric3d_root or root / "third_party/Metric3D",
        args.metric3d_checkpoint or root / "weights/teachers/metric3d/metric_depth_vit_large_800k.pth",
    )
    if args.json:
        print(json.dumps({
            "read_only": True, "model_loaded": False, "gpu_used": False,
            "supervision_sources": reports, "all_four_tasks_ready": "NOT_VERIFIED",
        }, indent=2))
    else:
        print("READ_ONLY=True model_load=False GPU=False directory_scan=False")
        for report in reports:
            print("{task}: {teacher} assets={asset_status} runtime={runtime_status}".format(**report))
            for alternatives in report["missing"]:
                print("  MISSING " + " OR ".join(alternatives))
            if "partial_download_bytes" in report:
                print("  resumable_partial_bytes=" + str(report["partial_download_bytes"]))
            print("  alignment=" + report["openscene_alignment"])
        print("ALL_FOUR_TASKS_READY=NOT_VERIFIED; Map uses GT, not a required MapTRv2 checkpoint")
    return 0 if all(report["asset_status"] == "FILES_PRESENT" for report in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
