"""Read-only four-teacher asset inventory, not an inference/readiness certificate."""

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


def inspect_teachers(navformer_root, maptr_root, maptr_checkpoint,
                     segformer_dir, metric3d_root, metric3d_checkpoint,
                     map_profile="centerline"):
    if map_profile not in ("centerline", "three_class"):
        raise ValueError("unknown MapTRv2 profile")
    config_name = (
        "maptrv2_nusc_r50_24ep_w_centerline.py"
        if map_profile == "centerline" else "maptrv2_nusc_r50_24ep.py"
    )
    return [
        inspect_files("agent", "Navformer UniAD track+map", (
            (navformer_root / "mmdet3d_plugin/__init__.py",),
            (navformer_root / "configs/navformer/track_map_nuplan_r50_navtrain.py",),
            (navformer_root / "data/alg_engine/ckpts/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth",),
        ), "USER_REPORTED_AGENT_VALIDATION; current files/runtime not revalidated"),
        inspect_files("map", "MapTRv2 " + map_profile, (
            (maptr_root / "projects/mmdet3d_plugin/maptr/detectors/maptrv2.py",),
            (maptr_root / "projects/configs/maptrv2" / config_name,),
            (maptr_checkpoint,),
        ), "UNVERIFIED; " + (
            "centerline checkpoint still requires nuPlan semantic/ROI audit"
            if map_profile == "centerline" else "three-class model has NO centerline"
        )),
        inspect_files("seg", "SegFormer-B5 Cityscapes (HF bundle)", (
            (segformer_dir / "config.json",),
            (segformer_dir / "preprocessor_config.json",),
            (segformer_dir / "model.safetensors", segformer_dir / "pytorch_model.bin"),
        ), "UNVERIFIED; 19 teacher classes, QUEST mapping not implemented"),
        inspect_files("depth", "Metric3Dv2 ViT-L", (
            (metric3d_root / "hubconf.py",),
            (metric3d_root / "mono/configs/HourglassDecoder/vit.raft5.large.py",),
            (metric3d_checkpoint,),
        ), "UNVERIFIED; image geometry and real inference evidence required"),
    ]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--navformer-root", type=Path, default=NAVFORMER_ROOT)
    parser.add_argument("--maptr-root", type=Path)
    parser.add_argument("--maptr-checkpoint", type=Path)
    parser.add_argument("--map-profile", choices=("centerline", "three_class"), default="centerline")
    parser.add_argument("--segformer-dir", type=Path)
    parser.add_argument("--metric3d-root", type=Path)
    parser.add_argument("--metric3d-checkpoint", type=Path)
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout; write no files")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = args.root.resolve()
    map_name = (
        "maptrv2_nusc_r50_24ep_w_centerline.pth"
        if args.map_profile == "centerline" else "maptrv2_nusc_r50_24ep.pth"
    )
    reports = inspect_teachers(
        args.navformer_root, args.maptr_root or root / "third_party/MapTR",
        args.maptr_checkpoint or root / "weights/teachers/maptrv2" / map_name,
        args.segformer_dir or root / "weights/teachers/segformer_cityscapes_b5",
        args.metric3d_root or root / "third_party/Metric3D",
        args.metric3d_checkpoint or root / "weights/teachers/metric3d/metric_depth_vit_large_800k.pth",
        args.map_profile,
    )
    if args.json:
        print(json.dumps({
            "read_only": True, "model_loaded": False, "gpu_used": False,
            "teachers": reports, "all_teachers_ready": "NOT_VERIFIED",
        }, indent=2))
    else:
        print("READ_ONLY=True model_load=False GPU=False directory_scan=False")
        for report in reports:
            print("{task}: {teacher} assets={asset_status} runtime={runtime_status}".format(**report))
            for alternatives in report["missing"]:
                print("  MISSING " + " OR ".join(alternatives))
            print("  alignment=" + report["openscene_alignment"])
        print("ALL_TEACHERS_READY=NOT_VERIFIED; existence is not checkpoint/inference/quality verification")
    return 0 if all(report["asset_status"] == "FILES_PRESENT" for report in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
