from __future__ import print_function

import argparse
import importlib
import pickle
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_streampetr_openscene import (  # noqa: E402
    CHECKPOINT_NAME,
    ROOT,
    STREAM_PETR_ROOT,
    first_existing,
    load_native_config,
    make_input,
    set_test_image_size,
)
from inspect_openscene_sequence import scene_and_timestamp  # noqa: E402


class TemporalSequenceState:
    def __init__(self):
        self.previous_scene_token = None
        self.previous_timestamp = None
        self.memory_valid = False
        self.pending = None

    def begin(self, scene_token, timestamp):
        if self.pending is not None:
            raise RuntimeError("complete the pending temporal frame before begin")
        scene_token = str(scene_token)
        timestamp = float(timestamp)
        new_scene = (
            self.previous_scene_token is None
            or scene_token != self.previous_scene_token
        )
        non_monotonic = (
            not new_scene
            and self.previous_timestamp is not None
            and timestamp <= self.previous_timestamp
        )
        reset_required = new_scene or non_monotonic or not self.memory_valid
        decision = {
            "scene_token": scene_token,
            "timestamp": timestamp,
            "new_scene": new_scene,
            "non_monotonic": non_monotonic,
            "reset_required": reset_required,
            "prev_exists": 0.0 if reset_required else 1.0,
        }
        self.pending = decision
        return decision

    def complete(self, success):
        if self.pending is None:
            raise RuntimeError("begin a temporal frame before complete")
        self.previous_scene_token = self.pending["scene_token"]
        self.previous_timestamp = self.pending["timestamp"]
        self.memory_valid = bool(success)
        self.pending = None


def default_output_dir(temporal):
    name = "streampetr_temporal" if temporal else "streampetr"
    return ROOT / "data/pseudo_labels" / name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export raw StreamPETR predictions for consecutive OpenScene samples"
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    temporal_group = parser.add_mutually_exclusive_group()
    temporal_group.add_argument(
        "--temporal", dest="temporal", action="store_true", help="Use scene-temporal inference"
    )
    temporal_group.add_argument(
        "--no-temporal",
        dest="temporal",
        action="store_false",
        help="Reset StreamPETR memory for every frame",
    )
    parser.set_defaults(temporal=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=first_existing(
            (
                ROOT / "data/openscene/meta_datas/meta_data_mini.pkl",
                ROOT
                / "data/openscene/meta_datas/openscene-v1.0/meta_datas/meta_data_mini.pkl",
            ),
            "OpenScene metadata",
        ),
    )
    parser.add_argument(
        "--camera-root", type=Path, default=ROOT / "data/openscene/sensor_blobs_mini"
    )
    parser.add_argument("--stream-petr-root", type=Path, default=STREAM_PETR_ROOT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=first_existing(
            (
                ROOT / "third_party/StreamPETR-main/ckpts" / CHECKPOINT_NAME,
                STREAM_PETR_ROOT / "ckpts" / CHECKPOINT_NAME,
                ROOT / "third_party/StreamPETR/StreamPETR-main/ckpts" / CHECKPOINT_NAME,
            ),
            "StreamPETR checkpoint",
        ),
    )
    parser.add_argument(
        "--mmdet3d-config-root",
        type=Path,
        default=None,
        help="Directory containing configs/_base_ if StreamPETR has no sibling mmdetection3d checkout",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def select_range(infos, start, count):
    if start < 0:
        raise ValueError("start must be non-negative, got {}".format(start))
    if count <= 0:
        raise ValueError("count must be positive, got {}".format(count))
    end = start + count
    if end > len(infos):
        raise IndexError(
            "requested samples [{}, {}) exceed metadata size {}".format(
                start, end, len(infos)
            )
        )
    return infos[start:end]


def target_token_set(target_infos):
    tokens = [str(info["token"]) for info in target_infos]
    if len(tokens) != len(set(tokens)):
        raise ValueError("target metadata slice contains duplicate tokens")
    return set(tokens)


def build_relevant_by_scene(all_infos, target_scenes):
    relevant_by_scene = {}
    for info in all_infos:
        scene_token = str(info["scene_token"])
        if scene_token in target_scenes:
            relevant_by_scene.setdefault(scene_token, []).append(info)
    return relevant_by_scene


def build_temporal_execution_plan(all_infos, target_infos):
    target_tokens = target_token_set(target_infos)
    ordered_target_scenes = []
    target_scenes = set()
    latest_target_timestamp_by_scene = {}
    for info in target_infos:
        scene_token = str(info["scene_token"])
        if scene_token not in target_scenes:
            target_scenes.add(scene_token)
            ordered_target_scenes.append(scene_token)
        timestamp = float(info["timestamp"])
        latest_target_timestamp_by_scene[scene_token] = max(
            timestamp,
            latest_target_timestamp_by_scene.get(scene_token, timestamp),
        )

    relevant_by_scene = build_relevant_by_scene(all_infos, target_scenes)
    plan = []
    for scene_token in ordered_target_scenes:
        scene_infos = relevant_by_scene.get(scene_token, [])
        scene_infos.sort(key=lambda info: float(info["timestamp"]))
        latest_target_timestamp = latest_target_timestamp_by_scene[scene_token]
        for info in scene_infos:
            if float(info["timestamp"]) > latest_target_timestamp:
                break
            plan.append(
                {
                    "info": info,
                    "target": str(info["token"]) in target_tokens,
                }
            )
    planned_targets = {
        str(item["info"]["token"]) for item in plan if item["target"]
    }
    if planned_targets != target_tokens:
        raise RuntimeError(
            "temporal plan target mismatch: missing={} extra={}".format(
                sorted(target_tokens - planned_targets),
                sorted(planned_targets - target_tokens),
            )
        )
    return plan


def validate_target_outputs(target_tokens, output_dir):
    target_tokens = set(target_tokens)
    output_tokens = {path.stem for path in output_dir.glob("*.pt")}
    if output_tokens != target_tokens:
        raise RuntimeError(
            "target/output token mismatch: missing_files={} extra_files={}".format(
                sorted(target_tokens - output_tokens),
                sorted(output_tokens - target_tokens),
            )
        )


def format_success_log(
    scene_token, timestamp, token, prev_exists, is_target, saved, detections
):
    return (
        "scene={} timestamp={} token={} prev_exists={} target={} saved={} "
        "detections={}".format(
            scene_token,
            timestamp,
            token,
            int(prev_exists),
            str(is_target).lower(),
            str(saved).lower(),
            detections,
        )
    )


def format_failure_log(
    scene_token, timestamp, token, prev_exists, is_target, error
):
    return (
        "scene={} timestamp={} token={} prev_exists={} target={} saved=false "
        "FAILED {}: {}".format(
            scene_token,
            timestamp,
            token,
            int(prev_exists),
            str(is_target).lower(),
            type(error).__name__,
            error,
        )
    )


def make_payload(token, pts_bbox):
    boxes = pts_bbox["boxes_3d"].tensor.detach().cpu()
    scores = pts_bbox["scores_3d"].detach().cpu()
    labels = pts_bbox["labels_3d"].detach().cpu()
    if boxes.shape[0] != scores.shape[0] or scores.shape[0] != labels.shape[0]:
        raise ValueError(
            "inconsistent prediction counts: boxes={}, scores={}, labels={}".format(
                boxes.shape[0], scores.shape[0], labels.shape[0]
            )
        )
    return {
        "token": str(token),
        "boxes_3d": boxes,
        "scores_3d": scores,
        "labels_3d": labels,
    }


def save_payload(payload, output_path, torch):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, str(temporary_path))
    temporary_path.replace(output_path)


def reset_temporal_memory(model):
    model.prev_scene_token = None
    head = getattr(model, "pts_bbox_head", None)
    if head is not None and hasattr(head, "reset_memory"):
        head.reset_memory()


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.temporal)

    import torch
    import mmcv
    import mmdet
    import mmdet3d
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmdet3d.core.bbox import get_box_type
    from mmdet3d.datasets.pipelines import Compose
    from mmdet3d.models import build_model

    if not torch.cuda.is_available():
        raise RuntimeError("StreamPETR requires CUDA in the Teacher runtime")

    with args.metadata.open("rb") as stream:
        metadata = pickle.load(stream)
    if "infos" not in metadata:
        raise KeyError("OpenScene metadata does not contain 'infos'")
    all_infos = metadata["infos"]
    target_infos = select_range(all_infos, args.start, args.count)
    target_tokens = target_token_set(target_infos)
    if args.temporal:
        execution_plan = build_temporal_execution_plan(all_infos, target_infos)
    else:
        execution_plan = [
            {"info": info, "target": True} for info in target_infos
        ]

    print(
        "runtime:",
        "torch={}".format(torch.__version__),
        "mmcv={}".format(mmcv.__version__),
        "mmdet={}".format(mmdet.__version__),
        "mmdet3d={}".format(mmdet3d.__version__),
        "gpu={}".format(torch.cuda.get_device_name(0)),
    )
    print("temporal_mode={}".format(str(args.temporal).lower()))
    print("target_selection=metadata_slice")
    print("selected_range=[{},{})".format(args.start, args.start + args.count))
    print("target_samples={}".format(len(target_tokens)))
    print("total_inference_frames={}".format(len(execution_plan)))
    print("output: {}".format(args.output_dir))

    cfg = load_native_config(args, mmcv)
    sys.path.insert(0, str(args.stream_petr_root.resolve()))
    importlib.import_module("projects.mmdet3d_plugin")
    import mmdet3d.datasets.pipelines  # noqa: F401

    box_type_3d, box_mode_3d = get_box_type("LiDAR")
    checkpoint_path = first_existing((args.checkpoint,), "StreamPETR checkpoint")
    cfg.model.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    model = model.cuda().eval()
    print("checkpoint loaded:", str(checkpoint_path))

    failed = 0
    failures = []
    scenes_processed = set()
    saved_tokens = set()
    warmup_frames_processed = 0
    total_inference_frames = 0
    resets = 0
    temporal_state = TemporalSequenceState()
    previous_item_scene = None
    for execution_index, item in enumerate(execution_plan):
        info = item["info"]
        is_target = bool(item["target"])
        token = str(info.get("token", "execution-{}".format(execution_index)))
        scene_token, timestamp = scene_and_timestamp(info)
        scenes_processed.add(scene_token)
        if args.temporal:
            temporal_decision = temporal_state.begin(scene_token, timestamp)
        else:
            temporal_decision = {
                "scene_token": scene_token,
                "timestamp": timestamp,
                "new_scene": previous_item_scene != scene_token,
                "non_monotonic": False,
                "reset_required": True,
                "prev_exists": 0.0,
            }
        previous_item_scene = scene_token
        if temporal_decision["non_monotonic"]:
            print(
                "WARNING execution_index={} token={} scene={} timestamp={} is not greater "
                "than previous timestamp {}; resetting temporal memory".format(
                    execution_index,
                    token,
                    scene_token,
                    timestamp,
                    temporal_state.previous_timestamp,
                ),
                flush=True,
            )
        if temporal_decision["reset_required"]:
            reset_temporal_memory(model)
            resets += 1
        try:
            total_inference_frames += 1
            raw = make_input(info, args.camera_root)
            raw["box_type_3d"] = box_type_3d
            raw["box_mode_3d"] = box_mode_3d
            set_test_image_size(cfg, raw["img_filename"])
            pipeline = Compose(cfg.test_pipeline)
            processed = pipeline(raw)
            if processed is None:
                raise RuntimeError("StreamPETR native test pipeline rejected the sample")
            data = scatter(collate([processed], samples_per_gpu=1), [0])[0]
            data["prev_exists"] = [[
                torch.tensor(temporal_decision["prev_exists"], device="cuda")
            ]]

            with torch.no_grad():
                outputs = model(return_loss=False, rescale=True, **data)
            if not isinstance(outputs, list) or len(outputs) != 1:
                raise RuntimeError(
                    "unexpected StreamPETR output: {}".format(type(outputs))
                )
            pts_bbox = outputs[0]["pts_bbox"]
            detections = int(pts_bbox["scores_3d"].shape[0])
            saved = False
            if is_target:
                payload = make_payload(token, pts_bbox)
                output_path = args.output_dir / "{}.pt".format(token)
                save_payload(payload, output_path, torch)
                saved_tokens.add(token)
                saved = True
            else:
                warmup_frames_processed += 1
            if args.temporal:
                temporal_state.complete(True)
        except Exception as error:
            failed += 1
            failures.append(
                (execution_index, token, type(error).__name__, str(error))
            )
            if args.temporal:
                temporal_state.complete(False)
            reset_temporal_memory(model)
            resets += 1
            torch.cuda.empty_cache()
            print(
                format_failure_log(
                    scene_token,
                    timestamp,
                    token,
                    temporal_decision["prev_exists"],
                    is_target,
                    error,
                ),
                flush=True,
            )
            continue
        print(
            format_success_log(
                scene_token,
                timestamp,
                token,
                temporal_decision["prev_exists"],
                is_target,
                saved,
                detections,
            ),
            flush=True,
        )

    print(
        "summary: target_samples={} target_saved={} warmup_frames_processed={} "
        "total_inference_frames={} scenes_processed={} failed_frames={} resets={} "
        "output={}".format(
            len(target_tokens),
            len(saved_tokens),
            warmup_frames_processed,
            total_inference_frames,
            len(scenes_processed),
            failed,
            resets,
            args.output_dir,
        )
    )
    for execution_index, token, error_type, message in failures:
        print(
            "failure: execution_index={} token={} {}: {}".format(
                execution_index, token, error_type, message
            )
        )
    validate_target_outputs(target_tokens, args.output_dir)
    print("target token set == output temporal .pt token set: true")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
