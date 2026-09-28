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
from inspect_openscene_sequence import scene_and_timestamp, sort_infos_temporally  # noqa: E402


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
    ordered_infos = sort_infos_temporally(all_infos) if args.temporal else list(all_infos)
    infos = select_range(ordered_infos, args.start, args.count)

    print(
        "runtime:",
        "torch={}".format(torch.__version__),
        "mmcv={}".format(mmcv.__version__),
        "mmdet={}".format(mmdet.__version__),
        "mmdet3d={}".format(mmdet3d.__version__),
        "gpu={}".format(torch.cuda.get_device_name(0)),
    )
    print("temporal_order={}".format(str(args.temporal).lower()))
    print("selected_range=[{},{})".format(args.start, args.start + args.count))
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

    succeeded = 0
    failed = 0
    failures = []
    scenes_processed = set()
    temporal_frames = 0
    first_frames = 0
    resets = 0
    temporal_state = TemporalSequenceState()
    previous_item_scene = None
    for offset, info in enumerate(infos):
        sample_index = args.start + offset
        token = str(info.get("token", "index-{}".format(sample_index)))
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
        if temporal_decision["new_scene"]:
            first_frames += 1
        if temporal_decision["prev_exists"] == 1.0:
            temporal_frames += 1
        if temporal_decision["non_monotonic"]:
            print(
                "WARNING index={} token={} scene={} timestamp={} is not greater "
                "than previous timestamp {}; resetting temporal memory".format(
                    sample_index,
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
            payload = make_payload(token, pts_bbox)
            output_path = args.output_dir / "{}.pt".format(token)
            save_payload(payload, output_path, torch)
            succeeded += 1
            if args.temporal:
                temporal_state.complete(True)
            print(
                "index={} token={} scene_token={} timestamp={} "
                "temporal_prev_exists={} new_scene={} detections={} file={}".format(
                    sample_index,
                    token,
                    scene_token,
                    timestamp,
                    int(temporal_decision["prev_exists"]),
                    str(temporal_decision["new_scene"]).lower(),
                    payload["scores_3d"].shape[0],
                    output_path,
                ),
                flush=True,
            )
        except Exception as error:
            failed += 1
            failures.append((sample_index, token, type(error).__name__, str(error)))
            if args.temporal:
                temporal_state.complete(False)
            reset_temporal_memory(model)
            resets += 1
            torch.cuda.empty_cache()
            print(
                "index={} token={} scene_token={} timestamp={} "
                "temporal_prev_exists={} new_scene={} FAILED {}: {}".format(
                    sample_index,
                    token,
                    scene_token,
                    timestamp,
                    int(temporal_decision["prev_exists"]),
                    str(temporal_decision["new_scene"]).lower(),
                    type(error).__name__,
                    error,
                ),
                flush=True,
            )

    print(
        "summary: attempted={} succeeded={} failed={} scenes_processed={} "
        "temporal_frames={} first_frames={} resets={} output={}".format(
            args.count,
            succeeded,
            failed,
            len(scenes_processed),
            temporal_frames,
            first_frames,
            resets,
            args.output_dir,
        )
    )
    for sample_index, token, error_type, message in failures:
        print(
            "failure: index={} token={} {}: {}".format(
                sample_index, token, error_type, message
            )
        )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
