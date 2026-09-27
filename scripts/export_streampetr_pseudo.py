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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export raw StreamPETR predictions for consecutive OpenScene samples"
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
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
        default=ROOT / "data/pseudo_labels/streampetr",
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
    infos = select_range(metadata["infos"], args.start, args.count)

    print(
        "runtime:",
        "torch={}".format(torch.__version__),
        "mmcv={}".format(mmcv.__version__),
        "mmdet={}".format(mmdet.__version__),
        "mmdet3d={}".format(mmdet3d.__version__),
        "gpu={}".format(torch.cuda.get_device_name(0)),
    )
    print(
        "range: [{}, {}) output: {}".format(
            args.start, args.start + args.count, args.output_dir
        )
    )

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
    for offset, info in enumerate(infos):
        sample_index = args.start + offset
        token = str(info.get("token", "index-{}".format(sample_index)))
        progress = "[{}/{}] index={} token={}".format(
            offset + 1, args.count, sample_index, token
        )
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
            data["prev_exists"] = [[torch.tensor(0.0, device="cuda")]]

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
            print(
                "{} OK detections={} file={}".format(
                    progress, payload["scores_3d"].shape[0], output_path
                ),
                flush=True,
            )
        except Exception as error:
            failed += 1
            failures.append((sample_index, token, type(error).__name__, str(error)))
            reset_temporal_memory(model)
            torch.cuda.empty_cache()
            print(
                "{} FAILED {}: {}".format(progress, type(error).__name__, error),
                flush=True,
            )

    print(
        "summary: attempted={} succeeded={} failed={} output={}".format(
            args.count, succeeded, failed, args.output_dir
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
