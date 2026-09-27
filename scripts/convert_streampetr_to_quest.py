from __future__ import print_function

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
STREAM_PETR_CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)
QUEST_CLASS_BY_NAME = {
    "car": 0,
    "truck": 0,
    "construction_vehicle": 0,
    "bus": 0,
    "trailer": 0,
    "motorcycle": 0,
    "bicycle": 0,
    "pedestrian": 1,
    "traffic_cone": 2,
    "barrier": 3,
}
STREAM_PETR_TO_QUEST = torch.tensor(
    [QUEST_CLASS_BY_NAME[name] for name in STREAM_PETR_CLASSES], dtype=torch.long
)
XY_RANGE = (-50.0, 50.0)
Z_RANGE = (-5.0, 5.0)
SIZE_NORM = (20.0, 10.0, 8.0)
VELOCITY_NORM = 20.0
MAX_AGENT_INSTANCES = 64


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw StreamPETR detections to QUEST Agent pseudo-labels"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data/pseudo_labels/streampetr",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/soft_labels"
    )
    parser.add_argument("--score-threshold", type=float, default=0.25)
    return parser.parse_args()


def load_payload(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def convert_predictions(payload, score_threshold=0.25):
    token = str(payload["token"])
    raw_boxes = torch.as_tensor(payload["boxes_3d"], dtype=torch.float32)
    raw_scores = torch.as_tensor(payload["scores_3d"], dtype=torch.float32)
    raw_labels = torch.as_tensor(payload["labels_3d"], dtype=torch.long)
    if raw_boxes.ndim != 2 or raw_boxes.shape[1] != 9:
        raise ValueError("boxes_3d must be [N,9], got {}".format(tuple(raw_boxes.shape)))
    if raw_scores.shape != (raw_boxes.shape[0],):
        raise ValueError(
            "scores_3d must be [N], got {} for N={}".format(
                tuple(raw_scores.shape), raw_boxes.shape[0]
            )
        )
    if raw_labels.shape != (raw_boxes.shape[0],):
        raise ValueError(
            "labels_3d must be [N], got {} for N={}".format(
                tuple(raw_labels.shape), raw_boxes.shape[0]
            )
        )
    if raw_labels.numel() and (
        int(raw_labels.min()) < 0 or int(raw_labels.max()) >= len(STREAM_PETR_CLASSES)
    ):
        raise ValueError("labels_3d contains an index outside the StreamPETR 10 classes")

    xy_min, xy_max = XY_RANGE
    z_min, z_max = Z_RANGE
    in_range = (
        (raw_boxes[:, 0] >= xy_min)
        & (raw_boxes[:, 0] <= xy_max)
        & (raw_boxes[:, 1] >= xy_min)
        & (raw_boxes[:, 1] <= xy_max)
        & (raw_boxes[:, 2] >= z_min)
        & (raw_boxes[:, 2] <= z_max)
    )
    keep = (raw_scores >= float(score_threshold)) & in_range
    raw_boxes = raw_boxes[keep]
    raw_scores = raw_scores[keep]
    raw_labels = raw_labels[keep]

    order = torch.argsort(raw_scores, descending=True)[:MAX_AGENT_INSTANCES]
    raw_boxes = raw_boxes[order]
    raw_labels = raw_labels[order]
    count = raw_boxes.shape[0]

    labels = torch.full((MAX_AGENT_INSTANCES,), -1, dtype=torch.long)
    boxes = torch.zeros((MAX_AGENT_INSTANCES, 8), dtype=torch.float32)
    velocity = torch.zeros((MAX_AGENT_INSTANCES, 3), dtype=torch.float32)
    if count:
        centers = raw_boxes[:, :3].clone()
        centers[:, 0:2] = (centers[:, 0:2] - xy_min) / (xy_max - xy_min)
        centers[:, 2] = (centers[:, 2] - z_min) / (z_max - z_min)
        centers.clamp_(0.0, 1.0)
        sizes = (
            raw_boxes[:, 3:6]
            / torch.tensor(SIZE_NORM, dtype=torch.float32).reshape(1, 3)
        ).clamp(0.0, 1.0)
        yaw = raw_boxes[:, 6]
        boxes[:count] = torch.cat(
            (centers, sizes, torch.sin(yaw[:, None]), torch.cos(yaw[:, None])),
            dim=1,
        )
        velocity[:count, :2] = raw_boxes[:, 7:9] / VELOCITY_NORM
        labels[:count] = STREAM_PETR_TO_QUEST[raw_labels]

    return {
        "token": token,
        "agent": {
            "labels": labels,
            "boxes": boxes,
            "velocity": velocity,
        },
    }


def convert_file(input_path, output_dir, score_threshold):
    payload = load_payload(input_path)
    if not isinstance(payload, dict):
        raise ValueError("pseudo-label file must contain a dict: {}".format(input_path))
    converted = convert_predictions(payload, score_threshold)
    token = converted["token"]
    if token != input_path.stem:
        raise ValueError(
            "pseudo-label token mismatch: {} != {}".format(token, input_path.stem)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "{}.pt".format(token)
    temporary_path = output_path.with_suffix(".pt.tmp")
    torch.save(converted, str(temporary_path))
    temporary_path.replace(output_path)
    return output_path, int((converted["agent"]["labels"] >= 0).sum())


def main():
    args = parse_args()
    input_paths = sorted(args.input_dir.glob("*.pt"))
    if not input_paths:
        raise RuntimeError("no StreamPETR pseudo-labels found in {}".format(args.input_dir))
    succeeded = 0
    failed = 0
    for index, input_path in enumerate(input_paths, start=1):
        try:
            output_path, count = convert_file(
                input_path, args.output_dir, args.score_threshold
            )
            succeeded += 1
            print(
                "[{}/{}] {} OK agents={} file={}".format(
                    index, len(input_paths), input_path.stem, count, output_path
                ),
                flush=True,
            )
        except Exception as error:
            failed += 1
            print(
                "[{}/{}] {} FAILED {}: {}".format(
                    index,
                    len(input_paths),
                    input_path.stem,
                    type(error).__name__,
                    error,
                ),
                flush=True,
            )
    print(
        "summary: input={} succeeded={} failed={} output={}".format(
            len(input_paths), succeeded, failed, args.output_dir
        )
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
