import torch

from scripts.evaluate_navformer_openscene import (
    filter_agent_gt,
    filter_navformer_predictions,
)


class Boxes:
    def __init__(self, tensor):
        self.tensor = tensor


def test_prediction_filter_maps_vehicle_and_pedestrian_only():
    boxes = torch.zeros(6, 9)
    boxes[4, 0] = 50.01
    scores = torch.tensor([0.24, 0.25, 0.80, 0.90, 0.90, 0.90])
    labels = torch.tensor([0, 0, 2, 1, 2, 3])

    result = filter_navformer_predictions(Boxes(boxes), scores, labels)

    torch.testing.assert_close(result["scores"], torch.tensor([0.25, 0.80]))
    assert result["labels"].tolist() == [0, 1]


def test_gt_filter_keeps_vehicle_and_pedestrian_within_50m():
    info = {
        "gt_boxes": torch.tensor(
            [
                [10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [20.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [51.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [5.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            ]
        ).numpy(),
        "gt_names": ["vehicle", "pedestrian", "pedestrian", "traffic_cone"],
    }

    result = filter_agent_gt(info)

    assert result["centers"].tolist() == [
        [10.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
    ]
    assert result["labels"].tolist() == [0, 1]
