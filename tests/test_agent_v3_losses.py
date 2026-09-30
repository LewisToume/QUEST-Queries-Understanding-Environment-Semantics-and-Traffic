import torch

from quest.losses import compute_agent_loss
from quest.teacher_adapters import (
    box_observability_mask,
    empty_canonical_agent,
    merge_canonical_agent_targets,
)


def make_target():
    target = empty_canonical_agent(4)
    target["labels"][0] = 0
    target["boxes_metric"][0] = torch.tensor([0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0])
    target["velocity_mps"][0] = torch.tensor([1.0, 0.0, 0.0])
    target["scores"][0] = 0.8
    target["class_support_mask"] = torch.tensor([True, True, False, False])
    target["valid_mask"][0] = True
    return {key: value.unsqueeze(0) for key, value in target.items()}


def make_predictions():
    cls = torch.zeros(1, 2, 5, requires_grad=True)
    boxes = torch.tensor(
        [[[0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.0, 1.0],
          [0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.0, 1.0]]],
        requires_grad=True,
    )
    velocity = torch.zeros(1, 2, 3, requires_grad=True)
    return {
        "proposal_objectness_logits": torch.zeros(1, 4, requires_grad=True),
        "proposal_xy_offsets": torch.zeros(1, 4, 2, requires_grad=True),
        "agent_cls_logits": cls,
        "agent_boxes": boxes,
        "agent_velocity": velocity,
        "agent_cls_logits_layers": cls.unsqueeze(0),
        "agent_boxes_layers": boxes.unsqueeze(0),
        "agent_velocity_layers": velocity.unsqueeze(0),
    }


def test_v3_agent_losses_are_split_finite_and_backward():
    predictions = make_predictions()
    losses = compute_agent_loss(
        predictions,
        make_target(),
        {"bev_h": 2, "bev_w": 2, "aux_layer_weights": [1.0]},
    )
    for key in (
        "proposal_objectness_loss", "proposal_offset_loss", "agent_cls_loss",
        "agent_center_loss", "agent_size_loss", "agent_yaw_loss",
        "agent_velocity_loss", "agent_loss",
    ):
        assert torch.isfinite(losses[key])
    losses["agent_loss"].backward()
    assert predictions["proposal_objectness_logits"].grad is not None
    assert predictions["agent_cls_logits"].grad is not None


def test_unsupported_teacher_classes_receive_no_classification_gradient():
    predictions = make_predictions()
    losses = compute_agent_loss(
        predictions,
        make_target(),
        {"bev_h": 2, "bev_w": 2, "aux_layer_weights": [1.0]},
    )
    losses["agent_loss"].backward()
    gradient = predictions["agent_cls_logits"].grad
    assert gradient[..., 2:4].eq(0).all()
    assert gradient[..., [0, 1, 4]].abs().sum() > 0


def test_proposal_warmup_disables_decoder_losses_only():
    losses = compute_agent_loss(
        make_predictions(),
        make_target(),
        {"bev_h": 2, "bev_w": 2, "aux_layer_weights": [1.0]},
        decoder_enabled=False,
    )
    assert losses["proposal_loss"] > 0
    assert losses["decoder_agent_loss"] == 0
    torch.testing.assert_close(losses["agent_loss"], losses["proposal_loss"])


def test_current_frame_box_observability_uses_positive_depth_and_image_bounds():
    projection = torch.tensor(
        [[10.0, 0.0, 50.0, 0.0], [0.0, 10.0, 50.0, 0.0],
         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    ).unsqueeze(0)
    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 2.0, 2.0, 2.0, 0.0],
         [0.0, 0.0, -10.0, 2.0, 2.0, 2.0, 0.0]]
    )
    assert box_observability_mask(boxes, projection, [(100, 100)]).tolist() == [True, False]


def test_hybrid_merge_keeps_hard_gt_and_adds_nonduplicate_teacher_targets():
    hard = empty_canonical_agent(4)
    hard["class_support_mask"][:] = True
    hard["labels"][0] = 0
    hard["boxes_metric"][0, :3] = torch.tensor([0.0, 0.0, 0.0])
    hard["scores"][0] = 1.0
    hard["valid_mask"][0] = True
    teacher = empty_canonical_agent(4)
    teacher["class_support_mask"][:2] = True
    teacher["labels"][:2] = torch.tensor([0, 1])
    teacher["boxes_metric"][0, :3] = torch.tensor([1.0, 0.0, 0.0])
    teacher["boxes_metric"][1, :3] = torch.tensor([10.0, 0.0, 0.0])
    teacher["scores"][:2] = 0.8
    teacher["valid_mask"][:2] = True
    merged = merge_canonical_agent_targets(hard, teacher, max_instances=4)
    assert merged["labels"][merged["valid_mask"]].tolist() == [0, 1]
    assert merged["scores"][0] == 1.0
