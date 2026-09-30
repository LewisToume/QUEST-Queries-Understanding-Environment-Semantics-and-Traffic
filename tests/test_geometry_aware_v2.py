from unittest.mock import patch

import torch

from quest.geometry import GeometryAwareBEVLift
from quest.heads import AgentHead
from quest.model import QUESTModel, load_quest_v2_checkpoint
from quest.queries import AgentDecoder


def test_lidar_to_camera_projection_uses_inverse_extrinsic_and_visibility():
    lift = GeometryAwareBEVLift(
        hidden_dim=4,
        bev_h=1,
        bev_w=1,
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        z_anchors=(-1.0, 1.0),
    )
    intrinsic = torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]
    ).reshape(1, 1, 3, 3)
    sensor2lidar = torch.eye(4).reshape(1, 1, 4, 4)
    sensor2lidar[0, 0, 0, 3] = 1.0

    grid, visible, depth = lift.project_reference_points(
        intrinsic, sensor2lidar, image_size=(4, 4)
    )

    assert depth[0, 0, 0].tolist() == [-1.0, 1.0]
    assert visible[0, 0, 0].tolist() == [False, True]
    torch.testing.assert_close(grid[0, 0, 0, 1], torch.tensor([-0.25, 0.25]))

    outside_intrinsic = intrinsic.clone()
    outside_intrinsic[..., 0, 2] = 10.0
    _, outside_visible, _ = lift.project_reference_points(
        outside_intrinsic, sensor2lidar, image_size=(4, 4)
    )
    assert not bool(outside_visible.any())


def test_geometry_lift_samples_visible_feature_and_returns_bev_tokens():
    lift = GeometryAwareBEVLift(
        hidden_dim=4,
        bev_h=1,
        bev_w=1,
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        z_anchors=(1.0,),
    )
    with torch.no_grad():
        lift.bev_embedding.zero_()
        lift.metric_position.zero_()
        for parameter in lift.ego_mlp.parameters():
            parameter.zero_()
    features = torch.ones(1, 1, 4, 2, 2)
    intrinsic = torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]
    ).reshape(1, 1, 3, 3)
    extrinsic = torch.eye(4).reshape(1, 1, 4, 4)

    tokens = lift(features, intrinsic, extrinsic, torch.zeros(1, 9), (4, 4))

    assert tuple(tokens.shape) == (1, 1, 4)
    torch.testing.assert_close(tokens, torch.ones_like(tokens))


def test_agent_spatial_references_and_reference_relative_box_center():
    decoder = AgentDecoder(
        hidden_dim=32,
        num_queries=100,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
    )
    references = decoder.reference_xyz.detach()
    assert tuple(references.shape) == (100, 3)
    assert torch.unique(references[:, 0]).numel() == 10
    assert torch.unique(references[:, 1]).numel() == 10
    assert references[:, 2].eq(0.5).all()

    head = AgentHead(hidden_dim=32, C_agent=4, D_box=8)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
    queries = torch.zeros(1, 100, 32)
    _, boxes, velocity = head(queries, references.unsqueeze(0))
    torch.testing.assert_close(boxes[0, :, :3], references)
    assert boxes[..., 3:6].eq(0.5).all()
    torch.testing.assert_close(
        torch.linalg.vector_norm(boxes[..., 6:8], dim=-1), torch.ones(1, 100)
    )
    assert velocity.eq(0).all()


def test_v1_or_unversioned_checkpoint_is_rejected():
    model = torch.nn.Linear(2, 2)
    try:
        load_quest_v2_checkpoint(model, {"model_state_dict": model.state_dict()})
    except ValueError as error:
        assert "refusing to load" in str(error)
    else:
        raise AssertionError("unversioned checkpoint was accepted")


def test_v2_trainable_parameter_count_is_in_target_range():
    class FrozenBackbone(torch.nn.Module):
        hidden_dim = 384
        patch_size = 14

        def __init__(self):
            super().__init__()
            self.placeholder = torch.nn.Parameter(
                torch.zeros(1), requires_grad=False
            )

    with patch("quest.model.FrozenDINOv2Backbone", return_value=FrozenBackbone()):
        model = QUESTModel()

    summary = model.parameter_summary()
    assert 20_000_000 <= summary["trainable"] <= 30_000_000
    assert summary["total"] == summary["trainable"] + summary["frozen"]
