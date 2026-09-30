from unittest.mock import patch

import torch

from quest.geometry import GeometryAwareBEVLift
from quest.heads import AgentHead
from quest.model import QUESTModel, load_quest_v3_checkpoint
from quest.queries import AgentDecoder


class FrozenBackbone(torch.nn.Module):
    hidden_dim = 384
    patch_size = 14

    def __init__(self):
        super().__init__()
        self.placeholder = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, images):
        patch_count = (images.shape[-2] // 14) * (images.shape[-1] // 14)
        mean = images.mean(dim=(1, 2, 3), keepdim=False)
        tokens = mean[:, None, None].expand(-1, patch_count + 1, self.hidden_dim)
        return tokens.contiguous()


def make_small_model():
    with patch("quest.model.FrozenDINOv2Backbone", return_value=FrozenBackbone()):
        return QUESTModel(
            bev_h=4,
            bev_w=4,
            bev_layers=1,
            agent_decoder_layers=2,
            map_decoder_layers=1,
            N_agent=4,
            N_map=2,
            seg_size=(4, 4),
            depth_size=(4, 4),
            dropout=0.0,
        )


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


def test_agent_decoder_has_no_learned_query_identity_and_returns_layers():
    decoder = AgentDecoder(
        hidden_dim=32,
        num_queries=4,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
    )
    names = {name for name, _ in decoder.named_parameters()}
    assert not any("query_embedding" in name for name in names)
    assert not any("reference_xyz" in name for name in names)
    memory = torch.randn(1, 16, 32)
    features = memory.transpose(1, 2).reshape(1, 32, 4, 4)
    references = torch.tensor(
        [[[0.125, 0.125, 0.5], [0.375, 0.375, 0.5], [0.625, 0.625, 0.5], [0.875, 0.875, 0.5]]]
    )
    layers = decoder(memory, features, references)
    assert len(layers) == 2
    assert all(tuple(layer.shape) == (1, 4, 32) for layer in layers)


def test_reference_relative_agent_head():
    head = AgentHead(hidden_dim=32, C_agent=4, D_box=8)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
    references = torch.tensor([[[0.2, 0.3, 0.5]]])
    _, boxes, velocity = head(torch.zeros(1, 1, 32), references)
    torch.testing.assert_close(boxes[..., :3], references)
    assert boxes[..., 3:6].eq(0.5).all()
    torch.testing.assert_close(boxes[..., 6:8], torch.tensor([[[0.0, 1.0]]]))
    assert velocity.eq(0).all()


def test_agent_outputs_depend_on_bev_and_proposals_are_spatial():
    torch.manual_seed(7)
    model = make_small_model().eval()
    real = torch.randn(1, 16, 384)
    zero = torch.zeros_like(real)
    real_features = real.transpose(1, 2).reshape(1, 384, 4, 4)
    zero_features = torch.zeros_like(real_features)
    shuffled = real[:, torch.randperm(16)]
    with torch.no_grad():
        output_real = model.forward_agent(real, real_features)
        output_zero = model.forward_agent(zero, zero_features)
        output_shuffled = model.forward_agent(shuffled, real_features)
    for key in ("proposal_objectness_logits", "agent_cls_logits", "agent_boxes"):
        assert not torch.allclose(output_real[key], output_zero[key])
        assert not torch.allclose(output_real[key], output_shuffled[key])
    assert float(output_real["proposal_spatial_std"].min()) > 0.0


def test_small_full_forward_has_v3_shapes_and_layer_outputs():
    model = make_small_model().eval()
    images = torch.randn(1, 8, 3, 28, 28)
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 8, 3, 3).clone()
    extrinsics = torch.eye(4).reshape(1, 1, 4, 4).expand(1, 8, 4, 4).clone()
    with torch.no_grad():
        output = model(images, intrinsics, extrinsics, torch.zeros(1, 9))
    assert tuple(output["proposal_objectness_logits"].shape) == (1, 16)
    assert tuple(output["agent_cls_logits"].shape) == (1, 4, 5)
    assert tuple(output["agent_cls_logits_layers"].shape) == (2, 1, 4, 5)
    assert tuple(output["agent_boxes"].shape) == (1, 4, 8)


def test_v1_v2_and_unversioned_checkpoints_are_rejected():
    model = torch.nn.Linear(2, 2)
    for version in (None, 1, 2):
        checkpoint = {"model_state_dict": model.state_dict()}
        if version is not None:
            checkpoint["architecture_version"] = version
        try:
            load_quest_v3_checkpoint(model, checkpoint)
        except ValueError as error:
            assert "refusing to load" in str(error)
        else:
            raise AssertionError(f"checkpoint version {version} was accepted")


def test_v3_trainable_parameter_count_remains_near_v2_capacity():
    with patch("quest.model.FrozenDINOv2Backbone", return_value=FrozenBackbone()):
        model = QUESTModel()
    summary = model.parameter_summary()
    assert 18_000_000 <= summary["trainable"] <= 30_000_000
    assert summary["total"] == summary["trainable"] + summary["frozen"]
