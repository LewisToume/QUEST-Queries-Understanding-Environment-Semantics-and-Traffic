from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn

from .backbone import FrozenDINOv2Backbone
from .bev import BEVEncoder
from .geometry import GeometryAwareBEVLift
from .heads import AgentHead, DepthHead, MapHead, SegHead
from .queries import AgentDecoder, MapDecoder
from .utils import load_yaml_config, project_root


QUEST_ARCHITECTURE_VERSION = 2
DEFAULT_CAMERA_NAMES = (
    "CAM_F0",
    "CAM_L0",
    "CAM_R0",
    "CAM_L1",
    "CAM_R1",
    "CAM_L2",
    "CAM_R2",
    "CAM_B0",
)


class QUESTModel(nn.Module):
    """QUEST V2 geometry-aware 8-camera perception student."""

    def __init__(
        self,
        architecture_version: int = QUEST_ARCHITECTURE_VERSION,
        backbone_name: str = "facebook/dinov2-small",
        local_backbone_dir: str = "weights/dinov2-small",
        hidden_dim: int = 384,
        camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
        bev_h: int = 32,
        bev_w: int = 32,
        x_range: tuple[float, float] = (-50.0, 50.0),
        y_range: tuple[float, float] = (-50.0, 50.0),
        z_anchors: Sequence[float] = (-1.0, 0.0, 1.0),
        bev_layers: int = 4,
        bev_attention_heads: int = 8,
        bev_ffn_dim: int = 1536,
        agent_decoder_layers: int = 4,
        map_decoder_layers: int = 2,
        decoder_attention_heads: int = 8,
        decoder_ffn_dim: int = 1536,
        dropout: float = 0.1,
        N_agent: int = 100,
        N_map: int = 50,
        C_seg: int = 6,
        seg_size: Tuple[int, int] = (64, 64),
        depth_size: Tuple[int, int] = (64, 64),
        C_agent: int = 4,
        D_box: int = 8,
        C_map: int = 4,
        P: int = 20,
    ) -> None:
        super().__init__()
        if architecture_version != QUEST_ARCHITECTURE_VERSION:
            raise ValueError(
                f"QUESTModel only supports architecture_version={QUEST_ARCHITECTURE_VERSION}"
            )
        if D_box != 8:
            raise ValueError("QUEST V2 Agent boxes require D_box=8")
        self.architecture_version = architecture_version
        self.hidden_dim = hidden_dim
        self.camera_names = tuple(camera_names)
        self.num_cameras = len(self.camera_names)
        self.backbone = FrozenDINOv2Backbone(backbone_name, local_backbone_dir)
        if self.backbone.hidden_dim != hidden_dim:
            raise ValueError(
                "QUEST V2 uses DINOv2-S native features directly: "
                f"backbone hidden={self.backbone.hidden_dim}, hidden_dim={hidden_dim}"
            )
        self.geometry_lift = GeometryAwareBEVLift(
            hidden_dim=hidden_dim,
            bev_h=bev_h,
            bev_w=bev_w,
            x_range=tuple(x_range),
            y_range=tuple(y_range),
            z_anchors=z_anchors,
        )
        self.bev_encoder = BEVEncoder(
            hidden_dim=hidden_dim,
            bev_h=bev_h,
            bev_w=bev_w,
            num_layers=bev_layers,
            num_attention_heads=bev_attention_heads,
            ffn_dim=bev_ffn_dim,
            dropout=dropout,
        )
        self.agent_decoder = AgentDecoder(
            hidden_dim=hidden_dim,
            num_queries=N_agent,
            num_layers=agent_decoder_layers,
            num_heads=decoder_attention_heads,
            ffn_dim=decoder_ffn_dim,
            dropout=dropout,
        )
        self.map_decoder = MapDecoder(
            hidden_dim=hidden_dim,
            num_queries=N_map,
            num_layers=map_decoder_layers,
            num_heads=decoder_attention_heads,
            ffn_dim=decoder_ffn_dim,
            dropout=dropout,
        )
        self.seg_head = SegHead(self.backbone.hidden_dim, C_seg, seg_size)
        self.depth_head = DepthHead(self.backbone.hidden_dim, depth_size)
        self.agent_head = AgentHead(hidden_dim, C_agent, D_box)
        self.map_head = MapHead(hidden_dim, C_map, P)

    @classmethod
    def from_yaml(cls, config_path: str | None = None) -> "QUESTModel":
        path = project_root() / "configs" / "model.yaml" if config_path is None else config_path
        return cls(**load_yaml_config(path)["model"])

    def parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {"total": total, "frozen": total - trainable, "trainable": trainable}

    def encode_image(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None = None,
        extrinsics: torch.Tensor | None = None,
        ego_state: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | tuple[int, int]]:
        if images.ndim != 5:
            raise ValueError(
                f"images must be [B, N_camera, 3, H, W], got {tuple(images.shape)}"
            )
        batch_size, num_cameras, channels, height, width = images.shape
        if num_cameras != self.num_cameras:
            raise ValueError(f"expected {self.num_cameras} cameras, got {num_cameras}")
        if channels != 3:
            raise ValueError(f"expected RGB images, got {channels} channels")
        if height % self.backbone.patch_size or width % self.backbone.patch_size:
            raise ValueError(
                f"image size {(height, width)} must be divisible by DINO patch size "
                f"{self.backbone.patch_size}"
            )
        if intrinsics is None or extrinsics is None or ego_state is None:
            raise ValueError(
                "QUEST V2 requires intrinsics, sensor2lidar extrinsics, and ego_state"
            )
        if intrinsics.shape != (batch_size, num_cameras, 3, 3):
            raise ValueError(f"invalid intrinsics shape: {tuple(intrinsics.shape)}")
        if extrinsics.shape != (batch_size, num_cameras, 4, 4):
            raise ValueError(f"invalid extrinsics shape: {tuple(extrinsics.shape)}")
        if ego_state.shape != (batch_size, 9):
            raise ValueError(f"invalid ego_state shape: {tuple(ego_state.shape)}")

        flat_images = images.reshape(batch_size * num_cameras, channels, height, width)
        backbone_output = self.backbone(flat_images)
        patch_grid_size = (
            height // self.backbone.patch_size,
            width // self.backbone.patch_size,
        )
        patch_tokens = backbone_output[:, 1:, :].reshape(
            batch_size, num_cameras, -1, self.backbone.hidden_dim
        )
        if patch_tokens.shape[2] != patch_grid_size[0] * patch_grid_size[1]:
            raise ValueError(
                "DINO patch tokens do not match input grid: "
                f"tokens={patch_tokens.shape[2]} grid={patch_grid_size}"
            )
        camera_features = patch_tokens.permute(0, 1, 3, 2).reshape(
            batch_size,
            num_cameras,
            self.hidden_dim,
            patch_grid_size[0],
            patch_grid_size[1],
        )
        lifted_tokens = self.geometry_lift(
            camera_features,
            intrinsics,
            extrinsics,
            ego_state,
            image_size=(height, width),
        )
        bev_tokens, bev_features = self.bev_encoder(lifted_tokens)
        return {
            "backbone_patch_tokens": patch_tokens,
            "backbone_feature_maps": camera_features,
            "patch_grid_size": patch_grid_size,
            "lifted_bev_tokens": lifted_tokens,
            "bev_tokens": bev_tokens,
            "bev_features": bev_features,
        }

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None = None,
        extrinsics: torch.Tensor | None = None,
        ego_state: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode_image(images, intrinsics, extrinsics, ego_state)
        agent_queries, reference_xyz = self.agent_decoder(encoded["bev_tokens"])
        map_queries = self.map_decoder(encoded["bev_tokens"])
        agent_cls, agent_boxes, agent_velocity = self.agent_head(
            agent_queries, reference_xyz
        )
        map_cls, map_points = self.map_head(map_queries)
        return {
            "seg_logits": self.seg_head(
                encoded["backbone_patch_tokens"], encoded["patch_grid_size"]
            ),
            "depth": self.depth_head(
                encoded["backbone_patch_tokens"], encoded["patch_grid_size"]
            ),
            "agent_cls_logits": agent_cls,
            "agent_boxes": agent_boxes,
            "agent_velocity": agent_velocity,
            "map_cls_logits": map_cls,
            "map_points": map_points,
            "bev_features": encoded["bev_features"],
        }


def load_quest_v2_checkpoint(
    model: QUESTModel, checkpoint: Mapping[str, Any]
) -> Any:
    version = checkpoint.get("architecture_version")
    if version != QUEST_ARCHITECTURE_VERSION:
        raise ValueError(
            "refusing to load a V1 or unversioned checkpoint into QUEST V2; "
            f"expected architecture_version={QUEST_ARCHITECTURE_VERSION}, got {version}"
        )
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("QUEST V2 checkpoint must contain model_state_dict")
    return model.load_state_dict(state_dict)


QUEST_Model = QUESTModel


if __name__ == "__main__":
    model = QUESTModel.from_yaml()
    summary = model.parameter_summary()
    print(f"architecture_version: {model.architecture_version}")
    print(f"total parameters: {summary['total']:,}")
    print(f"frozen parameters: {summary['frozen']:,}")
    print(f"trainable parameters: {summary['trainable']:,}")
