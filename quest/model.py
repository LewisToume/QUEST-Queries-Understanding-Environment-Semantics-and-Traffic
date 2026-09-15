from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn

from .backbone import FrozenDINOv2Backbone
from .bev import BEVEncoder
from .heads import AgentHead, DepthHead, MapHead, SegHead
from .queries import StructuredQueryDecoder
from .utils import load_yaml_config, project_root

DEFAULT_CAMERA_NAMES = (
    "CAM_F0",
    "CAM_B0",
    "CAM_L0",
    "CAM_L1",
    "CAM_L2",
    "CAM_R0",
    "CAM_R1",
    "CAM_R2",
)


class MultiViewFusion(nn.Module):
    """Fuse shared DINO features with camera identity and calibrated geometry."""

    def __init__(
        self,
        backbone_dim: int,
        hidden_dim: int,
        num_cameras: int = 8,
        num_layers: int = 1,
        num_attention_heads: int = 8,
        geometry_dim: int = 34,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.hidden_dim = hidden_dim
        self.geometry_dim = geometry_dim
        self.token_proj = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.camera_embed = nn.Embedding(num_cameras, hidden_dim)
        self.geometry_proj = nn.Sequential(
            nn.Linear(geometry_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def _geometry(
        self,
        batch_size: int,
        num_cameras: int,
        device: torch.device,
        dtype: torch.dtype,
        intrinsics: torch.Tensor | None,
        extrinsics: torch.Tensor | None,
        ego_state: torch.Tensor | None,
    ) -> torch.Tensor:
        if intrinsics is None:
            intrinsics_flat = torch.zeros(
                batch_size, num_cameras, 9, device=device, dtype=dtype
            )
        else:
            intrinsics_flat = intrinsics.to(device=device, dtype=dtype).reshape(
                batch_size, num_cameras, 9
            )
        if extrinsics is None:
            extrinsics_flat = torch.eye(4, device=device, dtype=dtype).reshape(1, 1, 16)
            extrinsics_flat = extrinsics_flat.expand(batch_size, num_cameras, -1)
        else:
            extrinsics_flat = extrinsics.to(device=device, dtype=dtype).reshape(
                batch_size, num_cameras, 16
            )
        if ego_state is None:
            ego = torch.zeros(batch_size, 9, device=device, dtype=dtype)
        else:
            ego = ego_state.to(device=device, dtype=dtype)
            if ego.ndim == 1:
                ego = ego.unsqueeze(0)
            if ego.shape[-1] < 9:
                ego = torch.cat(
                    [
                        ego,
                        torch.zeros(
                            batch_size, 9 - ego.shape[-1], device=device, dtype=dtype
                        ),
                    ],
                    dim=-1,
                )
            ego = ego[:, :9]
        geometry = torch.cat(
            [
                intrinsics_flat,
                extrinsics_flat,
                ego.unsqueeze(1).expand(batch_size, num_cameras, -1),
            ],
            dim=-1,
        )
        if geometry.shape[-1] != self.geometry_dim:
            raise ValueError(
                f"expected geometry_dim={self.geometry_dim}, got {geometry.shape[-1]}"
            )
        return geometry

    def forward(
        self,
        patch_tokens: torch.Tensor,
        intrinsics: torch.Tensor | None = None,
        extrinsics: torch.Tensor | None = None,
        ego_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_cameras, num_tokens, _ = patch_tokens.shape
        if num_cameras != self.num_cameras:
            raise ValueError(f"expected {self.num_cameras} cameras, got {num_cameras}")
        camera_ids = torch.arange(num_cameras, device=patch_tokens.device)
        camera_bias = self.camera_embed(camera_ids).reshape(
            1, num_cameras, 1, self.hidden_dim
        )
        geometry = self._geometry(
            batch_size,
            num_cameras,
            patch_tokens.device,
            patch_tokens.dtype,
            intrinsics,
            extrinsics,
            ego_state,
        )
        tokens = self.token_proj(patch_tokens)
        tokens = tokens + camera_bias + self.geometry_proj(geometry).unsqueeze(2)
        tokens = tokens.reshape(batch_size, num_cameras * num_tokens, self.hidden_dim)
        return self.encoder(tokens)


class QUESTModel(nn.Module):
    """8-camera multi-task perception model with shared BEV latent and four outputs:
    Semantic Segmentation, Depth, Agent, Vector Map.
    """

    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-small",
        local_backbone_dir: str = "weights/dinov2-small",
        hidden_dim: int = 256,
        camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
        fusion_layers: int = 1,
        fusion_attention_heads: int = 8,
        bev_h: int = 32,
        bev_w: int = 32,
        bev_layers: int = 2,
        bev_attention_heads: int = 8,
        decoder_layers: int = 2,
        decoder_attention_heads: int = 8,
        N_agent: int = 100,
        N_map: int = 50,
        C_seg: int = 6,
        seg_size: Tuple[int, int] = (64, 64),
        depth_size: Tuple[int, int] = (64, 64),
        C_agent: int = 10,
        D_box: int = 8,
        C_map: int = 4,
        P: int = 20,
    ) -> None:
        super().__init__()
        self.camera_names = tuple(camera_names)
        self.num_cameras = len(self.camera_names)
        self.backbone = FrozenDINOv2Backbone(backbone_name, local_backbone_dir)
        self.fusion = MultiViewFusion(
            self.backbone.hidden_dim,
            hidden_dim,
            self.num_cameras,
            fusion_layers,
            fusion_attention_heads,
        )
        self.bev_encoder = BEVEncoder(
            hidden_dim,
            bev_h,
            bev_w,
            bev_layers,
            bev_attention_heads,
        )
        self.decoder = StructuredQueryDecoder(
            hidden_dim,
            hidden_dim,
            N_agent,
            N_map,
            decoder_layers,
            decoder_attention_heads,
        )
        self.seg_head = SegHead(self.backbone.hidden_dim, C_seg, seg_size)
        self.depth_head = DepthHead(self.backbone.hidden_dim, depth_size)
        self.agent_head = AgentHead(hidden_dim, C_agent, D_box)
        self.map_head = MapHead(hidden_dim, C_map, P)

    @classmethod
    def from_yaml(cls, config_path: str | None = None) -> "QUESTModel":
        path = project_root() / "configs" / "model.yaml" if config_path is None else config_path
        return cls(**load_yaml_config(path)["model"])

    def encode_image(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None = None,
        extrinsics: torch.Tensor | None = None,
        ego_state: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError(f"images must be [B, 8, 3, H, W], got {tuple(images.shape)}")
        batch_size, num_cameras, channels, height, width = images.shape
        if num_cameras != self.num_cameras:
            raise ValueError(f"expected {self.num_cameras} cameras, got {num_cameras}")
        if channels != 3:
            raise ValueError(f"expected RGB images, got {channels} channels")
        flat_images = images.reshape(batch_size * num_cameras, channels, height, width)
        backbone_output = self.backbone(flat_images)
        patch_tokens = backbone_output[:, 1:, :].reshape(
            batch_size, num_cameras, -1, self.backbone.hidden_dim
        )
        fused_tokens = self.fusion(
            patch_tokens, intrinsics=intrinsics, extrinsics=extrinsics, ego_state=ego_state
        )
        bev_tokens, bev_features = self.bev_encoder(fused_tokens)
        return {
            "backbone_patch_tokens": patch_tokens,
            "fused_tokens": fused_tokens,
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
        decoded = self.decoder(encoded["bev_tokens"])
        agent_cls, agent_boxes, agent_velocity = self.agent_head(
            decoded["agent_queries"]
        )
        map_cls, map_points = self.map_head(decoded["map_queries"])
        return {
            "seg_logits": self.seg_head(encoded["backbone_patch_tokens"]),
            "depth": self.depth_head(encoded["backbone_patch_tokens"]),
            "agent_cls_logits": agent_cls,
            "agent_boxes": agent_boxes,
            "agent_velocity": agent_velocity,
            "map_cls_logits": map_cls,
            "map_points": map_points,
            "bev_features": encoded["bev_features"],
        }


QUEST_Model = QUESTModel


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QUESTModel.from_yaml().to(device).eval()
    images = torch.randn(1, 8, 3, 224, 224, device=device)
    intrinsics = torch.eye(3, device=device).reshape(1, 1, 3, 3).expand(1, 8, 3, 3)
    extrinsics = torch.eye(4, device=device).reshape(1, 1, 4, 4).expand(1, 8, 4, 4)
    ego_state = torch.zeros(1, 9, device=device)
    with torch.no_grad():
        outputs = model(images, intrinsics, extrinsics, ego_state)
    for name, value in outputs.items():
        print(f"{name}: {tuple(value.shape)}")
