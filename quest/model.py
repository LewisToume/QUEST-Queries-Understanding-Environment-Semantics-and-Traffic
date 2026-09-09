from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn

from .backbone import FrozenDINOv2Backbone
from .heads import AgentHead, FlowHead, MapHead, OCCHead, SegHead
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
    """Lightweight camera-aware fusion over shared DINOv2 tokens."""

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
            nn.Linear(backbone_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.camera_embed = nn.Embedding(num_cameras, hidden_dim)
        self.geometry_proj = nn.Sequential(
            nn.Linear(geometry_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _build_geometry(
        self,
        batch_size: int,
        num_cameras: int,
        device: torch.device,
        dtype: torch.dtype,
        intrinsics: torch.Tensor | None,
        extrinsics: torch.Tensor | None,
        ego_state: torch.Tensor | None,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if intrinsics is None:
            parts.append(torch.zeros(batch_size, num_cameras, 9, device=device, dtype=dtype))
        else:
            parts.append(intrinsics.to(device=device, dtype=dtype).reshape(batch_size, num_cameras, 9))

        if extrinsics is None:
            eye = torch.eye(4, device=device, dtype=dtype).reshape(1, 1, 16)
            parts.append(eye.expand(batch_size, num_cameras, 16))
        else:
            parts.append(extrinsics.to(device=device, dtype=dtype).reshape(batch_size, num_cameras, 16))

        if ego_state is None:
            parts.append(torch.zeros(batch_size, num_cameras, 9, device=device, dtype=dtype))
        else:
            ego = ego_state.to(device=device, dtype=dtype)
            if ego.ndim == 1:
                ego = ego.unsqueeze(0)
            ego = ego[:, :9]
            if ego.shape[-1] < 9:
                pad = torch.zeros(batch_size, 9 - ego.shape[-1], device=device, dtype=dtype)
                ego = torch.cat([ego, pad], dim=-1)
            parts.append(ego.unsqueeze(1).expand(batch_size, num_cameras, 9))

        geometry = torch.cat(parts, dim=-1)
        if geometry.shape[-1] != self.geometry_dim:
            raise ValueError(f"expected geometry_dim={self.geometry_dim}, got {geometry.shape[-1]}")
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

        device = patch_tokens.device
        dtype = patch_tokens.dtype
        memory = self.token_proj(patch_tokens)
        camera_ids = torch.arange(num_cameras, device=device)
        camera_bias = self.camera_embed(camera_ids).view(1, num_cameras, 1, self.hidden_dim)
        geometry = self._build_geometry(
            batch_size=batch_size,
            num_cameras=num_cameras,
            device=device,
            dtype=dtype,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            ego_state=ego_state,
        )
        geometry_bias = self.geometry_proj(geometry).unsqueeze(2)
        memory = memory + camera_bias + geometry_bias
        memory = memory.reshape(batch_size, num_cameras * num_tokens, self.hidden_dim)
        return self.encoder(memory)


class QUESTModel(nn.Module):
    """
    QUEST current-scene 8-camera perception model.

    Shared Frozen DINOv2 encodes each OpenScene camera view. A lightweight
    camera-aware Transformer fuses multi-view tokens before structured task
    queries decode Agent, Map, OCC, and Flow outputs.
    """

    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-small",
        local_backbone_dir: str = "weights/dinov2-small",
        hidden_dim: int = 256,
        camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
        fusion_layers: int = 1,
        fusion_attention_heads: int = 8,
        decoder_layers: int = 2,
        decoder_attention_heads: int = 8,
        num_seg_queries: int = 32,
        N_agent: int = 100,
        num_occ_queries: int = 50,
        num_flow_queries: int = 50,
        C_seg: int = 6,
        seg_size: Tuple[int, int] = (64, 64),
        C_agent: int = 10,
        D_box: int = 8,
        N_map: int = 50,
        C_map: int = 4,
        P: int = 20,
        C_occ: int = 4,
        C_flow: int = 2,
        X: int = 64,
        Y: int = 64,
        Z: int = 16,
    ) -> None:
        super().__init__()
        self.camera_names = tuple(camera_names)
        self.num_cameras = len(self.camera_names)

        self.backbone = FrozenDINOv2Backbone(
            backbone_name=backbone_name,
            local_backbone_dir=local_backbone_dir,
        )
        self.fusion = MultiViewFusion(
            backbone_dim=self.backbone.hidden_dim,
            hidden_dim=hidden_dim,
            num_cameras=self.num_cameras,
            num_layers=fusion_layers,
            num_attention_heads=fusion_attention_heads,
        )
        self.decoder = StructuredQueryDecoder(
            backbone_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_agent_queries=N_agent,
            num_map_queries=N_map,
            num_occ_queries=num_occ_queries,
            num_flow_queries=num_flow_queries,
            num_decoder_layers=decoder_layers,
            num_attention_heads=decoder_attention_heads,
        )

        self.seg_head = SegHead(
            hidden_dim=hidden_dim,
            C_seg=C_seg,
            seg_size=seg_size,
        )
        self.agent_head = AgentHead(
            hidden_dim=hidden_dim,
            C_agent=C_agent,
            D_box=D_box,
        )
        self.map_head = MapHead(
            hidden_dim=hidden_dim,
            C_map=C_map,
            P=P,
        )
        self.occ_head = OCCHead(
            hidden_dim=hidden_dim,
            C_occ=C_occ,
            occ_size=(X, Y, Z),
        )
        self.flow_head = FlowHead(
            hidden_dim=hidden_dim,
            C_flow=C_flow,
            flow_size=(X, Y, Z),
        )

    @classmethod
    def from_yaml(cls, config_path: str | None = None) -> "QUESTModel":
        path = project_root() / "configs" / "model.yaml" if config_path is None else config_path
        config = load_yaml_config(path)
        return cls(**config["model"])

    def encode_image(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None = None,
        extrinsics: torch.Tensor | None = None,
        ego_state: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError(f"images must be [B, 8, 3, H, W], got {tuple(images.shape)}")
        batch_size, num_cameras, channels, height, width = images.shape
        if num_cameras != self.num_cameras:
            raise ValueError(f"expected {self.num_cameras} cameras, got {num_cameras}")
        if channels != 3:
            raise ValueError(f"expected RGB images with 3 channels, got {channels}")

        flat_images = images.reshape(batch_size * num_cameras, channels, height, width)
        backbone_output = self.backbone(flat_images)
        patch_tokens = backbone_output[:, 1:, :].reshape(batch_size, num_cameras, -1, self.backbone.hidden_dim)
        fused_tokens = self.fusion(
            patch_tokens,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            ego_state=ego_state,
        )
        return {
            "backbone_tokens_with_cls": backbone_output.reshape(batch_size, num_cameras, -1, self.backbone.hidden_dim),
            "backbone_patch_tokens": patch_tokens,
            "fused_tokens": fused_tokens,
        }

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None = None,
        extrinsics: torch.Tensor | None = None,
        ego_state: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode_image(
            images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            ego_state=ego_state,
        )
        decoded = self.decoder(encoded["fused_tokens"])
        agent_cls_logits, agent_boxes, agent_velocity = self.agent_head(decoded["agent_queries"])
        map_cls_logits, map_points = self.map_head(decoded["map_queries"])
        return {
            "agent_cls_logits": agent_cls_logits,
            "agent_boxes": agent_boxes,
            "agent_velocity": agent_velocity,
            "map_cls_logits": map_cls_logits,
            "map_points": map_points,
            "occ_logits": self.occ_head(decoded["occ_queries"]),
            "flow_logits": self.flow_head(decoded["flow_queries"]),
        }


QUEST_Model = QUESTModel


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = QUESTModel.from_yaml().to(device).eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print("=" * 72)
    print("QUEST model smoke test")
    print(f"device                 : {device}")
    print(f"total params           : {total_params / 1e6:.2f} M")
    print(f"trainable params       : {trainable_params / 1e6:.2f} M")
    print(f"frozen backbone params : {frozen_params / 1e6:.2f} M")
    print("=" * 72)

    dummy_image = torch.randn(1, model.num_cameras, 3, 224, 224, device=device)
    dummy_intrinsics = torch.eye(3, device=device).view(1, 1, 3, 3).expand(1, model.num_cameras, 3, 3)
    dummy_extrinsics = torch.eye(4, device=device).view(1, 1, 4, 4).expand(1, model.num_cameras, 4, 4)
    dummy_ego_state = torch.zeros(1, 9, device=device)
    with torch.no_grad():
        encoded = model.encode_image(dummy_image, dummy_intrinsics, dummy_extrinsics, dummy_ego_state)
        outputs = model(dummy_image, dummy_intrinsics, dummy_extrinsics, dummy_ego_state)

    print("Backbone output shape:")
    print(f"  with CLS    : {tuple(encoded['backbone_tokens_with_cls'].shape)}")
    print(f"  patch only  : {tuple(encoded['backbone_patch_tokens'].shape)}")
    print(f"  fused tokens: {tuple(encoded['fused_tokens'].shape)}")
    print()
    print("Task head output shape:")
    for name, value in outputs.items():
        print(f"  {name:<17}: {tuple(value.shape)}")

    if torch.cuda.is_available():
        allocated_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
        reserved_mb = torch.cuda.memory_reserved(device) / 1024 / 1024
        print()
        print("CUDA memory:")
        print(f"  allocated : {allocated_mb:.2f} MB")
        print(f"  reserved  : {reserved_mb:.2f} MB")

    print("=" * 72)
