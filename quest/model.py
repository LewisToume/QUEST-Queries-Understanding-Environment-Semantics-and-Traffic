from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .backbone import FrozenDINOv2Backbone
from .heads import AgentHead, MapHead, OCCHead, SegHead
from .queries import StructuredQueryDecoder
from .utils import load_yaml_config, project_root


class QUESTModel(nn.Module):
    """
    QUEST research skeleton aligned with four local expert projects:
    - SegFormer: dense segmentation
    - StreamPETR: query-based object detection
    - MapTRv2: map element class + point set
    - FlashOCC: voxel semantic occupancy
    """

    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-small",
        local_backbone_dir: str = "weights/dinov2-small",
        hidden_dim: int = 256,
        num_seg_queries: int = 32,
        N_agent: int = 100,
        num_occ_queries: int = 50,
        C_seg: int = 6,
        seg_size: Tuple[int, int] = (64, 64),
        C_agent: int = 10,
        D_box: int = 8,
        N_map: int = 50,
        C_map: int = 4,
        P: int = 20,
        C_occ: int = 4,
        X: int = 64,
        Y: int = 64,
        Z: int = 16,
    ) -> None:
        super().__init__()

        self.backbone = FrozenDINOv2Backbone(
            backbone_name=backbone_name,
            local_backbone_dir=local_backbone_dir,
        )
        self.decoder = StructuredQueryDecoder(
            backbone_dim=self.backbone.hidden_dim,
            hidden_dim=hidden_dim,
            num_seg_queries=num_seg_queries,
            num_agent_queries=N_agent,
            num_map_queries=N_map,
            num_occ_queries=num_occ_queries,
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

    @classmethod
    def from_yaml(cls, config_path: str | None = None) -> "QUESTModel":
        path = project_root() / "configs" / "model.yaml" if config_path is None else config_path
        config = load_yaml_config(path)
        return cls(**config["model"])

    def encode_image(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        backbone_output = self.backbone(images)
        patch_tokens = backbone_output[:, 1:, :]
        return {
            "backbone_tokens_with_cls": backbone_output,
            "backbone_patch_tokens": patch_tokens,
        }

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encode_image(images)
        decoded = self.decoder(encoded["backbone_patch_tokens"])
        agent_cls_logits, agent_boxes = self.agent_head(decoded["agent_queries"])
        map_cls_logits, map_points = self.map_head(decoded["map_queries"])
        return {
            "seg_logits": self.seg_head(decoded["seg_queries"]),
            "agent_cls_logits": agent_cls_logits,
            "agent_boxes": agent_boxes,
            "map_cls_logits": map_cls_logits,
            "map_points": map_points,
            "occ_logits": self.occ_head(decoded["occ_queries"]),
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

    dummy_image = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        encoded = model.encode_image(dummy_image)
        outputs = model(dummy_image)

    print("Backbone output shape:")
    print(f"  with CLS    : {tuple(encoded['backbone_tokens_with_cls'].shape)}")
    print(f"  patch only  : {tuple(encoded['backbone_patch_tokens'].shape)}")
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
