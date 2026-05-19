from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegHead(nn.Module):
    """
    SegFormer-aligned dense segmentation head.

    Output:
    - seg_logits: [B, C_seg, H, W]
    """

    def __init__(
        self,
        hidden_dim: int,
        C_seg: int = 6,
        seg_size: Tuple[int, int] = (64, 64),
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.C_seg = C_seg
        self.seg_size = seg_size
        self.base_channels = base_channels
        self.seed_h = 8
        self.seed_w = 8

        self.seed_projector = nn.Sequential(
            nn.Linear(hidden_dim, base_channels * self.seed_h * self.seed_w),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, C_seg, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, seg_queries: torch.Tensor) -> torch.Tensor:
        batch_size = seg_queries.shape[0]
        pooled = seg_queries.mean(dim=1)
        seed = self.seed_projector(pooled).view(
            batch_size,
            self.base_channels,
            self.seed_h,
            self.seed_w,
        )
        seg_logits = self.decoder(seed)
        return F.interpolate(
            seg_logits,
            size=self.seg_size,
            mode="bilinear",
            align_corners=False,
        )


class AgentHead(nn.Module):
    """
    Single-frame StreamPETR-style object head.

    Outputs:
    - agent_cls_logits: [B, N_agent, C_agent + 1]
    - agent_boxes: [B, N_agent, D_box]
    """

    def __init__(
        self,
        hidden_dim: int,
        C_agent: int = 10,
        D_box: int = 8,
    ) -> None:
        super().__init__()
        if D_box < 8:
            raise ValueError("D_box must be at least 8")

        self.C_agent = C_agent
        self.D_box = D_box
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, C_agent + 1),
        )
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, D_box),
        )

    def forward(self, agent_queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        agent_cls_logits = self.cls_head(agent_queries)
        raw_boxes = self.box_head(agent_queries)

        box_center_size = raw_boxes[..., :6].sigmoid()
        box_yaw = F.normalize(raw_boxes[..., 6:8], dim=-1)
        if self.D_box == 8:
            agent_boxes = torch.cat([box_center_size, box_yaw], dim=-1)
        else:
            extra = raw_boxes[..., 8:]
            agent_boxes = torch.cat([box_center_size, box_yaw, extra], dim=-1)
        return agent_cls_logits, agent_boxes


class MapHead(nn.Module):
    """
    MapTRv2-style map-element head.

    Outputs:
    - map_cls_logits: [B, N_map, C_map + 1]
    - map_points: [B, N_map, P, 2]
    """

    def __init__(
        self,
        hidden_dim: int,
        C_map: int = 4,
        P: int = 20,
    ) -> None:
        super().__init__()
        self.C_map = C_map
        self.P = P
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, C_map + 1),
        )
        self.point_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, P * 2),
        )

    def forward(self, map_queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        map_cls_logits = self.cls_head(map_queries)
        map_points = self.point_head(map_queries).view(
            map_queries.shape[0],
            map_queries.shape[1],
            self.P,
            2,
        )
        map_points = map_points.sigmoid()
        return map_cls_logits, map_points


class OCCHead(nn.Module):
    """
    FlashOCC-style voxel semantic head.

    The previous version pooled all occupancy queries with a simple mean before
    decoding, which erased query-to-query differences too early. This version
    keeps the module lightweight but uses learned query-to-seed aggregation so
    different occupancy queries can contribute differently to different seed
    cells. It is still not a full FlashOCC reproduction, only a better OCC
    decoder for the current QUEST prototype stage.

    Output:
    - occ_logits: [B, C_occ, X, Y, Z]
    """

    def __init__(
        self,
        hidden_dim: int,
        C_occ: int = 4,
        occ_size: Tuple[int, int, int] = (64, 64, 16),
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.C_occ = C_occ
        self.occ_size = occ_size
        self.base_channels = base_channels
        self.seed_x = 4
        self.seed_y = 4
        self.seed_z = 2
        self.num_seed_cells = self.seed_x * self.seed_y * self.seed_z

        self.query_value_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, base_channels),
        )
        self.query_seed_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_seed_cells),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(base_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(16, C_occ, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, occ_queries: torch.Tensor) -> torch.Tensor:
        batch_size = occ_queries.shape[0]
        query_values = self.query_value_proj(occ_queries)
        seed_scores = self.query_seed_score(occ_queries)
        seed_weights = torch.softmax(seed_scores, dim=1)
        seed = torch.einsum("bns,bnc->bsc", seed_weights, query_values)
        seed = seed.transpose(1, 2).contiguous().view(
            batch_size,
            self.base_channels,
            self.seed_z,
            self.seed_y,
            self.seed_x,
        )
        occ_logits = self.decoder(seed)
        occ_logits = F.interpolate(
            occ_logits,
            size=(self.occ_size[2], self.occ_size[1], self.occ_size[0]),
            mode="trilinear",
            align_corners=False,
        )
        return occ_logits.permute(0, 1, 4, 3, 2).contiguous()
