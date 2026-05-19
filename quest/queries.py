from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class StructuredQueryDecoder(nn.Module):
    """
    Multi-task decoder with four explicit task query groups:
    - segmentation queries
    - agent queries
    - map queries
    - occupancy queries
    """

    def __init__(
        self,
        backbone_dim: int,
        hidden_dim: int = 256,
        num_seg_queries: int = 32,
        num_agent_queries: int = 100,
        num_map_queries: int = 50,
        num_occ_queries: int = 50,
        num_decoder_layers: int = 2,
        num_attention_heads: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_seg_queries = num_seg_queries
        self.num_agent_queries = num_agent_queries
        self.num_map_queries = num_map_queries
        self.num_occ_queries = num_occ_queries

        self.memory_proj = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.seg_query = nn.Parameter(torch.randn(num_seg_queries, hidden_dim) * 0.02)
        self.agent_query = nn.Parameter(
            torch.randn(num_agent_queries, hidden_dim) * 0.02
        )
        self.map_query = nn.Parameter(torch.randn(num_map_queries, hidden_dim) * 0.02)
        self.occ_query = nn.Parameter(torch.randn(num_occ_queries, hidden_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
        )

    def _expand_queries(self, batch_size: int) -> torch.Tensor:
        return torch.cat(
            [
                self.seg_query.unsqueeze(0).expand(batch_size, -1, -1),
                self.agent_query.unsqueeze(0).expand(batch_size, -1, -1),
                self.map_query.unsqueeze(0).expand(batch_size, -1, -1),
                self.occ_query.unsqueeze(0).expand(batch_size, -1, -1),
            ],
            dim=1,
        )

    def forward(self, backbone_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = backbone_tokens.shape[0]
        memory = self.memory_proj(backbone_tokens)
        all_queries = self._expand_queries(batch_size)
        decoded_queries = self.decoder(tgt=all_queries, memory=memory)

        split_sizes = [
            self.num_seg_queries,
            self.num_agent_queries,
            self.num_map_queries,
            self.num_occ_queries,
        ]
        seg_queries, agent_queries, map_queries, occ_queries = torch.split(
            decoded_queries,
            split_sizes,
            dim=1,
        )
        return {
            "seg_queries": seg_queries,
            "agent_queries": agent_queries,
            "map_queries": map_queries,
            "occ_queries": occ_queries,
        }
