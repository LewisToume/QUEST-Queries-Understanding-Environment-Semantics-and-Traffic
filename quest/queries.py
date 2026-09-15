from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class StructuredQueryDecoder(nn.Module):
    """Decode Agent and Vector Map queries from shared BEV tokens."""

    def __init__(
        self,
        backbone_dim: int,
        hidden_dim: int = 256,
        num_agent_queries: int = 100,
        num_map_queries: int = 50,
        num_decoder_layers: int = 2,
        num_attention_heads: int = 8,
    ) -> None:
        super().__init__()
        self.num_agent_queries = num_agent_queries
        self.num_map_queries = num_map_queries
        self.memory_proj = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.agent_query = nn.Parameter(torch.randn(num_agent_queries, hidden_dim) * 0.02)
        self.map_query = nn.Parameter(torch.randn(num_map_queries, hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_decoder_layers)

    def forward(self, bev_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = bev_tokens.shape[0]
        queries = torch.cat(
            [
                self.agent_query.unsqueeze(0).expand(batch_size, -1, -1),
                self.map_query.unsqueeze(0).expand(batch_size, -1, -1),
            ],
            dim=1,
        )
        decoded = self.decoder(tgt=queries, memory=self.memory_proj(bev_tokens))
        agent_queries, map_queries = torch.split(
            decoded, [self.num_agent_queries, self.num_map_queries], dim=1
        )
        return {"agent_queries": agent_queries, "map_queries": map_queries}
