from __future__ import annotations

import math

import torch
import torch.nn as nn


def _decoder_layer(
    hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float
) -> nn.TransformerDecoderLayer:
    return nn.TransformerDecoderLayer(
        d_model=hidden_dim,
        nhead=num_heads,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class AgentDecoder(nn.Module):
    """Decode spatial Agent queries independently from the shared BEV memory."""

    def __init__(
        self,
        hidden_dim: int = 384,
        num_queries: int = 100,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1536,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        grid_size = math.isqrt(num_queries)
        if grid_size * grid_size != num_queries:
            raise ValueError("Agent queries must form a square XY grid")
        coordinates = torch.linspace(
            0.5 / grid_size, 1.0 - 0.5 / grid_size, grid_size
        )
        try:
            grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        except TypeError:  # PyTorch 1.9
            grid_y, grid_x = torch.meshgrid(coordinates, coordinates)
        references = torch.stack(
            (grid_x.flatten(), grid_y.flatten(), torch.full((num_queries,), 0.5)),
            dim=-1,
        )
        self.query_embedding = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.reference_xyz = nn.Parameter(references)
        self.reference_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = nn.TransformerDecoder(
            _decoder_layer(hidden_dim, num_heads, ffn_dim, dropout),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, bev_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = bev_tokens.shape[0]
        references = self.reference_xyz.clamp(1e-4, 1.0 - 1e-4)
        queries = self.query_embedding + self.reference_mlp(references)
        decoded = self.decoder(
            tgt=queries.unsqueeze(0).expand(batch_size, -1, -1),
            memory=bev_tokens,
        )
        return self.norm(decoded), references.unsqueeze(0).expand(batch_size, -1, -1)


class MapDecoder(nn.Module):
    """Decode Vector Map queries independently from Agent queries."""

    def __init__(
        self,
        hidden_dim: int = 384,
        num_queries: int = 50,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 1536,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.query_embedding = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.decoder = nn.TransformerDecoder(
            _decoder_layer(hidden_dim, num_heads, ffn_dim, dropout),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, bev_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = bev_tokens.shape[0]
        queries = self.query_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        return self.norm(self.decoder(tgt=queries, memory=bev_tokens))
