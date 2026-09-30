from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Decode image-conditioned Agent proposals without learned query identities."""

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
        self.num_queries = int(num_queries)
        self.reference_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [
                _decoder_layer(hidden_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

    @staticmethod
    def sample_local_bev(
        bev_features: torch.Tensor, reference_xyz: torch.Tensor
    ) -> torch.Tensor:
        if bev_features.ndim != 4:
            raise ValueError("bev_features must be [B,C,H,W]")
        if reference_xyz.ndim != 3 or reference_xyz.shape[-1] != 3:
            raise ValueError("reference_xyz must be [B,Q,3]")
        if reference_xyz.shape[0] != bev_features.shape[0]:
            raise ValueError("reference_xyz batch does not match BEV features")
        grid = reference_xyz[..., :2].mul(2.0).sub(1.0).unsqueeze(2)
        sampled = F.grid_sample(
            bev_features,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return sampled.squeeze(-1).transpose(1, 2)

    def forward(
        self,
        bev_tokens: torch.Tensor,
        bev_features: torch.Tensor,
        reference_xyz: torch.Tensor,
    ) -> list[torch.Tensor]:
        if reference_xyz.shape[1] != self.num_queries:
            raise ValueError(
                f"expected {self.num_queries} Agent proposals, got {reference_xyz.shape[1]}"
            )
        local_bev = self.sample_local_bev(bev_features, reference_xyz)
        decoded = self.query_norm(local_bev + self.reference_mlp(reference_xyz))
        intermediates: list[torch.Tensor] = []
        for layer, norm in zip(self.layers, self.output_norms):
            decoded = layer(tgt=decoded, memory=bev_tokens)
            intermediates.append(norm(decoded))
        return intermediates


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
