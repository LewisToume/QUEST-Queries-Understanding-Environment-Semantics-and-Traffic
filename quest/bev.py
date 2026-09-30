from __future__ import annotations

import torch
import torch.nn as nn


class BEVEncoder(nn.Module):
    """Contextualize geometry-lifted metric BEV tokens."""

    def __init__(
        self,
        hidden_dim: int = 384,
        bev_h: int = 32,
        bev_w: int = 32,
        num_layers: int = 4,
        num_attention_heads: int = 8,
        ffn_dim: int = 1536,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, bev_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if bev_tokens.ndim != 3:
            raise ValueError(
                f"bev_tokens must be [B, H*W, C], got {tuple(bev_tokens.shape)}"
            )
        if bev_tokens.shape[1:] != (self.bev_h * self.bev_w, self.hidden_dim):
            raise ValueError(
                f"expected BEV tokens [B,{self.bev_h * self.bev_w},{self.hidden_dim}], "
                f"got {tuple(bev_tokens.shape)}"
            )
        batch_size = bev_tokens.shape[0]
        encoded = self.norm(self.encoder(bev_tokens))
        bev_features = encoded.transpose(1, 2).reshape(
            batch_size, self.hidden_dim, self.bev_h, self.bev_w
        )
        return encoded, bev_features
