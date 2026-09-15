from __future__ import annotations

import torch
import torch.nn as nn


class BEVEncoder(nn.Module):
    """Build a learned bird's-eye-view latent from fused camera tokens."""

    def __init__(
        self,
        hidden_dim: int = 256,
        bev_h: int = 32,
        bev_w: int = 32,
        num_layers: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_queries = nn.Parameter(torch.randn(bev_h * bev_w, hidden_dim) * 0.02)
        self.row_embed = nn.Parameter(torch.randn(bev_h, hidden_dim) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(bev_w, hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, camera_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if camera_tokens.ndim != 3:
            raise ValueError(
                f"camera_tokens must be [B, N, C], got {tuple(camera_tokens.shape)}"
            )
        batch_size = camera_tokens.shape[0]
        position = (
            self.row_embed[:, None, :] + self.col_embed[None, :, :]
        ).reshape(self.bev_h * self.bev_w, self.hidden_dim)
        queries = (self.bev_queries + position).unsqueeze(0).expand(batch_size, -1, -1)
        bev_tokens = self.norm(self.decoder(tgt=queries, memory=camera_tokens))
        bev_features = bev_tokens.transpose(1, 2).reshape(
            batch_size, self.hidden_dim, self.bev_h, self.bev_w
        )
        return bev_tokens, bev_features
