"""Navformer-inspired causal BEV fusion, not persistent-ID tracking reproduction."""

import torch
from torch import nn

from .geometry_depth import RayCrossViewAttention


class TemporalAgentBEV(nn.Module):
    def __init__(self, dim, options, max_gap_s):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(options["layers"])])
        self.attention = nn.ModuleList([RayCrossViewAttention(dim, options["heads"], options["attention_dim"], 9)
                                       for _ in range(options["layers"])])
        self.time = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.gates = nn.Parameter(torch.zeros(options["layers"], dim))
        self.max_gap_s = max_gap_s

    def forward(self, current, previous, history, x_range, y_range):
        b, c, h, w = current.shape
        if previous.shape != current.shape:
            raise ValueError("Agent current/previous BEV shapes differ")
        if (history["valid"].shape != (b,) or history["valid"].dtype != torch.bool
                or history["dt_s"].shape != (b,) or not bool(torch.isfinite(history["dt_s"]).all())
                or bool((history["valid"] & ((history["dt_s"] <= 0) | (history["dt_s"] > self.max_gap_s))).any())
                or bool((~history["valid"] & (history["dt_s"] != 0)).any())
                or history["previous_to_current"].shape != (b, 4, 4)
                or not bool(torch.isfinite(history["previous_to_current"]).all())):
            raise ValueError("Invalid causal Agent history")
        with torch.no_grad(), torch.autocast(device_type=current.device.type, enabled=False):
            transform = history["previous_to_current"].float()
            rotation = transform[:, :3, :3]
            if (not torch.allclose(rotation.transpose(-1, -2) @ rotation,
                                  torch.eye(3, device=current.device).expand(b, 3, 3), atol=1e-4, rtol=1e-4)
                    or not torch.allclose(torch.linalg.det(rotation), transform.new_ones(b), atol=1e-4)
                    or not torch.allclose(transform[:, 3], transform.new_tensor([0, 0, 0, 1]).expand(b, 4), atol=1e-5)):
                raise ValueError("Agent history transform must be rigid SE(3)")
            yy, xx = torch.meshgrid(torch.arange(h, device=current.device, dtype=torch.float32),
                                    torch.arange(w, device=current.device, dtype=torch.float32), indexing="ij")
            x = x_range[0] + (xx.flatten() + .5) * (x_range[1] - x_range[0]) / w
            y = y_range[0] + (yy.flatten() + .5) * (y_range[1] - y_range[0]) / h
            points = torch.stack((x, y, torch.zeros_like(x), torch.ones_like(x)), -1)
            inverse = torch.linalg.inv(transform)
            source = torch.einsum("bij,pj->bpi", inverse, points)
            grid = torch.stack((2 * (source[..., 0] - x_range[0]) / (x_range[1] - x_range[0]) - 1,
                                2 * (source[..., 1] - y_range[0]) / (y_range[1] - y_range[0]) - 1), -1)
            oy, ox = torch.meshgrid(torch.arange(-1, 2, device=current.device),
                                    torch.arange(-1, 2, device=current.device), indexing="ij")
            offsets = torch.stack((ox.flatten() * (2 / w), oy.flatten() * (2 / h)), -1)
            grid = grid[:, None, :, None] + offsets[None, None, None]
            bound = grid.new_tensor([1 - 1 / w, 1 - 1 / h])
            valid = (grid.abs() <= bound).all(-1) & history["valid"][:, None, None, None]
            grid = torch.where(valid[..., None], grid, torch.full_like(grid, 2.))
            geometry = (torch.zeros(b, 1, 1, device=current.device, dtype=torch.long), [grid], [valid])
        tokens = current.flatten(2).transpose(1, 2)[:, None]
        past = previous.flatten(2).transpose(1, 2)[:, None]
        time = self.time(history["dt_s"][:, None]).to(past.dtype)[:, None, None]
        projectable = valid.any(-1)[..., None]
        for index, (norm, attention) in enumerate(zip(self.norms, self.attention)):
            delta = attention(norm(tokens), (b, 1, h, w), geometry, source_tokens=norm(past) + time)
            tokens = tokens + self.gates[index].tanh() * delta * projectable
        features = tokens[:, 0].transpose(1, 2).reshape(b, c, h, w)
        return tokens[:, 0], features, valid.any(-1).float().mean()
