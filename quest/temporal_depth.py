"""Causal, motion-compensated two-frame extension of the trained Depth V2 branch."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .depth_geometry import project_raw, raw_camera_rays, sampling_geometry
from .geometry_depth import GeometryDepthFrontend, RayCrossViewAttention
from .temporal_agent import TemporalAgentBEV


@torch.no_grad()
def temporal_sampling_geometry(intrinsics, extrinsics, distortion, history, grid_size,
                               image_size, depths, source_cameras, max_gap_s):
    """Map current-camera rays into actual previous-camera pixels using relative SE(3)."""
    b, n = intrinsics.shape[:2]
    transform, valid_frame, dt = history["previous_to_current"].float(), history["valid"], history["dt_s"]
    if (transform.shape != (b, 4, 4) or valid_frame.shape != (b,) or valid_frame.dtype != torch.bool
            or dt.shape != (b,) or not bool(torch.isfinite(dt).all())
            or bool((valid_frame & ((dt <= 0) | (dt > max_gap_s))).any())
            or bool((~valid_frame & (dt != 0)).any())):
        raise ValueError("Invalid causal history transform/mask/time interval")
    rotation = transform[:, :3, :3]
    if (not bool(torch.isfinite(transform).all())
            or not torch.allclose(rotation.transpose(-1, -2) @ rotation,
                                  torch.eye(3, device=transform.device).expand(b, 3, 3), atol=1e-4, rtol=1e-4)
            or not torch.allclose(torch.linalg.det(rotation), transform.new_ones(b), atol=1e-4)
            or not torch.allclose(transform[:, 3], transform.new_tensor([0, 0, 0, 1]).expand(b, 4), atol=1e-5)):
        raise ValueError("previous_to_current must be a rigid previous-LiDAR to current-LiDAR transform")
    current_t = extrinsics.float()
    source_t = transform[:, None] @ history["extrinsics"].float()
    rays, ray_valid = raw_camera_rays(intrinsics.float(), distortion.float(), grid_size, image_size)
    forward = current_t[..., :3, 2]
    _, ids = (forward @ source_t[..., :3, 2].transpose(1, 2)).topk(source_cameras, dim=-1)
    points = rays[..., None, :] * depths.float()[None, None, None, :, None]
    points = torch.einsum("bnij,bnpdj->bnpdi", current_t[..., :3, :3], points) + current_t[..., None, None, :3, 3]
    inverse, batch = torch.linalg.inv(source_t), torch.arange(b, device=intrinsics.device)[:, None]
    grids, masks = [], []
    for rank in range(source_cameras):
        source = ids[..., rank]
        t = inverse[batch, source]
        xyz = torch.einsum("bnij,bnpdj->bnpdi", t[..., :3, :3], points) + t[..., None, None, :3, 3]
        grid, valid = project_raw(xyz, history["intrinsics"][batch, source],
                                  history["distortion"][batch, source], image_size)
        grids.append(grid)
        masks.append(valid & ray_valid[..., None] & valid_frame[:, None, None, None])
    return ids, grids, masks


class TemporalDepthBlock(nn.Module):
    def __init__(self, dim, options):
        super().__init__()
        self.current_norm, self.history_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.time_embedding = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.attention = RayCrossViewAttention(dim, options["heads"], options["attention_dim"], options["depth_samples"])
        # Exact V2 function at initialization; first update trains gates, then attention.
        self.gate = nn.Parameter(torch.zeros(dim))
        self.max_gap_s = options["temporal"]["max_gap_s"]

    def forward(self, current, previous, shape, geometry, dt_s):
        time = self.time_embedding((dt_s / self.max_gap_s)[:, None]).to(previous.dtype)
        source = self.history_norm(previous) + time[:, None, None]
        delta = self.attention(self.current_norm(current), shape, geometry, source_tokens=source)
        projectable = torch.stack(geometry[2]).any(0).any(-1)
        return current + self.gate.tanh() * delta * projectable[..., None]


class TemporalGeometryDepthFrontend(GeometryDepthFrontend):
    def __init__(self, dim, options):
        super().__init__(dim, options)
        self.temporal_blocks = nn.ModuleList([TemporalDepthBlock(dim, options) for _ in range(4)])
        self.agent_temporal = (TemporalAgentBEV(dim, options["temporal"]["agent_bev"], options["temporal"]["max_gap_s"])
                               if options["temporal"].get("agent_bev") is not None else None)

    def forward(self, feature_levels, intrinsics, extrinsics, distortion, image_size,
                camera_mask=None, history=None):
        if camera_mask is not None:
            raise ValueError("Temporal V3 currently requires all eight current cameras")
        if (history is None or len(feature_levels) != 4 or len(history["feature_levels"]) != 4
                or any(x.shape != feature_levels[-1].shape for x in (*feature_levels, *history["feature_levels"]))):
            raise ValueError("Temporal V3 requires explicit aligned current/history features and a validity mask")
        b, n, c, h, w = feature_levels[-1].shape
        mask = torch.ones(b, n, dtype=torch.bool, device=intrinsics.device)
        with torch.autocast(device_type=intrinsics.device.type, enabled=False):
            current_geometry = sampling_geometry(intrinsics, extrinsics, distortion, mask,
                                                  (h, w), image_size, self.ray_depths, self.options["neighbors"])
            previous_geometry = sampling_geometry(history["intrinsics"], history["extrinsics"], history["distortion"],
                                                   mask, (h, w), image_size, self.ray_depths, self.options["neighbors"])
            temporal_geometry = temporal_sampling_geometry(intrinsics, extrinsics, distortion, history,
                (h, w), image_size, self.ray_depths, self.options["temporal"]["source_cameras"],
                self.options["temporal"]["max_gap_s"])
        position = self.ray_embedding(current_geometry[3].to(feature_levels[-1].dtype))
        previous_position = self.ray_embedding(previous_geometry[3].to(feature_levels[-1].dtype))
        x, y, levels = torch.zeros_like(position), torch.zeros_like(previous_position), []
        shape = b, n, h, w
        for index, (projection, block, temporal, norm) in enumerate(zip(self.level_projections,
                self.geometry_blocks, self.temporal_blocks, self.output_norms)):
            x = x + projection(feature_levels[index].flatten(-2).transpose(-1, -2)) + position
            y = y + projection(history["feature_levels"][index].flatten(-2).transpose(-1, -2)) + previous_position
            if self.training and self.options["gradient_checkpointing"]:
                x = checkpoint(block, x, shape, current_geometry, use_reentrant=False)
                y = checkpoint(block, y, shape, previous_geometry, use_reentrant=False)
                x = checkpoint(temporal, x, y, shape, temporal_geometry, history["dt_s"], use_reentrant=False)
            else:
                x, y = block(x, shape, current_geometry), block(y, shape, previous_geometry)
                x = temporal(x, y, shape, temporal_geometry, history["dt_s"])
            levels.append(norm(x).reshape(b * n, h, w, c).permute(0, 3, 1, 2))
        depth = self.depth_decoder(levels, self.options["output_size"])
        return {"depth": depth.reshape(b, n, 1, *self.options["output_size"]),
                "depth_cross_view_coverage": torch.stack(current_geometry[2]).any(0).any(-1).float().mean(),
                "depth_temporal_coverage": torch.stack(temporal_geometry[2]).any(0).any(-1).float().mean(),
                "history_frame_fraction": history["valid"].float().mean()}
