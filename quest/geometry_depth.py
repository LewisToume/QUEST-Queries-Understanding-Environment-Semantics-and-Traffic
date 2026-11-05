"""DVGT-inspired multiview geometry encoder and DPT-style metric depth decoder.

Mechanism adaptation, not a replica of the temporal, prior-free DVGT teacher.
No OpenMMLab/teacher runtime dependencies. Existing spatial branches are bypassed.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .depth_geometry import sampling_geometry


GEOMETRY_DEPTH_VERSION = "geometry_dpt_depth_v2"
TEMPORAL_DEPTH_VERSION = "temporal_geometry_dpt_depth_v3"
SMALL_DVGT_VERSION = "dinov2_small_dvgt_depth_v5"
CALIBRATED_SMALL_DVGT_VERSION = "dinov2_calibrated_small_dvgt_depth_v6"
SMALL_DVGT_VERSIONS = (SMALL_DVGT_VERSION, CALIBRATED_SMALL_DVGT_VERSION)
TEMPORAL_DEPTH_VERSIONS = (TEMPORAL_DEPTH_VERSION, *SMALL_DVGT_VERSIONS)
GEOMETRY_DEPTH_VERSIONS = (GEOMETRY_DEPTH_VERSION, *TEMPORAL_DEPTH_VERSIONS)


def validate_geometry_depth_options(options):
    if options["version"] == CALIBRATED_SMALL_DVGT_VERSION:
        from .calibrated_small_dvgt import validate_calibrated_small_dvgt

        validate_calibrated_small_dvgt(options)
        return
    if options["version"] == SMALL_DVGT_VERSION:
        from .small_dvgt_depth import validate_small_dvgt

        validate_small_dvgt(options)
        return
    if (options["version"] not in GEOMETRY_DEPTH_VERSIONS or options["image_geometry"] != "raw_opencv"
            or options["preserve_spatial_features"] is not True
            or tuple(options["output_size"]) != (252, 448)):
        raise ValueError("Geometry Depth V2 requires explicit raw OpenCV geometry and unchanged spatial features")
    for key in ("geometry_layers", "heads", "attention_dim", "ffn_dim", "depth_samples", "neighbors", "dpt_features"):
        if not isinstance(options[key], int) or options[key] <= 0:
            raise ValueError("Expected positive geometry Depth option: " + key)
    if (options["geometry_layers"] != 4 or options["attention_dim"] % options["heads"]
            or 384 % options["heads"] or options["dpt_features"] < 8 or not 1 <= options["neighbors"] < 8
            or not math.isfinite(options["dropout"]) or not 0 <= options["dropout"] < 1
            or not 0 < options["min_depth_m"] <= options["ray_min_depth_m"] < options["ray_max_depth_m"]
            or not all(math.isfinite(options[k]) for k in ("min_depth_m", "ray_min_depth_m", "ray_max_depth_m"))):
        raise ValueError("Invalid geometry Depth attention/ray configuration")
    if any(not isinstance(options[key], bool) for key in ("amp_bf16", "gradient_checkpointing")):
        raise ValueError("Depth precision/checkpointing flags must be explicit booleans")
    if options["version"] == TEMPORAL_DEPTH_VERSION:
        temporal = options["temporal"]
        if (temporal["frames"] != 2 or temporal["causal"] is not True
                or temporal["missing_history"] != "mask"
                or not isinstance(temporal["source_cameras"], int)
                or not 1 <= temporal["source_cameras"] <= 8
                or not all(math.isfinite(temporal[k]) and temporal[k] > 0
                           for k in ("timestamp_scale", "max_gap_s"))):
            raise ValueError("Temporal Depth requires two causal frames and explicit history masking")
        agent = temporal.get("agent_bev")
        if agent is not None and (agent.get("enabled") is not True
                or any(not isinstance(agent[k], int) or agent[k] <= 0 for k in ("layers", "heads", "attention_dim"))
                or 384 % agent["heads"] or agent["attention_dim"] % agent["heads"]):
            raise ValueError("Invalid opt-in temporal Agent BEV profile")


class RayCrossViewAttention(nn.Module):
    def __init__(self, dim, heads, attention_dim, samples):
        super().__init__()
        self.heads, self.channels, self.samples = heads, attention_dim, samples
        self.query = nn.Linear(dim, attention_dim)
        self.key_value = nn.Conv2d(dim, attention_dim * 2, 1)
        self.project = nn.Linear(attention_dim, dim)
        self.depth_bias = nn.Parameter(torch.zeros(heads, samples))

    def forward(self, tokens, shape, geometry, source_tokens=None):
        b, n, h, w = shape
        source_ids, grids, masks = geometry[:3]
        p, c, heads = h * w, self.channels, self.heads
        maps = tokens.reshape(b * n, p, -1).transpose(1, 2).reshape(b * n, -1, h, w)
        kv = self.key_value(maps).reshape(b, n, c * 2, h, w)
        self_kv = kv.flatten(-2).transpose(-1, -2)
        if source_tokens is not None:
            source_maps = source_tokens.reshape(b * n, p, -1).transpose(1, 2).reshape(b * n, -1, h, w)
            kv = self.key_value(source_maps).reshape(b, n, c * 2, h, w)
        candidates, validity = [self_kv.unsqueeze(-2)], [torch.ones(b, n, p, 1, dtype=torch.bool, device=tokens.device)]
        batch = torch.arange(b, device=tokens.device)[:, None]
        for rank, (grid, valid) in enumerate(zip(grids, masks)):
            source = kv[batch, source_ids[..., rank]].reshape(b * n, c * 2, h, w)
            # CUDA grid_sample does not natively support bf16; keep sampling in fp32.
            sampled = F.grid_sample(source.float(), grid.reshape(b * n, p, self.samples, 2).float(),
                                    align_corners=False, padding_mode="zeros", mode="bilinear")
            candidates.append(sampled.to(tokens.dtype).reshape(b, n, c * 2, p, self.samples).permute(0, 1, 3, 4, 2))
            validity.append(valid)
        candidates = torch.cat(candidates, -2)
        valid = torch.cat(validity, -1)
        keys, values = candidates.chunk(2, -1)
        keys = keys.reshape(b, n, p, -1, heads, c // heads)
        values = values.reshape_as(keys)
        query = self.query(tokens).reshape(b, n, p, heads, c // heads)
        logits = (query.unsqueeze(-3).float() * keys.float()).sum(-1) / math.sqrt(c // heads)
        bias = torch.cat((self.depth_bias.new_zeros(heads, 1),
                          self.depth_bias.repeat(1, len(grids))), -1).T
        weights = (logits + bias).masked_fill(~valid[..., None], -float("inf")).softmax(-2)
        context = (weights[..., None] * values.float()).sum(-3).reshape(b, n, p, c)
        return self.project(context.to(tokens.dtype))


class GeometryDepthBlock(nn.Module):
    def __init__(self, dim, options):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.intra_view = nn.MultiheadAttention(dim, options["heads"], dropout=options["dropout"], batch_first=True)
        self.cross_view = RayCrossViewAttention(dim, options["heads"], options["attention_dim"], options["depth_samples"])
        self.ffn = nn.Sequential(nn.Linear(dim, options["ffn_dim"]), nn.GELU(),
                                 nn.Dropout(options["dropout"]), nn.Linear(options["ffn_dim"], dim))
        self.scales = nn.Parameter(torch.full((3, dim), .1))

    def forward(self, x, shape, geometry):
        b, n, h, w = shape
        y = self.norms[0](x).reshape(b * n, h * w, -1)
        x = x + self.scales[0] * self.intra_view(y, y, y, need_weights=False)[0].reshape_as(x)
        x = x + self.scales[1] * self.cross_view(self.norms[1](x), shape, geometry)
        return x + self.scales[2] * self.ffn(self.norms[2](x))


class ResidualConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1),
                                    nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.layers(x)


class DPTMetricDecoder(nn.Module):
    """Learned four-scale reassembly, skip residual fusion and positive metric output."""
    def __init__(self, dim, features, minimum):
        super().__init__()
        widths = (128, 256, 384, 384)
        self.projects = nn.ModuleList([nn.Conv2d(dim, width, 1) for width in widths])
        self.resize = nn.ModuleList([nn.ConvTranspose2d(widths[0], widths[0], 4, stride=4),
                                     nn.ConvTranspose2d(widths[1], widths[1], 2, stride=2), nn.Identity(),
                                     nn.Conv2d(widths[3], widths[3], 3, stride=2, padding=1)])
        self.scratch = nn.ModuleList([nn.Conv2d(width, features, 3, padding=1) for width in widths])
        self.skip_refine = nn.ModuleList([ResidualConv(features) for _ in widths])
        self.fuse_refine = nn.ModuleList([ResidualConv(features) for _ in widths])
        self.output = nn.Sequential(nn.Conv2d(features, features // 2, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(features // 2, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1))
        self.minimum = minimum
        nn.init.constant_(self.output[-1].bias, 10.)

    def forward_features(self, levels):
        pyramid = [scratch(resize(project(x))) for x, project, resize, scratch in
                   zip(levels, self.projects, self.resize, self.scratch)]
        value = None
        for index in range(3, -1, -1):
            skip = self.skip_refine[index](pyramid[index])
            value = skip if value is None else F.interpolate(value, skip.shape[-2:], mode="bilinear", align_corners=False) + skip
            value = self.fuse_refine[index](value)
        return value

    def forward(self, levels, output_size):
        value = self.forward_features(levels)
        logits = F.interpolate(self.output(value), output_size, mode="bilinear", align_corners=False)
        return F.softplus(logits.float()) + self.minimum


class GeometryDepthFrontend(nn.Module):
    def __init__(self, dim, options):
        super().__init__()
        validate_geometry_depth_options(options)
        self.options = dict(options)
        self.level_projections = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        self.ray_embedding = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        self.geometry_blocks = nn.ModuleList([GeometryDepthBlock(dim, options) for _ in range(4)])
        self.output_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(4)])
        self.depth_decoder = DPTMetricDecoder(dim, options["dpt_features"], options["min_depth_m"])
        self.register_buffer("ray_depths", torch.logspace(math.log10(options["ray_min_depth_m"]),
                                                        math.log10(options["ray_max_depth_m"]), options["depth_samples"]))

    def forward(self, feature_levels, intrinsics, extrinsics, distortion, image_size, camera_mask=None):
        if len(feature_levels) != 4 or any(level.shape != feature_levels[-1].shape for level in feature_levels):
            raise ValueError("Depth V2 needs four aligned DINO maps")
        b, n, c, h, w = feature_levels[-1].shape
        if camera_mask is None:
            camera_mask = torch.ones(b, n, dtype=torch.bool, device=intrinsics.device)
        if distortion is None or extrinsics is None:
            raise ValueError("Depth V2 requires sensor2lidar and actual OpenCV distortion, never implicit zeros")
        with torch.autocast(device_type=intrinsics.device.type, enabled=False):
            geometry = sampling_geometry(intrinsics, extrinsics, distortion, camera_mask.bool(),
                                         (h, w), image_size, self.ray_depths, self.options["neighbors"])
        position = self.ray_embedding(geometry[3].to(feature_levels[-1].dtype))
        x, levels = torch.zeros_like(position), []
        for feature, projection, block, norm in zip(feature_levels, self.level_projections,
                                                   self.geometry_blocks, self.output_norms):
            tokens = feature.flatten(-2).transpose(-1, -2)
            x = x + projection(tokens) + position
            if self.training and self.options["gradient_checkpointing"]:
                x = checkpoint(block, x, (b, n, h, w), geometry, use_reentrant=False)
            else:
                x = block(x, (b, n, h, w), geometry)
            levels.append(norm(x).reshape(b * n, h, w, c).permute(0, 3, 1, 2))
        depth = self.depth_decoder(levels, self.options["output_size"])
        coverage = torch.stack(geometry[2]).any(0).any(-1).float().mean()
        return {"depth": depth.reshape(b, n, 1, *self.options["output_size"]),
                "depth_cross_view_coverage": coverage}
