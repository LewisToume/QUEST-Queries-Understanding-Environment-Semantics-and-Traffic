"""V5-preserving multilevel and calibrated geometry additions; unchanged DINOv2."""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .depth_geometry import sampling_geometry
from .geometry_depth import RayCrossViewAttention
from .small_dvgt_depth import SmallDVGTDepthFrontend, validate_small_dvgt
from .temporal_depth import TemporalDepthBlock, temporal_sampling_geometry


def validate_calibrated_small_dvgt(options):
    validate_small_dvgt(options)
    cfg = options["calibration"]
    stages = cfg["round_indices"]
    if (len(stages) != 4 or list(stages) != sorted(set(stages))
            or stages[0] != 0 or stages[-1] >= options["geometry_layers"]
            or any(not isinstance(x, int) for x in stages)
            or not isinstance(cfg["depth_samples"], int) or cfg["depth_samples"] < 2
            or not isinstance(cfg["attention_dim"], int) or cfg["attention_dim"] <= 0
            or cfg["attention_dim"] % options["heads"]
            or not 1 <= cfg["neighbors"] < 8 or not 1 <= cfg["source_cameras"] <= 8
            or not all(math.isfinite(float(cfg[k])) for k in ("min_depth_m", "max_depth_m"))
            or not 0 < cfg["min_depth_m"] < cfg["max_depth_m"] == options["ray_max_depth_m"]):
        raise ValueError("Invalid calibrated DVGT ray ranges/attention/injection rounds")


class CalibratedSmallDVGTDepthFrontend(SmallDVGTDepthFrontend):
    def __init__(self, dim, options):
        validate_calibrated_small_dvgt(options)
        super().__init__(dim, options)
        cfg, width = options["calibration"], options["geometry_dim"]
        self.level_injections = nn.ModuleList([nn.Linear(dim, width) for _ in range(3)])
        self.injection_gates = nn.Parameter(torch.zeros(3, width))
        self.ray_embedding = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.Linear(width, width))
        self.position_gates = nn.Parameter(torch.zeros(4, width))
        self.calibrated_norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        self.calibrated_spatial = nn.ModuleList([RayCrossViewAttention(width, options["heads"],
            cfg["attention_dim"], cfg["depth_samples"]) for _ in range(4)])
        self.spatial_gates = nn.Parameter(torch.zeros(4, width))
        temporal_options = dict(heads=options["heads"], attention_dim=cfg["attention_dim"],
                                depth_samples=cfg["depth_samples"], temporal=options["temporal"])
        self.calibrated_temporal = nn.ModuleList([TemporalDepthBlock(width, temporal_options) for _ in range(4)])
        self.register_buffer("ray_depths", torch.logspace(math.log10(cfg["min_depth_m"]),
                                                        math.log10(cfg["max_depth_m"]), cfg["depth_samples"]))

    def prepare_tokens(self, maps, feature_levels, history, intrinsics, extrinsics, distortion, image_size):
        if (len(feature_levels) != 4 or len(history["feature_levels"]) != 4
                or any(x.shape != feature_levels[-1].shape for x in (*feature_levels, *history["feature_levels"]))
                or intrinsics is None or extrinsics is None or distortion is None):
            raise ValueError("V6 requires four real feature levels and explicit camera calibration")
        b, t, v, _, h, w = maps.shape
        cfg, past = self.options["calibration"], t == 2
        mask = torch.ones(b, v, device=maps.device, dtype=torch.bool)
        with torch.no_grad(), torch.autocast(device_type=maps.device.type, enabled=False):
            current = sampling_geometry(intrinsics, extrinsics, distortion, mask, (h, w),
                                         image_size, self.ray_depths, cfg["neighbors"])
            spatial = [current]
            temporal = None
            if past:
                previous = sampling_geometry(history["intrinsics"], history["extrinsics"], history["distortion"],
                    mask, (h, w), image_size, self.ray_depths, cfg["neighbors"])
                spatial = [previous, current]
                temporal = temporal_sampling_geometry(intrinsics, extrinsics, distortion, history,
                    (h, w), image_size, self.ray_depths, cfg["source_cameras"], self.options["temporal"]["max_gap_s"])
        extras = []
        for index, projection in enumerate(self.level_injections):
            feature = (torch.stack((history["feature_levels"][index], feature_levels[index]), 1)
                       if past else feature_levels[index][:, None])
            extras.append(projection(feature.flatten(-2).transpose(-1, -2)))
        position = torch.stack([self.ray_embedding(value[3].to(maps.dtype)) for value in spatial], 1)
        tokens = self.level_projections[0](maps.flatten(-2).transpose(-1, -2))
        return tokens, {"spatial": spatial, "temporal": temporal, "extras": extras,
                        "position": position, "dt_s": history["dt_s"]}

    def _refine_patches(self, patches, stage, shape, context):
        b, t, v, h, w = shape
        value = patches.reshape(b, t, v, h * w, -1)
        if stage < 3:
            value = value + self.injection_gates[stage].tanh() * context["extras"][stage]
        value = value + self.position_gates[stage].tanh() * context["position"]
        frames = []
        for index in range(t):
            frame = value[:, index]
            aligned = self.calibrated_spatial[stage](self.calibrated_norms[stage](frame),
                                                     (b, v, h, w), context["spatial"][index])
            # Outside-image rays fall back to the existing image features.
            visible = torch.stack(context["spatial"][index][2]).any(0).any(-1)[..., None]
            frames.append(frame + self.spatial_gates[stage].tanh() * aligned * visible)
        if t == 2:
            frames[-1] = self.calibrated_temporal[stage](frames[-1], frames[0], (b, v, h, w),
                                                        context["temporal"], context["dt_s"])
        return torch.stack(frames, 1).reshape_as(patches)

    def refine_tokens(self, tokens, index, shape, context):
        rounds = self.options["calibration"]["round_indices"]
        if index not in rounds:
            return tokens
        stage, start = rounds.index(index), self.geometry_blocks.patch_start_idx
        patches = tokens[:, start:]
        if self.training and self.options["gradient_checkpointing"]:
            patches = checkpoint(self._refine_patches, patches, stage, shape, context, use_reentrant=False)
        else:
            patches = self._refine_patches(patches, stage, shape, context)
        return torch.cat((tokens[:, :start], patches), 1)

    def geometry_diagnostics(self, context):
        current = context["spatial"][-1]
        coverage = torch.stack(current[2]).any(0).any(-1).float().mean()
        temporal = context["temporal"]
        return {"depth_cross_view_coverage": coverage,
                "depth_temporal_coverage": (coverage.new_zeros(()) if temporal is None
                    else torch.stack(temporal[2]).any(0).any(-1).float().mean())}
