"""Shared scene encoder/decoder for four jointly supervised QUEST readouts.

This is a new architecture, not a lossless conversion of four independent heads.
The old trained model is only an offline initialization/validation baseline.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .backbone import FrozenDINOv2Backbone
from .depth_geometry import raw_camera_rays
from .geometry import GeometryAwareBEVLift
from .geometry_depth import DPTMetricDecoder, TEMPORAL_DEPTH_VERSION
from .heads import AgentHead, AgentProposalHead
from .model import QUESTModel
from .spatial_map import PointMapDecoderLayer, WindowBEVLayer, inverse_sigmoid
from .temporal_depth import TemporalGeometryDepthFrontend


UNIFIED_SCENE_VERSION = "shared_scene_four_task_v1"


def sample_bev(features, xy):
    # grid_sample is not implemented for CUDA bf16.
    value = F.grid_sample(features.float(), xy.float().mul(2).sub(1),
                          align_corners=False, padding_mode="zeros")
    return value.to(features.dtype)


class SharedTemporalEncoder(TemporalGeometryDepthFrontend):
    def __init__(self, dim, options):
        super().__init__(dim, options)
        del self.depth_decoder
        del self.agent_temporal

    def forward(self, *args, **kwargs):
        return self.encode_levels(*args, **kwargs)


class SharedPyramidFusion(DPTMetricDecoder):
    def __init__(self, dim, features, minimum):
        super().__init__(dim, features, minimum)
        del self.output

    def forward(self, levels):
        return self.forward_features(levels)


class SharedStructuredQueryLayer(PointMapDecoderLayer):
    """One set of attention/FFN weights for object and polyline-point queries."""

    def forward(self, agent, vector, agent_xy, vector_xy, position, pyramid):
        b, qa, _, c = agent.shape
        qm, p = vector.shape[1:3]
        values = []
        for value, xy in ((agent, agent_xy), (vector, vector_xy)):
            normalized = self.norms[0](value)
            q = (normalized + position(xy)).flatten(0, 1)
            attended = self.point_attention(q, q, normalized.flatten(0, 1), need_weights=False)[0]
            values.append(value + self.dropout(attended.reshape_as(value)))
        agent, vector = values
        # Joint instance attention exchanges Agent/Map context, not two decoders.
        instances = torch.cat((self.norms[1](agent).mean(2), self.norms[1](vector).mean(2)), 1)
        locations = torch.cat((position(agent_xy).mean(2), position(vector_xy).mean(2)), 1)
        attended = self.instance_attention(instances + locations, instances + locations,
                                           instances, need_weights=False)[0]
        agent = agent + self.dropout(attended[:, :qa, None])
        vector = vector + self.dropout(attended[:, qa:, None])
        tokens = torch.cat((agent.flatten(1, 2), vector.flatten(1, 2)), 1).unsqueeze(2)
        xy = torch.cat((agent_xy.flatten(1, 2), vector_xy.flatten(1, 2)), 1).unsqueeze(2)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            spatial = self.spatial_attention((self.norms[2](tokens) + position(xy)).float(),
                                             xy.float(), [x.float() for x in pyramid])
        tokens = tokens + self.dropout(spatial.to(tokens.dtype))
        tokens = tokens + self.dropout(self.ffn(self.norms[3](tokens)))
        return tokens[:, :qa].reshape(b, qa, 1, c), tokens[:, qa:].reshape(b, qm, p, c)


class SharedSceneTrunk(nn.Module):
    def __init__(self, dim, geometry_options, profile, x_range, y_range, z_anchors):
        super().__init__()
        self.profile = dict(profile)
        self.geometry = SharedTemporalEncoder(dim, geometry_options)
        features = geometry_options["dpt_features"]
        self.pyramid_fusion = SharedPyramidFusion(dim, features, geometry_options["min_depth_m"])
        self.camera_projection = nn.Conv2d(features, dim, 1)
        h, w = profile["bev_h"], profile["bev_w"]
        self.lift = GeometryAwareBEVLift(dim, h, w, x_range, y_range, z_anchors)
        self.lift.ego_mlp.requires_grad_(False)
        self.bev_layers = nn.ModuleList([WindowBEVLayer(dim, h, w, profile["heads"], profile["ffn_dim"],
            profile["window_size"], profile["window_size"] // 2 if i % 2 else 0, profile["dropout"])
            for i in range(profile["bev_layers"])])
        self.bev_norm = nn.LayerNorm(dim)
        self.proposals = AgentProposalHead(dim)
        self.register_buffer("proposal_centers", QUESTModel._make_cell_centers(h, w))
        qa, qm, p = profile["agent_queries"], profile["map_queries"], profile["map_points"]
        if not 0 < qa <= h * w or min(qm, p) <= 0:
            raise ValueError("Invalid unified Agent/Map query capacities")
        side = math.ceil(math.sqrt(qm))
        ids = torch.linspace(0, side * side - 1, qm).round().long()
        centers = torch.stack(((ids % side + .5) / side, (ids // side + .5) / side), -1)
        self.map_reference_logits = nn.Parameter(inverse_sigmoid(centers))
        angles = torch.arange(p) * (2 * math.pi / p)
        self.map_point_offsets = nn.Parameter(torch.stack((angles.cos(), angles.sin()), -1) * .02)
        self.map_instance_embedding = nn.Parameter(torch.randn(qm, dim) * .02)
        self.map_point_embedding = nn.Parameter(torch.randn(p, dim) * .02)
        self.query_type = nn.Parameter(torch.randn(2, dim) * .02)
        self.reference_embedding = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.query_norm = nn.LayerNorm(dim)
        self.query_layers = nn.ModuleList([SharedStructuredQueryLayer(dim, profile["heads"],
            profile["ffn_dim"], profile["bev_levels"], profile["sampling_points"], profile["dropout"])
            for _ in range(profile["query_layers"])])
        self.query_output_norm = nn.LayerNorm(dim)
        # These messages feed the SAME dense features consumed by Seg and Depth.
        self.bev_to_camera = nn.Conv2d(dim, features, 1)
        self.query_to_camera = nn.Linear(dim, features)
        self.context_gate = nn.Parameter(torch.full((features,), .05))

    def camera_context(self, bev, intrinsics, extrinsics, distortion, grid_size, image_size):
        b, n = intrinsics.shape[:2]
        h, w = grid_size
        with torch.no_grad(), torch.autocast(device_type=bev.device.type, enabled=False):
            rays, ray_valid = raw_camera_rays(intrinsics.float(), distortion.float(), grid_size, image_size)
            depths = self.geometry.ray_depths.float()
            points = rays[..., None, :] * depths[None, None, None, :, None]
            points = torch.einsum("bnij,bnpdj->bnpdi", extrinsics[..., :3, :3].float(), points)
            points = points + extrinsics[..., None, None, :3, 3].float()
            lower = points.new_tensor([self.lift.x_range[0], self.lift.y_range[0]])
            span = points.new_tensor([self.lift.x_range[1] - self.lift.x_range[0],
                                      self.lift.y_range[1] - self.lift.y_range[0]])
            xy = (points[..., :2] - lower) / span
            valid = ((xy >= 0) & (xy <= 1)).all(-1) & ray_valid[..., None]
            xy = torch.where(valid[..., None], xy, torch.full_like(xy, 2.))
        contexts = []
        for camera in range(n):
            sampled = sample_bev(bev, xy[:, camera])
            weights = valid[:, camera, None].to(sampled.dtype)
            context = (sampled * weights).sum(-1) / weights.sum(-1).clamp_min(1)
            contexts.append(context.reshape(b, -1, h, w))
        return self.bev_to_camera(torch.stack(contexts, 1).flatten(0, 1))

    def forward(self, levels, previous_levels, images, intrinsics, extrinsics, distortion, history):
        b, n, dim, h, w = levels[-1].shape
        image_size = tuple(images.shape[-2:])
        previous = {**history, "feature_levels": previous_levels}
        encoded, diagnostics = self.geometry(levels, intrinsics, extrinsics, distortion, image_size, history=previous)
        dense = self.pyramid_fusion(encoded)
        camera = F.interpolate(self.camera_projection(dense), (h, w), mode="bilinear", align_corners=False)
        camera = camera.reshape(b, n, dim, h, w)
        with torch.autocast(device_type=images.device.type, enabled=False):
            lifted = self.lift(camera.float(), intrinsics.float(), extrinsics.float(),
                images.new_zeros(b, 9).float(), image_size, use_ego_state=False, distortion=distortion.float())
        value = lifted.reshape(b, self.profile["bev_h"], self.profile["bev_w"], dim)
        for layer in self.bev_layers:
            value = layer(value)
        bev = self.bev_norm(value).permute(0, 3, 1, 2).contiguous()
        tokens = bev.flatten(2).transpose(1, 2)
        objectness, offsets = self.proposals(tokens)
        qa = self.profile["agent_queries"]
        top = objectness.topk(qa, dim=1).indices
        reference_xy = (self.proposal_centers[top] + offsets.gather(1, top[..., None].expand(-1, -1, 2))
                        / offsets.new_tensor([self.profile["bev_w"], self.profile["bev_h"]])).clamp(1e-4, 1 - 1e-4)
        agent_xyz = torch.cat((reference_xy, torch.full_like(reference_xy[..., :1], .5)), -1)
        agent_xy = reference_xy.unsqueeze(2)
        vector_xy = (self.map_reference_logits.sigmoid()[:, None] + self.map_point_offsets[None]).clamp(1e-4, 1 - 1e-4)
        vector_xy = vector_xy[None].expand(b, -1, -1, -1)
        agent = sample_bev(bev, agent_xy).permute(0, 2, 3, 1) + self.query_type[0]
        vector = sample_bev(bev, vector_xy).permute(0, 2, 3, 1)
        vector = vector + self.map_instance_embedding[None, :, None] + self.map_point_embedding[None, None] + self.query_type[1]
        agent, vector = self.query_norm(agent), self.query_norm(vector)
        pyramid = [bev]
        for _ in range(1, self.profile["bev_levels"]):
            pyramid.append(F.avg_pool2d(pyramid[-1], 2))
        agent_layers, vector_layers = [], []
        for layer in self.query_layers:
            if self.training and self.profile["gradient_checkpointing"]:
                agent, vector = checkpoint(layer, agent, vector, agent_xy, vector_xy,
                                            self.reference_embedding, pyramid, use_reentrant=False)
            else:
                agent, vector = layer(agent, vector, agent_xy, vector_xy, self.reference_embedding, pyramid)
            agent_layers.append(self.query_output_norm(agent.squeeze(2)))
            vector_layers.append(self.query_output_norm(vector))
        # Equal task contribution: do not let 224*20 Map tokens dominate 100 Agent tokens.
        joint_context = .5 * (agent_layers[-1].mean(1) + vector_layers[-1].mean((1, 2)))
        spatial = self.camera_context(bev, intrinsics, extrinsics, distortion, (h, w), image_size)
        broadcast = self.query_to_camera(joint_context)[:, None, :, None, None].expand(b, n, -1, h, w).flatten(0, 1)
        context = F.interpolate(spatial + broadcast, dense.shape[-2:], mode="bilinear", align_corners=False)
        dense = dense + self.context_gate.tanh()[None, :, None, None] * context
        return {"dense_features": dense, "bev_features": bev, "agent_features_layers": agent_layers,
                "map_features_layers": vector_layers, "agent_reference_xyz": agent_xyz,
                "map_reference_xy": vector_xy, "proposal_objectness_logits": objectness,
                "proposal_xy_offsets": offsets, "proposal_indices": top, **diagnostics}


class UnifiedReadouts(nn.Module):
    def __init__(self, dim, features, minimum):
        super().__init__()
        self.minimum = minimum
        self.agent = AgentHead(dim, 4, 8)
        self.map_class = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, 4))
        self.map_point = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2))
        self.map_dense = nn.Conv2d(dim, 3, 1)
        self.seg = nn.Conv2d(features, 19, 1)
        self.depth = nn.Sequential(nn.Conv2d(features, features // 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(features // 2, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1))

    def forward(self, scene, batch_size, cameras, output_size):
        agents = [self.agent(x, scene["agent_reference_xyz"]) for x in scene["agent_features_layers"]]
        classes = torch.stack([x[0] for x in agents])
        boxes, velocity = torch.stack([x[1] for x in agents]), torch.stack([x[2] for x in agents])
        maps = scene["map_features_layers"]
        map_classes = torch.stack([self.map_class(x.mean(2)) for x in maps])
        points = torch.stack([(inverse_sigmoid(scene["map_reference_xy"]) + self.map_point(x)).sigmoid() for x in maps])
        dense = scene["dense_features"]
        seg = F.interpolate(self.seg(dense).float(), output_size, mode="bilinear", align_corners=False)
        depth = F.softplus(F.interpolate(self.depth(dense).float(), output_size, mode="bilinear", align_corners=False)) + self.minimum
        return {"agent_cls_logits_layers": classes, "agent_boxes_layers": boxes, "agent_velocity_layers": velocity,
                "agent_cls_logits": classes[-1], "agent_boxes": boxes[-1], "agent_velocity": velocity[-1],
                "map_cls_logits_layers": map_classes, "map_points_layers": points,
                "map_cls_logits": map_classes[-1], "map_points": points[-1],
                "map_dense_gt_logits": self.map_dense(scene["bev_features"]).float(),
                "seg_logits": seg.reshape(batch_size, cameras, 19, *output_size),
                "depth": depth.reshape(batch_size, cameras, 1, *output_size)}


class UnifiedQUEST(nn.Module):
    def __init__(self, model_config, profile, backbone=None):
        super().__init__()
        self.model_config, self.profile = copy.deepcopy(model_config), dict(profile)
        if profile["version"] != UNIFIED_SCENE_VERSION:
            raise ValueError("Unknown unified scene architecture")
        integer_options = ("bev_h", "bev_w", "bev_layers", "window_size", "bev_levels", "heads", "ffn_dim",
                           "query_layers", "sampling_points", "agent_queries", "map_queries", "map_points")
        if (any(not isinstance(profile[k], int) or isinstance(profile[k], bool) or profile[k] <= 0 for k in integer_options)
                or not math.isfinite(profile["dropout"]) or not 0 <= profile["dropout"] < 1
                or not isinstance(profile["gradient_checkpointing"], bool)):
            raise ValueError("Invalid shared-scene dimensions/dropout/checkpointing")
        dim = model_config["hidden_dim"]
        options = copy.deepcopy(model_config["dense_frontend"])
        if options["version"] != TEMPORAL_DEPTH_VERSION:
            raise ValueError("Shared-scene warm start requires selected Depth V3, not V5/V6")
        options["temporal"].pop("agent_bev", None)
        self.backbone = backbone if backbone is not None else FrozenDINOv2Backbone(
            model_config["backbone_name"], model_config["local_backbone_dir"])
        if self.backbone.hidden_dim != dim or len(model_config["camera_names"]) != 8:
            raise ValueError("Unified model requires native DINO width and eight ordered cameras")
        self.backbone.requires_grad_(False).eval()
        self.camera_names = tuple(model_config["camera_names"])
        self.layer_indices = tuple(options["layer_indices"])
        self.output_size = tuple(options["output_size"])
        self.bev_h, self.bev_w = profile["bev_h"], profile["bev_w"]
        if (profile["map_points"] != 20 or dim % profile["heads"] or
                any(x % profile["window_size"] or x % (2 ** (profile["bev_levels"] - 1))
                    for x in (self.bev_h, self.bev_w))):
            raise ValueError("Invalid unified point/grid/attention dimensions")
        self.trunk = SharedSceneTrunk(dim, options, profile, tuple(model_config["x_range"]),
                                     tuple(model_config["y_range"]), model_config["z_anchors"])
        self.heads = UnifiedReadouts(dim, options["dpt_features"], options["min_depth_m"])

    @property
    def geometry_lift(self):
        return self.trunk.lift

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def extract_levels(self, images):
        b, n, c, h, w = images.shape
        if n != len(self.camera_names) or c != 3 or h % self.backbone.patch_size or w % self.backbone.patch_size:
            raise ValueError("Invalid shared-scene RGB camera count/patch-aligned shape")
        with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
            _, levels = self.backbone.forward_intermediates(images.float().flatten(0, 1), self.layer_indices)
        return tuple(x[:, 1:].transpose(1, 2).reshape(b, n, self.backbone.hidden_dim,
                     h // self.backbone.patch_size, w // self.backbone.patch_size) for x in levels)

    def forward_from_levels(self, levels, previous, *, images, intrinsics, extrinsics, distortion,
                            history, ego_state=None, return_shared_features=False):
        if history is None or history["images"].shape != images.shape:
            raise ValueError("Require actual causal history buffers and cold-start mask")
        scene = self.trunk(levels, previous, images, intrinsics, extrinsics, distortion, history)
        output = self.heads(scene, images.shape[0], images.shape[1], tuple(images.shape[-2:]))
        output.update({key: scene[key] for key in ("proposal_objectness_logits", "proposal_xy_offsets",
            "proposal_indices", "agent_reference_xyz", "depth_cross_view_coverage", "depth_temporal_coverage",
            "history_frame_fraction")})
        if return_shared_features:
            output["shared_dense_features"] = scene["dense_features"]
        return output

    def forward(self, *, images, intrinsics, extrinsics, distortion, history, ego_state=None,
                return_shared_features=False):
        if history is None or history["images"].shape != images.shape:
            raise ValueError("Require actual causal history buffers and cold-start mask")
        levels, previous = self.extract_levels(images), self.extract_levels(history["images"])
        return self.forward_from_levels(levels, previous, images=images, intrinsics=intrinsics,
            extrinsics=extrinsics, distortion=distortion, history=history, ego_state=ego_state,
            return_shared_features=return_shared_features)

    def warm_start(self, source):
        """Explicit compatible transfers only; never claim prediction identity."""
        base = source.base
        if base.hidden_dim != self.model_config["hidden_dim"]:
            raise ValueError("Cannot change hidden width while warm-starting shared pretrained weights")
        transfers = {}

        def transfer(name, target, state):
            target.load_state_dict(state, strict=True)
            transfers[name] = len(state)

        transfer("backbone", self.backbone, base.backbone.state_dict())
        transfer("shared_geometry_and_time", self.trunk.geometry, {key: value for key, value in
            base.dense_frontend.state_dict().items() if not key.startswith(("depth_decoder.", "agent_temporal."))})
        transfer("shared_pyramid_fusion", self.trunk.pyramid_fusion, {key: value for key, value in
            base.dense_frontend.depth_decoder.state_dict().items() if not key.startswith("output.")})
        transfer("depth_readout", self.heads.depth, base.dense_frontend.depth_decoder.output.state_dict())
        transfer("agent_readout", self.heads.agent, base.agent_head.state_dict())
        transfer("agent_proposals", self.trunk.proposals, base.agent_proposal_head.state_dict())
        spatial = base.map_spatial
        if spatial is None:
            raise ValueError("Shared scene initialization requires the trained spatial Map baseline")
        self.trunk.lift.weight_mlp.load_state_dict(spatial.geometry_lift.weight_mlp.state_dict(), strict=True)
        with torch.no_grad():
            value = spatial.geometry_lift.bev_embedding.T.reshape(1, base.hidden_dim, spatial.h, spatial.w)
            value = F.interpolate(value, (self.bev_h, self.bev_w), mode="bilinear", align_corners=False)
            self.trunk.lift.bev_embedding.copy_(value.flatten(2).transpose(1, 2)[0])
        if len(self.trunk.bev_layers) != len(spatial.layers):
            raise ValueError("Shared BEV warm-start layer count must match spatial Map baseline")
        transfer("shared_bev_layers", self.trunk.bev_layers, spatial.layers.state_dict())
        transfer("shared_bev_norm", self.trunk.bev_norm, spatial.norm.state_dict())
        if len(self.trunk.query_layers) > len(base.map_decoder.layers):
            raise ValueError("Shared query layer count exceeds compatible warm-start layers")
        for i, layer in enumerate(self.trunk.query_layers):
            transfer("shared_query_layer_" + str(i), layer, base.map_decoder.layers[i].state_dict())
        transfer("shared_reference_embedding", self.trunk.reference_embedding, base.map_decoder.reference_mlp.state_dict())
        transfer("shared_query_input_norm", self.trunk.query_norm, base.map_decoder.input_norm.state_dict())
        transfer("shared_query_output_norm", self.trunk.query_output_norm, base.map_decoder.output_norms[-1].state_dict())
        transfer("map_class_readout", self.heads.map_class, base.map_head.cls_branches[-1].state_dict())
        transfer("map_point_readout", self.heads.map_point, base.map_head.point_branches[-1].state_dict())
        for target, old in ((self.trunk.map_reference_logits, base.map_decoder.reference_logits),
                            (self.trunk.map_point_offsets, base.map_decoder.point_offsets),
                            (self.trunk.map_instance_embedding, base.map_decoder.instance_embedding),
                            (self.trunk.map_point_embedding, base.map_decoder.point_embedding)):
            if target.shape != old.shape:
                raise ValueError("Shared query initialization changed certified capacity")
            with torch.no_grad():
                target.copy_(old)
        self.trunk.requires_grad_(True)
        self.trunk.lift.ego_mlp.requires_grad_(False)
        self.heads.requires_grad_(True)
        return {"transferred": transfers, "prediction_identity": False,
                "not_transferred": ["old_agent_decoder", "old_separate_seg_decoder", "old_coarse_bev", "old_temporal_agent"],
                "new": ["shared_camera_projection", "shared_feedback", "seg_1x1_readout", "map_dense_readout", "query_type"]}
