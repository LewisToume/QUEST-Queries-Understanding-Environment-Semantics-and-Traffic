"""SegFormer decoder adaptation on one frozen V3 QUEST, without a second image encoder."""

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .geometry_depth import TEMPORAL_DEPTH_VERSION


SEG_ON_V3_STAGE = "segformer_on_frozen_depth_v3_v1"


def full_base_fingerprint(base):
    digest = hashlib.sha256()
    for name, value in sorted(base.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class DINOSegFormerBranch(nn.Module):
    def __init__(self, hidden_dim, teacher_config):
        super().__init__()
        from transformers import SegformerConfig
        from transformers.models.segformer.modeling_segformer import SegformerDecodeHead

        config = SegformerConfig.from_dict(teacher_config)
        if config.num_labels != 19 or len(config.hidden_sizes) != 4:
            raise ValueError("Require four feature levels and Cityscapes19 output")
        widths = config.hidden_sizes
        self.projects = nn.ModuleList([nn.Conv2d(hidden_dim, width, 1) for width in widths])
        self.reassemble = nn.ModuleList([nn.ConvTranspose2d(widths[0], widths[0], 4, stride=4),
            nn.ConvTranspose2d(widths[1], widths[1], 2, stride=2), nn.Identity(),
            nn.Conv2d(widths[3], widths[3], 3, stride=2, padding=1)])
        self.norms = nn.ModuleList([nn.GroupNorm(8, width) for width in widths])
        self.rgb_detail = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32),
            nn.GELU(), nn.Conv2d(32, widths[0], 3, stride=2, padding=1), nn.GroupNorm(8, widths[0]), nn.GELU())
        self.decode_head = SegformerDecodeHead(config)

    def forward(self, levels, images):
        if len(levels) != 4 or any(x.shape != levels[-1].shape for x in levels):
            raise ValueError("Seg requires four aligned real DINO feature levels")
        b, n, c, h, w = levels[-1].shape
        pyramid = [norm(resize(project(x.reshape(b * n, c, h, w))))
                   for x, project, resize, norm in zip(levels, self.projects, self.reassemble, self.norms)]
        detail = self.rgb_detail(images.reshape(b * n, 3, *images.shape[-2:]))
        pyramid[0] = pyramid[0] + F.interpolate(detail, pyramid[0].shape[-2:], mode="bilinear", align_corners=False)
        logits = self.decode_head(pyramid)
        return F.interpolate(logits.float(), tuple(images.shape[-2:]), mode="bilinear", align_corners=False).reshape(
            b, n, 19, *images.shape[-2:])


def initialize_segformer_decoder(branch, bundle):
    bundle = Path(bundle)
    if (bundle / "model.safetensors").is_file():
        from safetensors.torch import load_file

        state = load_file(str(bundle / "model.safetensors"), device="cpu")
    else:
        state = torch.load(bundle / "pytorch_model.bin", map_location="cpu", weights_only=True)
    selected = {key.removeprefix("decode_head."): value for key, value in state.items() if key.startswith("decode_head.")}
    if not selected:
        raise ValueError("Teacher checkpoint contains no SegFormer decode_head weights")
    branch.decode_head.load_state_dict(selected, strict=True)
    return len(selected)


class SegOnFrozenV3(nn.Module):
    def __init__(self, base, branch):
        super().__init__()
        if base.dense_frontend_config["version"] != TEMPORAL_DEPTH_VERSION:
            raise ValueError("Seg must attach to the selected temporal Depth V3, not V5/V6")
        self.base, self.seg_branch = base, branch
        self.base.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        self.seg_branch.train(mode)
        return self

    def features(self, images):
        with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
            return self.base.extract_camera_features(images.float())

    def forward_seg(self, images):
        features = self.features(images)
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=images.device.type == "cuda"):
            return {"seg_logits": self.seg_branch(features["dense_feature_levels"], images)}

    def forward(self, *, images, intrinsics, extrinsics, ego_state, distortion, history):
        features = self.features(images)
        previous = self.features(history["images"])
        # Retain V3 joint-profile precision: shared DINO/spatial FP32; Depth bf16.
        with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
            encoded = self.base.encode_camera_features(features, intrinsics.float(), extrinsics.float(),
                ego_state.float(), tuple(images.shape[-2:]), use_ego_state=False)
            output = self.base.forward_agent(encoded["bev_tokens"], encoded["bev_features"])
            output.update(self.base.forward_map(encoded, intrinsics.float(), extrinsics.float(), ego_state.float(),
                                               tuple(images.shape[-2:]), include_dense=False))
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=images.device.type == "cuda"):
            with torch.no_grad():
                output.update(self.base._decode_dense(features, intrinsics, tuple(images.shape[-2:]), extrinsics,
                    distortion, history=history, history_features=previous))
            output["seg_logits"] = self.seg_branch(features["dense_feature_levels"], images)
        return output


def segmentation_loss(logits, targets, class_weights):
    numerator = logits.sum() * 0
    denominator = logits.new_zeros(())
    for index, item in enumerate(targets):
        seg = item["seg"]
        labels, valid = seg["labels"].to(logits.device).long(), seg["valid_mask"].to(logits.device)
        if not bool(valid.any()):
            raise ValueError("No valid Seg supervision in batch")
        safe = torch.where(valid, labels, 0)
        losses = F.cross_entropy(logits[index].float(), safe, weight=class_weights.float(), reduction="none")
        numerator = numerator + losses[valid].sum()
        denominator = denominator + class_weights[safe[valid]].sum()
    return numerator / denominator.clamp_min(1e-6)


def load_seg_on_v3(checkpoint, model_factory=None):
    from .model import QUESTModel
    from .seg_depth_training import validate_dense_checkpoint

    if checkpoint.get("stage") != SEG_ON_V3_STAGE:
        raise ValueError("Not a Seg-on-V3 checkpoint")
    metadata = checkpoint["base_metadata"]
    validate_dense_checkpoint(metadata)
    base = (model_factory or QUESTModel)(**metadata["model_config"])
    model = SegOnFrozenV3(base, DINOSegFormerBranch(base.hidden_dim, checkpoint["teacher_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if full_base_fingerprint(model.base) != checkpoint["base_sha256"]:
        raise ValueError("Frozen V3 state changed")
    return model
