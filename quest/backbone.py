from __future__ import annotations

import torch
from transformers import AutoModel

from .utils import project_root


def resolve_dinov2_source(
    model_name: str = "facebook/dinov2-small",
    local_backbone_dir: str = "weights/dinov2-small",
) -> tuple[str, bool]:
    """
    Prefer the local repo copy of DINOv2 weights for offline-friendly execution.
    """

    local_path = project_root() / local_backbone_dir
    if local_path.exists():
        return str(local_path.resolve()), True
    return model_name, False


def load_dinov2_backbone(
    model_name: str = "facebook/dinov2-small",
    local_backbone_dir: str = "weights/dinov2-small",
):
    source, local_files_only = resolve_dinov2_source(
        model_name=model_name,
        local_backbone_dir=local_backbone_dir,
    )
    model = AutoModel.from_pretrained(
        source,
        local_files_only=local_files_only,
    )

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    print(f"Loaded DINOv2 from: {source}")
    print(f"hidden_size        : {model.config.hidden_size}")
    print(f"patch_size         : {getattr(model.config, 'patch_size', 'unknown')}")
    print(f"parameters         : {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    print("backbone frozen    : True")
    return model


class FrozenDINOv2Backbone(torch.nn.Module):
    """Frozen DINOv2 backbone that returns token sequences."""

    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-small",
        local_backbone_dir: str = "weights/dinov2-small",
    ) -> None:
        super().__init__()
        source, local_files_only = resolve_dinov2_source(
            model_name=backbone_name,
            local_backbone_dir=local_backbone_dir,
        )
        self.model = AutoModel.from_pretrained(
            source,
            local_files_only=local_files_only,
        )
        self.hidden_dim = int(self.model.config.hidden_size)
        self.patch_size = int(getattr(self.model.config, "patch_size", 14))

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def train(self, mode: bool = True) -> "FrozenDINOv2Backbone":
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            outputs = self.model(pixel_values=images)
        return outputs.last_hidden_state


def test_dinov2_forward() -> None:
    print("=" * 60)
    print("DINOv2 backbone smoke test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_dinov2_backbone().to(device)

    input_tensor = torch.randn(1, 3, 224, 224, device=device)
    print(f"input shape         : {tuple(input_tensor.shape)}")

    with torch.no_grad():
        outputs = model(pixel_values=input_tensor)

    last_hidden_state = outputs.last_hidden_state
    expected_seq_len = (224 // 14) * (224 // 14) + 1
    expected_hidden_dim = model.config.hidden_size
    expected_shape = (1, expected_seq_len, expected_hidden_dim)

    print(f"output shape        : {tuple(last_hidden_state.shape)}")
    assert last_hidden_state.shape == expected_shape, (
        f"unexpected output shape, expected {expected_shape}, "
        f"got {tuple(last_hidden_state.shape)}"
    )

    print(f"patch token count   : {expected_seq_len - 1}")
    print(f"total token count   : {expected_seq_len}")
    print("=" * 60)


if __name__ == "__main__":
    test_dinov2_forward()
