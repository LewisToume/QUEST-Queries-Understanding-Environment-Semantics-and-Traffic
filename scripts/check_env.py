from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    print("QUEST environment check")
    print(f"project_root: {PROJECT_ROOT}")

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"[ERROR] torch import failed: {e}")
        return 1

    try:
        import transformers

        print(f"transformers: {transformers.__version__}")
    except Exception as e:
        print(f"[ERROR] transformers import failed: {e}")
        return 1

    weights_path = PROJECT_ROOT / "weights" / "dinov2-small"
    print(f"weights/dinov2-small exists: {weights_path.exists()}")
    if weights_path.exists():
        for name in ["config.json", "model.safetensors", "pytorch_model.bin"]:
            print(f"  {name}: {(weights_path / name).exists()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

