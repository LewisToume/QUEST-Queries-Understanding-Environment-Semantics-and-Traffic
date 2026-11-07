"""GradNorm-only controlled continuation; shares the audited continuation runner."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_unified_feature_kd import main


if __name__ == "__main__":
    raise SystemExit(main(["--config", "configs/unified_gradnorm_v1.yaml", *sys.argv[1:]]))
