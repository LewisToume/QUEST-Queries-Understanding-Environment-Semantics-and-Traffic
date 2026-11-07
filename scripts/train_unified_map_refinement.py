"""Map reference refinement continuation with the original four-task supervision."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_unified_feature_kd import main


if __name__ == "__main__":
    raise SystemExit(main(["--config", "configs/unified_map_refinement_v1.yaml", *sys.argv[1:]]))
