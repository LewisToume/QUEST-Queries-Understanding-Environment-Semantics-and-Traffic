from __future__ import annotations

from pathlib import Path

import yaml


def project_root() -> Path:
    """
    返回项目根目录（QUEST/）。

    约定:
    - 当前文件位于 QUEST/quest/utils.py
    - 因此根目录是 parents[1]
    """
    return Path(__file__).resolve().parents[1]


def weights_dir() -> Path:
    return project_root() / "weights"


def configs_dir() -> Path:
    return project_root() / "configs"


def data_dir() -> Path:
    return project_root() / "data"


def load_yaml_config(path: str | Path) -> dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}
