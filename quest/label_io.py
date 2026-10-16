from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


def load_record(path: Path) -> Mapping[str, Any]:
    try:
        record = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        record = torch.load(path, map_location="cpu")
    if not isinstance(record, Mapping):
        raise ValueError(f"label file must contain a mapping: {path}")
    return record
