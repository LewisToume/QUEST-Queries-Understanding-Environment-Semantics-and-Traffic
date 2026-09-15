from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _recursive_stack(items: list[Any]) -> Any:
    first = items[0]
    if isinstance(first, dict):
        if any(set(item) != set(first) for item in items):
            return items
        return {key: _recursive_stack([item[key] for item in items]) for key in first}
    if torch.is_tensor(first):
        return torch.stack(items, dim=0)
    if isinstance(first, (str, Path)):
        return [str(item) for item in items]
    if isinstance(first, (list, tuple)):
        if not all(item == first for item in items):
            return items
        return first
    if first is None:
        return items
    raise TypeError(f"Unsupported batch item type: {type(first)!r}")


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return _recursive_stack(batch)
