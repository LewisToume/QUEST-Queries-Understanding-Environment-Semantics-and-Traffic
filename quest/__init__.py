from __future__ import annotations

__all__ = [
    "QUESTModel",
    "QUEST_Model",
]


def __getattr__(name: str):
    if name in {"QUESTModel", "QUEST_Model"}:
        from .model import QUESTModel, QUEST_Model

        return {"QUESTModel": QUESTModel, "QUEST_Model": QUEST_Model}[name]
    raise AttributeError(f"module 'quest' has no attribute {name!r}")
