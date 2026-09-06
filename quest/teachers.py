from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TeacherSpec:
    task: str
    name: str
    enabled: bool = False
    outputs: tuple[str, ...] = field(default_factory=tuple)


class TeacherInterface:
    """
    Stage2 teacher interface placeholder.

    The current Stage1 path is GT supervised only. This interface records the
    task-to-teacher contract for later soft-label and feature distillation
    without invoking teacher models during real GT training.
    """

    def __init__(self, specs: Mapping[str, TeacherSpec]) -> None:
        self.specs = dict(specs)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TeacherInterface":
        specs: dict[str, TeacherSpec] = {}
        for task, raw in config.items():
            if not isinstance(raw, Mapping):
                continue
            specs[task] = TeacherSpec(
                task=task,
                name=str(raw.get("name", task)),
                enabled=bool(raw.get("enabled", False)),
                outputs=tuple(str(item) for item in raw.get("outputs", ())),
            )
        return cls(specs)

    def require_disabled_for_stage1(self) -> None:
        enabled = [f"{task}:{spec.name}" for task, spec in self.specs.items() if spec.enabled]
        if enabled:
            raise RuntimeError(
                "Stage1 must use OpenScene GT supervision only; disable teachers: "
                + ", ".join(enabled)
            )

    def export_contract(self) -> dict[str, dict[str, Any]]:
        return {
            task: {
                "name": spec.name,
                "enabled": spec.enabled,
                "outputs": list(spec.outputs),
            }
            for task, spec in self.specs.items()
        }
