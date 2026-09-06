from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from .openscene_dataset import OPENSCENE_CAMERA_NAMES
from .utils import project_root


TEACHER_CAMERA_ORDERS: dict[str, tuple[str, ...]] = {
    "StreamPETR": ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0"),
    "MapTRv2": ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0"),
    "OccNet": OPENSCENE_CAMERA_NAMES,
    "ViDAR": OPENSCENE_CAMERA_NAMES,
}

OPENSCENE_AGENT_CLASS_TO_STREAM_PETR = {
    "vehicle": "car",
    "pedestrian": "pedestrian",
    "traffic_cone": "traffic_cone",
    "generic_object": "barrier",
}

OPENSCENE_MAP_CLASS_NOTE = (
    "MapTR/MapTRv2 defaults are nuScenes vector-map classes. OpenScene/nuPlan "
    "map taxonomy must be verified before enabling map distillation."
)

OCCNET_CLASS_MAPPING_VERSION = "openscene_nuplan_occ11_to_quest_occ11_v0"
COORDINATE_CONVENTION = (
    "OpenScene sample uses camera sensor2lidar extrinsics; QUEST normalizes "
    "agent boxes into local lidar range and keeps occupancy as [C, X, Y, Z]."
)


class TeacherUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class TeacherSpec:
    task: str
    name: str
    enabled: bool = False
    repo_path: str | None = None
    config_path: str | None = None
    checkpoint_path: str | None = None
    outputs: tuple[str, ...] = field(default_factory=tuple)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TeacherCheckResult:
    task: str
    name: str
    repo_path: Path | None
    config_path: Path | None
    checkpoint_path: Path | None
    repo_ok: bool
    config_ok: bool
    checkpoint_ok: bool
    checkpoint_load_ok: bool
    dependencies_ok: bool
    inference_ok: bool
    frozen: bool
    status: str
    searched_paths: list[str] = field(default_factory=list)
    missing_dependencies: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    checkpoint_keys: list[str] = field(default_factory=list)
    raw_output_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    converted_output_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "name": self.name,
            "repo_path": str(self.repo_path) if self.repo_path else None,
            "config_path": str(self.config_path) if self.config_path else None,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "repo_ok": self.repo_ok,
            "config_ok": self.config_ok,
            "checkpoint_ok": self.checkpoint_ok,
            "checkpoint_load_ok": self.checkpoint_load_ok,
            "dependencies_ok": self.dependencies_ok,
            "inference_ok": self.inference_ok,
            "frozen": self.frozen,
            "status": self.status,
            "searched_paths": self.searched_paths,
            "missing_dependencies": self.missing_dependencies,
            "errors": self.errors,
            "checkpoint_keys": self.checkpoint_keys,
            "raw_output_shapes": self.raw_output_shapes,
            "converted_output_shapes": self.converted_output_shapes,
        }


def resolve_path(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root() / candidate
    return candidate


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def checkpoint_search_roots() -> list[Path]:
    root = project_root()
    return [root / "weights", root / "third_party"]


def find_checkpoints(name_patterns: Sequence[str]) -> list[Path]:
    matches: list[Path] = []
    lowered_patterns = [pattern.lower() for pattern in name_patterns]
    for root in checkpoint_search_roots():
        if not root.exists():
            continue
        for suffix in ("*.pth", "*.pt", "*.ckpt"):
            for path in root.rglob(suffix):
                lower_name = path.name.lower()
                if any(pattern in lower_name for pattern in lowered_patterns):
                    matches.append(path)
    return sorted(matches, key=lambda path: path.stat().st_size if path.exists() else 0, reverse=True)


def load_checkpoint_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_summary(checkpoint: Any) -> list[str]:
    if isinstance(checkpoint, Mapping):
        return [str(key) for key in list(checkpoint.keys())[:20]]
    return [type(checkpoint).__name__]


def tensor_shapes(obj: Any, prefix: str = "") -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    if torch.is_tensor(obj):
        shapes[prefix or "tensor"] = tuple(obj.shape)
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            shapes.update(tensor_shapes(value, child_prefix))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj[:10]):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            shapes.update(tensor_shapes(value, child_prefix))
    return shapes


def reorder_openscene_cameras(
    tensor: torch.Tensor,
    source_order: Sequence[str],
    target_order: Sequence[str],
) -> torch.Tensor:
    """Reorder camera-major tensors from OpenScene order to a teacher order."""

    indices = [source_order.index(camera_name) for camera_name in target_order]
    return tensor.index_select(dim=1, index=torch.as_tensor(indices, device=tensor.device))


def openscene_intrinsics_to_teacher(
    intrinsics: torch.Tensor,
    source_order: Sequence[str],
    target_order: Sequence[str],
) -> torch.Tensor:
    """OpenScene stores per-camera pinhole intrinsics; teachers receive the same matrices in their camera order."""

    return reorder_openscene_cameras(intrinsics, source_order, target_order)


def openscene_extrinsics_to_teacher(
    extrinsics: torch.Tensor,
    source_order: Sequence[str],
    target_order: Sequence[str],
) -> torch.Tensor:
    """OpenScene extrinsics are sensor-to-lidar transforms; adapters keep this convention explicit."""

    return reorder_openscene_cameras(extrinsics, source_order, target_order)


def openscene_images_to_teacher(
    images: torch.Tensor,
    source_order: Sequence[str],
    target_order: Sequence[str],
) -> torch.Tensor:
    """OpenScene image tensor is [B, 8, 3, H, W]; teachers receive [B, N_cam, 3, H, W]."""

    return reorder_openscene_cameras(images, source_order, target_order)


def stream_petr_output_to_quest(raw_output: Mapping[str, Any]) -> dict[str, Any]:
    """Convert StreamPETR lidar-coordinate detections to QUEST agent teacher fields after class mapping is verified."""

    raise TeacherUnavailableError("StreamPETR output conversion needs a real model output object; no inference output available.")


def maptr_output_to_quest(raw_output: Mapping[str, Any]) -> dict[str, Any]:
    """Convert MapTRv2 map vectors to normalized QUEST polylines after OpenScene map taxonomy is verified."""

    raise TeacherUnavailableError(OPENSCENE_MAP_CLASS_NOTE)


def occnet_output_to_quest(raw_output: Mapping[str, Any]) -> dict[str, Any]:
    """Convert OccNet occupancy logits to QUEST [B, C_occ, X, Y, Z] only after axis order and class mapping are verified."""

    raise TeacherUnavailableError("OccNet output conversion requires actual OccNet inference output and voxel metadata.")


def vidar_output_to_quest(raw_output: Mapping[str, Any]) -> dict[str, Any]:
    """Convert ViDAR future representation to QUEST flow only when the source tensor is verified as occupancy flow."""

    raise TeacherUnavailableError("ViDAR does not guarantee direct flow output; conversion must inspect real model output.")


class ExternalTeacher(nn.Module):
    required_modules = ("mmcv", "mmdet", "mmdet3d")
    checkpoint_patterns: tuple[str, ...] = ()
    output_keys: tuple[str, ...] = ()

    def __init__(self, spec: TeacherSpec) -> None:
        super().__init__()
        self.spec = spec
        self.repo_path = resolve_path(spec.repo_path)
        self.config_path = resolve_path(spec.config_path)
        self.checkpoint_path = resolve_path(spec.checkpoint_path)
        self.checkpoint: Any | None = None
        self.check_result = self._check()
        self.eval()
        self.requires_grad_(False)

    def _check(self) -> TeacherCheckResult:
        searched = [str(path) for path in checkpoint_search_roots()]
        repo_ok = bool(self.repo_path and self.repo_path.exists())
        config_ok = bool(self.config_path and self.config_path.exists())
        checkpoint_candidates = find_checkpoints(self.checkpoint_patterns)
        if self.checkpoint_path is None and checkpoint_candidates:
            self.checkpoint_path = checkpoint_candidates[0]
        checkpoint_ok = bool(self.checkpoint_path and self.checkpoint_path.exists())

        missing_dependencies = [module for module in self.required_modules if not module_available(module)]
        dependencies_ok = not missing_dependencies
        checkpoint_load_ok = False
        checkpoint_keys: list[str] = []
        errors: list[str] = []
        if checkpoint_ok and self.checkpoint_path is not None:
            try:
                self.checkpoint = load_checkpoint_cpu(self.checkpoint_path)
                checkpoint_load_ok = True
                checkpoint_keys = checkpoint_summary(self.checkpoint)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"checkpoint load failed: {type(exc).__name__}: {exc}")
        else:
            errors.append(f"checkpoint not found; searched: {searched}")

        if not repo_ok:
            errors.append(f"repo not found: {self.repo_path}")
        if not config_ok:
            errors.append(f"config not found: {self.config_path}")
        if not dependencies_ok:
            errors.append("missing dependencies: " + ", ".join(missing_dependencies))

        status = "READY_FOR_MODEL_BUILD" if repo_ok and config_ok and checkpoint_load_ok and dependencies_ok else "BLOCKED"
        return TeacherCheckResult(
            task=self.spec.task,
            name=self.spec.name,
            repo_path=self.repo_path,
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            repo_ok=repo_ok,
            config_ok=config_ok,
            checkpoint_ok=checkpoint_ok,
            checkpoint_load_ok=checkpoint_load_ok,
            dependencies_ok=dependencies_ok,
            inference_ok=False,
            frozen=all(not p.requires_grad for p in self.parameters()),
            status=status,
            searched_paths=searched,
            missing_dependencies=missing_dependencies,
            errors=errors,
            checkpoint_keys=checkpoint_keys,
        )

    def prepare_teacher_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        target_order = TEACHER_CAMERA_ORDERS.get(self.spec.name, OPENSCENE_CAMERA_NAMES)
        source_order = tuple(batch.get("camera_names", OPENSCENE_CAMERA_NAMES))
        return {
            "images": openscene_images_to_teacher(batch["images"], source_order, target_order),
            "intrinsics": openscene_intrinsics_to_teacher(batch["intrinsics"], source_order, target_order),
            "extrinsics": openscene_extrinsics_to_teacher(batch["extrinsics"], source_order, target_order),
            "ego_state": batch.get("ego_state"),
            "ego_pose": batch.get("ego_pose"),
            "metadata": batch.get("metadata"),
            "camera_order": target_order,
            "coordinate_convention": COORDINATE_CONVENTION,
        }

    def build_model(self) -> nn.Module:
        raise TeacherUnavailableError(
            f"{self.spec.name} cannot be built in this environment: {self.check_result.errors}"
        )

    @torch.no_grad()
    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        self.eval()
        if self.check_result.status != "READY_FOR_MODEL_BUILD":
            raise TeacherUnavailableError(f"{self.spec.name} BLOCKED: {'; '.join(self.check_result.errors)}")
        del batch
        raise TeacherUnavailableError(
            f"{self.spec.name} model construction is not enabled until MMDetection3D runtime integration is verified."
        )


class StreamPETRTeacher(ExternalTeacher):
    checkpoint_patterns = ("stream_petr", "streampetr")
    output_keys = ("cls_logits", "boxes_3d", "velocity", "scores", "valid_mask")


class MapTRv2Teacher(ExternalTeacher):
    checkpoint_patterns = ("maptr", "maptrv2")
    output_keys = ("cls_logits", "points", "scores", "valid_mask")


class OccNetTeacher(ExternalTeacher):
    checkpoint_patterns = ("occnet", "openocc", "occupancy")
    output_keys = ("occ_logits",)


class ViDARTeacher(ExternalTeacher):
    checkpoint_patterns = ("vidar",)
    output_keys = ("flow", "future_world", "valid_mask")


TEACHER_CLASSES = {
    "StreamPETR": StreamPETRTeacher,
    "MapTRv2": MapTRv2Teacher,
    "OccNet": OccNetTeacher,
    "ViDAR": ViDARTeacher,
}


def teacher_spec_from_config(task: str, raw: Mapping[str, Any]) -> TeacherSpec:
    extras = {
        key: value
        for key, value in raw.items()
        if key not in {"name", "enabled", "repo_path", "config_path", "checkpoint_path", "outputs"}
    }
    return TeacherSpec(
        task=task,
        name=str(raw.get("name", task)),
        enabled=bool(raw.get("enabled", False)),
        repo_path=raw.get("repo_path"),
        config_path=raw.get("config_path"),
        checkpoint_path=raw.get("checkpoint_path"),
        outputs=tuple(str(item) for item in raw.get("outputs", ())),
        extra=extras,
    )


def build_teacher(spec_or_config: TeacherSpec | Mapping[str, Any], task: str | None = None) -> ExternalTeacher:
    if isinstance(spec_or_config, TeacherSpec):
        spec = spec_or_config
    else:
        if task is None:
            raise ValueError("task is required when building a teacher from a config mapping")
        spec = teacher_spec_from_config(task, spec_or_config)
    teacher_cls = TEACHER_CLASSES.get(spec.name)
    if teacher_cls is None:
        raise ValueError(f"unsupported teacher name for task {spec.task}: {spec.name}")
    teacher = teacher_cls(spec)
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


def build_enabled_teachers(config: Mapping[str, Any]) -> dict[str, ExternalTeacher]:
    teachers: dict[str, ExternalTeacher] = {}
    for task, raw in config.items():
        if not isinstance(raw, Mapping):
            continue
        spec = teacher_spec_from_config(task, raw)
        if spec.enabled:
            teachers[task] = build_teacher(spec)
    return teachers


def build_all_teachers(config: Mapping[str, Any]) -> dict[str, ExternalTeacher]:
    teachers: dict[str, ExternalTeacher] = {}
    for task, raw in config.items():
        if not isinstance(raw, Mapping) or task == "segformer":
            continue
        spec = teacher_spec_from_config(task, raw)
        teachers[task] = build_teacher(spec)
    return teachers


class TeacherInterface:
    """Compatibility wrapper used by Stage1 to reject enabled teachers."""

    def __init__(self, specs: Mapping[str, TeacherSpec]) -> None:
        self.specs = dict(specs)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TeacherInterface":
        return cls(
            {
                task: teacher_spec_from_config(task, raw)
                for task, raw in config.items()
                if isinstance(raw, Mapping)
            }
        )

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
                "repo_path": spec.repo_path,
                "config_path": spec.config_path,
                "checkpoint_path": spec.checkpoint_path,
            }
            for task, spec in self.specs.items()
        }
