from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SPLIT_SCHEMA_VERSION = 1


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_hash(train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]) -> str:
    identity = {
        name: [(int(row["index"]), str(row["token"]), str(row["scene_token"]), str(row["city"]))
               for row in rows]
        for name, rows in (("train", train), ("validation", validation))
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_split(manifest: Mapping[str, Any], infos: Sequence[Mapping[str, Any]],
                   metadata_path: str | Path) -> None:
    if manifest.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError("Stage 3 split schema is unsupported")
    if manifest.get("metadata_sha256") != file_sha256(metadata_path):
        raise ValueError("Stage 3 split metadata SHA256 mismatch")
    train = manifest.get("train", {}).get("frames")
    validation = manifest.get("validation", {}).get("frames")
    if not isinstance(train, list) or not isinstance(validation, list):
        raise ValueError("Stage 3 split lacks full frame lists")
    if not validation:
        raise ValueError("Stage 3 split has no validation frames")
    if [row.get("index") for row in train] != list(range(5000)):
        raise ValueError("Stage 3 train split must preserve original indices 0-4999")
    indices: set[int] = set()
    tokens: set[str] = set()
    train_scenes = {str(row["scene_token"]) for row in train}
    for split_name, rows in (("train", train), ("validation", validation)):
        for row in rows:
            index = row.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(infos) or index in indices:
                raise ValueError(f"Stage 3 {split_name} has duplicate/out-of-range index {index}")
            info = infos[index]
            token = str(info["token"])
            if token != row.get("token") or str(info["scene_token"]) != row.get("scene_token"):
                raise ValueError(f"Stage 3 split index/token/scene mismatch at {index}")
            if not isinstance(row.get("city"), str) or not row["city"]:
                raise ValueError(f"Stage 3 split has no city at {index}")
            if token in tokens:
                raise ValueError(f"Stage 3 split has duplicate token {token}")
            indices.add(index)
            tokens.add(token)
            if split_name == "validation":
                if index < 5000 or str(info["scene_token"]) in train_scenes:
                    raise ValueError(f"Stage 3 validation overlaps training scene/index at {index}")
                if float(row.get("nearest_train_m", -1)) < 200:
                    raise ValueError(f"Stage 3 validation is not spatially isolated at {index}")
    if manifest.get("split_sha256") != split_hash(train, validation):
        raise ValueError("Stage 3 split frame-list hash mismatch")


def load_stage3_split(path: str | Path, infos: Sequence[Mapping[str, Any]],
                      metadata_path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    validate_split(manifest, infos, metadata_path)
    return manifest


def selected_frames(manifest: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    if split not in ("train", "validation"):
        raise ValueError(f"unknown Stage 3 split: {split}")
    return list(manifest[split]["frames"])
