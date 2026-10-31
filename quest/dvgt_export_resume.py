"""Read-only validation of completed DVGT labels before resuming an export."""

from pathlib import Path

import numpy as np
import torch

from .depth_teacher import validate_depth_record, validate_intrinsic
from .dvgt_depth import DVGT_SCHEMA_VERSION, camera_from_first_ego
from .label_io import load_record
from .openscene_dataset import OPENSCENE_CAMERA_NAMES


DVGT_INFERENCE_CONTRACT = "dvgt1_native_depth_export_v1"
# SHA256 of the reviewed DVGT exporter at f3bc817 (LF).
LEGACY_EXPORTER_SHA256 = "f5edc8382b984b1c4b3e37cff0ed6b031c669b58ed8ba121665f1575eccefe74"


def inference_signature(provenance):
    """Normalize only the reviewed orchestration-only exporter revision change."""
    contract = provenance.get("inference_contract")
    if contract is not None:
        if contract != DVGT_INFERENCE_CONTRACT:
            raise ValueError("Unknown DVGT inference contract")
        return contract
    if provenance.get("adapter_sha256") == LEGACY_EXPORTER_SHA256:
        return DVGT_INFERENCE_CONTRACT
    return provenance.get("adapter_sha256")


def validate_completed_record(record, index, infos, plan, cameras, metadata_hash,
                              split_hash, provenance, settings, geometry):
    info = infos[index]
    validate_depth_record(record, str(info["token"]), index, OPENSCENE_CAMERA_NAMES)
    if (record["schema_version"] != DVGT_SCHEMA_VERSION
            or record.get("training_label_scope") != "image_plane_candidate"
            or record.get("metadata_sha256") != metadata_hash
            or record.get("split_sha256") != split_hash
            or record.get("scene_token") != str(info["scene_token"])
            or record.get("timestamp") != float(info["timestamp"])):
        raise ValueError("Existing label metadata/split/scene/timestamp/scope differs")
    source = record["provenance"]
    if (inference_signature(source) != inference_signature(provenance)
            or source.get("adapter_sha256") not in (LEGACY_EXPORTER_SHA256, provenance["adapter_sha256"])):
        raise ValueError("Unreviewed existing DVGT exporter revision")
    for key, value in provenance.items():
        if key not in ("adapter_sha256", "inference_contract") and source.get(key) != value:
            raise ValueError("Existing label teacher/configuration differs: " + key)
    position = plan["indices"].index(index)
    expected = {"source_indices": plan["indices"],
                "tokens": [str(infos[i]["token"]) for i in plan["indices"]],
                "timestamps": [float(infos[i]["timestamp"]) for i in plan["indices"]],
                "scene_token": plan["scene_token"], "target_position": position,
                "uses_future": position < len(plan["indices"]) - 1}
    for key, value in expected.items():
        if record["sequence"].get(key) != value:
            raise ValueError("Existing label temporal context differs: " + key)
    depth = record["depth"]
    if (depth["image_geometry"] != geometry or depth["min_depth_m"] != settings["min_depth_m"]
            or depth["max_depth_m"] != settings["max_depth_m"]
            or [tuple(hw) for hw in record.get("source_hw", [])] != [tuple(c["source_hw"]) for c in cameras]
            or record.get("source_paths") != [str(c["path"]) for c in cameras]):
        raise ValueError("Existing label image geometry/range/source differs")
    for view, camera in enumerate(cameras):
        for key, current in (("intrinsics", camera["scaled_intrinsic"]),
                             ("sensor2lidar", camera["sensor2lidar"]),
                             ("distortion", camera["distortion"])):
            saved = depth[key][view].double()
            current = torch.as_tensor(current, dtype=torch.float64)
            if saved.shape != current.shape or not torch.allclose(saved, current, atol=1e-5, rtol=1e-5):
                raise ValueError("Existing label calibration differs: " + camera["name"] + " " + key)
        validate_intrinsic(record["teacher_intrinsics"][view])
        transform = camera_from_first_ego(infos[plan["indices"][0]], info, camera["sensor2lidar"])
        saved = np.asarray(record["camera_from_first_ego_rdf"][view])
        if saved.shape != (4, 4) or not np.allclose(saved, transform, atol=1e-8, rtol=1e-8):
            raise ValueError("Existing label camera/ego transform differs: " + camera["name"])


def inspect_outputs(targets, infos, plans, output_dir, audit_dir, *, resume, candidate,
                    cameras_by_index, metadata_hash, split_hash, provenance, settings, geometry):
    """Skip only validated final .pt files; never overwrite partial files or diagnostics."""
    completed, blockers = set(), []
    by_target = {index: plan for plan in plans for index in plan["targets"]}
    for checked, index in enumerate(targets, 1):
        token = str(infos[index]["token"])
        final = Path(output_dir) / (token + ".pt")
        if resume and final.is_file():
            try:
                validate_completed_record(load_record(final), index, infos, by_target[index],
                                          cameras_by_index[index], metadata_hash, split_hash,
                                          provenance, settings, geometry)
            except Exception as error:
                blockers.append(f"EXISTING_LABEL_INVALID: token={token} {type(error).__name__}: {error}")
            else:
                completed.add(token)
        else:
            paths = [Path(audit_dir) / (token + ".json"), Path(audit_dir) / (token + ".png")]
            if candidate:
                paths += [final, Path(output_dir) / (token + ".pt.tmp")]
            blockers.extend("OUTPUT_EXISTS: " + str(path) for path in paths if path.exists())
        if resume and checked % 250 == 0:
            print(f"resume_check={checked}/{len(targets)} validated={len(completed)} "
                  f"blockers={len(blockers)}", flush=True)
    pending = [{**plan, "targets": [i for i in plan["targets"] if str(infos[i]["token"]) not in completed]}
               for plan in plans]
    return completed, [plan for plan in pending if plan["targets"]], blockers
