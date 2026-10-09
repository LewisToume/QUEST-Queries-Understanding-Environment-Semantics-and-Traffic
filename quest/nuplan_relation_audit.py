from __future__ import annotations

from collections import Counter
import inspect
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .vector_map_labels import require_lidar2global


BASELINE_RELATION_AUDIT_VERSION = "nuplan_baseline_relation_v1"
RELATION_COLUMNS = {"LANE": "lane_fid", "LANE_CONNECTOR": "lane_connector_fid"}


def _relation_values(series: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pandas as pd

    missing = np.asarray(series.isna(), dtype=bool)
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    valid = np.isfinite(values) & (values >= 0) & (values <= 2**53)
    valid &= np.equal(values, np.floor(np.where(np.isfinite(values), values, 0)))
    invalid_non_null = ~missing & ~valid
    return values, valid, invalid_non_null


def _intersecting_positions(frame: Any, polygon: Any) -> np.ndarray:
    try:
        return np.asarray(frame.sindex.query(polygon, predicate="intersects"), dtype=np.int64)
    except (AttributeError, ImportError, TypeError, ValueError):
        return np.flatnonzero(np.asarray(frame.geometry.intersects(polygon), dtype=bool))


class NuPlanBaselineRelationAudit:
    """Audit the ID relations used by nuPlan get_all_rows_with_value, without patching it."""

    def __init__(self, maps_db: Any) -> None:
        self.maps_db = maps_db
        self._cache: dict[str, dict[str, Any]] = {}

    def _city(self, location: str) -> dict[str, Any]:
        if location not in self._cache:
            paths = self.maps_db.load_vector_layer(location, "baseline_paths")
            if "geometry" not in paths or "fid" not in paths:
                raise ValueError(f"{location} baseline_paths lacks geometry/fid")
            unlocatable = int(paths.geometry.isna().sum()) + int(paths.geometry.is_empty.sum())
            if unlocatable:
                raise ValueError(f"{location} baseline_paths has {unlocatable} unlocatable geometries; ROI impact unknown")
            values, valid, invalid = {}, {}, {}
            for column in RELATION_COLUMNS.values():
                if column not in paths:
                    raise ValueError(f"{location} baseline_paths lacks {column}")
                values[column], valid[column], invalid[column] = _relation_values(paths[column])
            counts = {
                name: Counter(values[column][valid[column]].astype(np.int64).tolist())
                for name, column in RELATION_COLUMNS.items()
            }
            self._cache[location] = {
                "paths": paths, "values": values, "valid": valid, "invalid": invalid,
                "counts": counts,
            }
        return self._cache[location]

    def audit(self, location: str, info: Mapping[str, Any], map_api: Any,
              xy_range: tuple[float, float, float, float]) -> dict[str, Any]:
        from shapely.geometry import Polygon, box
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer

        city = self._city(location)
        matrix = require_lidar2global(info)
        x0, y0, x1, y1 = xy_range
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        projected = corners @ matrix[:2, :2].T + matrix[:2, 3]
        roi_global = Polygon(projected)
        if roi_global.is_empty or not roi_global.is_valid or roi_global.area <= 0:
            raise ValueError("LiDAR ROI has no valid projected map footprint")
        invalid_paths = city["paths"].loc[~city["paths"].geometry.is_valid]
        for fid, geometry in invalid_paths.geometry.items():
            bounds = geometry.bounds
            if (len(bounds) != 4 or not all(np.isfinite(value) for value in bounds)
                    or box(*bounds).intersects(roi_global)):
                raise ValueError(f"{location} invalid baseline fid={fid} may affect current ROI")
        roi_rows = _intersecting_positions(city["paths"], roi_global)
        radius = max(abs(x0), abs(x1), abs(y0), abs(y1)) * 2**0.5 + 10.0
        layers = [getattr(SemanticMapLayer, name) for name in RELATION_COLUMNS]
        objects = map_api.get_proximal_map_objects(
            Point2D(float(matrix[0, 3]), float(matrix[1, 3])), radius, layers
        )
        candidate_ids = {}
        for name, layer in zip(RELATION_COLUMNS, layers):
            candidates = objects.get(layer, [])
            if any(item is None for item in candidates):
                raise ValueError(f"{location} {name} candidate could not be constructed")
            ids = [int(item.id) for item in candidates]
            if any(value < 0 or value > 2**53 for value in ids):
                raise ValueError(f"{location} {name} candidate ID is outside auditable integer range")
            if len(ids) != len(set(ids)):
                raise ValueError(f"{location} {name} candidate IDs are duplicated")
            candidate_ids[name] = set(ids)
        diagnostics: dict[str, Any] = {
            "version": BASELINE_RELATION_AUDIT_VERSION,
            "status": "unverified", "roi_baseline_rows": len(roi_rows),
            "candidate_objects": {name: len(ids) for name, ids in candidate_ids.items()},
            "fields": {}, "unresolved_roi_rows": [], "missing_candidate_relations": [],
            "unresolved_global_fields": [],
        }
        for name, column in RELATION_COLUMNS.items():
            values = city["values"][column]
            valid = city["valid"][column]
            invalid = city["invalid"][column]
            diagnostics["fields"][column] = {
                "null_rows": int(np.asarray(city["paths"][column].isna(), dtype=bool).sum()),
                "invalid_non_null_rows": int(invalid.sum()),
                "invalid_non_null_roi_rows": int(invalid[roi_rows].sum()),
                "valid_association_rows": int(valid.sum()),
                "invalid_examples": [str(item) for item in city["paths"].loc[invalid, column].head(5)],
            }
            if bool(invalid.any()):
                # A fractional/non-numeric value outside the ROI could still cast to a
                # queried ID, so its impact cannot be dismissed by spatial location.
                diagnostics["unresolved_global_fields"].append(column)
            for object_id in candidate_ids[name]:
                count = city["counts"][name][object_id]
                if count != 1:
                    diagnostics["missing_candidate_relations"].append(
                        {"layer": name, "id": object_id, "baseline_rows": count}
                    )
        for position in roi_rows:
            links = []
            has_invalid = False
            for name, column in RELATION_COLUMNS.items():
                has_invalid |= bool(city["invalid"][column][position])
                if city["valid"][column][position]:
                    links.append((name, int(city["values"][column][position])))
            if has_invalid or len(links) != 1 or links[0][1] not in candidate_ids[links[0][0]]:
                diagnostics["unresolved_roi_rows"].append(str(city["paths"].iloc[position]["fid"]))
        if (diagnostics["missing_candidate_relations"] or diagnostics["unresolved_roi_rows"]
                or diagnostics["unresolved_global_fields"]):
            raise ValueError(f"nuPlan baseline relation audit cannot rule out missing map elements: {diagnostics}")
        diagnostics["status"] = "verified_no_baseline_relation_omission_in_roi"
        return diagnostics


def classify_invalid_cast_warnings(caught: list[Any]) -> list[dict[str, Any]]:
    from nuplan.common.maps.nuplan_map.utils import get_all_rows_with_value

    source_lines, first_line = inspect.getsourcelines(get_all_rows_with_value)
    cast_lines = {first_line + offset for offset, line in enumerate(source_lines)
                  if ".astype(int)" in line}
    source_path = Path(inspect.getsourcefile(get_all_rows_with_value)).resolve()
    if not cast_lines:
        raise RuntimeError("nuPlan get_all_rows_with_value cast site changed; relation audit needs review")
    counts: Counter[tuple[str, int, str]] = Counter()
    for warning in caught:
        if "invalid value encountered in cast" not in str(warning.message):
            continue
        source = Path(warning.filename).resolve()
        if source != source_path or warning.lineno not in cast_lines:
            raise RuntimeError(f"invalid cast from unaudited source {source}:{warning.lineno}")
        counts[("nuplan_map/utils.py", int(warning.lineno), str(warning.message))] += 1
    return [{"source": source, "line": line, "message": message, "count": count}
            for (source, line, message), count in sorted(counts.items())]


def validate_relation_audit(diagnostics: Mapping[str, Any]) -> None:
    if (diagnostics.get("version") != BASELINE_RELATION_AUDIT_VERSION
            or diagnostics.get("status") != "verified_no_baseline_relation_omission_in_roi"
            or diagnostics.get("unresolved_roi_rows") != []
            or diagnostics.get("missing_candidate_relations") != []
            or diagnostics.get("unresolved_global_fields") != []):
        raise ValueError("nuPlan baseline relation audit is missing or unresolved")
    warnings = diagnostics.get("invalid_cast_warnings")
    if (not isinstance(warnings, list)
            or diagnostics.get("invalid_cast_warning_count") != sum(
                item.get("count", 0) for item in warnings if isinstance(item, dict)
            )):
        raise ValueError("nuPlan invalid-cast warning provenance is incomplete")
    if any(not isinstance(item, dict) or item.get("source") != "nuplan_map/utils.py"
           or not isinstance(item.get("line"), int)
           or not isinstance(item.get("count"), int) or item["count"] <= 0
           or "invalid value encountered in cast" not in str(item.get("message"))
           for item in warnings):
        raise ValueError("nuPlan invalid-cast warning source is not audited")
    fields = diagnostics.get("fields")
    if not isinstance(fields, dict) or set(fields) != set(RELATION_COLUMNS.values()):
        raise ValueError("nuPlan baseline relation fields are incomplete")
    for column in RELATION_COLUMNS.values():
        field = fields[column]
        if (not isinstance(field, dict)
                or not all(isinstance(field.get(key), int) and field[key] >= 0 for key in
                           ("null_rows", "invalid_non_null_rows", "invalid_non_null_roi_rows",
                            "valid_association_rows"))
                or field["invalid_non_null_rows"] != 0
                or field["invalid_non_null_roi_rows"] != 0):
            raise ValueError(f"nuPlan {column} has unresolved invalid IDs in the ROI")
    candidates = diagnostics.get("candidate_objects")
    if (not isinstance(candidates, dict) or set(candidates) != set(RELATION_COLUMNS)
            or not all(isinstance(value, int) and value >= 0 for value in candidates.values())
            or not isinstance(diagnostics.get("roi_baseline_rows"), int)
            or diagnostics["roi_baseline_rows"] < 0):
        raise ValueError("nuPlan baseline relation audit counts are incomplete")
