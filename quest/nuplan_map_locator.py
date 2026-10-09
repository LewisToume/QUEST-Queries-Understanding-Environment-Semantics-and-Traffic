from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from .vector_map_labels import require_lidar2global


def projected_map_bounds(gpkg_path: str | Path) -> tuple[float, float, float, float]:
    """Read the real GPKG layer extents and project them like the nuPlan map API."""
    from pyproj import CRS, Transformer

    with sqlite3.connect(f"file:{Path(gpkg_path).resolve().as_posix()}?mode=ro", uri=True) as db:
        row = db.execute("SELECT value FROM meta WHERE key = 'projectedCoordSystem'").fetchone()
        if row is None or not row[0]:
            raise ValueError(f"GPKG has no projectedCoordSystem: {gpkg_path}")
        destination = CRS.from_user_input(row[0])
        layers = db.execute(
            "SELECT c.min_x, c.min_y, c.max_x, c.max_y, s.organization, "
            "s.organization_coordsys_id, s.definition FROM gpkg_contents c "
            "JOIN gpkg_spatial_ref_sys s ON c.srs_id = s.srs_id "
            "WHERE c.data_type = 'features'"
        ).fetchall()
    bounds = []
    for x0, y0, x1, y1, organization, identifier, definition in layers:
        if None in (x0, y0, x1, y1) or not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            continue
        if x0 > x1 or y0 > y1:
            raise ValueError(f"GPKG has inverted geographic extent: {gpkg_path}")
        source = CRS.from_user_input(
            f"EPSG:{identifier}" if organization == "EPSG" else definition
        )
        box = Transformer.from_crs(source, destination, always_xy=True).transform_bounds(
            x0, y0, x1, y1, densify_pts=21
        )
        if not all(math.isfinite(v) for v in box):
            raise ValueError(f"GPKG extent did not project finitely: {gpkg_path}")
        bounds.append(box)
    if not bounds:
        raise ValueError(f"GPKG has no usable feature extents: {gpkg_path}")
    return (min(b[0] for b in bounds), min(b[1] for b in bounds),
            max(b[2] for b in bounds), max(b[3] for b in bounds))


class NuPlanMapLocator:
    def __init__(self, maps_db: Any) -> None:
        self.bounds = {
            str(location): projected_map_bounds(maps_db.get_gpkg_path_and_store_on_disk(location))
            for location in maps_db.get_locations()
        }
        if not self.bounds:
            raise ValueError("nuPlan maps DB has no locations")
        self.scene_locations: dict[str, str] = {}

    def resolve(self, info: Mapping[str, Any]) -> str:
        x, y = require_lidar2global(info)[:2, 3]
        matches = [location for location, (x0, y0, x1, y1) in self.bounds.items()
                   if x0 <= x <= x1 and y0 <= y <= y1]
        if len(matches) != 1:
            raise ValueError(f"global XY ({x:.3f}, {y:.3f}) matches {len(matches)} nuPlan maps: {matches}")
        location = matches[0]
        stated = info.get("map_location")
        if stated and str(stated) != location:
            raise ValueError(f"metadata map_location={stated} disagrees with projected XY map={location}")
        scene = str(info.get("scene_token", ""))
        if not scene:
            raise ValueError("OpenScene frame has no scene_token for map consistency check")
        previous = self.scene_locations.setdefault(scene, location)
        if previous != location:
            raise ValueError(f"scene_token={scene} changed map from {previous} to {location}")
        return location


MAP_LAYER_AUDIT_VERSION = "gpkg_source_vs_api_roi_v1"
MAP_EXPORT_REQUIRED_LAYERS = (
    "baseline_paths", "lanes_polygons", "lane_connectors",
    "gen_lane_connectors_scaled_width_polygons", "lane_groups_polygons",
    "intersections", "crosswalks",
)
MAP_EXPORT_OPTIONAL_LAYERS = ("carpark_areas",)


def _bad_geometry_rows(frame: Any) -> dict[str, Any]:
    bad = frame.geometry.isna() | frame.geometry.is_empty | ~frame.geometry.is_valid
    return {str(fid): geometry for fid, geometry in frame.loc[bad, "geometry"].items()}


def _compare_map_layer(gpkg_path: str | Path, layer: str, maps_db: Any,
                       location: str, source_count: int) -> dict[str, Any]:
    import pyogrio
    from pyproj import Transformer
    from shapely.ops import transform

    # Match the official loader's fid_as_index so row identity, not just length, is checked.
    source = pyogrio.read_dataframe(str(gpkg_path), layer=layer, fid_as_index=True)
    loaded = maps_db.load_vector_layer(location, layer)
    source.index = source.index.map(str)
    loaded_ids = {str(fid) for fid in loaded.index}
    source_ids = set(source.index)
    if (len(source) != source_count or len(source_ids) != len(source)
            or len(loaded_ids) != len(loaded) or source_ids != loaded_ids):
        raise ValueError(
            f"map layer {location}/{layer} lost or added records: "
            f"gpkg_rows={source_count} raw_rows={len(source)} api_rows={len(loaded)} "
            f"missing_ids={sorted(source_ids - loaded_ids)[:10]} "
            f"extra_ids={sorted(loaded_ids - source_ids)[:10]}"
        )
    if source.crs is None or loaded.crs is None:
        raise ValueError(f"map layer {location}/{layer} lacks source or API CRS")
    source_bad = _bad_geometry_rows(source)
    loaded_bad = _bad_geometry_rows(loaded)
    new_bad = set(loaded_bad) - set(source_bad)
    if new_bad:
        raise ValueError(f"map layer {location}/{layer} gained invalid geometry: {sorted(new_bad)[:10]}")
    projector = Transformer.from_crs(source.crs, loaded.crs, always_xy=True)
    existing_bad = {}
    for fid, geometry in source_bad.items():
        if geometry is None or geometry.is_empty:
            raise ValueError(f"map layer {location}/{layer} source geometry fid={fid} has no locatable footprint")
        try:
            source_projected = transform(projector.transform, geometry)
            loaded_geometry = loaded.loc[fid].geometry
            existing_bad[fid] = (source_projected, loaded_geometry if fid in loaded_bad else None)
        except Exception as error:
            raise ValueError(f"map layer {location}/{layer} cannot locate source-invalid fid={fid}") from error
    return {"source_rows": source_count, "api_rows": len(loaded),
            "source_invalid_rows": len(source_bad), "api_invalid_rows": len(loaded_bad),
            "existing_invalid": existing_bad}


def _frame_layer_impact(layer: str, comparison: Mapping[str, Any], roi_global: Any) -> dict[str, Any]:
    from shapely.geometry import box

    outside = []
    for fid, geometries in comparison["existing_invalid"].items():
        for geometry in geometries:
            if geometry is None:
                continue
            try:
                bounds = geometry.bounds
                if (geometry.is_empty or len(bounds) != 4
                        or not all(math.isfinite(value) for value in bounds)
                        or box(*bounds).intersects(roi_global)):
                    raise ValueError(f"map layer {layer} source-invalid fid={fid} affects this frame ROI")
            except ValueError:
                raise
            except Exception as error:
                raise ValueError(f"map layer {layer} cannot determine ROI impact of fid={fid}") from error
        outside.append(fid)
    return {"source_rows": comparison["source_rows"], "api_rows": comparison["api_rows"],
            "source_invalid_rows": comparison["source_invalid_rows"],
            "api_invalid_rows": comparison["api_invalid_rows"],
            "preexisting_invalid_outside_roi": len(outside),
            "preexisting_invalid_examples": sorted(outside)[:10],
            "new_invalid_rows": 0, "missing_rows": 0}


def check_map_layer_counts(
    maps_db: Any, location: str, info: Mapping[str, Any],
    xy_range: tuple[float, float, float, float], *,
    cache: dict[str, Any], all_map_layers: bool = False,
) -> dict[str, Any]:
    """Cache source/API differences per city, but certify old defects per frame ROI."""
    import numpy as np
    from shapely.geometry import Polygon

    if location not in cache:
        gpkg_path = maps_db.get_gpkg_path_and_store_on_disk(location)
        layers = MAP_EXPORT_REQUIRED_LAYERS + MAP_EXPORT_OPTIONAL_LAYERS
        if all_map_layers:
            layers += ("stop_polygons", "boundaries")
        with sqlite3.connect(f"file:{Path(gpkg_path).resolve().as_posix()}?mode=ro", uri=True) as db:
            available = {row[0] for row in db.execute(
                "SELECT table_name FROM gpkg_contents WHERE data_type = 'features'"
            )}
            if not set(MAP_EXPORT_REQUIRED_LAYERS).issubset(available):
                raise ValueError(f"{location} lacks required source layers: {set(MAP_EXPORT_REQUIRED_LAYERS) - available}")
            comparisons = {}
            for layer in layers:
                if layer not in available:
                    continue
                # Layer names are fixed above, never interpolated from metadata or CLI.
                count = int(db.execute(f'SELECT COUNT(*) FROM "{layer}"').fetchone()[0])
                comparisons[layer] = _compare_map_layer(gpkg_path, layer, maps_db, location, count)
        cache[location] = comparisons
    matrix = require_lidar2global(info)
    x0, y0, x1, y1 = xy_range
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    roi_global = Polygon(corners @ matrix[:2, :2].T + matrix[:2, 3])
    if roi_global.is_empty or not roi_global.is_valid or roi_global.area <= 0:
        raise ValueError("LiDAR ROI has no valid projected map footprint")
    per_layer = {layer: _frame_layer_impact(layer, result, roi_global)
                 for layer, result in cache[location].items()}
    diagnostics = {"version": MAP_LAYER_AUDIT_VERSION, "status": "verified_for_frame_roi",
                   "per_layer": per_layer}
    warnings = {name: row["preexisting_invalid_outside_roi"]
                for name, row in per_layer.items() if row["preexisting_invalid_outside_roi"]}
    if warnings:
        print(f"map_layer_preexisting_invalid_outside_roi location={location} token={info['token']} counts={warnings}")
    return diagnostics


def validate_map_layer_audit(diagnostics: Mapping[str, Any]) -> None:
    if (diagnostics.get("version") != MAP_LAYER_AUDIT_VERSION
            or diagnostics.get("status") != "verified_for_frame_roi"):
        raise ValueError("nuPlan map-layer ROI audit is missing or obsolete")
    layers = diagnostics.get("per_layer")
    if not isinstance(layers, Mapping) or not set(MAP_EXPORT_REQUIRED_LAYERS).issubset(layers):
        raise ValueError("nuPlan map-layer audit does not cover all exported required layers")
    for layer, result in layers.items():
        if not isinstance(result, Mapping):
            raise ValueError(f"nuPlan map-layer audit has no counts for {layer}")
        fields = ("source_rows", "api_rows", "source_invalid_rows", "api_invalid_rows",
                  "preexisting_invalid_outside_roi", "new_invalid_rows", "missing_rows")
        if any(not isinstance(result.get(key), int) or result[key] < 0 for key in fields):
            raise ValueError(f"nuPlan map-layer audit has invalid counts for {layer}")
        if (result["source_rows"] != result["api_rows"]
                or result["api_invalid_rows"] > result["source_invalid_rows"]
                or result["new_invalid_rows"]
                or result["missing_rows"] or result["source_invalid_rows"]
                != result["preexisting_invalid_outside_roi"]):
            raise ValueError(f"nuPlan map-layer audit leaves unresolved map geometry in {layer}")
