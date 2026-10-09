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


def check_map_layer_counts(maps_db: Any, location: str, *, all_map_layers: bool = False) -> None:
    """Compare source and API-loaded rows; warnings alone cannot prove no loss."""
    gpkg_path = maps_db.get_gpkg_path_and_store_on_disk(location)
    required = ("baseline_paths", "lanes_polygons", "lane_connectors",
                "gen_lane_connectors_scaled_width_polygons")
    layers = required + (("lane_groups_polygons", "intersections", "crosswalks",
                          "carpark_areas") if all_map_layers else ())
    with sqlite3.connect(f"file:{Path(gpkg_path).resolve().as_posix()}?mode=ro", uri=True) as db:
        available = {row[0] for row in db.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type = 'features'"
        )}
        if not set(required).issubset(available):
            raise ValueError(f"{location} is missing baseline/lane source layers: {set(required) - available}")
        for layer in layers:
            if layer not in available:
                continue
            # Table names come only from gpkg_contents, never user input.
            source_count = int(db.execute(f'SELECT COUNT(*) FROM "{layer}"').fetchone()[0])
            loaded = maps_db.load_vector_layer(location, layer)
            loaded_count = len(loaded)
            missing_geometry = int(loaded.geometry.isna().sum())
            empty_geometry = int(loaded.geometry.is_empty.sum())
            invalid_geometry = int((~loaded.geometry.is_valid).sum()) if layer in required else 0
            print(f"map_cast_audit location={location} layer={layer} source_rows={source_count} "
                  f"api_rows={loaded_count} null_geometry={missing_geometry} "
                  f"empty_geometry={empty_geometry} invalid_required_geometry={invalid_geometry}")
            if source_count != loaded_count or missing_geometry or empty_geometry or invalid_geometry:
                raise ValueError(f"map layer {layer} may have lost valid geometry during load; inspect source")
