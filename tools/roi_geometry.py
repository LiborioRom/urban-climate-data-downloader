from __future__ import annotations

import math

from pyproj import Transformer
from shapely.affinity import rotate as rotate_geometry
from shapely.geometry import box, mapping, shape

from schemas.conversation import ROIState
from schemas.request import ROIRequest, ROISpecification


PROJECTED_CRS = "EPSG:25830"
TO_PROJECTED = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
TO_WGS84 = Transformer.from_crs(PROJECTED_CRS, "EPSG:4326", always_xy=True)


def half_side_from_area(area_km2: float) -> float:
    if area_km2 <= 0:
        raise ValueError("La superficie del ROI debe ser positiva.")
    return math.sqrt(area_km2 * 1_000_000) / 2


def roi_polygon_projected(roi: ROIState):
    x, y = TO_PROJECTED.transform(roi.center_lon, roi.center_lat)
    polygon = box(x - roi.half_side_m, y - roi.half_side_m, x + roi.half_side_m, y + roi.half_side_m)
    return rotate_geometry(polygon, roi.rotation_deg, origin=(x, y), use_radians=False)


def roi_polygon_wgs84(roi: ROIState) -> dict:
    polygon = roi_polygon_projected(roi)
    coordinates = [TO_WGS84.transform(x, y) for x, y in polygon.exterior.coords]
    return mapping(type(polygon)(coordinates))


def create_roi(
    center_lat: float,
    center_lon: float,
    *,
    half_side_m: float | None = None,
    area_km2: float | None = None,
    place_name: str | None = None,
    display_name: str | None = None,
    boundary_geojson: dict | None = None,
    osm_type: str | None = None,
    osm_id: int | None = None,
    rotation_deg: float = 0.0,
    provenance: str = "geocoder",
) -> ROIState:
    half_side = half_side_m or (half_side_from_area(area_km2) if area_km2 else None)
    if half_side is None:
        raise ValueError("Se necesita area_km2 o half_side_m para construir el ROI.")
    roi = ROIState(
        center_lat=center_lat, center_lon=center_lon, half_side_m=half_side,
        rotation_deg=rotation_deg % 90.0,
        place_name=place_name, display_name=display_name, boundary_geojson=boundary_geojson,
        osm_type=osm_type, osm_id=osm_id, provenance=provenance,
    )
    return _with_updated_coverage(roi)


def _transform_geometry(geometry, transformer: Transformer):
    from shapely.ops import transform
    return transform(transformer.transform, geometry)


def _with_updated_coverage(roi: ROIState) -> ROIState:
    coverage = None
    if roi.boundary_geojson:
        try:
            boundary = _transform_geometry(shape(roi.boundary_geojson), TO_PROJECTED)
            if boundary.area > 0:
                covered = boundary.intersection(roi_polygon_projected(roi)).area
                coverage = 100 * covered / boundary.area
        except (TypeError, ValueError):
            pass
    return roi.model_copy(update={"coverage_percent": coverage})


def center_from_geojson(geojson: dict, fallback_lat: float, fallback_lon: float) -> tuple[float, float]:
    try:
        geometry = shape(geojson)
        projected = _transform_geometry(geometry, TO_PROJECTED)
        point = projected.centroid
        lon, lat = TO_WGS84.transform(point.x, point.y)
        return lat, lon
    except (TypeError, ValueError):
        return fallback_lat, fallback_lon


def move_roi(roi: ROIState, direction: str, distance_m: float) -> ROIState:
    if distance_m <= 0:
        raise ValueError("La distancia debe ser positiva.")
    offsets = {"north": (0, distance_m), "south": (0, -distance_m), "east": (distance_m, 0), "west": (-distance_m, 0)}
    if direction not in offsets:
        raise ValueError(f"Dirección no soportada: {direction}")
    x, y = TO_PROJECTED.transform(roi.center_lon, roi.center_lat)
    dx, dy = offsets[direction]
    lon, lat = TO_WGS84.transform(x + dx, y + dy)
    moved = roi.model_copy(deep=True, update={"center_lat": lat, "center_lon": lon, "confirmed": False})
    return _with_updated_coverage(moved)


def resize_roi(roi: ROIState, scale_factor: float) -> ROIState:
    if scale_factor <= 0:
        raise ValueError("El factor de escala debe ser positivo.")
    resized = roi.model_copy(deep=True, update={"half_side_m": roi.half_side_m * scale_factor, "confirmed": False})
    return _with_updated_coverage(resized)


def update_roi_from_geojson(roi: ROIState, geojson: dict) -> ROIState:
    """Recupera centro y giro tras arrastrar o rotar el cuadrado en Leaflet."""
    if geojson.get("type") == "Feature":
        geojson = geojson.get("geometry") or {}
    geometry = shape(geojson)
    if geometry.geom_type != "Polygon" or geometry.is_empty or not geometry.is_valid:
        raise ValueError("El editor cartográfico no devolvió un polígono válido.")
    projected = _transform_geometry(geometry, TO_PROJECTED)
    expected_area = (2 * roi.half_side_m) ** 2
    if expected_area <= 0 or abs(projected.area - expected_area) / expected_area > 0.15:
        raise ValueError("El cuadrado editado no conserva las dimensiones previstas.")

    centroid = projected.centroid
    lon, lat = TO_WGS84.transform(centroid.x, centroid.y)
    coordinates = list(projected.exterior.coords)
    first, second = next(
        (start, end)
        for start, end in zip(coordinates, coordinates[1:])
        if math.dist(start, end) > 1e-6
    )
    rotation = math.degrees(math.atan2(second[1] - first[1], second[0] - first[0])) % 90.0
    if math.isclose(rotation, 90.0, abs_tol=1e-6):
        rotation = 0.0
    updated = roi.model_copy(
        deep=True,
        update={
            "center_lat": lat,
            "center_lon": lon,
            "rotation_deg": rotation,
            "confirmed": False,
            "provenance": "interactive_map",
        },
    )
    return _with_updated_coverage(updated)


def as_request(roi: ROIState) -> ROIRequest:
    return ROIRequest(
        specification=ROISpecification.CENTER_AND_HALF_SIDE,
        place_name=roi.place_name,
        center_lat=roi.center_lat,
        center_lon=roi.center_lon,
        half_side_m=roi.half_side_m,
        rotation_deg=roi.rotation_deg,
    )
