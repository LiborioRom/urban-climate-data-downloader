import pytest
from shapely.geometry import shape

from tools.roi_geometry import (
    TO_PROJECTED,
    create_roi,
    half_side_from_area,
    move_roi,
    resize_roi,
    roi_polygon_projected,
    roi_polygon_wgs84,
    update_roi_from_geojson,
)


def test_two_square_kilometres_produces_expected_half_side():
    assert half_side_from_area(2.0) == pytest.approx(707.10678, rel=1e-6)


def test_move_roi_uses_metric_projection():
    roi = create_roi(37.4, -5.98, area_km2=2.0)
    moved = move_roi(roi, "west", 500)
    old_x, old_y = TO_PROJECTED.transform(roi.center_lon, roi.center_lat)
    new_x, new_y = TO_PROJECTED.transform(moved.center_lon, moved.center_lat)
    assert moved.center_lon < roi.center_lon
    assert new_x == pytest.approx(old_x - 500, abs=0.01)
    assert new_y == pytest.approx(old_y, abs=0.01)
    assert moved.half_side_m == roi.half_side_m
    assert not moved.confirmed


def test_resize_changes_linear_size_and_quadratic_area():
    roi = create_roi(37.4, -5.98, area_km2=2.0)
    resized = resize_roi(roi, 2)
    assert resized.half_side_m == pytest.approx(2 * roi.half_side_m)
    assert resized.area_km2 == pytest.approx(4 * roi.area_km2)


def test_coverage_is_recomputed_after_move():
    boundary = {
        "type": "Polygon",
        "coordinates": [[[-5.99, 37.39], [-5.97, 37.39], [-5.97, 37.41], [-5.99, 37.41], [-5.99, 37.39]]],
    }
    roi = create_roi(37.4, -5.98, area_km2=1.0, boundary_geojson=boundary)
    moved = move_roi(roi, "east", 5000)
    assert roi.coverage_percent is not None and roi.coverage_percent > 0
    assert moved.coverage_percent == pytest.approx(0.0)


def test_rotated_roi_preserves_square_area():
    roi = create_roi(37.4, -5.98, half_side_m=750, rotation_deg=32)
    polygon = roi_polygon_projected(roi)

    assert polygon.area == pytest.approx(1500**2, rel=1e-9)
    assert shape(roi_polygon_wgs84(roi)).is_valid


def test_geojson_editor_recovers_center_and_rotation():
    roi = create_roi(37.4, -5.98, half_side_m=750, rotation_deg=27)
    updated = update_roi_from_geojson(roi, {"type": "Feature", "properties": {}, "geometry": roi_polygon_wgs84(roi)})

    assert updated.center_lat == pytest.approx(roi.center_lat, abs=1e-7)
    assert updated.center_lon == pytest.approx(roi.center_lon, abs=1e-7)
    assert updated.rotation_deg == pytest.approx(27, abs=0.05)
    assert updated.half_side_m == roi.half_side_m
    assert not updated.confirmed
