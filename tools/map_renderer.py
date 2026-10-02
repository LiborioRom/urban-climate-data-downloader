from __future__ import annotations

import folium

from schemas.conversation import ROIState
from tools.roi_geometry import roi_polygon_wgs84


def render_roi_map(roi: ROIState, node_spacing_m: float | None = None) -> folium.Map:
    map_view = folium.Map(location=[roi.center_lat, roi.center_lon], zoom_start=15, tiles="OpenStreetMap")
    if roi.boundary_geojson:
        folium.GeoJson(
            roi.boundary_geojson,
            name="Lugar de referencia",
            style_function=lambda _: {"color": "#2563eb", "weight": 2, "fillOpacity": 0.08},
        ).add_to(map_view)
    folium.GeoJson(
        roi_polygon_wgs84(roi),
        name="ROI propuesto",
        style_function=lambda _: {"color": "#dc2626", "weight": 3, "fillColor": "#ef4444", "fillOpacity": 0.15},
        tooltip=f"ROI {roi.area_km2:.3f} km²",
    ).add_to(map_view)
    folium.Marker(
        [roi.center_lat, roi.center_lon],
        tooltip=f"Centro: {roi.center_lat:.6f}, {roi.center_lon:.6f}",
        icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
    ).add_to(map_view)
    folium.LayerControl().add_to(map_view)
    return map_view

