from __future__ import annotations

import json

import folium
from branca.element import MacroElement, Template
from folium.elements import JSCSSMixin
from folium.plugins import Geocoder

from schemas.conversation import ROIState
from tools.roi_geometry import roi_polygon_wgs84


class EditableROIPlugin(JSCSSMixin, MacroElement):
    """Añade arrastre y rotación Geoman a un único ROI controlado por Python."""

    default_js = [
        (
            "leaflet_geoman_js",
            "https://unpkg.com/@geoman-io/leaflet-geoman-free@2.18.3/dist/leaflet-geoman.min.js",
        )
    ]
    default_css = [
        (
            "leaflet_geoman_css",
            "https://unpkg.com/@geoman-io/leaflet-geoman-free@2.18.3/dist/leaflet-geoman.css",
        )
    ]

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        var drawnItems = L.featureGroup().addTo({{ this._parent.get_name() }});
        window.drawnItems = drawnItems;
        var roiGeometry = {{ this.roi_geojson | safe }};
        var roiLayer = L.geoJSON(roiGeometry, {
            bubblingMouseEvents: false,
            style: {
                color: "#dc2626",
                weight: 4,
                fillColor: "#ef4444",
                fillOpacity: 0.18
            }
        }).getLayers()[0];
        roiLayer.options.pmIgnore = false;
        roiLayer.feature = {
            type: "Feature",
            properties: {kind: "editable_roi"},
            geometry: roiGeometry
        };
        roiLayer.bindTooltip("Arrastra para mover · usa ↻ para rotar");
        drawnItems.addLayer(roiLayer);

        {{ this._parent.get_name() }}.pm.setLang("es");
        {{ this._parent.get_name() }}.pm.addControls({
            position: "topleft",
            drawMarker: false,
            drawCircleMarker: false,
            drawPolyline: false,
            drawRectangle: false,
            drawPolygon: false,
            drawCircle: false,
            drawText: false,
            editMode: false,
            dragMode: true,
            cutPolygon: false,
            removalMode: false,
            rotateMode: true
        });
        {{ this._parent.get_name() }}.pm.enableGlobalDragMode();

        function publishRoiGeometry(event) {
            roiLayer.feature.geometry = roiLayer.toGeoJSON().geometry;
            {{ this._parent.get_name() }}.fire("draw:edited", {
                layer: roiLayer,
                layers: drawnItems,
                layerType: "polygon",
                sourceEvent: event
            });
        }
        roiLayer.on("pm:dragend", publishRoiGeometry);
        roiLayer.on("pm:rotateend", publishRoiGeometry);
        {% endmacro %}
        """
    )

    def __init__(self, roi: ROIState):
        super().__init__()
        self._name = "EditableROIPlugin"
        self.roi_geojson = json.dumps(roi_polygon_wgs84(roi), ensure_ascii=False)


def render_editable_roi_map(roi: ROIState | None, saved_rois: list[tuple[str, ROIState]] | None = None) -> folium.Map:
    saved_rois = saved_rois or []
    fallback = saved_rois[0][1] if saved_rois else None
    center_roi = roi or fallback
    center = [center_roi.center_lat, center_roi.center_lon] if center_roi else [37.3891, -5.9845]
    map_view = folium.Map(location=center, zoom_start=15 if roi else 12, tiles="OpenStreetMap")
    Geocoder(
        collapsed=False,
        position="topright",
        add_marker=True,
        zoom=16,
        provider="nominatim",
        placeholder="Buscar ubicación…",
        provider_options={"geocodingQueryParams": {"countrycodes": "es", "accept-language": "es"}},
    ).add_to(map_view)
    for name, saved_roi in saved_rois:
        folium.GeoJson(
            roi_polygon_wgs84(saved_roi),
            name=name,
            style_function=lambda _feature: {
                "color": "#2563eb", "weight": 3, "fillColor": "#60a5fa", "fillOpacity": 0.12,
            },
            tooltip=f"ROI guardado: {name}",
        ).add_to(map_view)
    if roi:
        EditableROIPlugin(roi).add_to(map_view)
    return map_view
