from schemas.conversation import ROIState
from tools.editable_map import render_editable_roi_map


def test_editable_map_contains_search_drag_and_rotation_controls():
    roi = ROIState(center_lat=37.4, center_lon=-5.98, half_side_m=750, rotation_deg=15)
    html = render_editable_roi_map(roi).get_root().render()

    assert "leaflet-geoman" in html
    assert "enableGlobalDragMode" in html
    assert 'roiLayer.on("pm:dragend"' in html
    assert 'roiLayer.on("pm:rotateend"' in html
    assert "Control.Geocoder" in html


def test_editable_map_shows_saved_rois_without_making_them_editable():
    active = ROIState(center_lat=37.4, center_lon=-5.98, half_side_m=500)
    saved = ROIState(center_lat=37.36, center_lon=-5.99, half_side_m=600, confirmed=True)

    html = render_editable_roi_map(active, [("Heliópolis", saved)]).get_root().render()

    assert "ROI guardado: Heli" in html
    assert html.count("EditableROIPlugin") == 0
    assert html.count("var roiLayer") == 1
