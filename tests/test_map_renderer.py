from tools.map_renderer import render_roi_map
from tools.roi_geometry import create_roi


def test_roi_map_contains_layers_and_center():
    roi = create_roi(37.3546, -5.9846, area_km2=2.0, display_name="Heliópolis, Sevilla")
    html = render_roi_map(roi).get_root().render()
    assert "OpenStreetMap" in html
    assert "ROI propuesto" in html
    assert "37.3546" in html
