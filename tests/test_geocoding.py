import json

from tools.geocoding import choose_candidate, geocode_place


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return [{
            "display_name": "Heliópolis, Bellavista-La Palmera, Sevilla, España",
            "lat": "37.3500", "lon": "-5.9810", "osm_type": "relation", "osm_id": 123,
            "category": "place", "type": "neighbourhood", "importance": 0.6,
            "boundingbox": ["37.34", "37.36", "-5.99", "-5.97"],
            "geojson": {"type": "Polygon", "coordinates": [[[-5.99, 37.34], [-5.97, 37.34], [-5.97, 37.36], [-5.99, 37.36], [-5.99, 37.34]]]},
        }]


def test_geocoder_uses_cache_and_candidate_selection(tmp_path):
    calls = []

    def requester(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeResponse()

    first = geocode_place("Heliópolis, Sevilla", tmp_path, requester=requester)
    second = geocode_place("Heliópolis, Sevilla", tmp_path, requester=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache not used")))
    assert len(calls) == 1
    assert first == second
    assert choose_candidate(first, "Heliópolis Sevilla").osm_id == 123

