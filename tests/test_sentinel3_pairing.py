import pandas as pd

from tools.sentinel3_pairing import PAIR_COLUMNS, build_landsat_sentinel_pairs


def test_pairing_requires_same_date_close_hour_and_nearest_pixel():
    landsat = pd.DataFrame({
        "node_id": ["n1", "n2", "n3"],
        "date": ["2024-07-01", "2024-07-01", "2024-07-02"],
        "acquisition_dt_utc": ["2024-07-01T10:45Z", "2024-07-01T10:45Z", "2024-07-02T10:45Z"],
        "latitude": [37.0, 37.01, 37.0],
        "longitude": [-6.0, -5.99, -6.0],
        "LST_K": [310.0, 311.0, 312.0],
    })
    sentinel = pd.DataFrame({
        "date": ["2024-07-01", "2024-07-01", "2024-07-02"],
        "acquisition_dt_utc": ["2024-07-01T10:15Z", "2024-07-01T10:15Z", "2024-07-02T14:00Z"],
        "latitude": [37.0, 37.01, 37.0],
        "longitude": [-6.0, -5.99, -6.0],
        "LST_1km": [305.0, 306.0, 307.0],
    })
    result = build_landsat_sentinel_pairs(landsat, sentinel, max_hour_difference=2)
    assert result.columns.tolist() == PAIR_COLUMNS
    assert result["nodo_id"].tolist() == ["n1", "n2"]
    assert result["lst_sentinel_3"].tolist() == [305.0, 306.0]
    assert result["hour_sentinel"].tolist() == [10.25, 10.25]
