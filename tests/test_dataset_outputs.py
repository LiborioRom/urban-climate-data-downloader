import numpy as np
import pandas as pd

from tools.dataset_outputs import (
    LANDSAT_PREFERRED_COLS,
    SENTINEL_PREFERRED_COLS,
    build_landsat_final,
    build_sentinel_final,
)


def _common_values():
    return {
        "date": "2024-07-15", "acquisition_dt_utc": "2024-07-15T10:00:00Z",
        "latitude": 37.4, "longitude": -5.98,
        "Tair_C": 30.0, "RH": 35.0, "Rsol_Wm2": 700.0, "wind_speed": 2.0,
        "rain_3d": 0.0, "day_of_year": 197, "sin_doy": -0.2, "cos_doy": -0.9,
    }


def test_landsat_has_exact_columns_and_order():
    row = {
        **_common_values(), "node_id": "n1", "LST_K": 310.0, "NDVI": 0.4,
        "DEM": 12.0, "NDBI": 0.1, "ALBEDO": 0.2, "aspect_ratio": 0.3,
        "has_buildings": 1, "row": 0, "col": 0, "row_norm": 0.0,
        "col_norm": 0.0, "scene_id": "must_be_removed",
    }
    result = build_landsat_final(pd.DataFrame([row]))
    assert list(result.columns) == LANDSAT_PREFERRED_COLS
    assert "scene_id" not in result


def test_sentinel_is_renamed_filtered_and_ndvi_is_filled_from_nearest_date():
    rows = []
    for when, ndvi, lat in [
        ("2024-07-10T10:00:00Z", 0.25, 37.4000),
        ("2024-07-15T10:00:00Z", np.nan, 37.4001),
        ("2024-07-25T10:00:00Z", 0.80, 37.4000),
    ]:
        rows.append({
            **_common_values(), "date": when[:10], "acquisition_dt_utc": when,
            "s3_pixel_id": when[:10], "s3_row": 10, "s3_col": 20,
            "latitude": lat, "LST_S3_K": 305.0, "NDVI_S2_mean_1km": ndvi,
            "NDBI_S2_mean_1km": 0.1, "ALBEDO_S2_mean_1km": 0.2,
            "DEM_mean_1km": 11.0, "aspect_ratio_mean_1km": 0.3,
            "has_buildings_mean_1km": 0.7, "s3_exception": 0,
        })
    result, filled = build_sentinel_final(pd.DataFrame(rows))
    assert list(result.columns) == SENTINEL_PREFERRED_COLS
    assert "LST_1km" in result and "LST_S3_K" not in result
    assert "s3_exception" not in result
    assert filled == 1
    assert result.loc[result.date.eq("2024-07-15"), "NDVI"].iloc[0] == 0.25


def test_sentinel_ndvi_remains_blank_when_no_temporal_value_exists():
    row = {
        **_common_values(), "acquisition_dt_utc": "2024-07-15T10:00:00Z",
        "s3_pixel_id": "p1", "s3_row": 1, "s3_col": 1, "LST_S3_K": 305.0,
        "NDVI_S2_mean_1km": np.nan, "NDBI_S2_mean_1km": 0.1,
        "ALBEDO_S2_mean_1km": 0.2, "DEM_mean_1km": 11.0,
        "aspect_ratio_mean_1km": 0.3, "has_buildings_mean_1km": 0.7,
    }
    result, filled = build_sentinel_final(pd.DataFrame([row]))
    assert filled == 0
    assert result.NDVI.isna().all()
