import pandas as pd

from tools.era5_land import (
    _regional_frame,
    contiguous_day_ranges,
    estimated_era5_pixels,
    expanded_era5_days,
    split_era5_ranges,
)
from tools.scientific_variables import era5_gee_band_names, era5_short_names


def test_earth_engine_band_mapping_uses_hourly_flow_bands():
    selected = ["air_temperature", "solar_radiation", "rain_3d"]
    pairs = dict(zip(era5_short_names(selected), era5_gee_band_names(selected)))
    assert pairs["t2m"] == "temperature_2m"
    assert pairs["ssrd"] == "surface_solar_radiation_downwards_hourly"
    assert pairs["tp"] == "total_precipitation_hourly"


def test_expanded_days_and_ranges_preserve_separated_seasons():
    targets = pd.to_datetime(["2020-05-01T10:00Z", "2020-05-02T10:00Z", "2021-05-01T10:00Z"])
    days = expanded_era5_days(targets)
    ranges = contiguous_day_ranges(days)
    assert len(ranges) == 2
    assert ranges[0][0] == pd.Timestamp("2020-04-29T00:00Z")
    assert ranges[0][1] == pd.Timestamp("2020-05-04T00:00Z")


def test_large_era5_range_is_split_by_estimated_value_count():
    ranges = [(pd.Timestamp("2020-05-01T00:00Z"), pd.Timestamp("2020-11-01T00:00Z"))]
    area = [37.55, -6.15, 37.25, -5.85]
    chunks = split_era5_ranges(ranges, area_nwse=area, band_count=30)
    assert len(chunks) > 1
    assert chunks[0][0] == ranges[0][0]
    assert chunks[-1][1] == ranges[0][1]
    assert estimated_era5_pixels(area) >= 9


def test_get_region_rows_are_reduced_to_one_regional_row_per_hour():
    rows = [
        ["id", "longitude", "latitude", "time", "t2m", "tp"],
        ["a", -6.0, 37.4, 1_594_771_200_000, 300.0, 0.001],
        ["b", -5.9, 37.4, 1_594_771_200_000, 302.0, 0.003],
    ]
    frame = _regional_frame(rows, ["t2m", "tp"])
    assert len(frame) == 1
    assert frame.loc[0, "t2m"] == 301.0
    assert frame.loc[0, "tp"] == 0.002
