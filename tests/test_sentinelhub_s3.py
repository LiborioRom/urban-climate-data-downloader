import numpy as np
import pandas as pd
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import box

from tools.sentinelhub_s3 import (
    S3_COLLECTION,
    aligned_output_grid,
    aligned_grid_cells,
    choose_s3_products,
    geotiff_to_s3_pixels,
    landsat_reference_hour,
    process_request_payload,
    statistics_request_payload,
    statistics_response_rows,
)


def test_aligned_grid_keeps_exact_one_kilometre_pixels():
    bounds, width, height = aligned_output_grid((100, 200, 1600, 1800), 1000)
    assert (width, height) == (2, 2)
    assert bounds[2] - bounds[0] == width * 1000
    assert bounds[3] - bounds[1] == height * 1000


def test_process_payload_requests_official_l2_lst_without_f1():
    payload = process_request_payload(
        (0, 0, 2000, 2000), "EPSG:25830", 2, 2, "2024-07-15T10:00:00Z"
    )
    assert payload["input"]["data"][0]["type"] == S3_COLLECTION
    assert payload["input"]["data"][0]["processing"]["upsampling"] == "NEAREST"
    assert '"LST"' in payload["evalscript"]
    assert '"F1"' not in payload["evalscript"]
    assert payload["output"]["width"] == 2


def test_products_can_be_limited_to_nearest_landsat_dates():
    acquisitions = pd.to_datetime(
        ["2024-07-01T09:00Z", "2024-07-02T10:00Z", "2024-07-10T10:00Z"], utc=True
    )
    products = pd.DataFrame({
        "Id": ["a", "b", "c"], "Name": ["a", "b", "c"],
        "acquisition_dt_utc": acquisitions,
        "date": acquisitions.date, "hour": acquisitions.hour,
    })
    selected, daily_count = choose_s3_products(
        products, morning_start_hour=7, morning_end_hour=13, target_hour=10,
        reference_datetimes=["2024-07-03T10:00Z"], tolerance_days=3,
    )
    assert daily_count == 3
    assert selected["Id"].tolist() == ["b"]


def test_daily_products_use_minutes_and_can_fill_intermediate_days():
    acquisitions = pd.to_datetime(
        ["2024-07-01T09:50Z", "2024-07-01T10:10Z", "2024-07-02T10:05Z"], utc=True
    )
    products = pd.DataFrame({
        "Id": ["a", "b", "c"], "Name": ["a", "b", "c"],
        "acquisition_dt_utc": acquisitions,
        "date": acquisitions.date, "hour": acquisitions.hour,
    })
    selected, daily_count = choose_s3_products(
        products, morning_start_hour=7, morning_end_hour=13, target_hour=10.2,
    )
    assert daily_count == 2
    assert selected["Id"].tolist() == ["b", "c"]
    assert landsat_reference_hour(["2024-07-01T10:30Z", "2024-07-02T11:30Z"]) == 11.0


def test_statistics_payload_batches_daily_intervals_and_parser_reads_means():
    cells = aligned_grid_cells((0, 0, 2000, 2000), 2, 2)
    assert cells[0] == {"row": 0, "col": 0, "bounds": [0.0, 1000.0, 1000.0, 2000.0], "x": 500.0, "y": 1500.0}
    payload = statistics_request_payload(
        cells[0]["bounds"], "EPSG:25830", "2024-07-01", "2024-07-08", 10.5
    )
    assert payload["aggregation"]["aggregationInterval"] == {"of": "P1D"}
    assert payload["aggregation"]["resx"] == 1000.0
    assert 'mosaicking: "TILE"' in payload["aggregation"]["evalscript"]

    response = {"data": [{
        "interval": {"from": "2024-07-01T00:00:00Z"},
        "outputs": {"data": {"bands": {
            name: {"stats": {"mean": float(index + 1)}}
            for index, name in enumerate(("LST", "LST_uncertainty", "NDVI", "CLOUD", "BAYES", "POINTING", "CONFIDENCE"))
        }}},
    }]}
    rows = statistics_response_rows(response)
    assert rows[0]["LST"] == 1.0
    assert rows[0]["CONFIDENCE"] == 7.0


def test_small_multiband_geotiff_is_converted_to_official_lst_pixels():
    values = np.zeros((8, 2, 2), dtype="float32")
    values[0] = 305.0  # LST
    values[1] = 2.0    # incertidumbre
    values[2] = 0.3    # NDVI
    values[7] = 1.0    # dataMask
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff", width=2, height=2, count=8, dtype="float32",
            crs="EPSG:25830", transform=from_origin(0, 2000, 1000, 1000),
        ) as dataset:
            dataset.write(values)
        content = memory.read()

    frame = geotiff_to_s3_pixels(
        content,
        product_id="item-1",
        product_name="product-1",
        acquisition_time="2024-07-15T10:00Z",
        roi_projected=box(100, 1100, 900, 1900),
        projected_crs="EPSG:25830",
    )
    assert len(frame) == 1
    assert frame.loc[0, "LST_S3_K"] == 305.0
    assert frame.loc[0, "LST_uncertainty_1km"] == 2.0
    assert frame.loc[0, "NDVI_S3"] == pytest.approx(0.3)
