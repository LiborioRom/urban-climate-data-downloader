from __future__ import annotations

import io
import math
from collections.abc import Iterable, Sequence
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.windows import Window
from shapely.geometry import box


S3_COLLECTION = "sentinel-3-slstr-l2"
S3_STATISTICS_URL = "https://sh.dataspace.copernicus.eu/statistics/v1"
S3_PROCESS_BANDS = (
    "LST",
    "LST_uncertainty",
    "NDVI",
    "CLOUD",
    "BAYES",
    "POINTING",
    "CONFIDENCE",
    "dataMask",
)
S3_PROCESS_OUTPUT_COLUMNS = {
    "LST": "LST_S3_K",
    "LST_uncertainty": "LST_uncertainty_1km",
    "NDVI": "NDVI_S3",
    "CLOUD": "s3_cloud",
    "BAYES": "s3_bayes",
    "POINTING": "s3_pointing",
    "CONFIDENCE": "s3_confidence",
}
S3_PROCESS_SUPPORTED_VARIABLES = {
    "LST",
    "sentinel3_lst_uncertainty",
    "sentinel3_ndvi",
    "sentinel3_quality",
}


def aligned_output_grid(
    bounds: Sequence[float], resolution_m: float = 1000.0
) -> tuple[list[float], int, int]:
    """Amplía simétricamente el bbox para que cada píxel mida exactamente resolution_m."""
    xmin, ymin, xmax, ymax = map(float, bounds)
    if resolution_m <= 0 or xmax <= xmin or ymax <= ymin:
        raise ValueError("Los límites y la resolución de Sentinel-3 deben ser positivos.")
    width = max(1, math.ceil((xmax - xmin) / resolution_m))
    height = max(1, math.ceil((ymax - ymin) / resolution_m))
    center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    span_x, span_y = width * resolution_m, height * resolution_m
    aligned = [
        center_x - span_x / 2,
        center_y - span_y / 2,
        center_x + span_x / 2,
        center_y + span_y / 2,
    ]
    return aligned, width, height


def process_evalscript() -> str:
    returned = ",\n        ".join(f"sample.{band}" for band in S3_PROCESS_BANDS)
    quoted = ", ".join(f'"{band}"' for band in S3_PROCESS_BANDS)
    return f"""//VERSION=3
function setup() {{
  return {{
    input: [{{ bands: [{quoted}] }}],
    output: {{ bands: {len(S3_PROCESS_BANDS)}, sampleType: \"FLOAT32\" }}
  }};
}}
function evaluatePixel(sample) {{
  return [
        {returned}
  ];
}}
"""


def process_request_payload(
    bounds: Sequence[float],
    crs: str,
    width: int,
    height: int,
    acquisition_time: Any,
) -> dict[str, Any]:
    acquisition = pd.Timestamp(acquisition_time)
    if acquisition.tzinfo is None:
        acquisition = acquisition.tz_localize("UTC")
    else:
        acquisition = acquisition.tz_convert("UTC")
    start = acquisition - pd.Timedelta(seconds=30)
    end = acquisition + pd.Timedelta(minutes=5)
    epsg = str(crs).upper().removeprefix("EPSG:")
    return {
        "input": {
            "bounds": {
                "bbox": list(map(float, bounds)),
                "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"},
            },
            "data": [{
                "type": S3_COLLECTION,
                "dataFilter": {
                    "timeRange": {
                        "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    },
                    "mosaickingOrder": "mostRecent",
                },
                "processing": {"upsampling": "NEAREST"},
            }],
        },
        "output": {
            "width": int(width),
            "height": int(height),
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": process_evalscript(),
    }


def catalogue_rows(features: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        properties = feature.get("properties", {})
        acquisition = pd.to_datetime(properties.get("datetime"), utc=True, errors="coerce")
        if pd.isna(acquisition):
            continue
        item_id = str(feature.get("id", "")).strip()
        if not item_id:
            continue
        rows.append({
            "Id": item_id,
            "Name": properties.get("sentinel3:product_id") or item_id,
            "acquisition_dt_utc": acquisition,
            "date": acquisition.date(),
            "hour": acquisition.hour,
            "cloud_cover": properties.get("eo:cloud_cover"),
        })
    return pd.DataFrame(rows)


def choose_s3_products(
    products: pd.DataFrame,
    *,
    morning_start_hour: int,
    morning_end_hour: int,
    target_hour: float,
    reference_datetimes: Iterable[Any] | None = None,
    tolerance_days: int = 3,
) -> tuple[pd.DataFrame, int]:
    if products.empty:
        raise RuntimeError("No se encontraron adquisiciones Sentinel-3 L2 LST para el ROI y periodo.")
    unique = products.drop_duplicates("Id").copy()
    morning = unique[unique["hour"].between(morning_start_hour, morning_end_hour)].copy()
    if morning.empty:
        raise RuntimeError("No se encontraron adquisiciones Sentinel-3 en la franja horaria indicada.")
    acquisition_minutes = (
        morning["acquisition_dt_utc"].dt.hour * 60
        + morning["acquisition_dt_utc"].dt.minute
        + morning["acquisition_dt_utc"].dt.second / 60
    )
    morning["hour_distance"] = (acquisition_minutes - float(target_hour) * 60).abs()
    daily = (
        morning.sort_values(["date", "hour_distance", "acquisition_dt_utc"])
        .groupby("date", as_index=False).first()
        .sort_values("acquisition_dt_utc")
    )
    daily_count = len(daily)
    if reference_datetimes is None:
        return daily.reset_index(drop=True), daily_count

    selected = []
    references = pd.to_datetime(pd.Series(list(reference_datetimes)).dropna().unique(), utc=True)
    for reference in references:
        day_distance = (daily["acquisition_dt_utc"].dt.normalize() - reference.normalize()).abs().dt.days
        candidates = daily.loc[day_distance <= tolerance_days].copy()
        if candidates.empty:
            continue
        candidates["reference_distance"] = (candidates["acquisition_dt_utc"] - reference).abs()
        selected.append(candidates.sort_values(["reference_distance", "acquisition_dt_utc"]).iloc[0])
    if not selected:
        raise RuntimeError("No se encontró Sentinel-3 próximo a ninguna adquisición Landsat.")
    result = pd.DataFrame(selected).drop_duplicates("Id").sort_values("acquisition_dt_utc")
    return result.reset_index(drop=True), daily_count


def landsat_reference_hour(datetimes: Iterable[Any], fallback: float = 10.0) -> float:
    """Devuelve la hora UTC decimal mediana de las adquisiciones Landsat."""
    values = pd.to_datetime(pd.Series(list(datetimes)), utc=True, errors="coerce").dropna()
    if values.empty:
        return float(fallback)
    minutes = values.dt.hour * 60 + values.dt.minute + values.dt.second / 60
    return float(minutes.median() / 60)


def statistics_evalscript(
    target_hour_utc: float,
    morning_start_hour: int = 7,
    morning_end_hour: int = 13,
) -> str:
    """Selecciona por píxel la escena diaria más próxima a la hora Landsat."""
    target_minutes = int(round(float(target_hour_utc) * 60))
    start_minutes = int(morning_start_hour) * 60
    end_minutes = int(morning_end_hour) * 60 + 59
    bands = ", ".join(f'"{name}"' for name in S3_PROCESS_BANDS)
    outputs = ", ".join(f'"{name}"' for name in S3_PROCESS_BANDS[:-1])
    returned = ", ".join(f"sample.{name}" for name in S3_PROCESS_BANDS[:-1])
    return f"""//VERSION=3
function setup() {{
  return {{
    input: [{{ bands: [{bands}] }}],
    mosaicking: "TILE",
    output: [
      {{ id: "data", bands: [{outputs}], sampleType: "FLOAT32" }},
      {{ id: "dataMask", bands: 1 }}
    ]
  }};
}}
function minutesUTC(value) {{
  var date = new Date(value);
  return date.getUTCHours() * 60 + date.getUTCMinutes() + date.getUTCSeconds() / 60;
}}
function evaluatePixel(samples, scenes) {{
  var best = -1;
  var bestDistance = 1e20;
  for (var i = 0; i < samples.length; i++) {{
    var minute = minutesUTC(scenes.tiles[i].date);
    if (minute < {start_minutes} || minute > {end_minutes}) continue;
    var distance = Math.abs(minute - {target_minutes});
    if (distance < bestDistance) {{ best = i; bestDistance = distance; }}
  }}
  if (best < 0) return {{ data: [NaN, NaN, NaN, NaN, NaN, NaN, NaN], dataMask: [0] }};
  var sample = samples[best];
  return {{ data: [{returned}], dataMask: [sample.dataMask] }};
}}
"""


def statistics_request_payload(
    bounds: Sequence[float],
    crs: str,
    start: Any,
    end: Any,
    target_hour_utc: float,
    *,
    resolution_m: float = 1000.0,
    morning_start_hour: int = 7,
    morning_end_hour: int = 13,
) -> dict[str, Any]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    epsg = str(crs).upper().removeprefix("EPSG:")
    return {
        "input": {
            "bounds": {
                "bbox": list(map(float, bounds)),
                "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"},
            },
            "data": [{"type": S3_COLLECTION}],
        },
        "aggregation": {
            "timeRange": {
                "from": start_ts.strftime("%Y-%m-%dT00:00:00Z"),
                "to": end_ts.strftime("%Y-%m-%dT00:00:00Z"),
            },
            "aggregationInterval": {"of": "P1D"},
            "evalscript": statistics_evalscript(
                target_hour_utc, morning_start_hour, morning_end_hour
            ),
            "resx": float(resolution_m),
            "resy": float(resolution_m),
        },
    }


def aligned_grid_cells(
    bounds: Sequence[float], width: int, height: int
) -> list[dict[str, Any]]:
    """Describe cada celda en el mismo orden fila-columna que un GeoTIFF norte-arriba."""
    xmin, ymin, xmax, ymax = map(float, bounds)
    cell_width = (xmax - xmin) / int(width)
    cell_height = (ymax - ymin) / int(height)
    cells = []
    for row in range(int(height)):
        top = ymax - row * cell_height
        bottom = top - cell_height
        for col in range(int(width)):
            left = xmin + col * cell_width
            right = left + cell_width
            cells.append({
                "row": row,
                "col": col,
                "bounds": [left, bottom, right, top],
                "x": (left + right) / 2,
                "y": (bottom + top) / 2,
            })
    return cells


def statistics_response_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convierte la respuesta diaria de Statistical API en valores medios por celda."""
    rows: list[dict[str, Any]] = []
    output_names = S3_PROCESS_BANDS[:-1]
    for item in payload.get("data", []):
        interval = item.get("interval", {})
        bands = item.get("outputs", {}).get("data", {}).get("bands", {})
        row: dict[str, Any] = {"date": interval.get("from")}
        for name in output_names:
            row[name] = bands.get(name, {}).get("stats", {}).get("mean", np.nan)
        if not pd.isna(row["LST"]):
            rows.append(row)
    return rows


def geotiff_to_s3_pixels(
    content: bytes,
    *,
    product_id: str,
    product_name: str,
    acquisition_time: Any,
    roi_projected: Any,
    projected_crs: str,
    max_uncertainty_k: float = 5.0,
) -> pd.DataFrame:
    """Convierte el pequeño GeoTIFF remoto en píxeles SLSTR de 1 km que intersectan el ROI."""
    with MemoryFile(content) as memory:
        with memory.open() as src:
            if src.count != len(S3_PROCESS_BANDS):
                raise RuntimeError(
                    f"Sentinel Hub devolvió {src.count} bandas; se esperaban {len(S3_PROCESS_BANDS)}."
                )
            values = src.read().astype("float64")
            transform = src.transform
            raster_crs = src.crs or projected_crs

    band = {name: values[index] for index, name in enumerate(S3_PROCESS_BANDS)}
    lst = band["LST"]
    uncertainty = band["LST_uncertainty"]
    confidence = np.nan_to_num(band["CONFIDENCE"], nan=0).round().astype(np.uint16)
    valid = np.isfinite(lst) & (band["dataMask"] > 0)
    valid &= (lst >= 223.0) & (lst <= 360.0)
    valid &= np.isfinite(uncertainty) & (uncertainty >= 0) & (uncertainty <= max_uncertainty_k)
    valid &= (confidence & (1 << 14)) == 0  # summary_cloud del producto SL_2_LST___

    to_wgs84 = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
    acquisition = pd.to_datetime(acquisition_time, utc=True)
    rows: list[dict[str, Any]] = []
    for row, col in zip(*np.where(valid)):
        left, bottom, right, top = rasterio.windows.bounds(Window(col, row, 1, 1), transform)
        if not box(left, bottom, right, top).intersects(roi_projected):
            continue
        x, y = rasterio.transform.xy(transform, row, col, offset="center")
        lon, lat = to_wgs84.transform(x, y)
        record = {
            "s3_row": int(row),
            "s3_col": int(col),
            "latitude": float(lat),
            "longitude": float(lon),
            "LST_S3_K": float(lst[row, col]),
            "LST_S3_uncertainty_K": float(uncertainty[row, col]),
            "LST_uncertainty_1km": float(uncertainty[row, col]),
            "NDVI_S3": float(band["NDVI"][row, col]),
            "s3_cloud": int(round(band["CLOUD"][row, col])),
            "s3_bayes": int(round(band["BAYES"][row, col])),
            "s3_pointing": int(round(band["POINTING"][row, col])),
            "s3_confidence": int(confidence[row, col]),
            "quality_flag_1km": int(confidence[row, col]),
            "s3_product_id": product_id,
            "s3_product_name": product_name,
            "acquisition_dt_utc": acquisition,
        }
        record["s3_pixel_id"] = f"{acquisition:%Y%m%dT%H%M}_{row}_{col}"
        rows.append(record)
    return pd.DataFrame(rows)
