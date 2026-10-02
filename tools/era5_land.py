from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from tools.scientific_variables import era5_gee_band_names, era5_short_names


ERA5_GEE_COLLECTION = "ECMWF/ERA5_LAND/HOURLY"
ERA5_GEE_SCALE_M = 11132
ERA5_CACHE_VERSION = 1


def expanded_era5_days(target_times: Iterable[Any]) -> list[pd.Timestamp]:
    """Incluye dos días previos y el posterior para lluvia acumulada y cierre diario."""
    days: set[pd.Timestamp] = set()
    for timestamp in pd.to_datetime(list(target_times), utc=True):
        day = timestamp.normalize()
        for offset in (-2, -1, 0, 1):
            days.add(day + pd.Timedelta(days=offset))
    return sorted(days)


def contiguous_day_ranges(days: Sequence[pd.Timestamp]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Convierte días sueltos en intervalos contiguos [inicio, fin exclusivo)."""
    if not days:
        return []
    normalized = set()
    for day in days:
        timestamp = pd.Timestamp(day)
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        normalized.add(timestamp.normalize())
    ordered = sorted(normalized)
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = previous = ordered[0]
    for day in ordered[1:]:
        if day - previous > pd.Timedelta(days=1):
            ranges.append((start, previous + pd.Timedelta(days=1)))
            start = day
        previous = day
    ranges.append((start, previous + pd.Timedelta(days=1)))
    return ranges


def estimated_era5_pixels(area_nwse: Sequence[float]) -> int:
    north, west, south, east = map(float, area_nwse)
    latitude_cells = max(1, math.ceil(abs(north - south) / 0.1) + 1)
    longitude_cells = max(1, math.ceil(abs(east - west) / 0.1) + 1)
    return latitude_cells * longitude_cells


def split_era5_ranges(
    ranges: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
    *,
    area_nwse: Sequence[float],
    band_count: int,
    target_values: int = 750_000,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Divide consultas para mantenerse holgadamente por debajo del límite de getRegion."""
    values_per_day = 24 * estimated_era5_pixels(area_nwse) * max(1, band_count + 4)
    max_days = max(1, min(184, target_values // max(values_per_day, 1)))
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in ranges:
        cursor = start
        while cursor < end:
            chunk_end = min(end, cursor + pd.Timedelta(days=max_days))
            chunks.append((cursor, chunk_end))
            cursor = chunk_end
    return chunks


def _cache_path(
    cache_dir: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    area_nwse: Sequence[float],
    bands: Sequence[str],
) -> Path:
    signature = {
        "version": ERA5_CACHE_VERSION,
        "collection": ERA5_GEE_COLLECTION,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "area": [round(float(value), 6) for value in area_nwse],
        "bands": list(bands),
    }
    digest = hashlib.sha1(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    return cache_dir / f"era5_gee_{start:%Y%m%d}_{end:%Y%m%d}_{digest}.csv.gz"


def _get_region_with_retry(value: Any, *, attempts: int = 5) -> list[list[Any]]:
    for attempt in range(attempts):
        try:
            return value.getInfo()
        except Exception as exc:
            text = str(exc).lower()
            transient = any(token in text for token in ("429", "quota", "concurrent", "timeout", "timed out"))
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("Earth Engine no devolvió ERA5-Land.")


def _regional_frame(rows: list[list[Any]], short_names: Sequence[str]) -> pd.DataFrame:
    if not rows or len(rows) == 1:
        return pd.DataFrame(columns=["era5_dt_utc", *short_names])
    frame = pd.DataFrame(rows[1:], columns=rows[0])
    missing = [name for name in short_names if name not in frame]
    if missing:
        raise RuntimeError(f"Earth Engine no devolvió las bandas ERA5-Land esperadas: {missing}")
    frame["era5_dt_utc"] = pd.to_datetime(frame["time"], unit="ms", utc=True)
    for column in short_names:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.groupby("era5_dt_utc", as_index=False)[list(short_names)].mean()


def download_era5_land_gee(
    target_times: Iterable[Any],
    *,
    area_nwse: Sequence[float],
    selected_variables: Iterable[str],
    cache_dir: Path,
    overwrite: bool = False,
    ee_module: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Obtiene ERA5-Land por lotes temporales y conserva solo la media regional comprimida."""
    if ee_module is None:
        import ee as ee_module

    short_names = era5_short_names(selected_variables)
    gee_bands = era5_gee_band_names(selected_variables)
    days = expanded_era5_days(target_times)
    chunks = split_era5_ranges(
        contiguous_day_ranges(days), area_nwse=area_nwse, band_count=len(gee_bands)
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    north, west, south, east = map(float, area_nwse)
    region = ee_module.Geometry.Rectangle([west, south, east, north], proj="EPSG:4326", geodesic=False)
    collection = ee_module.ImageCollection(ERA5_GEE_COLLECTION)
    frames: list[pd.DataFrame] = []
    downloaded_chunks = 0

    for start, end in chunks:
        target = _cache_path(
            cache_dir, start=start, end=end, area_nwse=area_nwse, bands=gee_bands
        )
        if target.exists() and not overwrite:
            frame = pd.read_csv(target, parse_dates=["era5_dt_utc"])
        else:
            table = (
                collection.filterDate(start.isoformat(), end.isoformat())
                .select(gee_bands, short_names)
                .getRegion(region, ERA5_GEE_SCALE_M)
            )
            frame = _regional_frame(_get_region_with_retry(table), short_names)
            temporary = target.with_suffix(target.suffix + ".part")
            frame.to_csv(temporary, index=False, compression="gzip")
            temporary.replace(target)
            downloaded_chunks += 1
        frames.append(frame)

    if not frames:
        raise RuntimeError("No hay intervalos ERA5-Land que descargar.")
    regional = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("era5_dt_utc")
        .sort_values("era5_dt_utc")
        .reset_index(drop=True)
    )
    info = {
        "source": "earth_engine",
        "requested_days": len(days),
        "requested_chunks": len(chunks),
        "downloaded_chunks": downloaded_chunks,
        "downloaded_hours": int(regional["era5_dt_utc"].nunique()),
        "cache_files": len(chunks),
    }
    return regional, info
