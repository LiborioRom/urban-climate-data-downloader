from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from rules.variable_catalog import VARIABLE_CATALOG, output_columns_for
from schemas.request import DatasetPurpose, DatasetRequest


IDENTITY_COLUMNS = [
    "area_id", "dataset_purpose", "node_id", "date", "timestamp_utc",
    "latitude", "longitude", "x", "y", "row", "col", "row_norm", "col_norm",
    "day_of_year", "sin_doy", "cos_doy",
]

MATCH_COLUMNS = [
    "sentinel3_pixel_id", "sentinel3_datetime_utc", "sentinel3_latitude",
    "sentinel3_longitude", "sentinel3_distance_m", "s3_day_difference",
    "lst_time_difference_hours",
]

QUALITY_COLUMNS = ["landsat_valid", "sentinel3_valid", "training_ready"]

CORE_DOWNSCALING_COLUMNS = [
    "LST_K", "LST_1km", "NDVI", "Tair_C", "RH", "Rsol_Wm2", "wind_speed",
    "rain_3d", "DEM", "NDBI", "ALBEDO", "aspect_ratio", "has_buildings",
]

STATIC_COLUMNS = {
    "DEM", "aspect_ratio", "has_buildings", "building_height_m", "bld_footprint_m2",
    "sky_view_factor",
}
for _spec in VARIABLE_CATALOG.values():
    if _spec.get("temporal_kind") == "static":
        STATIC_COLUMNS.update(_spec.get("output_columns", {}).get("landsat", []))

INTERNAL_CSV_NAMES = [
    "dataset_landsat_sin_downscaling.csv",
    "dataset_sentinel3_1km_sin_downscaling.csv",
    "requested_dataset.csv",
    "nodos_malla.csv",
    "errores_landsat.csv",
    "errores_sentinel3.csv",
]


def _stable_area_id(request: DatasetRequest) -> str:
    roi = request.roi
    if not roi:
        return "roi_unknown"
    if roi.area_id:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", roi.area_id).strip("_") or "roi_unknown"
    geometry = {
        "center_lat": roi.center_lat,
        "center_lon": roi.center_lon,
        "half_side_m": roi.half_side_m,
        "rotation_deg": roi.rotation_deg,
        "bbox": roi.bbox,
    }
    digest = hashlib.sha1(json.dumps(geometry, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"roi_{digest}"


def _as_date(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out.loc[out["date"].notna()].copy()


def _first_observation_per_node_date(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in ("date", "node_id", "acquisition_dt_utc") if column in frame]
    return frame.sort_values(sort_columns).drop_duplicates(["date", "node_id"], keep="first")


def _nearest_s3_date(reference: pd.Timestamp, available: pd.DatetimeIndex, tolerance_days: int) -> pd.Timestamp | None:
    if available.empty:
        return None
    offsets = np.abs((available - reference).days)
    position = int(np.argmin(offsets))
    return available[position] if int(offsets[position]) <= tolerance_days else None


def _nearest_positions(
    fine_lat: np.ndarray,
    fine_lon: np.ndarray,
    coarse_lat: np.ndarray,
    coarse_lon: np.ndarray,
    *,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve el vecino grueso más cercano sin exigir SciPy."""
    positions = np.empty(len(fine_lat), dtype=int)
    distances = np.empty(len(fine_lat), dtype=float)
    earth_radius_m = 6_371_008.8
    coarse_lat_rad = np.deg2rad(coarse_lat)
    coarse_lon_rad = np.deg2rad(coarse_lon)
    for start in range(0, len(fine_lat), chunk_size):
        stop = min(start + chunk_size, len(fine_lat))
        lat1 = np.deg2rad(fine_lat[start:stop])[:, None]
        lon1 = np.deg2rad(fine_lon[start:stop])[:, None]
        dlat = coarse_lat_rad[None, :] - lat1
        dlon = coarse_lon_rad[None, :] - lon1
        hav = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(coarse_lat_rad)[None, :] * np.sin(dlon / 2) ** 2
        angular = 2 * np.arctan2(np.sqrt(hav), np.sqrt(np.maximum(0.0, 1 - hav)))
        selected = np.argmin(angular, axis=1)
        positions[start:stop] = selected
        distances[start:stop] = angular[np.arange(stop - start), selected] * earth_radius_m
    return positions, distances


def _temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    day = pd.to_datetime(out["date"]).dt.dayofyear
    out["day_of_year"] = day
    out["sin_doy"] = np.sin(2 * np.pi * day / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * day / 365.25)
    return out


def _sentinel_column_name(column: str, occupied: set[str]) -> str:
    if column == "LST_1km":
        return column
    return f"{column}_1km" if column in occupied else column


def _general_scientific_columns(request: DatasetRequest, available: list[str]) -> list[str]:
    requested = []
    for stream in ("landsat", "sentinel3"):
        for column in output_columns_for(request.variables, stream):
            requested.extend([column, f"{column}_1km"])
    return [column for column in dict.fromkeys(requested) if column in available]


def build_unified_dataset(
    landsat: pd.DataFrame,
    sentinel3: pd.DataFrame,
    nodes: pd.DataFrame,
    request: DatasetRequest,
) -> pd.DataFrame:
    """Armoniza las dos mallas en una tabla ancha con una fila por nodo fino y fecha."""
    if nodes.empty:
        raise ValueError("No hay nodos finos con los que construir el dataset unificado.")
    landsat = _first_observation_per_node_date(_as_date(landsat))
    sentinel3 = _as_date(sentinel3)
    nodes = nodes.drop_duplicates("node_id").copy()

    reference_dates = pd.DatetimeIndex(
        pd.concat([landsat["date"], sentinel3["date"]], ignore_index=True).dropna().drop_duplicates().sort_values()
    )
    if reference_dates.empty:
        raise ValueError("Las salidas no contienen fechas válidas para construir el dataset unificado.")

    calendar = pd.DataFrame({"date": reference_dates})
    calendar["_join"] = 1
    node_columns = [column for column in ("node_id", "latitude", "longitude", "x", "y", "row", "col", "row_norm", "col_norm") if column in nodes]
    node_grid = nodes[node_columns].copy()
    node_grid["_join"] = 1
    unified = calendar.merge(node_grid, on="_join", how="inner").drop(columns="_join")

    spatial_identity = {"latitude", "longitude", "x", "y", "row", "col", "row_norm", "col_norm", "day_of_year", "sin_doy", "cos_doy"}
    landsat_values = [column for column in landsat.columns if column not in spatial_identity and column not in {"date", "node_id"}]
    landsat_part = landsat[["date", "node_id", *landsat_values]].copy()
    if "acquisition_dt_utc" in landsat_part:
        landsat_part = landsat_part.rename(columns={"acquisition_dt_utc": "landsat_datetime_utc"})
    unified = unified.merge(landsat_part, on=["date", "node_id"], how="left")
    for column in STATIC_COLUMNS.intersection(unified.columns):
        unified[column] = unified.groupby("node_id")[column].transform(
            lambda values: values.dropna().iloc[0] if values.notna().any() else np.nan
        )

    s3_dates = pd.DatetimeIndex(sentinel3["date"].drop_duplicates().sort_values())
    tolerance = request.parameters.s3_max_day_difference
    matched_parts: list[pd.DataFrame] = []
    sentinel_skip = {"date", "node_id", "latitude", "longitude", "row", "col", "row_norm", "col_norm", "day_of_year", "sin_doy", "cos_doy"}
    sentinel_values = [column for column in sentinel3.columns if column not in sentinel_skip and column != "acquisition_dt_utc"]
    occupied = set(unified.columns)
    renamed_values = {column: _sentinel_column_name(column, occupied) for column in sentinel_values}

    for reference_date, fine in unified.groupby("date", sort=True):
        matched_date = _nearest_s3_date(reference_date, s3_dates, tolerance)
        if matched_date is None:
            matched_parts.append(fine)
            continue
        coarse = sentinel3.loc[sentinel3["date"].eq(matched_date)].dropna(subset=["latitude", "longitude"])
        if coarse.empty:
            matched_parts.append(fine)
            continue
        positions, distances = _nearest_positions(
            fine["latitude"].to_numpy(float), fine["longitude"].to_numpy(float),
            coarse["latitude"].to_numpy(float), coarse["longitude"].to_numpy(float),
        )
        selected = coarse.iloc[positions].reset_index(drop=True)
        part = fine.reset_index(drop=True).copy()
        part["sentinel3_pixel_id"] = selected["node_id"].to_numpy()
        part["sentinel3_latitude"] = selected["latitude"].to_numpy()
        part["sentinel3_longitude"] = selected["longitude"].to_numpy()
        part["sentinel3_distance_m"] = distances
        part["s3_day_difference"] = int(abs((matched_date - reference_date).days))
        if "acquisition_dt_utc" in selected:
            part["sentinel3_datetime_utc"] = selected["acquisition_dt_utc"].astype(str).to_numpy()
        else:
            part["sentinel3_datetime_utc"] = matched_date.strftime("%Y-%m-%d")
        for source, target in renamed_values.items():
            part[target] = selected[source].to_numpy()
        matched_parts.append(part)

    unified = pd.concat(matched_parts, ignore_index=True, sort=False)
    unified["area_id"] = _stable_area_id(request)
    unified["dataset_purpose"] = request.dataset_purpose.value
    if "landsat_datetime_utc" in unified:
        landsat_time = pd.to_datetime(unified["landsat_datetime_utc"], utc=True, errors="coerce")
    else:
        landsat_time = pd.Series(pd.NaT, index=unified.index, dtype="datetime64[ns, UTC]")
        legacy_landsat = unified.get("LST_K", pd.Series(index=unified.index, dtype=float)).notna()
        landsat_time.loc[legacy_landsat] = (
            pd.to_datetime(unified.loc[legacy_landsat, "date"], utc=True, errors="coerce")
            + pd.Timedelta(hours=12)
        )
    sentinel_raw = unified.get("sentinel3_datetime_utc", pd.Series(pd.NaT, index=unified.index))
    sentinel_time = pd.to_datetime(sentinel_raw, utc=True, errors="coerce")
    reference_time = landsat_time.fillna(sentinel_time).fillna(pd.to_datetime(unified["date"], utc=True))
    unified["timestamp_utc"] = reference_time.astype(str)
    unified["lst_time_difference_hours"] = (sentinel_time - landsat_time).abs().dt.total_seconds() / 3600
    unified["landsat_valid"] = unified.get("LST_K", pd.Series(index=unified.index, dtype=float)).notna()
    unified["sentinel3_valid"] = unified.get("LST_1km", pd.Series(index=unified.index, dtype=float)).notna()
    unified["training_ready"] = unified["landsat_valid"] & unified["sentinel3_valid"]
    unified = _temporal_features(unified)
    unified["date"] = pd.to_datetime(unified["date"]).dt.strftime("%Y-%m-%d")

    leading = [column for column in [*IDENTITY_COLUMNS, *MATCH_COLUMNS, "LST_K", "LST_1km", *QUALITY_COLUMNS] if column in unified]
    remaining = [column for column in unified.columns if column not in leading and column not in {"landsat_datetime_utc", "time_start", "scene_id"}]
    if request.dataset_purpose == DatasetPurpose.GENERAL_DOWNLOAD:
        scientific = _general_scientific_columns(request, remaining + leading)
        remaining = [column for column in remaining if column in scientific]
        leading = [column for column in leading if column not in {"LST_K", "LST_1km"} or column in scientific]
    else:
        preferred = [column for column in CORE_DOWNSCALING_COLUMNS if column in remaining or column in leading]
        leading = list(dict.fromkeys([*leading, *preferred]))
        remaining = [column for column in remaining if column not in leading]
    return unified.loc[:, list(dict.fromkeys([*leading, *remaining]))].sort_values(["date", "row", "col"]).reset_index(drop=True)


def count_source_errors(output_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source, filename in (("landsat", "errores_landsat.csv"), ("sentinel3", "errores_sentinel3.csv")):
        path = output_dir / filename
        if not path.exists():
            counts[source] = 0
            continue
        try:
            counts[source] = len(pd.read_csv(path))
        except Exception:
            counts[source] = 1
    return counts


def build_quality_report(
    frame: pd.DataFrame,
    request: DatasetRequest,
    source_errors: dict[str, int],
    source_quality: dict | None = None,
) -> dict:
    ignored = set(IDENTITY_COLUMNS + MATCH_COLUMNS + QUALITY_COLUMNS)
    scientific = [column for column in frame.columns if column not in ignored]
    missing = []
    for column in scientific:
        count = int(frame[column].isna().sum())
        if count:
            missing.append({"column": column, "count": count, "percent": round(100 * count / len(frame), 2)})
    missing.sort(key=lambda item: (-item["count"], item["column"]))
    total_cells = len(frame) * len(scientific)
    total_missing = sum(item["count"] for item in missing)
    ready = int(frame.get("training_ready", pd.Series(False, index=frame.index)).sum())
    if request.dataset_purpose == DatasetPurpose.LST_DOWNSCALING and ready == 0:
        status = "insufficient"
        summary = "No hay parejas válidas de LST Landsat y Sentinel-3 para entrenar el downscaling."
    elif missing or any(source_errors.values()):
        status = "with_gaps"
        summary = f"La descarga terminó con {total_missing:,} valores vacíos y {sum(source_errors.values())} elementos fallidos."
    else:
        status = "complete"
        summary = "La descarga terminó sin carencias detectadas en las variables científicas."
    return {
        "status": status,
        "summary": summary,
        "rows": int(len(frame)),
        "nodes": int(frame["node_id"].nunique()) if "node_id" in frame else 0,
        "dates": int(frame["date"].nunique()) if "date" in frame else 0,
        "scientific_completeness_percent": round(100 * (1 - total_missing / total_cells), 2) if total_cells else 100.0,
        "training_ready_rows": ready,
        "training_ready_percent": round(100 * ready / len(frame), 2) if len(frame) else 0.0,
        "missing_by_column": missing,
        "source_errors": source_errors,
        "source_quality": source_quality or {},
    }


def archive_intermediate_csvs(output_dir: Path) -> Path:
    target_dir = output_dir / "cache" / "intermediate"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in INTERNAL_CSV_NAMES:
        source = output_dir / name
        if source.exists():
            target = target_dir / name
            if target.exists():
                target.unlink()
            shutil.move(str(source), str(target))
    return target_dir
