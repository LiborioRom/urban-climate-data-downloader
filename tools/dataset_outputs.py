from __future__ import annotations

import numpy as np
import pandas as pd

from rules.variable_catalog import output_columns_for
from tools.scientific_variables import S2_OPTIONAL_BANDS


LANDSAT_PREFERRED_COLS = [
    "node_id", "date", "acquisition_dt_utc", "latitude", "longitude", "LST_K", "NDVI", "Tair_C",
    "RH", "Rsol_Wm2", "wind_speed", "rain_3d", "DEM", "NDBI", "ALBEDO",
    "aspect_ratio", "has_buildings", "row", "col", "row_norm", "col_norm",
    "day_of_year", "sin_doy", "cos_doy",
]

SENTINEL_PREFERRED_COLS = [
    "node_id", "date", "acquisition_dt_utc", "latitude", "longitude", "LST_1km", "NDVI", "Tair_C",
    "RH", "Rsol_Wm2", "wind_speed", "rain_3d", "DEM", "NDBI", "ALBEDO",
    "aspect_ratio", "has_buildings", "row", "col", "row_norm", "col_norm",
    "day_of_year", "sin_doy", "cos_doy",
]

SENTINEL_RENAME_MAP = {
    "s3_pixel_id": "node_id",
    "LST_S3_K": "LST_1km",
    "NDVI_S2_mean_1km": "NDVI",
    "NDBI_S2_mean_1km": "NDBI",
    "ALBEDO_S2_mean_1km": "ALBEDO",
    "DEM_mean_1km": "DEM",
    "aspect_ratio_mean_1km": "aspect_ratio",
    "has_buildings_mean_1km": "has_buildings",
    "building_height_m_mean_1km": "building_height_m",
    "bld_footprint_m2_mean_1km": "bld_footprint_m2",
    "sky_view_factor_mean_1km": "sky_view_factor",
    "s3_row": "row",
    "s3_col": "col",
}
SENTINEL_RENAME_MAP.update({
    f"{column}_S2_mean_1km": column
    for column in S2_OPTIONAL_BANDS.values()
})


def _require_and_select(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label}: faltan columnas finales obligatorias: {missing}")
    return frame.loc[:, columns].copy()


def build_landsat_final(frame: pd.DataFrame, requested_variables: list[str] | None = None) -> pd.DataFrame:
    """Conserva el núcleo acordado y añade solo variables opcionales solicitadas."""
    optional = output_columns_for(requested_variables or [], "landsat")
    columns = list(dict.fromkeys([*LANDSAT_PREFERRED_COLS, *optional]))
    return _require_and_select(frame, columns, "Landsat")


def _normalized_axis(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    minimum, maximum = numeric.min(), numeric.max()
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(0.0, index=values.index)
    return (numeric - minimum) / (maximum - minimum)


def fill_ndvi_from_nearest_time(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Imputa NDVI faltante desde la fecha disponible más próxima y el píxel más cercano.

    La fecha candidata debe ser distinta de la fecha objetivo. Dentro de esa adquisición
    se usa el píxel geográficamente más cercano, porque las filas/columnas de una pasada
    SLSTR no constituyen identificadores espaciales estables entre productos.
    """
    out = frame.copy()
    if "NDVI" not in out or not out["NDVI"].isna().any():
        return out, 0

    timestamps = pd.to_datetime(out["acquisition_dt_utc"], utc=True, errors="coerce")
    valid = out["NDVI"].notna() & timestamps.notna()
    missing = out["NDVI"].isna() & timestamps.notna()
    if not valid.any() or not missing.any():
        return out, 0

    valid_times = pd.Index(timestamps[valid].unique()).sort_values()
    filled = 0
    for target_time in pd.Index(timestamps[missing].unique()).sort_values():
        other_times = valid_times[valid_times != target_time]
        if other_times.empty:
            continue
        nearest_time = other_times[np.argmin(np.abs(other_times - target_time))]
        candidates = out.loc[valid & timestamps.eq(nearest_time), ["latitude", "longitude", "NDVI"]].dropna()
        target_indices = out.index[missing & timestamps.eq(target_time)]
        if candidates.empty:
            continue

        candidate_lat = candidates["latitude"].to_numpy(float)
        candidate_lon = candidates["longitude"].to_numpy(float)
        for index in target_indices:
            lat = float(out.at[index, "latitude"])
            lon = float(out.at[index, "longitude"])
            lon_scale = np.cos(np.deg2rad(lat))
            distance_sq = (candidate_lat - lat) ** 2 + ((candidate_lon - lon) * lon_scale) ** 2
            nearest_position = int(np.argmin(distance_sq))
            out.at[index, "NDVI"] = candidates.iloc[nearest_position]["NDVI"]
            filled += 1
    return out, filled


def build_sentinel_final(frame: pd.DataFrame, requested_variables: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    """Armoniza Sentinel-3 al esquema solicitado y completa NDVI temporalmente."""
    out = frame.rename(columns=SENTINEL_RENAME_MAP).copy()
    out["row_norm"] = _normalized_axis(out["row"])
    out["col_norm"] = _normalized_axis(out["col"])
    out, filled = fill_ndvi_from_nearest_time(out)
    optional = output_columns_for(requested_variables or [], "sentinel3")
    columns = list(dict.fromkeys([*SENTINEL_PREFERRED_COLS, *optional]))
    return _require_and_select(out, columns, "Sentinel-3"), filled
