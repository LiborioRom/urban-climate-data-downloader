from __future__ import annotations

import numpy as np
import pandas as pd


PAIR_COLUMNS = [
    "nodo_id",
    "date",
    "lst_landsat",
    "lst_sentinel_3",
    "hour_landsat",
    "hour_sentinel",
]


def _decimal_hour(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, utc=True, errors="coerce")
    return timestamps.dt.hour + timestamps.dt.minute / 60 + timestamps.dt.second / 3600


def build_landsat_sentinel_pairs(
    landsat: pd.DataFrame,
    sentinel3: pd.DataFrame,
    *,
    max_hour_difference: float = 2.0,
) -> pd.DataFrame:
    """Empareja cada nodo Landsat con el píxel Sentinel-3 más cercano del mismo día."""
    required_landsat = {"node_id", "date", "acquisition_dt_utc", "latitude", "longitude", "LST_K"}
    required_sentinel = {"date", "acquisition_dt_utc", "latitude", "longitude", "LST_1km"}
    if not required_landsat.issubset(landsat) or not required_sentinel.issubset(sentinel3):
        return pd.DataFrame(columns=PAIR_COLUMNS)

    fine = landsat.copy()
    coarse = sentinel3.copy()
    fine["date"] = pd.to_datetime(fine["date"], errors="coerce").dt.normalize()
    coarse["date"] = pd.to_datetime(coarse["date"], errors="coerce").dt.normalize()
    fine["hour_landsat"] = _decimal_hour(fine["acquisition_dt_utc"])
    coarse["hour_sentinel"] = _decimal_hour(coarse["acquisition_dt_utc"])
    parts: list[pd.DataFrame] = []

    for date, day_fine in fine.groupby("date", sort=True):
        day_coarse = coarse.loc[coarse["date"].eq(date)].dropna(
            subset=["latitude", "longitude", "LST_1km", "hour_sentinel"]
        )
        if day_coarse.empty:
            continue
        landsat_hour = float(day_fine["hour_landsat"].median())
        acquisitions = day_coarse.groupby("acquisition_dt_utc")["hour_sentinel"].first()
        closest_time = (acquisitions - landsat_hour).abs().idxmin()
        selected = day_coarse.loc[day_coarse["acquisition_dt_utc"].eq(closest_time)].reset_index(drop=True)
        sentinel_hour = float(selected["hour_sentinel"].iloc[0])
        if abs(sentinel_hour - landsat_hour) > float(max_hour_difference):
            continue

        fine_lat = day_fine["latitude"].to_numpy(float)[:, None]
        fine_lon = day_fine["longitude"].to_numpy(float)[:, None]
        coarse_lat = selected["latitude"].to_numpy(float)[None, :]
        coarse_lon = selected["longitude"].to_numpy(float)[None, :]
        lon_scale = np.cos(np.deg2rad(fine_lat))
        distances = (fine_lat - coarse_lat) ** 2 + ((fine_lon - coarse_lon) * lon_scale) ** 2
        nearest = np.argmin(distances, axis=1)

        parts.append(pd.DataFrame({
            "nodo_id": day_fine["node_id"].astype(str).to_numpy(),
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "lst_landsat": pd.to_numeric(day_fine["LST_K"], errors="coerce").to_numpy(),
            "lst_sentinel_3": pd.to_numeric(selected.iloc[nearest]["LST_1km"], errors="coerce").to_numpy(),
            "hour_landsat": day_fine["hour_landsat"].to_numpy(float),
            "hour_sentinel": np.full(len(day_fine), sentinel_hour),
        }))

    if not parts:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    result = pd.concat(parts, ignore_index=True).dropna(subset=["lst_landsat", "lst_sentinel_3"])
    return result.loc[:, PAIR_COLUMNS].sort_values(["date", "nodo_id"]).reset_index(drop=True)
