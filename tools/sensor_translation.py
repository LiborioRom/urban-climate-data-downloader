from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from schemas.request import DatasetRequest
from tools.date_windows import effective_date_windows


S2_SOURCE_COLUMNS = {
    "NDVI_S2": "NDVI_S2_100m",
    "NDBI_S2": "NDBI_S2_100m",
    "ALBEDO_S2": "ALBEDO_S2_100m",
}

COARSE_FEATURES = [
    "LST_1km",
    "lst_time_difference_hours",
    "Tair_C_1km",
    "RH_1km",
    "Rsol_Wm2_1km",
    "wind_speed_1km",
    "rain_3d_log_1km",
    "sin_doy",
    "cos_doy",
]

SPATIAL_FEATURES = [
    "NDVI_S2_delta",
    "NDBI_S2_delta",
    "ALBEDO_S2_delta",
    "DEM_delta",
    "aspect_ratio_delta",
    "has_buildings_delta",
]

DIRECT_FEATURES = [*COARSE_FEATURES, *SPATIAL_FEATURES]


@dataclass
class TranslationBundle:
    name: str
    coarse_kind: str
    coarse_model: object | None
    coarse_bias: float
    spatial_model: object | None
    direct_model: object | None
    coarse_features: list[str]
    spatial_features: list[str]
    direct_features: list[str]


def find_latest_multizone_run(project_root: Path) -> Path:
    """Localiza la descarga multizona completa más reciente."""
    candidates = sorted(
        (project_root / "runs").glob("*/combined/metadata_multizona.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for metadata_path in candidates:
        quality_path = metadata_path.with_name("informe_calidad_multizona.json")
        dataset_path = metadata_path.with_name("dataset_multizona.csv")
        if not quality_path.exists() or not dataset_path.exists():
            continue
        try:
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if quality.get("status") in {"complete", "partial"} and quality.get("areas_completed", 0) > 0:
            return metadata_path.parents[1]
    raise FileNotFoundError("No existe una descarga multizona terminada.")


def resolve_area_s2_cache(area_dir: Path, shared_cache: Path) -> Path:
    """Reconstruye la clave de caché Sentinel-2 usada por un ROI concreto."""
    from tools.scientific_variables import s2_band_names

    output_dir = area_dir / "data_downloads"
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    request = DatasetRequest.model_validate(metadata["request"])
    intermediate = output_dir / "cache" / "intermediate"
    nodes = pd.read_csv(intermediate / "nodos_malla.csv")
    sentinel3 = pd.read_csv(
        intermediate / "dataset_sentinel3_1km_sin_downscaling.csv",
        usecols=["acquisition_dt_utc"],
    )
    acquisitions = pd.DatetimeIndex(
        pd.to_datetime(
            sentinel3["acquisition_dt_utc"], utc=True, errors="coerce", format="mixed"
        ).dropna().unique()
    ).sort_values()
    roi = request.roi
    static_prefix = (
        f"{roi.center_lat:.5f}_{roi.center_lon:.5f}_{roi.half_side_m:.0f}_{roi.rotation_deg:.2f}_"
        f"{request.target_resolution_m:.0f}_{int(request.parameters.include_urban_morphology)}_"
    ).replace("-", "m").replace(".", "p")
    static_matches = sorted((shared_cache / "static").glob(f"node_static_{static_prefix}*.csv"))
    if static_matches:
        node_hash = int(static_matches[-1].stem.removeprefix(f"node_static_{static_prefix}"))
    else:
        node_hash = int(pd.util.hash_pandas_object(
            nodes[["node_id", "latitude", "longitude"]].astype({"node_id": str}), index=False
        ).sum())
    signature = {
        "version": 3,
        "acquisitions": [pd.Timestamp(item).isoformat() for item in acquisitions],
        "bands": list(s2_band_names(request.variables)),
        "node_hash": node_hash,
        "max_cloud": request.parameters.max_sentinel2_cloud,
        "max_day_difference": request.parameters.s2_max_day_difference,
        "date_windows": [
            (
                pd.Timestamp(window.start_date, tz="UTC").isoformat(),
                (pd.Timestamp(window.end_date, tz="UTC") + pd.Timedelta(days=1)).isoformat(),
            )
            for window in effective_date_windows(request)
        ],
    }
    digest = hashlib.sha1(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    target = shared_cache / "sentinel2" / f"sentinel2_batch_{digest}.csv"
    if not target.exists():
        raise FileNotFoundError(f"No se encontró la caché Sentinel-2 correspondiente a {area_dir.name}: {target}")
    return target


def _group_columns(frame: pd.DataFrame) -> list[str]:
    columns = ["date", "sentinel3_pixel_id"]
    if "area_id" in frame:
        columns.insert(0, "area_id")
    return columns


def load_s2_cache(path: Path) -> pd.DataFrame:
    """Carga la caché tabular y consolida escenas solapadas por nodo y fecha."""
    columns = ["node_id", "s2_source_time_start", *S2_SOURCE_COLUMNS]
    raw = pd.read_csv(path, usecols=columns)
    raw["node_id"] = raw["node_id"].astype(str)
    raw["s2_date"] = (
        pd.to_datetime(pd.to_numeric(raw.pop("s2_source_time_start"), errors="coerce"), unit="ms", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    raw = raw.rename(columns=S2_SOURCE_COLUMNS)
    values = list(S2_SOURCE_COLUMNS.values())
    return raw.groupby(["s2_date", "node_id"], as_index=False)[values].mean()


def _s2_date_catalog(s2: pd.DataFrame) -> pd.DataFrame:
    values = list(S2_SOURCE_COLUMNS.values())
    complete = s2[values].notna().all(axis=1)
    return (
        s2.assign(_complete=complete)
        .groupby("s2_date", as_index=False)
        .agg(s2_coverage=("_complete", "mean"))
    )


def choose_s2_dates(
    reference_dates: pd.Series | pd.DatetimeIndex,
    s2: pd.DataFrame,
    *,
    max_days: int = 20,
    preferred_coverage: float = 0.80,
) -> pd.DataFrame:
    """Elige la escena clara más próxima sin solicitar datos nuevos."""
    catalog = _s2_date_catalog(s2)
    rows: list[dict] = []
    for reference in pd.DatetimeIndex(pd.to_datetime(reference_dates).unique()).sort_values():
        candidates = catalog.copy()
        candidates["s2_gap_days"] = (candidates["s2_date"] - reference).abs().dt.days
        candidates = candidates.loc[candidates["s2_gap_days"].le(max_days)]
        if candidates.empty:
            rows.append({"date": reference, "s2_date": pd.NaT, "s2_gap_days": np.nan, "s2_coverage": 0.0})
            continue
        preferred = candidates.loc[candidates["s2_coverage"].ge(preferred_coverage)]
        pool = preferred if not preferred.empty else candidates
        selected = pool.sort_values(["s2_gap_days", "s2_coverage"], ascending=[True, False]).iloc[0]
        rows.append({
            "date": reference,
            "s2_date": selected["s2_date"],
            "s2_gap_days": int(selected["s2_gap_days"]),
            "s2_coverage": float(selected["s2_coverage"]),
        })
    return pd.DataFrame(rows)


def attach_fine_s2(
    frame: pd.DataFrame,
    s2: pd.DataFrame,
    *,
    max_days: int = 20,
    preferred_coverage: float = 0.80,
) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["node_id"] = out["node_id"].astype(str)
    choices = choose_s2_dates(out["date"], s2, max_days=max_days, preferred_coverage=preferred_coverage)
    out = out.merge(choices, on="date", how="left")
    return out.merge(s2, on=["s2_date", "node_id"], how="left")


def prepare_translation_rows(
    unified: pd.DataFrame,
    s2: pd.DataFrame,
    *,
    max_time_difference_hours: float = 1.5,
    max_s2_days: int = 20,
) -> pd.DataFrame:
    """Prepara pares reales Sentinel-3/Landsat y predictores disponibles en producción."""
    frame = unified.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    valid = frame["LST_K"].notna() & frame["LST_1km"].notna()
    if "s3_day_difference" in frame:
        valid &= pd.to_numeric(frame["s3_day_difference"], errors="coerce").eq(0)
    if "lst_time_difference_hours" in frame:
        valid &= pd.to_numeric(frame["lst_time_difference_hours"], errors="coerce").le(max_time_difference_hours)
    frame = attach_fine_s2(frame.loc[valid].copy(), s2, max_days=max_s2_days)
    return engineer_translation_features(frame, require_target=True)


def prepare_prediction_rows(
    unified: pd.DataFrame,
    s2: pd.DataFrame,
    *,
    max_s2_days: int = 20,
) -> pd.DataFrame:
    """Prepara todas las adquisiciones Sentinel-3 reales para traducirlas a 100 m."""
    frame = unified.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    valid = frame["LST_1km"].notna()
    if "s3_day_difference" in frame:
        valid &= pd.to_numeric(frame["s3_day_difference"], errors="coerce").eq(0)
    frame = attach_fine_s2(frame.loc[valid].copy(), s2, max_days=max_s2_days)
    return engineer_translation_features(frame, require_target=False)


def engineer_translation_features(frame: pd.DataFrame, *, require_target: bool) -> pd.DataFrame:
    out = frame.copy()
    group_columns = _group_columns(out)
    fine_columns = {
        "NDVI_S2_100m": "NDVI_S2_delta",
        "NDBI_S2_100m": "NDBI_S2_delta",
        "ALBEDO_S2_100m": "ALBEDO_S2_delta",
        "DEM": "DEM_delta",
        "aspect_ratio": "aspect_ratio_delta",
        "has_buildings": "has_buildings_delta",
    }
    for fine, delta in fine_columns.items():
        numeric = pd.to_numeric(out[fine], errors="coerce")
        coarse_mean = numeric.groupby([out[column] for column in group_columns]).transform("mean")
        out[delta] = numeric - coarse_mean
    if require_target:
        out["LST_group_mean_K"] = out.groupby(group_columns)["LST_K"].transform("mean")
        out["LST_spatial_anomaly_K"] = out["LST_K"] - out["LST_group_mean_K"]
    return out.sort_values(["date", "sentinel3_pixel_id", "node_id"]).reset_index(drop=True)


def _ridge(alpha: float = 10.0):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=alpha))


def spatial_estimators(random_state: int = 42) -> dict[str, object]:
    return {
        "ridge": _ridge(10.0),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestRegressor(
                n_estimators=350,
                max_depth=12,
                min_samples_leaf=8,
                max_features=0.85,
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            ExtraTreesRegressor(
                n_estimators=350,
                max_depth=14,
                min_samples_leaf=6,
                max_features=0.85,
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        "xgboost": XGBRegressor(
            n_estimators=450,
            max_depth=2,
            learning_rate=0.025,
            min_child_weight=20,
            subsample=0.80,
            colsample_bytree=0.90,
            reg_lambda=5.0,
            reg_alpha=0.20,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def direct_estimators(random_state: int = 42) -> dict[str, object]:
    return {
        "direct_ridge": _ridge(10.0),
        "direct_xgboost": XGBRegressor(
            n_estimators=500,
            max_depth=2,
            learning_rate=0.025,
            min_child_weight=20,
            subsample=0.80,
            colsample_bytree=0.90,
            reg_lambda=5.0,
            reg_alpha=0.20,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def _date_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("date")["date"].transform("size").astype(float)
    return (1.0 / counts * counts.mean()).to_numpy()


def _fit_with_optional_weights(model: object, x: pd.DataFrame, y: pd.Series, weights: np.ndarray) -> object:
    fitted = clone(model)
    if hasattr(fitted, "steps"):
        final_step = fitted.steps[-1][0]
        fitted.fit(x, y, **{f"{final_step}__sample_weight": weights})
    else:
        fitted.fit(x, y, sample_weight=weights)
    return fitted


def _coarse_table(frame: pd.DataFrame) -> pd.DataFrame:
    aggregations = {column: "mean" for column in COARSE_FEATURES if column in frame}
    aggregations["LST_group_mean_K"] = "mean"
    return frame.groupby(_group_columns(frame), as_index=False).agg(aggregations)


def fit_translation_bundle(frame: pd.DataFrame, name: str, random_state: int = 42) -> TranslationBundle:
    spatial_models = spatial_estimators(random_state)
    direct_models = direct_estimators(random_state)
    weights = _date_balanced_weights(frame)
    if name in direct_models:
        target = frame["LST_K"] - frame["LST_1km"]
        model = _fit_with_optional_weights(direct_models[name], frame[DIRECT_FEATURES], target, weights)
        return TranslationBundle(name, "direct", None, 0.0, None, model, COARSE_FEATURES, SPATIAL_FEATURES, DIRECT_FEATURES)

    coarse_kind, spatial_name = name.split("+", maxsplit=1)
    coarse = _coarse_table(frame)
    fitted_coarse_features = COARSE_FEATURES
    if coarse_kind == "bias":
        coarse_model = None
        coarse_bias = float((coarse["LST_group_mean_K"] - coarse["LST_1km"]).mean())
    elif coarse_kind == "linear":
        fitted_coarse_features = ["LST_1km"]
        coarse_model = _ridge(0.01)
        coarse_model.fit(coarse[fitted_coarse_features], coarse["LST_group_mean_K"])
        coarse_bias = 0.0
    elif coarse_kind == "ridge":
        coarse_model = _ridge(10.0)
        coarse_model.fit(coarse[COARSE_FEATURES], coarse["LST_group_mean_K"])
        coarse_bias = 0.0
    else:
        raise ValueError(f"Calibrador desconocido: {coarse_kind}")
    spatial_model = _fit_with_optional_weights(
        spatial_models[spatial_name], frame[SPATIAL_FEATURES], frame["LST_spatial_anomaly_K"], weights
    )
    return TranslationBundle(name, coarse_kind, coarse_model, coarse_bias, spatial_model, None, fitted_coarse_features, SPATIAL_FEATURES, DIRECT_FEATURES)


def predict_translation(bundle: TranslationBundle, frame: pd.DataFrame) -> np.ndarray:
    group_columns = _group_columns(frame)
    if bundle.coarse_kind == "direct":
        return frame["LST_1km"].to_numpy(float) + bundle.direct_model.predict(frame[bundle.direct_features])

    coarse = frame.groupby(group_columns, as_index=False).agg({column: "mean" for column in bundle.coarse_features})
    if bundle.coarse_kind == "bias":
        coarse["_coarse_prediction"] = coarse["LST_1km"] + bundle.coarse_bias
    else:
        coarse["_coarse_prediction"] = bundle.coarse_model.predict(coarse[bundle.coarse_features])
    keys = pd.MultiIndex.from_frame(frame[group_columns])
    coarse_lookup = coarse.set_index(group_columns)["_coarse_prediction"]
    coarse_prediction = coarse_lookup.reindex(keys).to_numpy(float)

    anomaly = bundle.spatial_model.predict(frame[bundle.spatial_features])
    anomaly_series = pd.Series(anomaly, index=frame.index)
    centered = anomaly_series - anomaly_series.groupby([frame[column] for column in group_columns]).transform("mean")
    return coarse_prediction + centered.to_numpy(float)


def metric_row(name: str, frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    observed = frame["LST_K"].to_numpy(float)
    absolute = np.abs(prediction - observed)
    by_date = pd.Series(absolute, index=frame.index).groupby(frame["date"]).mean()
    return {
        "modelo": name,
        "MAE_K": mean_absolute_error(observed, prediction),
        "MAE_macro_fecha_K": float(by_date.mean()),
        "RMSE_K": mean_squared_error(observed, prediction) ** 0.5,
        "R2": r2_score(observed, prediction),
        "Bias_K": float(np.mean(prediction - observed)),
        "n_fechas": int(frame["date"].nunique()),
        "n_filas": int(len(frame)),
    }


def candidate_names() -> list[str]:
    return [
        "bias+ridge",
        "bias+random_forest",
        "bias+extra_trees",
        "bias+xgboost",
        "linear+ridge",
        "linear+random_forest",
        "linear+extra_trees",
        "linear+xgboost",
        "ridge+random_forest",
        "ridge+extra_trees",
        "ridge+xgboost",
        "direct_xgboost",
    ]
