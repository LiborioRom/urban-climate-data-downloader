from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "LST_K"
COARSE_LST_COLUMN = "LST_1km"
RESIDUAL_TARGET_COLUMN = "Delta_LST"
ORIGINAL_DOWNSCALING_FEATURES = [
    "LST_1km",
    "NDVI_1km",
    "NDVI_delta",
    "NDBI_delta",
    "ALBEDO_delta",
    "aspect_ratio_delta",
]
ORIGINAL_FINE_COARSE_PAIRS = {
    "NDVI_delta": ("NDVI", "NDVI_1km"),
    "NDBI_delta": ("NDBI", "NDBI_1km"),
    "ALBEDO_delta": ("ALBEDO", "ALBEDO_1km"),
    "aspect_ratio_delta": ("aspect_ratio", "aspect_ratio_1km"),
}
NON_FEATURE_COLUMNS = {
    TARGET_COLUMN,
    "area_id", "dataset_purpose", "node_id", "date", "timestamp_utc",
    "sentinel3_pixel_id", "sentinel3_datetime_utc", "landsat_datetime_utc",
    "latitude", "longitude", "x", "y",
    "sentinel3_latitude", "sentinel3_longitude",
    "sentinel3_distance_m", "s3_day_difference", "lst_time_difference_hours",
    "landsat_valid", "sentinel3_valid", "training_ready", "target_source",
    "LST_downscaled_K", "target_lst_k",
}


def find_latest_downscaling_dataset(project_root: Path) -> Path:
    """Devuelve la descarga TFM-GNN terminada más reciente, nunca una salida general."""
    candidates = sorted(
        (project_root / "runs").glob("*/data_downloads/dataset_unificado.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for dataset_path in candidates:
        metadata_path = dataset_path.with_name("metadata.json")
        quality_path = dataset_path.with_name("informe_calidad.json")
        if not metadata_path.exists() or not quality_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            purpose = metadata.get("request", {}).get("dataset_purpose")
        except (OSError, json.JSONDecodeError):
            continue
        if purpose == "lst_downscaling":
            return dataset_path
    raise FileNotFoundError(
        "No existe una descarga terminada con el propósito Proyecto TFM-GNN."
    )


def add_rain_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Añade los derivados de lluvia cuando se carga un CSV generado por una versión anterior."""
    out = frame.copy()
    if "rain_3d" not in out:
        return out
    rain = pd.to_numeric(out["rain_3d"], errors="coerce")
    if "rain_3d_log" not in out:
        out["rain_3d_log"] = np.log1p(rain.clip(lower=0))
    if "is_rainy" not in out:
        out["is_rainy"] = np.where(rain.isna(), np.nan, (rain > 0).astype("int8"))
    return out


def prepare_training_rows(frame: pd.DataFrame, warm_months: list[int] | None = None) -> pd.DataFrame:
    required = {"node_id", "date", TARGET_COLUMN, COARSE_LST_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "El CSV no corresponde al propósito de downscaling; faltan columnas: " + ", ".join(missing)
        )
    if "dataset_purpose" in frame:
        purposes = set(frame["dataset_purpose"].dropna().astype(str))
        if purposes and purposes != {"lst_downscaling"}:
            raise ValueError("El CSV no fue generado con el propósito Proyecto TFM-GNN.")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.loc[out["date"].notna()].copy()
    if warm_months:
        out = out.loc[out["date"].dt.month.isin(warm_months)].copy()
    if out.empty:
        raise ValueError("No quedan observaciones después del filtro temporal.")
    out["training_ready"] = out[TARGET_COLUMN].notna() & out[COARSE_LST_COLUMN].notna()
    return out.sort_values(["date", "node_id"]).reset_index(drop=True)


def add_original_downscaling_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce las variables y el objetivo residual usados en el proyecto original."""
    required = {TARGET_COLUMN, COARSE_LST_COLUMN, "NDVI_1km"}
    for fine_column, coarse_column in ORIGINAL_FINE_COARSE_PAIRS.values():
        required.update((fine_column, coarse_column))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Faltan variables auxiliares del downscaling original: " + ", ".join(missing)
        )
    out = frame.copy()
    for delta_column, (fine_column, coarse_column) in ORIGINAL_FINE_COARSE_PAIRS.items():
        out[delta_column] = (
            pd.to_numeric(out[fine_column], errors="coerce")
            - pd.to_numeric(out[coarse_column], errors="coerce")
        )
    out[RESIDUAL_TARGET_COLUMN] = (
        pd.to_numeric(out[TARGET_COLUMN], errors="coerce")
        - pd.to_numeric(out[COARSE_LST_COLUMN], errors="coerce")
    )
    return out


def prepare_coincident_training_rows(
    frame: pd.DataFrame,
    warm_months: list[int] | None = None,
) -> pd.DataFrame:
    """Conserva pares Landsat–Sentinel-3 del mismo día con todas las variables originales."""
    out = prepare_training_rows(frame, warm_months=warm_months)
    out = add_original_downscaling_features(out)
    if "s3_day_difference" in out:
        same_day = pd.to_numeric(out["s3_day_difference"], errors="coerce").eq(0)
    else:
        same_day = pd.Series(True, index=out.index)
    complete = out[ORIGINAL_DOWNSCALING_FEATURES + [RESIDUAL_TARGET_COLUMN]].notna().all(axis=1)
    out["training_ready"] = same_day & complete
    if not out["training_ready"].any():
        raise ValueError(
            "No hay pares Landsat–Sentinel-3 del mismo día con las variables auxiliares completas."
        )
    return out.sort_values(["date", "node_id"]).reset_index(drop=True)


def select_numeric_features(frame: pd.DataFrame) -> list[str]:
    numeric = frame.select_dtypes(include=["number", "bool"]).columns
    features = [column for column in numeric if column not in NON_FEATURE_COLUMNS]
    if COARSE_LST_COLUMN not in features:
        raise ValueError("LST_1km debe formar parte de las variables predictoras.")
    empty = [column for column in features if frame[column].notna().sum() == 0]
    features = [column for column in features if column not in empty]
    if "rain_3d_log" in features:
        features = [column for column in features if column != "rain_3d"]
    if "rain_3d_log_1km" in features:
        features = [column for column in features if column != "rain_3d_1km"]
    return features


def assign_temporal_splits(frame: pd.DataFrame) -> pd.Series:
    """Separa por años completos cuando es posible y nunca mezcla nodos de una misma fecha."""
    dates = pd.DatetimeIndex(frame.loc[frame["training_ready"], "date"].drop_duplicates().sort_values())
    if len(dates) < 5:
        raise ValueError("Se necesitan al menos cinco fechas emparejadas para crear train, validación y test.")
    years = sorted(set(dates.year))
    split_by_date: dict[pd.Timestamp, str] = {}
    if len(years) >= 3:
        validation_year, test_year = years[-2], years[-1]
        for value in dates:
            split_by_date[value] = "test" if value.year == test_year else "validation" if value.year == validation_year else "train"
    elif len(years) == 2:
        test_year = years[-1]
        earlier = dates[dates.year != test_year]
        cut = max(1, int(np.floor(len(earlier) * 0.8)))
        validation_dates = set(earlier[cut:]) or {earlier[-1]}
        for value in dates:
            split_by_date[value] = "test" if value.year == test_year else "validation" if value in validation_dates else "train"
    else:
        train_end = max(1, int(np.floor(len(dates) * 0.6)))
        validation_end = max(train_end + 1, int(np.floor(len(dates) * 0.8)))
        validation_end = min(validation_end, len(dates) - 1)
        for index, value in enumerate(dates):
            split_by_date[value] = "train" if index < train_end else "validation" if index < validation_end else "test"
    result = frame["date"].map(split_by_date).fillna("prediction_only")
    for required in ("train", "validation", "test"):
        if required not in set(result):
            raise ValueError(f"La división temporal no ha podido crear el conjunto {required}.")
    return result


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_008.8
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp = p2 - p1
    dl = np.deg2rad(lon2 - lon1)
    value = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * radius * np.arctan2(np.sqrt(value), np.sqrt(max(0.0, 1 - value))))


def build_grid_edges(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"node_id", "row", "col", "latitude", "longitude"}
    if not required.issubset(frame.columns):
        raise ValueError("Se necesitan node_id, row, col, latitude y longitude para construir el grafo.")
    nodes = frame[list(required)].drop_duplicates("node_id").dropna(subset=["row", "col"])
    lookup = {(int(row.row), int(row.col)): row for row in nodes.itertuples(index=False)}
    edges = []
    for (row_index, col_index), source in lookup.items():
        for neighbour in ((row_index + 1, col_index), (row_index, col_index + 1)):
            target = lookup.get(neighbour)
            if target is None:
                continue
            distance = _haversine_m(source.latitude, source.longitude, target.latitude, target.longitude)
            edges.extend([
                {"source_node_id": source.node_id, "target_node_id": target.node_id, "distance_m": distance},
                {"source_node_id": target.node_id, "target_node_id": source.node_id, "distance_m": distance},
            ])
    return pd.DataFrame(edges, columns=["source_node_id", "target_node_id", "distance_m"])


def export_neural_ready(
    frame: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    training = frame.loc[frame["split"].isin(["train", "validation", "test"])].copy()
    train = training.loc[training["split"].eq("train")]
    means = train[feature_columns].mean(numeric_only=True)
    scales = train[feature_columns].std(numeric_only=True).replace(0, 1).fillna(1)
    identifiers = [column for column in ("area_id", "node_id", "date", "latitude", "longitude", "row", "col", "split") if column in frame]
    outputs: dict[str, Path] = {}
    for split in ("train", "validation", "test"):
        part = frame.loc[frame["split"].eq(split), identifiers + feature_columns + [TARGET_COLUMN]].copy()
        part.loc[:, feature_columns] = (part[feature_columns] - means) / scales
        target = output_dir / f"{split}.csv"
        part.to_csv(target, index=False)
        outputs[split] = target
    edges = build_grid_edges(frame)
    edges_path = output_dir / "edges.csv"
    edges.to_csv(edges_path, index=False)
    outputs["edges"] = edges_path
    normalization_path = output_dir / "normalization.json"
    normalization_path.write_text(
        json.dumps({"features": feature_columns, "mean": means.to_dict(), "std": scales.to_dict()}, indent=2),
        encoding="utf-8",
    )
    outputs["normalization"] = normalization_path
    return outputs
