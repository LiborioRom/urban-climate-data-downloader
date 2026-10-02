from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from tools.downscaling_training import build_grid_edges
from tools.sensor_translation import fit_translation_bundle, predict_translation


def grouped_date_oof(
    frame: pd.DataFrame,
    model_name: str,
    *,
    max_splits: int = 5,
    random_state: int = 42,
) -> pd.Series:
    """Predicciones fuera de muestra sin repartir una fecha entre train y validación."""
    dates = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    n_dates = dates.nunique()
    if n_dates < 3:
        raise ValueError("Se necesitan al menos tres fechas para la validación agrupada.")
    splitter = GroupKFold(n_splits=min(max_splits, n_dates), shuffle=True, random_state=random_state)
    predictions = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold, (train_position, valid_position) in enumerate(splitter.split(frame, groups=dates), start=1):
        train = frame.iloc[train_position]
        valid = frame.iloc[valid_position]
        bundle = fit_translation_bundle(train, model_name, random_state=random_state + fold)
        predictions.iloc[valid_position] = predict_translation(bundle, valid)
    if predictions.isna().any():
        raise RuntimeError("La validación agrupada no generó una predicción para cada fila.")
    return predictions


def macro_date_mae(frame: pd.DataFrame, prediction: pd.Series | np.ndarray) -> float:
    errors = np.abs(np.asarray(prediction, dtype=float) - frame["LST_K"].to_numpy(float))
    by_date = pd.DataFrame({
        "date": pd.to_datetime(frame["date"]).to_numpy(),
        "error": errors,
    }).groupby("date")["error"].mean()
    return float(by_date.mean())


def choose_model_scope(
    general_mae: float,
    local_mae: float | None,
    *,
    minimum_local_improvement: float = 0.02,
) -> str:
    """Prefiere el modelo general salvo mejora local reproducible por validación."""
    if local_mae is None or not np.isfinite(local_mae):
        return "general"
    return "local" if local_mae <= general_mae * (1 - minimum_local_improvement) else "general"


def build_gnn_ready_dataset(
    unified: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    predicted_lst: np.ndarray,
    *,
    area_id: str,
    model_name: str,
    model_scope: str,
    test_mae_k: float | None,
    plausible_target_range_k: tuple[float, float] = (270.0, 350.0),
) -> pd.DataFrame:
    """Une observaciones y LST sintética conservando explícitamente su procedencia."""
    base = unified.copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    predicted = prediction_rows.copy()
    predicted["date"] = pd.to_datetime(predicted["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    predicted["LST_model_K"] = np.asarray(predicted_lst, dtype=float)
    extra_columns = [
        "node_id", "date", "LST_model_K", "NDVI_S2_100m", "NDBI_S2_100m",
        "ALBEDO_S2_100m", "s2_date", "s2_gap_days", "s2_coverage",
    ]
    extra_columns = [column for column in extra_columns if column in predicted]
    predicted = predicted[extra_columns].drop_duplicates(["node_id", "date"], keep="first")
    out = base.merge(predicted, on=["node_id", "date"], how="left")
    observed = pd.to_numeric(out["LST_K"], errors="coerce").notna()
    synthetic = ~observed & pd.to_numeric(out["LST_model_K"], errors="coerce").notna()
    out["LST_target_K"] = pd.to_numeric(out["LST_K"], errors="coerce").where(observed, out["LST_model_K"])
    out["target_source"] = np.select(
        [observed, synthetic], ["landsat_observed", f"sentinel3_downscaled_{model_scope}"], default="unavailable"
    )
    out["target_is_observed"] = observed.astype("int8")
    out["target_is_synthetic"] = synthetic.astype("int8")
    minimum_k, maximum_k = plausible_target_range_k
    out["target_physical_range_ok"] = out["LST_target_K"].between(minimum_k, maximum_k).astype("int8")
    out["target_training_eligible"] = (
        out["LST_target_K"].notna() & out["LST_target_K"].between(minimum_k, maximum_k)
    ).astype("int8")
    out["downscaling_model"] = model_name
    out["downscaling_scope"] = model_scope
    out["downscaling_test_mae_k"] = test_mae_k
    out["area_id"] = area_id
    out["global_node_id"] = area_id + "__" + out["node_id"].astype(str)
    return out.sort_values(["date", "row", "col"]).reset_index(drop=True)


def export_area_outputs(frame: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset_gnn_ready.csv"
    frame.to_csv(dataset_path, index=False)
    edges = build_grid_edges(frame)
    edges_path = output_dir / "edges.csv"
    edges.to_csv(edges_path, index=False)
    summary = {
        "rows": int(len(frame)),
        "nodes": int(frame["node_id"].nunique()),
        "dates": int(frame["date"].nunique()),
        "observed_targets": int(frame["target_is_observed"].sum()),
        "synthetic_targets": int(frame["target_is_synthetic"].sum()),
        "unavailable_targets": int(frame["LST_target_K"].isna().sum()),
        "out_of_range_targets": int(
            (frame["LST_target_K"].notna() & frame["target_physical_range_ok"].eq(0)).sum()
        ),
        "training_eligible_targets": int(frame["target_training_eligible"].sum()),
        "model": str(frame["downscaling_model"].iloc[0]),
        "scope": str(frame["downscaling_scope"].iloc[0]),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"dataset": dataset_path, "edges": edges_path, "summary": summary_path}
