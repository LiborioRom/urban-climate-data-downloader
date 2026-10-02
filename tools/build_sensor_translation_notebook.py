from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "05_traduccion_sentinel3_landsat.ipynb"


def md(value: str):
    return nbf.v4.new_markdown_cell(value.strip())


def code(value: str):
    return nbf.v4.new_code_cell(value.strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}
notebook["cells"] = [
    md(
        """
# Traducción Sentinel-3 → Landsat a 100 m

Este experimento está diseñado para el objetivo real del TFM: generar una aproximación Landsat a 100 m a partir de Sentinel-3, con pocas fechas de referencia y sin utilizar variables que desaparezcan cuando no hay Landsat.

La estrategia separa dos problemas:

1. **Calibración entre sensores:** estima la media Landsat correspondiente a cada píxel Sentinel-3.
2. **Detalle espacial:** reparte esa temperatura dentro del píxel mediante Sentinel-2 a 100 m y morfología urbana estática.

Las anomalías espaciales se recentran a media cero dentro de cada píxel Sentinel-3. Así, el detalle de 100 m no cambia la media calibrada. El modelo se elige exclusivamente con fechas anteriores a 2025 mediante validación agrupada por fecha; 2025 queda intacto como test temporal.

Referencias metodológicas: [downscaling diario de Sentinel-3](https://doi.org/10.3390/rs14225752), [Random Forest para downscaling regional](https://doi.org/10.1016/j.rse.2016.02.038) y [validación de LST SLSTR](https://doi.org/10.3390/rs13112228).
"""
    ),
    code(
        """
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import GroupKFold

from tools.downscaling_training import find_latest_downscaling_dataset
from tools.sensor_translation import (
    candidate_names,
    fit_translation_bundle,
    load_s2_cache,
    metric_row,
    predict_translation,
    prepare_prediction_rows,
    prepare_translation_rows,
)

PROJECT_ROOT = Path.cwd()
TEST_YEAR = 2025
MAX_SENSOR_TIME_DIFFERENCE_H = 1.5
MAX_S2_GAP_DAYS = 20
RANDOM_STATE = 42

SOURCE_CSV = find_latest_downscaling_dataset(PROJECT_ROOT)
S2_CACHE = max((PROJECT_ROOT / "data_downloads" / "cache" / "sentinel2").glob("sentinel2_batch_*.csv"), key=lambda p: p.stat().st_mtime)
RUN_DIR = SOURCE_CSV.parents[1]
OUTPUT_DIR = SOURCE_CSV.parent / "model_ready" / "sensor_translation"
MODEL_DIR = RUN_DIR / "models" / "sensor_translation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Dataset: {SOURCE_CSV}")
print(f"Sentinel-2 fino reutilizado: {S2_CACHE}")
print(f"Salidas: {OUTPUT_DIR}")
"""
    ),
    md("## 1. Preparación de pares reales y covariables utilizables"),
    code(
        """
unified = pd.read_csv(SOURCE_CSV, low_memory=False)
s2 = load_s2_cache(S2_CACHE)
pairs = prepare_translation_rows(
    unified,
    s2,
    max_time_difference_hours=MAX_SENSOR_TIME_DIFFERENCE_H,
    max_s2_days=MAX_S2_GAP_DAYS,
)

pretest = pairs.loc[pairs["date"].dt.year.lt(TEST_YEAR)].copy()
test = pairs.loc[pairs["date"].dt.year.eq(TEST_YEAR)].copy()
if pretest["date"].nunique() < 10 or test["date"].nunique() < 3:
    raise ValueError("No hay suficientes fechas independientes para la validación temporal prevista.")

coverage = (
    pairs.assign(year=pairs["date"].dt.year)
    .groupby("year")
    .agg(filas=("node_id", "size"), fechas=("date", "nunique"),
         desfase_S2_mediano_dias=("s2_gap_days", "median"), cobertura_S2_media=("s2_coverage", "mean"))
)
print(f"Pares utilizables: {len(pairs):,} filas, {pairs['date'].nunique()} fechas")
print(f"Selección/entrenamiento: {pretest['date'].nunique()} fechas; test final: {test['date'].nunique()} fechas")
display(coverage.round(3))
"""
    ),
    md(
        """
## 2. Selección de arquitectura sin mirar 2025

Se comparan modelos lineales, Random Forest, Extra Trees y XGBoost. Cada pliegue contiene fechas completas: ningún nodo de una fecha aparece simultáneamente en entrenamiento y validación. La métrica de selección es el MAE medio por fecha, para que una adquisición con más píxeles válidos no domine el resultado.
"""
    ),
    code(
        """
groups = pretest["date"].dt.strftime("%Y-%m-%d")
cv = GroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
names = candidate_names()
oof = {name: pd.Series(np.nan, index=pretest.index, dtype=float) for name in names}

started = time.perf_counter()
for fold, (train_pos, valid_pos) in enumerate(cv.split(pretest, groups=groups), start=1):
    fold_train = pretest.iloc[train_pos]
    fold_valid = pretest.iloc[valid_pos]
    for name in names:
        bundle = fit_translation_bundle(fold_train, name, random_state=RANDOM_STATE + fold)
        oof[name].loc[fold_valid.index] = predict_translation(bundle, fold_valid)
    print(f"Pliegue {fold}/5 terminado")

cv_seconds = time.perf_counter() - started
cv_rows = [metric_row(name, pretest, prediction.loc[pretest.index].to_numpy()) for name, prediction in oof.items()]
cv_table = pd.DataFrame(cv_rows).sort_values(["MAE_macro_fecha_K", "RMSE_K"]).reset_index(drop=True)
selected_name = str(cv_table.iloc[0]["modelo"])

print(f"Validación cruzada terminada en {cv_seconds:.1f} s")
print(f"Arquitectura seleccionada: {selected_name}")
display(cv_table.round(4))
"""
    ),
    md("## 3. Test temporal final sobre 2025"),
    code(
        """
selected_pretest_bundle = fit_translation_bundle(pretest, selected_name, random_state=RANDOM_STATE)
selected_test_prediction = predict_translation(selected_pretest_bundle, test)

raw_prediction = test["LST_1km"].to_numpy(float)
mean_bias = float((pretest["LST_K"] - pretest["LST_1km"]).mean())
bias_prediction = raw_prediction + mean_bias

test_table = pd.DataFrame([
    metric_row("Sentinel-3 sin corregir", test, raw_prediction),
    metric_row("Corrección media entrenada", test, bias_prediction),
    metric_row(selected_name, test, selected_test_prediction),
])
baseline_mae = float(test_table.loc[test_table["modelo"].eq("Sentinel-3 sin corregir"), "MAE_K"].iloc[0])
test_table["mejora_MAE_vs_S3_%"] = 100 * (baseline_mae - test_table["MAE_K"]) / baseline_mae

test_predictions = test[[
    "area_id", "node_id", "date", "latitude", "longitude", "sentinel3_pixel_id",
    "LST_K", "LST_1km", "s2_date", "s2_gap_days", "lst_time_difference_hours",
]].copy()
test_predictions["LST_predicted_K"] = selected_test_prediction
test_predictions["error_K"] = test_predictions["LST_predicted_K"] - test_predictions["LST_K"]

display(test_table.round(4))
"""
    ),
    md(
        """
## 4. Modelo de producción y traducción de todas las fechas Sentinel-3

Tras cerrar el test, la arquitectura elegida se reentrena con las 35 fechas coincidentes. Se traduce cada adquisición Sentinel-3 real del CSV. No se realizan peticiones a ninguna API: Sentinel-2 se recupera de la caché existente.

El CSV de producción no sustituye a Landsat observado. `LST_predicted_K` es una etiqueta sintética con incertidumbre equivalente al error del test anterior.
"""
    ),
    code(
        """
production_bundle = fit_translation_bundle(pairs, selected_name, random_state=RANDOM_STATE)
prediction_rows = prepare_prediction_rows(unified, s2, max_s2_days=MAX_S2_GAP_DAYS)
prediction_rows["LST_predicted_K"] = predict_translation(production_bundle, prediction_rows)
prediction_rows["target_source"] = "sentinel3_to_landsat_model"
prediction_rows["downscaling_model"] = selected_name

production_columns = [
    "area_id", "node_id", "date", "timestamp_utc", "latitude", "longitude", "row", "col",
    "sentinel3_pixel_id", "sentinel3_datetime_utc", "LST_1km", "LST_predicted_K",
    "NDVI_S2_100m", "NDBI_S2_100m", "ALBEDO_S2_100m", "s2_date", "s2_gap_days",
    "DEM", "aspect_ratio", "has_buildings", "Tair_C_1km", "RH_1km", "Rsol_Wm2_1km",
    "wind_speed_1km", "rain_3d_log_1km", "sin_doy", "cos_doy", "target_source", "downscaling_model",
]
production_columns = [column for column in production_columns if column in prediction_rows]
production = prediction_rows[production_columns].copy()

model_path = MODEL_DIR / "sentinel3_to_landsat_bundle.joblib"
cv_path = MODEL_DIR / "comparacion_modelos_cv.csv"
test_metrics_path = MODEL_DIR / "metricas_test_2025.csv"
test_predictions_path = OUTPUT_DIR / "predicciones_test_2025.csv"
production_path = OUTPUT_DIR / "dataset_lst_100m_traducido.csv"
metadata_path = MODEL_DIR / "metadata.json"

joblib.dump(production_bundle, model_path)
cv_table.to_csv(cv_path, index=False)
test_table.to_csv(test_metrics_path, index=False)
test_predictions.to_csv(test_predictions_path, index=False)
production.to_csv(production_path, index=False)

metadata = {
    "source_csv": str(SOURCE_CSV),
    "s2_cache": str(S2_CACHE),
    "selected_architecture": selected_name,
    "selection_metric": "MAE_macro_fecha_K",
    "selection_dates": int(pretest["date"].nunique()),
    "test_year": TEST_YEAR,
    "test_dates": int(test["date"].nunique()),
    "production_training_dates": int(pairs["date"].nunique()),
    "translated_dates": int(production["date"].nunique()),
    "translated_rows": int(len(production)),
    "max_sensor_time_difference_hours": MAX_SENSOR_TIME_DIFFERENCE_H,
    "max_s2_gap_days": MAX_S2_GAP_DAYS,
    "test_metrics": test_table.to_dict(orient="records"),
}
metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

print(f"Modelo: {model_path}")
print(f"Métricas finales: {test_metrics_path}")
print(f"Predicciones de test: {test_predictions_path}")
print(f"Dataset traducido: {production_path}")
print(f"Producción: {len(production):,} filas en {production['date'].nunique()} fechas")
"""
    ),
    md(
        """
## Interpretación y siguiente paso

- El resultado de 2025 es la estimación honesta de generalización temporal; no debe sustituirse por el ajuste de producción.
- Para entrenar la GNN, `LST_predicted_K` puede actuar como objetivo sintético frecuente y `LST_K` como referencia observada escasa.
- La GNN debe conocer la procedencia de la etiqueta y ponderar más las observaciones Landsat reales.
- Añadir fechas de invierno solo sería recomendable si el sistema final debe operar también fuera de mayo-octubre. Para el objetivo cálido actual, aumentar estaciones y años cálidos sería más informativo.
"""
    ),
]

nbf.write(notebook, OUTPUT)
print(OUTPUT)
