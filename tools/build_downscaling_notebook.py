from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "02_downscaling_xgboost.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}
notebook["cells"] = [
    markdown(
        """
# Downscaling Landsat–Sentinel-3 con XGBoost

Proyecto TFM-GNN. Este notebook reproduce las variables auxiliares del proyecto original y entrena un modelo residual únicamente con fechas Landsat y Sentinel-3 coincidentes. La separación es temporal: los años antiguos entrenan, el penúltimo valida y el último evalúa.

Las fechas Sentinel-3 sin observación Landsat no pueden entrenar el downscaling. Además, las variables espectrales finas actuales solo existen en fechas Landsat; por eso esta versión genera predicciones evaluables para los pares coincidentes y no inventa covariables para fechas intermedias.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from tools.downscaling_training import (
    ORIGINAL_DOWNSCALING_FEATURES,
    RESIDUAL_TARGET_COLUMN,
    add_rain_features,
    assign_temporal_splits,
    export_neural_ready,
    find_latest_downscaling_dataset,
    prepare_coincident_training_rows,
)

PROJECT_ROOT = Path.cwd()
EXPECTED_YEAR_RANGE = (2013, 2025)
WARM_MONTHS = [5, 6, 7, 8, 9, 10]
RANDOM_STATE = 42
USE_GPU = False  # En este volumen la CPU ofrece prácticamente el mismo tiempo y menos complejidad.

SOURCE_CSV = find_latest_downscaling_dataset(PROJECT_ROOT)
RUN_DIR = SOURCE_CSV.parents[1]
MODEL_DIR = RUN_DIR / "models"
OUTPUT_DIR = SOURCE_CSV.parent / "model_ready"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Fuente: {SOURCE_CSV}")
print(f"Salida: {OUTPUT_DIR}")
"""
    ),
    markdown("## 1. Pares coincidentes y variables del proyecto original"),
    code(
        """
raw = add_rain_features(pd.read_csv(SOURCE_CSV, low_memory=False))
prepared = prepare_coincident_training_rows(raw, warm_months=WARM_MONTHS)
prepared["split"] = assign_temporal_splits(prepared)
model_data = prepared.loc[prepared["training_ready"]].copy()

coverage = pd.DataFrame({
    "filas": model_data.groupby("split").size(),
    "fechas": model_data.groupby("split")["date"].nunique(),
    "año_inicial": model_data.groupby("split")["date"].min().dt.year,
    "año_final": model_data.groupby("split")["date"].max().dt.year,
}).reindex(["train", "validation", "test"])

print(f"Filas brutas: {len(raw):,}")
print(f"Pares coincidentes completos: {len(model_data):,} en {model_data['date'].nunique()} fechas")
print(f"Diferencia horaria máxima Landsat–Sentinel-3: {model_data['lst_time_difference_hours'].max():.2f} h")
display(coverage)
display(pd.DataFrame({"variables del modelo": ORIGINAL_DOWNSCALING_FEATURES}))
"""
    ),
    markdown(
        """
## 2. Variables contextuales compatibles

La tabla mide disponibilidad y correlación lineal absoluta con el residual térmico. Es exploratoria: una correlación alta no demuestra causalidad y estas variables no entran todavía en el modelo principal.
"""
    ),
    code(
        """
if {"DEM", "DEM_1km"}.issubset(prepared.columns):
    prepared["DEM_delta"] = prepared["DEM"] - prepared["DEM_1km"]
if {"has_buildings", "has_buildings_1km"}.issubset(prepared.columns):
    prepared["has_buildings_delta"] = prepared["has_buildings"] - prepared["has_buildings_1km"]

candidate_columns = [
    "DEM_delta", "has_buildings_delta", "Tair_C", "RH", "Rsol_Wm2",
    "wind_speed", "rain_3d_log", "is_rainy", "sin_doy", "cos_doy",
]
eligible = prepared.loc[prepared["training_ready"]].copy()
candidate_rows = []
for column in candidate_columns:
    if column not in eligible:
        candidate_rows.append({"variable": column, "disponibilidad_%": 0.0, "valores_unicos": 0, "corr_abs_residual": np.nan})
        continue
    values = pd.to_numeric(eligible[column], errors="coerce")
    valid = values.notna() & eligible[RESIDUAL_TARGET_COLUMN].notna()
    correlation = values[valid].corr(eligible.loc[valid, RESIDUAL_TARGET_COLUMN]) if valid.sum() > 2 and values[valid].nunique() > 1 else np.nan
    candidate_rows.append({
        "variable": column,
        "disponibilidad_%": round(100 * values.notna().mean(), 2),
        "valores_unicos": int(values.nunique(dropna=True)),
        "corr_abs_residual": abs(correlation) if pd.notna(correlation) else np.nan,
    })

candidate_report = pd.DataFrame(candidate_rows).sort_values(
    ["disponibilidad_%", "corr_abs_residual"], ascending=[False, False], na_position="last"
)
display(candidate_report)

def make_model():
    return XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.1,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        device="cuda" if USE_GPU else "cpu",
    )

morphology = [column for column in ["DEM_delta", "has_buildings_delta"] if column in eligible]
weather_calendar = [
    column for column in [
        "Tair_C", "RH", "Rsol_Wm2", "wind_speed", "rain_3d_log",
        "is_rainy", "sin_doy", "cos_doy",
    ] if column in eligible
]
feature_sets = {
    "Original": ORIGINAL_DOWNSCALING_FEATURES,
    "Original + morfología": ORIGINAL_DOWNSCALING_FEATURES + morphology,
    "Original + meteo/calendario": ORIGINAL_DOWNSCALING_FEATURES + weather_calendar,
    "Original + todo el contexto": ORIGINAL_DOWNSCALING_FEATURES + morphology + weather_calendar,
}
context_experiments = []
experiment_train = eligible.loc[eligible["split"].eq("train")]
experiment_validation = eligible.loc[eligible["split"].eq("validation")]
for name, experiment_features in feature_sets.items():
    experiment = make_model()
    experiment.fit(
        experiment_train[experiment_features],
        experiment_train[RESIDUAL_TARGET_COLUMN],
        verbose=False,
    )
    predicted = experiment_validation["LST_1km"] + experiment.predict(experiment_validation[experiment_features])
    context_experiments.append({
        "configuración": name,
        "n_variables": len(experiment_features),
        "MAE_validacion_K": mean_absolute_error(experiment_validation["LST_K"], predicted),
        "RMSE_validacion_K": mean_squared_error(experiment_validation["LST_K"], predicted) ** 0.5,
    })
context_experiment_table = pd.DataFrame(context_experiments).sort_values("RMSE_validacion_K")
display(context_experiment_table.round(4))
"""
    ),
    markdown("## 3. Entrenamiento residual y evaluación frente al baseline"),
    code(
        """
features = ORIGINAL_DOWNSCALING_FEATURES
train = model_data.loc[model_data["split"].eq("train")]
validation = model_data.loc[model_data["split"].eq("validation")]
test = model_data.loc[model_data["split"].eq("test")].copy()

model = make_model()

started = time.perf_counter()
model.fit(
    train[features],
    train[RESIDUAL_TARGET_COLUMN],
    eval_set=[(validation[features], validation[RESIDUAL_TARGET_COLUMN])],
    verbose=False,
)
training_seconds = time.perf_counter() - started

test["baseline_LST_K"] = test["LST_1km"]
test["Delta_LST_pred"] = model.predict(test[features])
test["LST_downscaled_K"] = test["LST_1km"] + test["Delta_LST_pred"]

def metrics(name, observed, predicted):
    error = predicted - observed
    return {
        "modelo": name,
        "MAE_K": mean_absolute_error(observed, predicted),
        "RMSE_K": mean_squared_error(observed, predicted) ** 0.5,
        "R2": r2_score(observed, predicted),
        "Bias_K": error.mean(),
    }

metrics_table = pd.DataFrame([
    metrics("Baseline Sentinel-3 1 km", test["LST_K"], test["baseline_LST_K"]),
    metrics("XGBoost downscaling 100 m", test["LST_K"], test["LST_downscaled_K"]),
])
baseline_mae = metrics_table.loc[0, "MAE_K"]
baseline_rmse = metrics_table.loc[0, "RMSE_K"]
metrics_table["mejora_MAE_%"] = 100 * (baseline_mae - metrics_table["MAE_K"]) / baseline_mae
metrics_table["mejora_RMSE_%"] = 100 * (baseline_rmse - metrics_table["RMSE_K"]) / baseline_rmse

print(f"Entrenamiento CPU: {training_seconds:.2f} s")
display(metrics_table.round(4))
"""
    ),
    markdown("## 4. Importancia, artefactos reproducibles y datos para modelos neuronales"),
    code(
        """
importance = pd.DataFrame({
    "variable": features,
    "importancia": model.feature_importances_,
}).sort_values("importancia", ascending=False)
display(importance)

model_data["Delta_LST_pred"] = model.predict(model_data[features])
model_data["LST_downscaled_K"] = model_data["LST_1km"] + model_data["Delta_LST_pred"]

model_path = MODEL_DIR / "xgboost_lst_downscaling.json"
model.save_model(model_path)
joblib.dump(model, MODEL_DIR / "xgboost_lst_downscaling.joblib")
metrics_path = MODEL_DIR / "metricas_test.csv"
importance_path = MODEL_DIR / "importancia_variables.csv"
predictions_path = OUTPUT_DIR / "predicciones_test.csv"
coincident_path = OUTPUT_DIR / "dataset_downscaling_coincidente.csv"
candidate_path = MODEL_DIR / "variables_contextuales_candidatas.csv"
context_experiment_path = MODEL_DIR / "comparativa_variables_contextuales.csv"

metrics_table.to_csv(metrics_path, index=False)
importance.to_csv(importance_path, index=False)
candidate_report.to_csv(candidate_path, index=False)
context_experiment_table.to_csv(context_experiment_path, index=False)
test.to_csv(predictions_path, index=False)
model_data.to_csv(coincident_path, index=False)
neural_outputs = export_neural_ready(model_data, features, OUTPUT_DIR / "gnn")

metadata = {
    "source_csv": str(SOURCE_CSV),
    "coincidence_rule": "same calendar date; s3_day_difference == 0",
    "features": features,
    "target": RESIDUAL_TARGET_COLUMN,
    "training_seconds_cpu": training_seconds,
    "rows": int(len(model_data)),
    "dates": int(model_data["date"].nunique()),
    "test_year": int(test["date"].dt.year.max()),
    "metrics": metrics_table.to_dict(orient="records"),
}
(MODEL_DIR / "downscaling_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Modelo: {model_path}")
print(f"Métricas: {metrics_path}")
print(f"Predicciones test: {predictions_path}")
print(f"Dataset coincidente: {coincident_path}")
print("Salidas neuronales:")
for name, path in neural_outputs.items():
    print(f"  {name}: {path}")
"""
    ),
    markdown(
        """
## 5. Qué probar después

- **Disponibles ya:** `DEM_delta` y `has_buildings_delta` aportan contraste espacial; ERA5-Land (`Tair_C`, `RH`, radiación, viento y lluvia) y el ciclo anual aportan contexto temporal. Deben compararse por validación temporal, no añadirse automáticamente.
- **Compatibles con las fuentes actuales:** EVI, SAVI, NDMI, BSI, NDWI/MNDWI, reflectancias y `urban_index` pueden derivarse de Landsat L2 y Sentinel-2 en las mismas fechas.
- **Urbanismo estático:** altura y huella de edificios, `sky_view_factor`, densidad viaria y fracción impermeable pueden añadirse desde OSM/CNIG si se dispone de cobertura fiable.
- **Control de calidad:** incertidumbre y banderas SLSTR deben usarse para filtrar o ponderar muestras, no como sustitutos de la señal física.

Para aplicar el modelo a días sin Landsat será necesario conservar covariables finas por nodo procedentes de Sentinel-2 u otra fuente. El CSV actual no contiene NDVI/NDBI/ALBEDO finos en esos días.
"""
    ),
]

nbf.write(notebook, OUTPUT)
print(OUTPUT)
