from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "06_downscaling_multizona.ipynb"


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
# Downscaling multizona Sentinel-3 → Landsat

Este notebook reutiliza el framework seleccionado en el experimento anterior (`bias+random_forest`) y decide por área entre:

- **modelo generalista**, entrenado con todas las zonas; y
- **modelo local**, entrenado exclusivamente con la zona correspondiente.

La decisión se toma con predicciones fuera de muestra agrupadas por fecha y sin consultar 2025. El modelo local solo se acepta si dispone de suficientes fechas históricas y mejora al generalista al menos un 2 %. Después se evalúa una única vez sobre 2025, se reentrena para producción y se exporta un CSV por área preparado para la futura GNN.
"""
    ),
    code(
        """
from pathlib import Path
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd
from IPython.display import display

from tools.multizone_downscaling import (
    build_gnn_ready_dataset,
    choose_model_scope,
    export_area_outputs,
    grouped_date_oof,
    macro_date_mae,
)
from tools.sensor_translation import (
    find_latest_multizone_run,
    fit_translation_bundle,
    load_s2_cache,
    metric_row,
    predict_translation,
    prepare_prediction_rows,
    prepare_translation_rows,
    resolve_area_s2_cache,
)

PROJECT_ROOT = Path.cwd()
SOURCE_RUN = None  # Path explícita opcional; None usa la descarga multizona más reciente.
TEST_YEAR = 2025
MODEL_NAME = "bias+random_forest"
MAX_SENSOR_TIME_DIFFERENCE_H = 1.5
MAX_S2_GAP_DAYS = 20
MIN_LOCAL_PRETEST_DATES = 12
MIN_LOCAL_IMPROVEMENT = 0.02
RANDOM_STATE = 42
PLAUSIBLE_TARGET_RANGE_K = (270.0, 350.0)  # Control conservador para LST diurna de mayo–octubre.

RUN_DIR = Path(SOURCE_RUN) if SOURCE_RUN else find_latest_multizone_run(PROJECT_ROOT)
RUN_METADATA = json.loads((RUN_DIR / "combined" / "metadata_multizona.json").read_text(encoding="utf-8"))
SHARED_CACHE = Path(RUN_METADATA["shared_cache"])
OUTPUT_DIR = RUN_DIR / "model_ready_multizona"
MODEL_DIR = RUN_DIR / "models" / "multizone_downscaling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

AREA_DIRS = {
    path.name: path for path in sorted((RUN_DIR / "areas").iterdir())
    if path.is_dir() and (path / "data_downloads" / "dataset_unificado.csv").exists()
}
print(f"Descarga: {RUN_DIR}")
print(f"Áreas completas: {len(AREA_DIRS)}")
print(f"Salida: {OUTPUT_DIR}")
"""
    ),
    md("## 1. Pares coincidentes y caché Sentinel-2 correcta por área"),
    code(
        """
pairs_by_area = {}
area_resources = {}
coverage_rows = []

for area_id, area_dir in AREA_DIRS.items():
    unified_path = area_dir / "data_downloads" / "dataset_unificado.csv"
    s2_path = resolve_area_s2_cache(area_dir, SHARED_CACHE)
    unified = pd.read_csv(unified_path, low_memory=False)
    s2 = load_s2_cache(s2_path)
    pairs = prepare_translation_rows(
        unified, s2,
        max_time_difference_hours=MAX_SENSOR_TIME_DIFFERENCE_H,
        max_s2_days=MAX_S2_GAP_DAYS,
    ).reset_index(drop=True)
    pairs["area_id"] = area_id
    pairs_by_area[area_id] = pairs
    area_resources[area_id] = {"dataset": unified_path, "s2_cache": s2_path}
    pretest_dates = pairs.loc[pairs["date"].dt.year.lt(TEST_YEAR), "date"].nunique()
    test_dates = pairs.loc[pairs["date"].dt.year.eq(TEST_YEAR), "date"].nunique()
    coverage_rows.append({
        "area_id": area_id,
        "filas_pares": len(pairs),
        "fechas_pares": pairs["date"].nunique(),
        "fechas_pretest": pretest_dates,
        "fechas_test_2025": test_dates,
        "modelo_local_evaluable": pretest_dates >= MIN_LOCAL_PRETEST_DATES,
        "cache_s2": s2_path.name,
    })
    del unified, s2
    gc.collect()

coverage = pd.DataFrame(coverage_rows)
coverage.to_csv(MODEL_DIR / "cobertura_por_area.csv", index=False)
display(coverage)
"""
    ),
    md(
        """
## 2. Comparación local–general sin utilizar 2025

Las fechas completas forman los pliegues: una fecha nunca aparece simultáneamente en entrenamiento y validación. Las dos áreas con solo tres fechas anteriores a 2025 quedan asignadas al modelo generalista, porque un Random Forest local no sería defendible con ese tamaño.
"""
    ),
    code(
        """
pretest_by_area = {
    area_id: frame.loc[frame["date"].dt.year.lt(TEST_YEAR)].copy().reset_index(drop=True)
    for area_id, frame in pairs_by_area.items()
}
pooled_pretest = pd.concat(pretest_by_area.values(), ignore_index=True)

started = time.perf_counter()
general_oof = grouped_date_oof(pooled_pretest, MODEL_NAME, random_state=RANDOM_STATE)
selection_rows = []
local_oof_by_area = {}
for area_id, local_frame in pretest_by_area.items():
    area_positions = pooled_pretest.index[pooled_pretest["area_id"].eq(area_id)]
    general_mae = macro_date_mae(pooled_pretest.loc[area_positions], general_oof.loc[area_positions])
    local_mae = None
    if local_frame["date"].nunique() >= MIN_LOCAL_PRETEST_DATES:
        local_oof = grouped_date_oof(local_frame, MODEL_NAME, random_state=RANDOM_STATE)
        local_oof_by_area[area_id] = local_oof
        local_mae = macro_date_mae(local_frame, local_oof)
    scope = choose_model_scope(
        general_mae, local_mae, minimum_local_improvement=MIN_LOCAL_IMPROVEMENT,
    )
    selection_rows.append({
        "area_id": area_id,
        "fechas_pretest": local_frame["date"].nunique(),
        "MAE_CV_general_K": general_mae,
        "MAE_CV_local_K": local_mae,
        "mejora_local_%": None if local_mae is None else 100 * (general_mae - local_mae) / general_mae,
        "modelo_seleccionado": scope,
    })

selection = pd.DataFrame(selection_rows).sort_values("area_id")
selection.to_csv(MODEL_DIR / "seleccion_local_vs_general.csv", index=False)
print(f"Validación terminada en {time.perf_counter() - started:.1f} s")
display(selection.round(4))
"""
    ),
    md("## 3. Evaluación temporal final sobre 2025"),
    code(
        """
general_pretest_bundle = fit_translation_bundle(pooled_pretest, MODEL_NAME, random_state=RANDOM_STATE)
test_metric_rows = []

for area_id, pairs in pairs_by_area.items():
    test = pairs.loc[pairs["date"].dt.year.eq(TEST_YEAR)].copy().reset_index(drop=True)
    if test.empty:
        continue
    general_prediction = predict_translation(general_pretest_bundle, test)
    local_prediction = None
    if pretest_by_area[area_id]["date"].nunique() >= MIN_LOCAL_PRETEST_DATES:
        local_bundle = fit_translation_bundle(pretest_by_area[area_id], MODEL_NAME, random_state=RANDOM_STATE)
        local_prediction = predict_translation(local_bundle, test)
    selected_scope = selection.set_index("area_id").at[area_id, "modelo_seleccionado"]
    selected_prediction = local_prediction if selected_scope == "local" else general_prediction
    candidates = [
        ("baseline_sentinel3", test["LST_1km"].to_numpy(float)),
        ("general", general_prediction),
    ]
    if local_prediction is not None:
        candidates.append(("local", local_prediction))
    for scope, prediction in candidates:
        row = metric_row(scope, test, prediction)
        row.update({"area_id": area_id, "seleccionado": scope == selected_scope})
        test_metric_rows.append(row)

test_metrics = pd.DataFrame(test_metric_rows)
test_metrics.to_csv(MODEL_DIR / "metricas_test_2025_por_area.csv", index=False)
display(test_metrics.round(4))
"""
    ),
    md(
        """
## 4. Modelos de producción y CSV final por área

Después de cerrar la evaluación, los modelos se reentrenan con todos los pares, incluido 2025. Cada CSV conserva todas las filas originales y añade:

- `LST_model_K`: traducción Sentinel-3 → Landsat cuando es posible;
- `LST_target_K`: Landsat observado con prioridad y, si falta, LST sintética;
- `target_source`, `target_is_observed` y `target_is_synthetic`;
- `target_physical_range_ok` y `target_training_eligible` para excluir extremos sin borrar su trazabilidad;
- covariables Sentinel-2 finas y trazabilidad del modelo.

Las filas sin Landsat ni Sentinel-3 válido se conservan, pero su objetivo queda vacío para que el siguiente notebook pueda usarlas solo como contexto o descartarlas explícitamente.
"""
    ),
    code(
        """
all_pairs = pd.concat(pairs_by_area.values(), ignore_index=True)
general_production_bundle = fit_translation_bundle(all_pairs, MODEL_NAME, random_state=RANDOM_STATE)
joblib.dump(general_production_bundle, MODEL_DIR / "modelo_general.joblib", compress=3)

selection_lookup = selection.set_index("area_id")["modelo_seleccionado"].to_dict()
selected_test_mae = (
    test_metrics.loc[test_metrics["seleccionado"]]
    .set_index("area_id")["MAE_K"].to_dict()
)
manifest_areas = []

for area_id, resources in area_resources.items():
    scope = selection_lookup[area_id]
    if scope == "local":
        production_bundle = fit_translation_bundle(pairs_by_area[area_id], MODEL_NAME, random_state=RANDOM_STATE)
        joblib.dump(production_bundle, MODEL_DIR / f"modelo_{area_id}.joblib", compress=3)
    else:
        production_bundle = general_production_bundle

    unified = pd.read_csv(resources["dataset"], low_memory=False)
    s2 = load_s2_cache(resources["s2_cache"])
    prediction_rows = prepare_prediction_rows(unified, s2, max_s2_days=MAX_S2_GAP_DAYS)
    prediction_rows["area_id"] = area_id
    translated = predict_translation(production_bundle, prediction_rows)
    final = build_gnn_ready_dataset(
        unified, prediction_rows, translated,
        area_id=area_id,
        model_name=MODEL_NAME,
        model_scope=scope,
        test_mae_k=selected_test_mae.get(area_id),
        plausible_target_range_k=PLAUSIBLE_TARGET_RANGE_K,
    )
    outputs = export_area_outputs(final, OUTPUT_DIR / "areas" / area_id)
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    summary.update({
        "area_id": area_id,
        "dataset": str(outputs["dataset"]),
        "edges": str(outputs["edges"]),
        "s2_cache": str(resources["s2_cache"]),
        "test_mae_k": selected_test_mae.get(area_id),
    })
    manifest_areas.append(summary)
    print(f"✓ {area_id}: {summary['synthetic_targets']:,} objetivos sintéticos · modelo {scope}")
    del unified, s2, prediction_rows, final
    gc.collect()

manifest = {
    "source_run": str(RUN_DIR),
    "framework": MODEL_NAME,
    "selection_rule": f"local si mejora MAE CV >= {100 * MIN_LOCAL_IMPROVEMENT:.1f}% y hay >= {MIN_LOCAL_PRETEST_DATES} fechas pretest",
    "test_year": TEST_YEAR,
    "plausible_target_range_k": PLAUSIBLE_TARGET_RANGE_K,
    "areas": manifest_areas,
}
(OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

final_summary = pd.DataFrame(manifest_areas)
final_summary.to_csv(OUTPUT_DIR / "resumen_salidas.csv", index=False)
display(final_summary)
print(f"Manifest: {OUTPUT_DIR / 'manifest.json'}")
"""
    ),
    md(
        """
## Uso posterior

Cada `areas/<area_id>/dataset_gnn_ready.csv` puede alimentar un experimento GNN independiente. El siguiente notebook debe separar temporalmente por fechas, ajustar normalización únicamente con train y distinguir objetivos observados de sintéticos. `edges.csv` contiene la conectividad espacial local de cada malla.
"""
    ),
]

nbf.write(notebook, OUTPUT)
print(OUTPUT)
