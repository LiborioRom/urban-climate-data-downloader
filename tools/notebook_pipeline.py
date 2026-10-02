from __future__ import annotations

import json
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from schemas.validation import AcquisitionPlan
from rules.variable_catalog import variable_label
from tools.credentials import credential_status
from tools.date_windows import describe_date_selection, effective_date_windows
from tools.unified_dataset import (
    archive_intermediate_csvs,
    build_quality_report,
    build_unified_dataset,
    count_source_errors,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_NOTEBOOK = PROJECT_ROOT / "01_descarga_datos_local.ipynb"
ProgressCallback = Callable[..., None]

def _parameter_source(plan: AcquisitionPlan, output_dir: Path, cache_dir: Path | None = None) -> str:
    request = plan.request
    assert request.roi and request.start_date and request.end_date and request.target_resolution_m
    roi = request.roi
    date_windows = effective_date_windows(request)
    if not date_windows:
        raise ValueError("No hay ventanas temporales válidas para ejecutar.")
    use_original_nodes = (
        abs(request.target_resolution_m - 100) < 1e-9
        and roi.place_name == "Sevilla"
        and roi.center_lat == 37.4035
        and roi.center_lon == -5.9810
        and roi.half_side_m == 750
        and roi.rotation_deg == 0
    )
    assignments = {
        "START_DATE": str(request.start_date),
        "END_DATE": str(request.end_date),
        "DATE_WINDOWS": [(str(window.start_date), str(window.end_date)) for window in date_windows],
        "ALLOWED_MONTHS": list(range(1, 13)),
        "CENTER_LAT": roi.center_lat,
        "CENTER_LON": roi.center_lon,
        "HALF_SIDE_M": roi.half_side_m,
        "ROI_ROTATION_DEG": roi.rotation_deg,
        "NODE_SPACING_M": request.target_resolution_m,
        "MAX_LANDSAT_CLOUD": request.parameters.max_landsat_cloud,
        "MAX_SENTINEL2_CLOUD": request.parameters.max_sentinel2_cloud,
        "S2_MAX_DAY_DIFFERENCE": request.parameters.s2_max_day_difference,
        "S3_REFERENCE_TOLERANCE_DAYS": request.parameters.s3_max_day_difference,
        "INCLUDE_URBAN_MORPHOLOGY": request.parameters.include_urban_morphology,
        "OVERWRITE_OUTPUTS": request.parameters.overwrite_outputs,
        "REQUESTED_VARIABLES": request.variables,
    }
    lines = ["# Parámetros inyectados por el wrapper determinista (valores Pydantic validados)."]
    lines.extend(f"{key} = {value!r}" for key, value in assignments.items())
    lines.extend([
        f"OUTPUT_DIR = Path({str(output_dir)!r})",
        f"CACHE_DIR = Path({str(cache_dir or PROJECT_ROOT / 'data_downloads' / 'cache')!r})",
        "LANDSAT_OUTPUT = OUTPUT_DIR / 'dataset_landsat_sin_downscaling.csv'",
        "SENTINEL3_OUTPUT = OUTPUT_DIR / 'dataset_sentinel3_1km_sin_downscaling.csv'",
        "PAIRED_OUTPUT = OUTPUT_DIR / 'dataset_pares_landsat_sentinel3.csv'",
        "NODES_OUTPUT = OUTPUT_DIR / 'nodos_malla.csv'",
        "NODES_CSV = Path.cwd() / 'nodos_originales_100m.csv'" if use_original_nodes else "NODES_CSV = None",
    ])
    return "\n".join(lines)


def prepare_notebook(plan: AcquisitionPlan, run_dir: Path, cache_dir: Path | None = None) -> Path:
    if not plan.executable:
        raise ValueError("El plan contiene errores y no puede ejecutarse.")
    import nbformat
    from nbformat.warnings import MissingIDFieldWarning

    output_dir = run_dir / "data_downloads"
    output_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingIDFieldWarning)
        notebook = nbformat.read(CANONICAL_NOTEBOOK, as_version=4)
    injection = nbformat.v4.new_code_cell(_parameter_source(plan, output_dir, cache_dir))
    injection.metadata["tags"] = ["injected-parameters"]
    notebook.cells.insert(5, injection)
    target = run_dir / "pipeline_parametrizado.ipynb"
    nbformat.write(notebook, target)
    return target


def _write_unified_outputs(plan: AcquisitionPlan, output_dir: Path) -> tuple[Path, Path, dict]:
    landsat = pd.read_csv(output_dir / "dataset_landsat_sin_downscaling.csv")
    sentinel3 = pd.read_csv(output_dir / "dataset_sentinel3_1km_sin_downscaling.csv")
    nodes = pd.read_csv(output_dir / "nodos_malla.csv")
    source_errors = count_source_errors(output_dir)
    source_quality_path = output_dir / "calidad_fuentes.json"
    source_quality = (
        json.loads(source_quality_path.read_text(encoding="utf-8"))
        if source_quality_path.exists() else {}
    )
    unified = build_unified_dataset(landsat, sentinel3, nodes, plan.request)
    unified_path = output_dir / "dataset_unificado.csv"
    unified.to_csv(unified_path, index=False)
    quality = build_quality_report(unified, plan.request, source_errors, source_quality)
    quality_path = output_dir / "informe_calidad.json"
    quality_path.write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")
    return unified_path, quality_path, quality


def _write_metadata(
    plan: AcquisitionPlan,
    output_dir: Path,
    unified_csv: Path,
    quality_report: Path,
    quality: dict,
    package_zip: Path | None,
    plan_report: Path | None = None,
    paired_csv: Path | None = None,
) -> Path:
    configured_credentials = credential_status()
    used_credentials = [
        name for name in ("EE_PROJECT", "SH_CLIENT_ID", "SH_CLIENT_SECRET")
        if configured_credentials.get(name)
    ]
    era5_source = quality.get("source_quality", {}).get("era5_land", {}).get("source")
    if era5_source == "cds" and configured_credentials.get("CDS_API_KEY"):
        used_credentials.append("CDS_API_KEY")
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": plan.request.model_dump(mode="json"),
        "sources": [item.model_dump(mode="json") for item in plan.variable_plans],
        "harmonization_steps": plan.harmonization_steps,
        "execution_profile": plan.execution_profile,
        "outputs": {
            "unified_csv": unified_csv.name,
            "quality_report": quality_report.name,
            "package_zip": package_zip.name if package_zip else None,
            "plan_report": plan_report.name if plan_report else None,
            "paired_csv": paired_csv.name if paired_csv and paired_csv.exists() else None,
        },
        "quality_summary": quality,
        "internal_intermediate_directory": "cache/intermediate",
        "credential_names_used": used_credentials,
    }
    target = output_dir / "metadata.json"
    target.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def _write_package(package_path: Path, files: list[Path | None]) -> Path:
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path and path.exists():
                archive.write(path, arcname=path.name)
    return package_path


def _write_plan_report(plan: AcquisitionPlan, output_dir: Path) -> Path:
    request = plan.request
    assert request.roi and request.start_date and request.end_date and request.target_resolution_m
    roi = request.roi
    lines = [
        "INFORME DEL PROCESO DE ADQUISICIÓN",
        "",
        f"Propósito del dataset: {request.dataset_purpose.value}.",
        f"Periodo procesado: {describe_date_selection(request)}.",
        f"Resolución de la malla de salida: {request.target_resolution_m:g} m.",
        (
            f"Área: cuadrado centrado en {roi.center_lat:.6f}, {roi.center_lon:.6f}, "
            f"con {2 * roi.half_side_m:.1f} m de lado y giro de {roi.rotation_deg:.1f}°."
        ),
        "",
        "VARIABLES Y PROCEDENCIA",
    ]
    for item in plan.variable_plans:
        source = item.dataset_or_product or item.source
        native = f"; resolución nativa aproximada: {item.native_resolution_m:g} m" if item.native_resolution_m else ""
        transformations = "; ".join(item.transformations) or "sin transformación adicional declarada"
        lines.append(
            f"- {variable_label(item.variable)}: fuente {item.source} ({source}){native}. "
            f"Tratamiento: {transformations}."
        )
    lines.extend(["", "OPERACIONES REALIZADAS"])
    for step in plan.steps:
        variables = ", ".join(variable_label(name) for name in step.variables)
        suffix = f" Variables: {variables}." if variables else ""
        lines.append(f"{step.order}. {step.action}.{suffix}")
    lines.extend(["", "CRITERIOS DE ARMONIZACIÓN"])
    lines.extend(f"- {item}." for item in plan.harmonization_steps)
    relevant_issues = [issue for issue in plan.issues if issue.severity.value != "ERROR"]
    if relevant_issues:
        lines.extend(["", "AVISOS CIENTÍFICOS"])
        lines.extend(f"- {issue.message}" for issue in relevant_issues)
    lines.extend([
        "",
        "SALIDAS",
        "- dataset_unificado.csv contiene una fila por nodo fino y fecha de referencia.",
        "- dataset_pares_landsat_sentinel3.csv contiene coincidencias diarias y horarias para entrenar el downscaling.",
        "- informe_calidad.json resume cobertura, valores vacíos y elementos fallidos.",
        "- Las salidas Landsat y Sentinel-3 se conservan como intermedios internos.",
        "- metadata.json conserva los parámetros, fuentes y trazabilidad de la ejecución.",
    ])
    target = output_dir / "informe_plan.txt"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _progress_stage(plan: AcquisitionPlan, source: str) -> tuple[int, str] | None:
    def selected(*source_ids: str) -> str:
        names = [
            variable_label(item.variable)
            for item in plan.variable_plans
            if item.source in source_ids
        ]
        return ", ".join(dict.fromkeys(names))

    stages = (
        ("to_projected = Transformer", 10, "Área y malla espacial preparadas."),
        ("def init_earth_engine", 18, "Servicios de datos autenticados."),
        ("static_df =", 30, f"Variables estáticas y urbanas completadas: {selected('srtm', 'cnig_wcs', 'openstreetmap', 'osm_cnig_derived') or 'ninguna solicitada'}."),
        ("landsat_df = download_landsat_rows()", 46, f"Lote tabular Landsat completado: {selected('landsat_8_9') or 'datos auxiliares'}."),
        ("s3_products, s3_daily_count = search_s3_products", 55, "Sentinel-3 diario filtrado; iniciando lotes temporales de 1 km."),
        ("s3_pixels = download_all_s3_pixels", 66, f"Lotes diarios Sentinel-3 completados: {selected('sentinel_3_slstr_l2') or 'datos auxiliares'}."),
        ("s2_nodes = download_sentinel2_batch", 78, f"Lote tabular Sentinel-2 completado: {selected('sentinel_2_l2a') or 'datos auxiliares'}."),
        ("era5_hourly, era5_daily, era5_info = load_shared_era5", 94, f"ERA5-Land compartido y ambos CSV completados: {selected('era5_land') or 'datos auxiliares'}."),
        ("def validate_output", 98, "Salidas científicas verificadas."),
    )
    return next(((percent, message) for marker, percent, message in stages if marker in source), None)


def execute_plan(
    plan: AcquisitionPlan,
    run_dir: Path,
    *,
    include_plan_report: bool = False,
    progress_callback: ProgressCallback | None = None,
    cache_dir: Path | None = None,
    create_package: bool = True,
) -> dict[str, Path]:
    """Ejecuta únicamente tras confirmación de la UI; nunca recibe texto o código del LLM."""
    notebook_path = prepare_notebook(plan, run_dir, cache_dir)
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(notebook_path, as_version=4)
    def on_cell_executed(*, cell, cell_index, execute_reply) -> None:
        succeeded = execute_reply.get("content", {}).get("status") == "ok"
        if succeeded and progress_callback and (stage := _progress_stage(plan, cell.source)):
            progress_callback(*stage)

    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        on_cell_executed=on_cell_executed,
    )
    if progress_callback:
        progress_callback(3, "Notebook parametrizado y listo para ejecutarse.")
    client.execute()
    nbformat.write(notebook, notebook_path)
    output_dir = run_dir / "data_downloads"
    if progress_callback:
        progress_callback(99, "Fuentes armonizadas en el CSV único y calidad evaluada.")
    unified_csv, quality_report, quality = _write_unified_outputs(plan, output_dir)
    plan_report = _write_plan_report(plan, output_dir) if include_plan_report else None
    package_zip = output_dir / "descarga_datos.zip" if create_package else None
    paired_csv = output_dir / "dataset_pares_landsat_sentinel3.csv"
    metadata = _write_metadata(
        plan, output_dir, unified_csv, quality_report, quality, package_zip, plan_report, paired_csv,
    )
    if package_zip:
        _write_package(package_zip, [unified_csv, paired_csv, metadata, quality_report, plan_report])
    archive_intermediate_csvs(output_dir)
    if progress_callback:
        progress_callback(100, "Dataset unificado, metadatos e informe de calidad preparados.")
    outputs = {
        "notebook": notebook_path,
        "unified_csv": unified_csv,
        "quality_report": quality_report,
        "metadata": metadata,
    }
    if package_zip:
        outputs["package_zip"] = package_zip
    if plan_report:
        outputs["plan_report"] = plan_report
    if paired_csv.exists():
        outputs["paired_csv"] = paired_csv
    return outputs
