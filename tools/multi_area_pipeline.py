from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from agent.planner import build_plan
from schemas.request import DatasetRequest, ROIRequest
from schemas.validation import AcquisitionPlan
from tools.notebook_pipeline import execute_plan
from tools.resilient_json import write_json_resilient


ProgressCallback = Callable[..., None]


def _area_identity(area: ROIRequest, index: int) -> tuple[str, str]:
    area_id = area.area_id or f"area_{index:02d}"
    area_name = area.area_name or area.place_name or area_id
    return area_id, area_name


def _child_request(request: DatasetRequest, area: ROIRequest) -> DatasetRequest:
    return request.model_copy(deep=True, update={"roi": area.model_copy(deep=True), "areas": []})


def _append_csv(source: Path, target: Path, area_id: str, area_name: str) -> dict[str, object]:
    rows = 0
    nodes: set[str] = set()
    dates: set[str] = set()
    first = not target.exists()
    for chunk in pd.read_csv(source, chunksize=100_000):
        chunk["area_id"] = area_id
        chunk.insert(1, "area_name", area_name)
        node_column = "node_id" if "node_id" in chunk else "nodo_id" if "nodo_id" in chunk else None
        if node_column:
            chunk.insert(2, "global_node_id", area_id + "__" + chunk[node_column].astype(str))
            nodes.update(chunk[node_column].astype(str).unique())
        if "date" in chunk:
            dates.update(chunk["date"].astype(str).unique())
        chunk.to_csv(target, mode="w" if first else "a", header=first, index=False)
        first = False
        rows += len(chunk)
    return {"rows": rows, "nodes": len(nodes), "dates": sorted(dates)}


def execute_multi_area_plan(
    plan: AcquisitionPlan,
    run_dir: Path,
    *,
    cache_dir: Path,
    include_plan_report: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Path]:
    """Ejecuta ROIs aislados en paralelo y genera una salida multizona única."""
    areas = plan.request.areas or ([plan.request.roi] if plan.request.roi else [])
    if not areas:
        raise ValueError("La solicitud multizona no contiene áreas.")

    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    status_lock = threading.Lock()
    area_states: dict[str, dict[str, object]] = {}
    area_outputs: dict[str, dict[str, Path]] = {}
    for index, area in enumerate(areas, start=1):
        area_id, area_name = _area_identity(area, index)
        area_states[area_id] = {"area_id": area_id, "area_name": area_name, "state": "queued", "percent": 0, "message": "En cola."}

    def publish(message: str) -> None:
        completed_fraction = sum(float(item["percent"]) for item in area_states.values()) / (100 * len(area_states))
        overall = min(98, max(2, round(2 + 96 * completed_fraction)))
        details = {"areas": list(area_states.values())}
        write_json_resilient(run_dir / "estado_areas.json", details)
        if progress_callback:
            progress_callback(overall, message, details)

    def run_area(index: int, area: ROIRequest) -> tuple[str, dict[str, Path]]:
        area_id, area_name = _area_identity(area, index)
        child_dir = run_dir / "areas" / area_id
        child_plan = build_plan(_child_request(plan.request, area))
        if not child_plan.executable:
            raise ValueError(f"El ROI '{area_name}' no produjo un plan ejecutable.")

        def child_progress(percent: int, message: str) -> None:
            with status_lock:
                area_states[area_id].update(state="running", percent=percent, message=message)
                publish(f"Procesando {area_name}: {message}")

        with status_lock:
            area_states[area_id].update(state="running", percent=1, message="Preparando área.")
            publish(f"Iniciando {area_name}.")
        outputs = execute_plan(
            child_plan,
            child_dir,
            include_plan_report=include_plan_report,
            progress_callback=child_progress,
            cache_dir=cache_dir,
            create_package=False,
        )
        with status_lock:
            area_states[area_id].update(state="complete", percent=100, message="Área completada.")
            publish(f"Área completada: {area_name}.")
        return area_id, outputs

    failures: list[dict[str, str]] = []
    max_workers = min(plan.request.parameters.max_parallel_areas, len(areas))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="roi") as pool:
        futures = {pool.submit(run_area, index, area): (index, area) for index, area in enumerate(areas, start=1)}
        for future in as_completed(futures):
            index, area = futures[future]
            area_id, area_name = _area_identity(area, index)
            try:
                completed_id, outputs = future.result()
                area_outputs[completed_id] = outputs
            except Exception as exc:
                failures.append({"area_id": area_id, "area_name": area_name, "error": str(exc)})
                with status_lock:
                    area_states[area_id].update(state="error", message=str(exc)[-500:])
                    publish(f"No se pudo completar {area_name}; continúan las demás zonas.")

    if not area_outputs:
        raise RuntimeError("Ningún ROI pudo completarse. Revisa descarga.log y los estados por área.")

    combined_dir = run_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    unified_path = combined_dir / "dataset_multizona.csv"
    paired_path = combined_dir / "pares_landsat_sentinel3_multizona.csv"
    summaries: list[dict[str, object]] = []
    for index, area in enumerate(areas, start=1):
        area_id, area_name = _area_identity(area, index)
        outputs = area_outputs.get(area_id)
        if not outputs:
            continue
        item = {"area_id": area_id, "area_name": area_name}
        item.update(_append_csv(outputs["unified_csv"], unified_path, area_id, area_name))
        quality_path = outputs.get("quality_report")
        item["quality"] = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path else {}
        summaries.append(item)
        if outputs.get("paired_csv"):
            _append_csv(outputs["paired_csv"], paired_path, area_id, area_name)

    status = "partial" if failures else "complete"
    quality = {
        "status": status,
        "summary": (
            f"Se completaron {len(summaries)} de {len(areas)} áreas."
            if failures else f"Se completaron las {len(areas)} áreas seleccionadas."
        ),
        "areas_requested": len(areas),
        "areas_completed": len(summaries),
        "areas_failed": failures,
        "rows": sum(int(item["rows"]) for item in summaries),
        "per_area": summaries,
    }
    quality_path = combined_dir / "informe_calidad_multizona.json"
    quality_path.write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parallel_workers": max_workers,
        "shared_cache": str(cache_dir),
        "areas": list(area_states.values()),
        "request": plan.request.model_dump(mode="json"),
    }
    metadata_path = combined_dir / "metadata_multizona.json"
    metadata_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if progress_callback:
        progress_callback(100, quality["summary"], {"areas": list(area_states.values())})
    outputs = {
        "unified_csv": unified_path,
        "quality_report": quality_path,
        "metadata": metadata_path,
    }
    if paired_path.exists():
        outputs["paired_csv"] = paired_path
    return outputs
