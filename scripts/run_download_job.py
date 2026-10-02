from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.resilient_json import write_json_resilient

def write_status(path: Path, payload: dict) -> None:
    write_json_resilient(path, payload)


def try_write_status(path: Path, payload: dict) -> bool:
    """El estado visual nunca debe detener una descarga científica válida."""
    try:
        write_status(path, payload)
        return True
    except OSError as exc:
        print(f"AVISO: no se pudo actualizar el estado de descarga: {exc}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--include-plan-report", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    events: list[dict[str, object]] = []

    latest_details: dict[str, object] = {}

    def update(percent: int, message: str, details: dict[str, object] | None = None) -> None:
        if details:
            latest_details.update(details)
        if not events or events[-1]["percent"] != percent:
            events.append({"percent": percent, "message": message})
        try_write_status(args.status, {
            "state": "running",
            "percent": percent,
            "message": message,
            "events": events,
            "started_at": started_at,
            "elapsed_seconds": time.perf_counter() - started,
            **latest_details,
        })

    try:
        from agent.planner import build_plan
        from schemas.request import DatasetRequest
        from tools.notebook_pipeline import execute_plan
        from tools.multi_area_pipeline import execute_multi_area_plan
        from tools.storage import shared_cache_directory

        request = DatasetRequest.model_validate_json(args.request.read_text(encoding="utf-8"))
        plan = build_plan(request)
        if not plan.executable:
            raise ValueError("La solicitud dejó de ser ejecutable antes de iniciar el notebook.")
        update(1, "Solicitud validada; preparando el notebook.")
        cache_dir = args.cache_dir or shared_cache_directory()
        if request.areas:
            outputs = execute_multi_area_plan(
                plan,
                args.run_dir,
                cache_dir=cache_dir,
                include_plan_report=args.include_plan_report,
                progress_callback=update,
            )
        else:
            outputs = execute_plan(
                plan,
                args.run_dir,
                include_plan_report=args.include_plan_report,
                progress_callback=update,
                cache_dir=cache_dir,
            )
    except Exception as exc:
        traceback.print_exc()
        try_write_status(args.status, {
            "state": "error",
            "percent": events[-1]["percent"] if events else 0,
            "message": "La descarga se ha detenido.",
            "events": events,
            "started_at": started_at,
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc)[-12000:],
            **latest_details,
        })
        return 1

    final_state = "complete"
    try:
        final_quality = json.loads(Path(outputs["quality_report"]).read_text(encoding="utf-8"))
        if final_quality.get("status") == "partial":
            final_state = "partial"
    except (KeyError, OSError, json.JSONDecodeError, TypeError):
        pass
    try_write_status(args.status, {
        "state": final_state,
        "percent": 100,
        "message": "Dataset generado con incidencias parciales." if final_state == "partial" else "Dataset generado correctamente.",
        "events": events,
        "started_at": started_at,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {key: str(value) for key, value in outputs.items()},
        **latest_details,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
