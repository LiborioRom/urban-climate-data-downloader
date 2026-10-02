from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from schemas.validation import AcquisitionPlan
from tools.resilient_json import write_json_resilient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "run_download_job.py"


def _write_json(path: Path, payload: dict) -> None:
    write_json_resilient(path, payload)


def load_download_status(status_path: str | Path) -> dict:
    path = Path(status_path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {
            "state": "starting",
            "percent": 0,
            "message": "Iniciando el proceso de descarga…",
            "events": [],
        }


def start_background_download(
    plan: AcquisitionPlan,
    run_dir: Path,
    *,
    include_plan_report: bool,
    cache_dir: Path | None = None,
) -> dict:
    """Inicia el notebook fuera del ciclo de Streamlit para resistir sus reruns."""
    run_dir.mkdir(parents=True, exist_ok=True)
    request_path = run_dir / "solicitud_descarga.json"
    status_path = run_dir / "estado_descarga.json"
    log_path = run_dir / "descarga.log"
    request_path.write_text(plan.request.model_dump_json(indent=2), encoding="utf-8")
    started_at = datetime.now(timezone.utc).isoformat()
    _write_json(status_path, {
        "state": "starting",
        "percent": 0,
        "message": "Iniciando el proceso de descarga…",
        "events": [],
        "started_at": started_at,
    })
    command = [
        sys.executable,
        str(WORKER_SCRIPT),
        "--request", str(request_path),
        "--run-dir", str(run_dir),
        "--status", str(status_path),
    ]
    if include_plan_report:
        command.append("--include-plan-report")
    if cache_dir:
        command.extend(["--cache-dir", str(cache_dir)])
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
    return {
        "pid": process.pid,
        "run_dir": str(run_dir),
        "status_path": str(status_path),
        "log_path": str(log_path),
        "started_at": started_at,
        "announced": False,
    }
