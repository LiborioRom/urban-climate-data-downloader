import json
import os

from agent.planner import build_plan
from schemas.request import DatasetRequest, ROIRequest, ROISpecification
from tools import background_download
from tools.resilient_json import write_json_resilient


def _valid_plan():
    return build_plan(DatasetRequest(
        roi=ROIRequest(
            specification=ROISpecification.CENTER_AND_HALF_SIDE,
            center_lat=37.4,
            center_lon=-5.98,
            half_side_m=750,
        ),
        start_date="2024-07-15",
        end_date="2024-07-15",
        variables=["LST"],
        target_resolution_m=100,
    ))


def test_background_job_persists_request_without_credentials_and_starts_worker(tmp_path, monkeypatch):
    captured = {}

    class Process:
        pid = 1234

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(background_download.subprocess, "Popen", fake_popen)
    job = background_download.start_background_download(
        _valid_plan(), tmp_path, include_plan_report=True,
    )

    request_text = (tmp_path / "solicitud_descarga.json").read_text(encoding="utf-8")
    status = json.loads((tmp_path / "estado_descarga.json").read_text(encoding="utf-8"))
    assert "SH_CLIENT_SECRET" not in request_text
    assert "--include-plan-report" in captured["command"]
    assert captured["kwargs"].get("shell") is not True
    assert status["state"] == "starting"
    assert job["pid"] == 1234


def test_status_reader_tolerates_an_incomplete_atomic_update(tmp_path):
    status_path = tmp_path / "estado_descarga.json"
    status_path.write_text("{", encoding="utf-8")

    status = background_download.load_download_status(status_path)

    assert status["state"] == "starting"


def test_status_writer_retries_a_temporary_windows_lock(tmp_path, monkeypatch):
    target = tmp_path / "estado_descarga.json"
    original_replace = os.replace
    attempts = 0

    def locked_twice(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise PermissionError(5, "Acceso denegado")
        return original_replace(source, destination)

    monkeypatch.setattr("tools.resilient_json.os.replace", locked_twice)
    write_json_resilient(target, {"state": "running", "percent": 98}, initial_delay_seconds=0)

    assert attempts == 3
    assert json.loads(target.read_text(encoding="utf-8"))["percent"] == 98
    assert not list(tmp_path.glob("*.tmp"))


def test_status_writer_uses_direct_fallback_when_replace_stays_locked(tmp_path, monkeypatch):
    target = tmp_path / "estado_descarga.json"
    target.write_text('{"state": "starting"}', encoding="utf-8")

    def always_locked(source, destination):
        raise PermissionError(5, "Acceso denegado")

    monkeypatch.setattr("tools.resilient_json.os.replace", always_locked)
    write_json_resilient(
        target,
        {"state": "complete", "percent": 100},
        retries=2,
        initial_delay_seconds=0,
    )

    assert json.loads(target.read_text(encoding="utf-8"))["state"] == "complete"
