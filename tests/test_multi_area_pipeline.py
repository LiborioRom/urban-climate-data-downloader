import json
import threading
import time
from pathlib import Path

import pandas as pd

from agent.planner import build_plan
from rules.validation_rules import validate_request
from schemas.request import DatasetRequest, PipelineParameters, ROIRequest, ROISpecification
from tools import multi_area_pipeline


def _area(area_id: str, latitude: float) -> ROIRequest:
    return ROIRequest(
        specification=ROISpecification.CENTER_AND_HALF_SIDE,
        area_id=area_id,
        area_name=area_id.replace("_", " ").title(),
        center_lat=latitude,
        center_lon=-5.98,
        half_side_m=500,
    )


def _request(areas: list[ROIRequest]) -> DatasetRequest:
    return DatasetRequest(
        areas=areas,
        start_date="2024-07-15",
        end_date="2024-07-15",
        variables=["LST"],
        target_resolution_m=100,
        parameters=PipelineParameters(max_parallel_areas=2),
    )


def test_multi_area_request_uses_first_roi_for_planning_and_rejects_duplicate_ids():
    valid = validate_request(_request([_area("norte", 37.42), _area("sur", 37.36)]))
    assert valid.request.roi.area_id == "norte"
    assert not any(issue.code == "MISSING_ROI" for issue in valid.issues)

    duplicated = validate_request(_request([_area("zona", 37.42), _area("zona", 37.36)]))
    assert any(issue.code == "DUPLICATE_AREA_ID" for issue in duplicated.issues)


def test_multi_area_executor_runs_bounded_parallel_and_consolidates(tmp_path, monkeypatch):
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def fake_execute(plan, run_dir, **_kwargs):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.05)
        output_dir = Path(run_dir) / "data_downloads"
        output_dir.mkdir(parents=True, exist_ok=True)
        area_id = plan.request.roi.area_id
        unified = output_dir / "dataset_unificado.csv"
        pd.DataFrame({
            "area_id": [area_id], "node_id": ["n0"], "date": ["2024-07-15"], "LST_K": [310.0],
        }).to_csv(unified, index=False)
        quality = output_dir / "informe_calidad.json"
        quality.write_text(json.dumps({"status": "complete", "summary": "Correcto."}), encoding="utf-8")
        with lock:
            active -= 1
        return {"unified_csv": unified, "quality_report": quality}

    monkeypatch.setattr(multi_area_pipeline, "execute_plan", fake_execute)
    request = _request([_area("norte", 37.42), _area("sur", 37.36), _area("este", 37.39)])
    outputs = multi_area_pipeline.execute_multi_area_plan(
        build_plan(request), tmp_path / "run", cache_dir=tmp_path / "cache",
    )

    combined = pd.read_csv(outputs["unified_csv"])
    quality = json.loads(outputs["quality_report"].read_text(encoding="utf-8"))
    assert maximum_active == 2
    assert set(combined["area_id"]) == {"norte", "sur", "este"}
    assert set(combined["global_node_id"]) == {"norte__n0", "sur__n0", "este__n0"}
    assert quality["areas_completed"] == 3
    assert quality["status"] == "complete"
