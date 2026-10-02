from agent.planner import build_plan
from schemas.request import DatasetRequest, ROIRequest, ROISpecification
from tools.notebook_pipeline import _progress_stage, _write_plan_report


def _plan():
    request = DatasetRequest(
        roi=ROIRequest(
            specification=ROISpecification.CENTER_AND_HALF_SIDE,
            center_lat=37.4,
            center_lon=-5.98,
            half_side_m=750,
        ),
        start_date="2024-07-15",
        end_date="2024-07-20",
        variables=["LST", "air_temperature"],
        target_resolution_m=100,
    )
    return build_plan(request)


def test_optional_plan_report_is_readable_and_records_provenance(tmp_path):
    target = _write_plan_report(_plan(), tmp_path)
    text = target.read_text(encoding="utf-8")

    assert "Periodo procesado: 2024-07-15 → 2024-07-20" in text
    assert "VARIABLES Y PROCEDENCIA" in text
    assert "Landsat" in text
    assert "ERA5" in text
    assert "SALIDAS" in text


def test_progress_stages_report_completed_source_groups():
    plan = _plan()
    percent, message = _progress_stage(plan, "landsat_df = download_landsat_rows()")

    assert percent == 46
    assert "Landsat" in message
    assert "Temperatura superficial" in message
