from agent.conversation import interpret_request
from agent.planner import build_plan


def codes(plan):
    return {issue.code for issue in plan.issues}


def test_case_1_lst_ndvi_sevilla_may_october_2024_100m():
    request = interpret_request(
        "Quiero LST y NDVI de Sevilla entre mayo y octubre de 2024 a 100 m",
    )
    plan = build_plan(request)
    assert request.variables == ["LST", "NDVI"]
    assert str(request.start_date) == "2024-05-01"
    assert str(request.end_date) == "2024-10-31"
    assert request.target_resolution_m == 100
    assert request.roi and request.roi.place_name == "Sevilla"
    assert plan.executable
    assert {item.source for item in plan.variable_plans} == {"landsat_8_9"}


def test_case_2_air_temperature_at_100m_warns_about_era5_support():
    request = interpret_request(
        "Temperatura del aire de Sevilla entre mayo y octubre de 2024 a 100 m",
    )
    plan = build_plan(request)
    assert request.variables == ["air_temperature"]
    assert "FINER_THAN_NATIVE" in codes(plan)
    assert "ERA5_REGIONAL_CONTEXT" in codes(plan)
    assert plan.executable


def test_case_3_buildings_and_roads_recognizes_sources_and_todo():
    request = interpret_request(
        "Edificios y carreteras de Sevilla entre mayo y octubre de 2024 a 100 m",
    )
    plan = build_plan(request)
    assert request.variables == ["buildings", "roads"]
    assert {item.source for item in plan.variable_plans} == {"openstreetmap", "cnig_wcs"}
    assert "VECTOR_RASTERIZATION" in codes(plan)
    assert "NOT_IMPLEMENTED" in codes(plan)
    assert not plan.executable


def test_case_4_incomplete_request_reports_missing_fields():
    request = interpret_request("Quiero datos de Sevilla")
    plan = build_plan(request)
    assert request.roi is not None
    assert {"MISSING_START_DATE", "MISSING_END_DATE", "MISSING_VARIABLES", "MISSING_RESOLUTION"} <= codes(plan)
    assert not plan.executable
