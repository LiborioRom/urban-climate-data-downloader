import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from agent.dialogue_manager import _is_variable_question, process_message
from rules.variable_catalog import VARIABLE_CATALOG, describe_variable, mentioned_variables
from schemas.conversation import ConversationState, VariableCatalogueSelection, VariableKnowledgeAnswer
from tools.scientific_variables import (
    deaccumulate_era5_land,
    derive_era5,
    era5_request_variables,
)
from tools.urban_morphology import approximate_sky_view_factor


REQUIRED_CARD_FIELDS = {
    "display_name", "category", "unit", "short_description", "interpretation",
    "aliases", "sources", "variable_type", "temporal_kind", "output_columns",
    "limitations", "calculation_cost", "availability",
}


def test_every_variable_has_a_complete_technical_card():
    for name, specification in VARIABLE_CATALOG.items():
        assert REQUIRED_CARD_FIELDS <= specification.keys(), name
        assert specification["short_description"], name
        assert specification["interpretation"], name
        assert specification["sources"], name


def test_aliases_cover_new_satellite_era5_and_urban_variables():
    found = mentioned_variables(
        "Quiero NDMI, presión superficial, sky view factor y densidad de red viaria"
    )
    assert {"NDMI", "surface_pressure", "sky_view_factor", "road_length_density"} <= set(found)


def test_chat_passes_the_relevant_card_to_qwen(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "agent.dialogue_manager.ollama_status",
        lambda: {"running": True, "model_available": True},
    )

    def fake_chat(response_model, **kwargs):
        captured.update(kwargs)
        assert response_model is VariableKnowledgeAnswer
        return VariableKnowledgeAnswer(
            answerable_from_cards=True,
            answer="Qwen explica NDMI usando exclusivamente la ficha.",
            referenced_variables=["NDMI"],
        )

    monkeypatch.setattr("agent.dialogue_manager.structured_chat", fake_chat)
    result = process_message("¿Qué significa NDMI?", ConversationState(), [])
    assert result.provider == "ollama_variable_grounded"
    assert result.assistant_message == "Qwen explica NDMI usando exclusivamente la ficha."
    assert '"canonical_name": "NDMI"' in captured["system_prompt"]
    assert '"formula": "(NIR - SWIR1) / (NIR + SWIR1)"' in captured["system_prompt"]
    assert captured["history"] is None


def test_chat_returns_qwen_insufficiency_for_unknown_information(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "agent.dialogue_manager.ollama_status",
        lambda: {"running": True, "model_available": True},
    )

    def fake_chat(response_model, **kwargs):
        captured.update(kwargs)
        return VariableKnowledgeAnswer(
            answerable_from_cards=False,
            answer="Las fichas no contienen información suficiente para responder esa pregunta.",
            missing_information="No existe una ficha para la variable mencionada.",
        )

    monkeypatch.setattr("agent.dialogue_manager.structured_chat", fake_chat)
    result = process_message("¿Cómo se obtiene la temperatura radiante de Marte?", ConversationState(), [])
    assert result.provider == "ollama_variable_grounded"
    assert "no contienen información suficiente" in result.assistant_message
    assert '"cards": []' in captured["system_prompt"]


def test_variable_question_does_not_fall_back_to_unverified_science(monkeypatch):
    monkeypatch.setattr(
        "agent.dialogue_manager.ollama_status",
        lambda: {"running": False, "model_available": False},
    )
    result = process_message("¿Qué significa NDMI?", ConversationState(), [])
    assert result.provider == "ollama_variable_unavailable"
    assert "Qwen no está disponible" in result.assistant_message
    assert "respuesta no fundamentada" in result.assistant_message


def test_source_selection_command_is_not_misclassified_as_a_question():
    assert not _is_variable_question("Usa la fuente Landsat para NDVI")
    assert _is_variable_question("¿Cuál es la fuente de NDVI?")
    assert _is_variable_question("¿Qué relación demostrada tiene NDMI con la mortalidad humana?")


def test_qwen_answer_is_rejected_if_it_references_an_unretrieved_variable(monkeypatch):
    monkeypatch.setattr(
        "agent.dialogue_manager.ollama_status",
        lambda: {"running": True, "model_available": True},
    )
    monkeypatch.setattr(
        "agent.dialogue_manager.structured_chat",
        lambda *args, **kwargs: VariableKnowledgeAnswer(
            answerable_from_cards=True,
            answer="Respuesta inventada.",
            referenced_variables=["surface_pressure"],
        ),
    )
    result = process_message("¿Qué significa NDMI?", ConversationState(), [])
    assert result.provider == "ollama_variable_ungrounded"
    assert "he descartado" in result.assistant_message


def test_thematic_catalogue_question_uses_qwen_selection_then_full_cards(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "agent.dialogue_manager.ollama_status",
        lambda: {"running": True, "model_available": True},
    )

    def fake_chat(response_model, **kwargs):
        calls.append((response_model, kwargs))
        if response_model is VariableCatalogueSelection:
            return VariableCatalogueSelection(
                selected_variables=["LST", "air_temperature", "skin_temperature"]
            )
        return VariableKnowledgeAnswer(
            answerable_from_cards=True,
            answer="Las principales variables térmicas son LST, temperatura del aire y temperatura de piel.",
            referenced_variables=["LST", "air_temperature", "skin_temperature"],
        )

    monkeypatch.setattr("agent.dialogue_manager.structured_chat", fake_chat)
    result = process_message(
        "¿Qué variables hay relacionadas con temperatura?", ConversationState(), []
    )
    assert result.provider == "ollama_variable_grounded"
    assert [call[0] for call in calls] == [VariableCatalogueSelection, VariableKnowledgeAnswer]
    assert '"canonical_name":"LST"' in calls[0][1]["system_prompt"]
    assert '"formula"' in calls[1][1]["system_prompt"]


def test_era5_dependencies_and_physical_conversions():
    requested = era5_request_variables([
        "surface_pressure", "soil_moisture_level_2", "rain_3d_log", "is_rainy",
    ])
    assert {"surface_pressure", "volumetric_soil_water_layer_2", "total_precipitation"} <= set(requested)
    hourly, daily = derive_era5(pd.DataFrame({
        "era5_dt_utc": pd.to_datetime(["2024-07-01T10:00Z"]),
        "t2m": [303.15], "d2m": [293.15], "u10": [3.0], "v10": [4.0],
        "ssrd": [3_600_000.0], "tp": [0.002], "sp": [101_325.0], "swvl2": [0.25],
    }))
    assert np.isclose(hourly.loc[0, "Tair_C"], 30.0)
    assert np.isclose(hourly.loc[0, "wind_speed"], 5.0)
    assert np.isclose(hourly.loc[0, "Rsol_Wm2"], 1000.0)
    assert np.isclose(hourly.loc[0, "surface_pressure_hPa"], 1013.25)
    assert np.isclose(hourly.loc[0, "soil_moisture_l2"], 0.25)
    assert np.isclose(daily.loc[0, "rain_1d"], 2.0)
    assert np.isclose(daily.loc[0, "rain_3d_log"], np.log1p(2.0))
    assert daily.loc[0, "is_rainy"] == 1


def test_era5_land_accumulations_are_converted_to_hourly_increments():
    frame = pd.DataFrame({
        "era5_dt_utc": pd.to_datetime([
            "2024-07-01T22:00Z", "2024-07-01T23:00Z", "2024-07-02T00:00Z",
            "2024-07-02T01:00Z", "2024-07-02T02:00Z",
        ]),
        "ssrd": [10.0, 15.0, 20.0, 2.0, 5.0],
        "tp": [0.010, 0.015, 0.020, 0.002, 0.005],
    })

    hourly = deaccumulate_era5_land(frame, ["ssrd", "tp"])

    assert np.isnan(hourly.loc[0, "tp"])
    assert np.allclose(hourly.loc[1:, "ssrd"], [5.0, 5.0, 2.0, 3.0])
    assert np.allclose(hourly.loc[1:, "tp"], [0.005, 0.005, 0.002, 0.003])


def test_midnight_precipitation_is_assigned_to_previous_day():
    _, daily = derive_era5(pd.DataFrame({
        "era5_dt_utc": pd.to_datetime(["2024-07-01T23:00Z", "2024-07-02T00:00Z", "2024-07-02T01:00Z"]),
        "tp": [0.001, 0.002, 0.004],
    }))

    assert np.isclose(daily.loc[daily.day.eq(pd.Timestamp("2024-07-01", tz="UTC")), "rain_1d"].iloc[0], 3.0)
    assert np.isclose(daily.loc[daily.day.eq(pd.Timestamp("2024-07-02", tz="UTC")), "rain_1d"].iloc[0], 4.0)


def test_logarithmic_and_binary_rain_have_grounded_catalogue_cards():
    assert VARIABLE_CATALOG["rain_3d_log"]["formula"] == "ln(rain_3d + 1)"
    assert VARIABLE_CATALOG["is_rainy"]["formula"] == "1 if rain_3d > 0 else 0"


def test_svf_returns_open_sky_for_empty_height_raster(tmp_path):
    raster = tmp_path / "empty_buildings.tif"
    data = np.zeros((100, 100), dtype="float32")
    with rasterio.open(
        raster, "w", driver="GTiff", width=100, height=100, count=1,
        dtype="float32", crs="EPSG:25830", transform=from_origin(0, 100, 1, 1),
    ) as dst:
        dst.write(data, 1)
    nodes = pd.DataFrame({"x": [50.0], "y": [50.0]})
    svf = approximate_sky_view_factor(
        nodes, raster, max_radius_m=30, radial_step_m=2, azimuth_count=16
    )
    assert np.isclose(svf.iloc[0], 1.0)


def test_describe_variable_marks_planned_urban_metrics():
    description = describe_variable("shadow_fraction", technical=True)
    assert "Planificada" in description
    assert "sombra" in description.lower()
