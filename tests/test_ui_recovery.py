import json
from datetime import date

from streamlit.testing.v1 import AppTest

from schemas.conversation import ConversationState, ROIState
from schemas.request import DatasetPurpose, DatasetRequest, ROIRequest, ROISpecification
from schemas.request import DateSelectionMode
from ui.app import TFM_GNN_VARIABLES


def test_missing_credentials_are_explained_in_chat_without_running_notebook(monkeypatch):
    for name in ("EE_PROJECT", "SH_CLIENT_ID", "SH_CLIENT_SECRET", "CDS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    roi = ROIState(center_lat=37.4, center_lon=-5.98, half_side_m=750, confirmed=True)
    request = DatasetRequest(
        roi=ROIRequest(specification=ROISpecification.CENTER_AND_HALF_SIDE, center_lat=37.4, center_lon=-5.98, half_side_m=750),
        start_date="2024-07-15", end_date="2024-07-15", variables=["LST"], target_resolution_m=100,
    )

    app = AppTest.from_file("app.py")
    app.session_state["conversation_state"] = ConversationState(request=request, active_roi=roi)
    app.session_state["chat_messages"] = []
    app.session_state["active_view"] = "🛠️ Petición editable"
    app.run(timeout=20)
    review = next(box for box in app.checkbox if box.label.startswith("He revisado"))
    review.check()
    app.run(timeout=20)
    generate = next(button for button in app.button if button.label == "Generar dataset")
    generate.click()
    app.run(timeout=20)

    chat = "\n".join(message["content"] for message in app.session_state["chat_messages"])
    assert "No puedo iniciar la descarga" in chat
    assert "SH_CLIENT_ID" in chat
    assert "MISSING_CREDENTIALS_PREFLIGHT" in chat
    assert app.session_state["editor_start_date"].isoformat() == "2024-07-15"
    assert app.session_state["editor_variables"] == ["LST"]
    assert set(app.session_state["_missing_fields"]) == {"EE_PROJECT", "SH_CLIENT_ID", "SH_CLIENT_SECRET"}


def test_editable_values_survive_switching_to_chat_and_back():
    app = AppTest.from_file("app.py")
    app.session_state["active_view"] = "🛠️ Petición editable"
    app.run(timeout=20)
    app.date_input[0].set_value(date(2024, 7, 1))
    app.date_input[1].set_value(date(2024, 7, 2))
    app.multiselect[0].select("LST")
    app.number_input[0].set_value(250)
    app.number_input[1].set_value(1800)
    app.toggle[0].set_value(False)
    app.toggle[1].set_value(True)
    app.run(timeout=20)

    app.button_group[0].set_value("💬 Conversación")
    app.run(timeout=20)
    app.button_group[0].set_value("🛠️ Petición editable")
    app.run(timeout=20)

    assert [widget.value for widget in app.date_input] == [date(2024, 7, 1), date(2024, 7, 2)]
    assert app.multiselect[0].value == ["LST"]
    assert [widget.value for widget in app.number_input] == [250.0, 1800.0]
    assert [widget.value for widget in app.toggle] == [False, True]


def test_recurring_window_updates_the_request_with_written_year_range():
    app = AppTest.from_file("app.py")
    app.session_state["active_view"] = "🛠️ Petición editable"
    app.run(timeout=20)
    mode = next(widget for widget in app.selectbox if widget.label == "Selección temporal")
    mode.set_value(DateSelectionMode.RECURRING_WINDOW.value)
    app.run(timeout=20)
    start_year = next(widget for widget in app.number_input if widget.label == "Año inicial")
    end_year = next(widget for widget in app.number_input if widget.label == "Año final")
    start_year.set_value(2020)
    end_year.set_value(2022)
    app.run(timeout=20)

    request = app.session_state["conversation_state"].request
    assert request.date_selection_mode == DateSelectionMode.RECURRING_WINDOW
    assert [(str(item.start_date), str(item.end_date)) for item in request.date_windows] == [
        ("2020-05-01", "2020-10-31"),
        ("2021-05-01", "2021-10-31"),
        ("2022-05-01", "2022-10-31"),
    ]


def test_tfm_purpose_applies_project_variables_and_temporal_preset():
    app = AppTest.from_file("app.py")
    app.session_state["active_view"] = "🛠️ Petición editable"
    app.run(timeout=20)
    purpose = next(widget for widget in app.selectbox if widget.label == "Propósito del dataset")
    purpose.set_value(DatasetPurpose.LST_DOWNSCALING.value)
    app.run(timeout=20)

    request = app.session_state["conversation_state"].request
    assert request.dataset_purpose == DatasetPurpose.LST_DOWNSCALING
    assert request.variables == TFM_GNN_VARIABLES
    assert request.target_resolution_m == 100
    assert request.date_selection_mode == DateSelectionMode.RECURRING_WINDOW
    assert request.date_windows[0].start_date.isoformat() == "2013-05-01"
    assert request.date_windows[-1].start_date.year == 2025
    assert request.date_windows[-1].end_date.month == 10
    assert request.date_windows[-1].end_date.day == 31


def test_download_status_is_shown_in_the_editable_output_section(tmp_path):
    status_path = tmp_path / "estado_descarga.json"
    status_path.write_text(json.dumps({
        "state": "running",
        "percent": 46,
        "message": "Landsat completado.",
        "events": [{"percent": 46, "message": "Landsat completado."}],
        "started_at": "2026-09-14T12:00:00+00:00",
    }), encoding="utf-8")
    app = AppTest.from_file("app.py")
    app.session_state["_download_job"] = {
        "status_path": str(status_path), "announced": False,
    }
    app.session_state["active_view"] = "🛠️ Petición editable"
    app.run(timeout=20)
    assert any(item.value == "Descarga en curso" for item in app.subheader)
