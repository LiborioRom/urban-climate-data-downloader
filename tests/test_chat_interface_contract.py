from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_chat_renders_current_user_turn_and_typing_indicator():
    source = (ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    handler = source[source.index("def _handle_prompt"):source.index("def _render_conversation_view")]
    assert 'with st.chat_message("user")' in handler
    assert 'with st.chat_message("assistant")' in handler
    assert "typing-indicator" in handler
    assert "thinking.empty()" in handler
    assert "time.perf_counter()" in handler
    assert "elapsed_seconds=elapsed_seconds" in handler


def test_pending_turn_is_declared_before_chat_input():
    source = (ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    view = source[source.index("def _render_conversation_view"):source.index("def _render_editor_view")]
    assert view.index("pending_turn = st.container()") < view.index("st.chat_input(")
    assert "with pending_turn:" in view


def test_chat_css_aligns_user_and_assistant_on_opposite_sides():
    source = (ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    assert "stChatMessageAvatarUser" in source
    assert "margin-left: auto" in source
    assert "stChatMessageAvatarAssistant" in source
    assert "margin-right: auto" in source
    assert ".response-time" in source


def test_brief_changelog_has_one_date_and_short_change_lines():
    lines = (ROOT / "CAMBIOS_BREVES.txt").read_text(encoding="utf-8").splitlines()
    assert lines.count("2026-09-12") == 1
    changes = [line.removeprefix("- ") for line in lines if line.startswith("- ")]
    assert changes
    assert all(len(change.rstrip(".").split()) <= 20 for change in changes)
    instructions = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "CAMBIOS_BREVES.txt" in instructions
    assert "20 palabras" in instructions


def test_editable_request_uses_bidirectional_map_instead_of_coordinate_fields():
    source = (ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    form = source[source.index("def _request_form"):source.index("def _render_variable_catalog")]
    editor = source[source.index("def _render_interactive_roi_editor"):source.index("def _summary_item")]

    assert "Latitud central" not in form
    assert "Longitud central" not in form
    assert "st_folium(" in editor
    assert '"last_geocoder_result"' in editor
    assert '"last_active_drawing"' in editor


def test_editor_has_one_execution_mode_and_optional_map_and_report():
    source = (ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    form = source[source.index("def _request_form"):source.index("def _render_variable_catalog")]

    assert 'selectbox("Modo"' not in form
    assert 'key="show_roi_map"' in source
    assert "Incluir informe del plan propuesto (.txt)" in source
    assert "start_background_download" in source
    assert "@st.fragment(run_every=1.0)" in source
    assert "_render_missing_field_styles" in source


def test_editor_restores_widgets_and_places_map_toggle_immediately_before_map():
    source = (ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    prime = source[source.index("def _prime_editor_widgets"):source.index("def _request_form")]
    editor = source[source.index("def _render_interactive_roi_editor"):source.index("def _summary_item")]

    assert "all(key in st.session_state for key in widget_keys)" in prime
    assert editor.index('show_map = st.toggle("Mostrar mapa"') < editor.index("map_data = st_folium(")
    between = editor[editor.index('show_map = st.toggle("Mostrar mapa"'):editor.index("map_data = st_folium(")]
    assert "number_input(" not in between


def test_checks_include_a_short_explanation_of_their_scope():
    source = (ROOT / "ui" / "app.py").read_text(encoding="utf-8")

    assert "Verifican que no falten campos" in source
    assert "Solo los errores impiden descargar" in source


def test_live_download_panel_is_rendered_immediately_after_output_card():
    source = (ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    editor = source[source.index("def _render_editor_view"):source.index("def main")]

    assert editor.index('key="execution_card"') < editor.index("_render_download_monitor()")
    assert editor.index("_render_download_monitor()") < editor.index("_render_missing_field_styles()")


def test_editor_exposes_dataset_purpose_and_quality_summary():
    source = (ROOT / "ui" / "app.py").read_text(encoding="utf-8")

    assert '"Propósito del dataset"' in source
    assert "Calidad de la descarga" in source
    assert "_render_quality_summary(quality)" in source
    assert '"Cobertura por fuente"' in source
    assert "fallback opcional" in source


def test_editor_exposes_recurring_date_windows():
    source = (ROOT / "ui" / "app.py").read_text(encoding="utf-8")

    assert '"Ventana anual en años seleccionados"' in source
    assert '"Año inicial"' in source and '"Año final"' in source
    assert 'key="editor_season_years"' not in source
    assert "build_recurring_windows" in source
    assert '"Mes inicial"' in source and '"Día inicial"' in source
    assert '"Mes final"' in source and '"Día final"' in source
    assert "se ignora el año" not in source


def test_editor_exposes_general_and_tfm_project_purposes():
    source = (ROOT / "ui" / "app.py").read_text(encoding="utf-8")

    assert '"Propósito general"' in source
    assert '"Proyecto TFM-GNN"' in source
    assert "TFM_GNN_VARIABLES" in source
    assert "_apply_purpose_preset" in source
