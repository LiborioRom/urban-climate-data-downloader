from agent.dialogue_manager import _geocoder_query, _heuristic_turn, interpret_turn, process_message
from schemas.conversation import AgentTurn, ClarificationAnswer, ConversationState, Intent, ROIDefinition, ROIState


def test_symbolic_fallback_understands_metric_roi_move(monkeypatch):
    monkeypatch.setattr("agent.dialogue_manager.ollama_status", lambda: {"running": False, "model_available": False})
    state = ConversationState(active_roi=ROIState(center_lat=37.4, center_lon=-5.98, half_side_m=750))
    result = process_message("desplázalo 500 m a la izquierda", state, [])
    assert result.state.last_intent == Intent.MOVE_ROI
    assert result.state.active_roi.center_lon < -5.98
    assert "500 m" in result.assistant_message


def test_vague_move_asks_for_distance():
    state = ConversationState(active_roi=ROIState(center_lat=37.4, center_lon=-5.98, half_side_m=750))
    turn = _heuristic_turn("muévelo un poco a la derecha", state)
    assert turn.intent == Intent.MOVE_ROI
    assert turn.roi_adjustment.distance_m is None


def test_symbolic_roi_keeps_original_place_spelling():
    turn = _heuristic_turn(
        "Quiero un área de 2 km² que rodee el barrio de Heliópolis en Sevilla",
        ConversationState(),
    )
    assert turn.intent == Intent.DEFINE_ROI
    assert turn.roi_definition.place_query == "Heliópolis en Sevilla"
    assert _geocoder_query(turn.roi_definition.place_query) == "Heliópolis, Sevilla"


def test_geocoder_cleans_conversational_neighbourhood_query():
    assert _geocoder_query("barrio de Heliópolis, en sevilla") == "Heliópolis, sevilla"


def test_combined_request_keeps_variables_and_roi_when_model_omits_patch(monkeypatch):
    monkeypatch.setattr("agent.dialogue_manager.ollama_status", lambda: {"running": True, "model_available": True})
    monkeypatch.setattr(
        "agent.dialogue_manager.structured_chat",
        lambda *args, **kwargs: AgentTurn(
            intent=Intent.DEFINE_ROI,
            roi_definition=ROIDefinition(place_query="barrio de Heliópolis, en sevilla", area_km2=2),
        ),
    )
    turn, provider = interpret_turn(
        "Quiero LST, NDVI y DEM de un cuadrado de 2 km² con centro en Heliópolis, Sevilla",
        ConversationState(),
        [],
    )
    assert provider == "ollama"
    assert turn.roi_definition.area_km2 == 2
    assert turn.request_patch.variables == ["LST", "NDVI", "elevation"]


def test_combined_request_is_complete_in_symbolic_fallback(monkeypatch):
    monkeypatch.setattr("agent.dialogue_manager.ollama_status", lambda: {"running": False, "model_available": False})
    turn, provider = interpret_turn(
        "Quiero LST, NDVI y DEM de un cuadrado de 2 km² con centro en el barrio de Heliópolis, en Sevilla",
        ConversationState(),
        [],
    )
    assert provider == "symbolic_fallback"
    assert turn.intent == Intent.DEFINE_ROI
    assert turn.request_patch.variables == ["LST", "NDVI", "elevation"]


def test_request_updates_preserve_existing_state(monkeypatch):
    monkeypatch.setattr("agent.dialogue_manager.ollama_status", lambda: {"running": False, "model_available": False})
    state = ConversationState()
    first = process_message("NDVI a 100 m", state, []).state
    second = process_message("entre mayo y octubre de 2024", first, []).state
    assert second.request.variables == ["NDVI"]
    assert second.request.target_resolution_m == 100
    assert str(second.request.start_date) == "2024-05-01"
    assert str(second.request.end_date) == "2024-10-31"


def test_incoherent_define_roi_uses_qwen_clarification_without_mutating_state(monkeypatch):
    monkeypatch.setattr("agent.dialogue_manager.ollama_status", lambda: {"running": True, "model_available": True})
    calls = []

    def fake_chat(schema, **kwargs):
        calls.append((schema, kwargs))
        if schema is AgentTurn:
            return AgentTurn(
                intent=Intent.DEFINE_ROI,
                roi_definition=ROIDefinition(place_query="ayer"),
            )
        return ClarificationAnswer(message="No he entendido el intervalo indicado. ¿Quieres modificar las fechas de descarga?")

    monkeypatch.setattr("agent.dialogue_manager.structured_chat", fake_chat)
    state = ConversationState()
    result = process_message(
        "quiero que se descarguen entre antes de ayer y ayer",
        state,
        [{"role": "assistant", "content": "¿Qué fechas necesitas?"}],
    )

    assert len(calls) == 2
    assert calls[1][0] is ClarificationAnswer
    assert calls[1][1]["history"] == [{"role": "assistant", "content": "¿Qué fechas necesitas?"}]
    assert result.provider == "ollama_clarification"
    assert result.state.last_intent == Intent.CLARIFY
    assert result.state.active_roi is None
    assert result.state.request.roi is None
    assert "modificar las fechas" in result.assistant_message
    assert "Falta definir el área" not in result.assistant_message
    assert "superficie" not in result.assistant_message


def test_clarification_has_fixed_fallback_if_second_qwen_call_fails(monkeypatch):
    monkeypatch.setattr("agent.dialogue_manager.ollama_status", lambda: {"running": True, "model_available": True})
    call_count = 0

    def fake_chat(schema, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AgentTurn(intent=Intent.EXECUTE)
        raise RuntimeError("invalid clarification")

    monkeypatch.setattr("agent.dialogue_manager.structured_chat", fake_chat)
    result = process_message("no era eso", ConversationState(), [])

    assert result.provider == "clarification_fallback"
    assert result.state.last_intent == Intent.CLARIFY
    assert result.execute_requested is False
    assert "suficiente seguridad" in result.assistant_message
    assert "Falta definir el área" not in result.assistant_message


def test_variable_download_request_does_not_execute_in_symbolic_fallback(monkeypatch):
    monkeypatch.setattr("agent.dialogue_manager.ollama_status", lambda: {"running": False, "model_available": False})
    result = process_message("quiero descargar NDVI y DEM", ConversationState(), [])

    assert result.execute_requested is False
    assert result.state.last_intent == Intent.CREATE_OR_UPDATE_REQUEST
    assert result.state.request.variables == ["NDVI", "elevation"]


def test_direct_clarify_intent_never_applies_an_attached_patch(monkeypatch):
    monkeypatch.setattr("agent.dialogue_manager.ollama_status", lambda: {"running": True, "model_available": True})
    monkeypatch.setattr(
        "agent.dialogue_manager.structured_chat",
        lambda *args, **kwargs: AgentTurn(
            intent=Intent.CLARIFY,
            request_patch={"variables": ["NDVI"], "variable_action": "add"},
            assistant_message="¿Puedes reformularlo?",
        ),
    )

    result = process_message("no era eso", ConversationState(), [])

    assert result.state.request.variables == []
    assert "actualizado" not in result.assistant_message
    assert "Falta definir el área" not in result.assistant_message
