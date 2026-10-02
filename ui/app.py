from __future__ import annotations

import json
import os
import re
import time
import calendar
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit_folium import st_folium

from agent.dialogue_manager import process_message
from agent.error_explainer import explain_exception, explain_missing_credentials
from agent.local_llm import ollama_status
from agent.planner import build_plan
from schemas.conversation import ConversationState, ROIState
from schemas.request import DateSelectionMode, DatasetPurpose, DatasetRequest, ROIRequest
from tools.background_download import load_download_status, start_background_download
from tools.credentials import credential_status, missing_required_credentials, set_session_credentials
from tools.date_windows import build_recurring_windows, describe_date_selection
from tools.editable_map import render_editable_roi_map
from tools.map_renderer import render_roi_map
from tools.roi_geometry import as_request, create_roi, resize_roi, update_roi_from_geojson
from tools.storage import get_data_root, new_run_directory, shared_cache_directory
from rules.variable_catalog import (
    VARIABLE_CATALOG,
    describe_variable,
    selectable_variables,
    variable_label,
    variables_by_category,
)


CHAT_VIEW = "💬 Conversación"
EDITOR_VIEW = "🛠️ Petición editable"
VARIABLE_LABELS = {name: spec["display_name"] for name, spec in VARIABLE_CATALOG.items()}
PURPOSE_LABELS = {
    DatasetPurpose.GENERAL_DOWNLOAD.value: "Propósito general",
    DatasetPurpose.LST_DOWNSCALING.value: "Proyecto TFM-GNN",
}
DATE_MODE_LABELS = {
    DateSelectionMode.CONTINUOUS.value: "Intervalo continuo",
    DateSelectionMode.RECURRING_WINDOW.value: "Ventana anual en años seleccionados",
}

TFM_GNN_VARIABLES = [
    "LST", "NDVI", "NDBI", "ALBEDO", "air_temperature", "relative_humidity",
    "solar_radiation", "wind_speed", "rain_3d_log", "is_rainy", "elevation",
    "aspect_ratio", "buildings",
]


def _load_streamlit_secrets_into_environment() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    names = (
        "OLLAMA_MODEL", "OLLAMA_HOST", "OLLAMA_NUM_CTX", "NOMINATIM_USER_AGENT",
        "ERA5_SOURCE", "EE_PROJECT", "SH_CLIENT_ID", "SH_CLIENT_SECRET", "CDS_API_KEY",
        "LST_DATA_ROOT",
    )
    for name in names:
        try:
            value = st.secrets.get(name)
        except (FileNotFoundError, KeyError, StreamlitSecretNotFoundError):
            value = None
        if value and not os.getenv(name):
            os.environ[name] = str(value)


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #f8fafc 0%, #ffffff 28%); }
        div[data-testid="stMainBlockContainer"] { max-width: 1500px; padding-top: 2.5rem; }
        h1 { letter-spacing: -0.035em; font-size: 2.8rem !important; }
        h2 { font-size: 2rem !important; line-height: 1.25 !important; }
        h3 { font-size: 1.62rem !important; line-height: 1.3 !important; }
        div[data-testid="stMainBlockContainer"] p,
        div[data-testid="stMainBlockContainer"] li,
        div[data-testid="stMainBlockContainer"] label,
        div[data-testid="stMainBlockContainer"] button,
        div[data-testid="stMainBlockContainer"] input,
        div[data-testid="stMainBlockContainer"] textarea {
            font-size: 1.16rem !important;
            line-height: 1.55 !important;
        }
        div[data-testid="stSegmentedControl"] label,
        div[data-testid="stSegmentedControl"] p {
            font-size: 1.28rem !important;
            font-weight: 600 !important;
        }
        div[data-testid="stChatMessage"] {
            padding: 1.05rem 1.2rem;
            border: 1px solid #e6eaf0;
            border-radius: 18px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.045);
            margin-bottom: .85rem;
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
            width: fit-content;
            max-width: 82%;
            margin-left: auto;
            margin-right: 0;
            flex-direction: row-reverse;
            background: #dff1ff;
            border-color: #bddcf3;
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
            width: fit-content;
            max-width: 92%;
            margin-left: 0;
            margin-right: auto;
            background: #ffffff;
        }
        div[data-testid="stChatMessageContent"] p,
        div[data-testid="stChatMessageContent"] li {
            font-size: 1.25rem !important;
            line-height: 1.65 !important;
        }
        div[data-testid="stChatInput"] textarea {
            min-height: 96px !important;
            font-size: 1.22rem !important;
            line-height: 1.45 !important;
            padding-top: 1rem !important;
        }
        div[data-testid="stChatInput"] { margin-top: 1rem; }
        .typing-indicator {
            display: inline-flex;
            align-items: center;
            gap: .36rem;
            min-height: 1.7rem;
            padding: .15rem .2rem;
        }
        .typing-indicator span {
            width: .55rem;
            height: .55rem;
            border-radius: 999px;
            background: #64748b;
            animation: typing-bounce 1.15s infinite ease-in-out;
        }
        .typing-indicator span:nth-child(2) { animation-delay: .16s; }
        .typing-indicator span:nth-child(3) { animation-delay: .32s; }
        @keyframes typing-bounce {
            0%, 60%, 100% { transform: translateY(0); opacity: .42; }
            30% { transform: translateY(-.35rem); opacity: 1; }
        }
        .response-time {
            display: block;
            margin-top: .28rem;
            color: #94a3b8;
            font-size: .72rem;
            line-height: 1;
            text-align: right;
            font-variant-numeric: tabular-nums;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 18px;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
            border-color: #e2e8f0;
        }
        .app-subtitle { color: #64748b; margin: -.55rem 0 1.35rem; font-size: 1.08rem; }
        .section-kicker {
            color: #2563eb; font-size: .82rem; font-weight: 750; letter-spacing: .08em;
            text-transform: uppercase; margin-bottom: -.4rem;
        }
        .download-time {
            display: inline-flex; align-items: center; gap: .4rem; padding: .45rem .8rem;
            border-radius: 999px; color: #166534; background: #dcfce7; font-weight: 650;
        }
        .st-key-live_summary_panel h3 { font-size: 1.85rem !important; }
        .st-key-live_summary_panel p,
        .st-key-live_summary_panel label,
        .st-key-live_summary_panel button {
            font-size: 1.22rem !important;
            line-height: 1.55 !important;
        }
        section[data-testid="stSidebar"] h3 { font-size: 1.25rem !important; }
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] button,
        section[data-testid="stSidebar"] input {
            font-size: 0.98rem !important;
            line-height: 1.45 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


FIELD_SELECTORS = {
    "start_date": ".st-key-date_selection_card",
    "end_date": ".st-key-date_selection_card",
    "variables": ".st-key-editor_variables",
    "resolution": ".st-key-editor_resolution",
    "roi": ".st-key-roi_editor_card",
    "roi_confirmation": ".st-key-roi_editor_card",
    "review_confirmation": ".st-key-review_confirmation",
    "EE_PROJECT": ".st-key-credential_EE_PROJECT",
    "SH_CLIENT_ID": ".st-key-credential_SH_CLIENT_ID",
    "SH_CLIENT_SECRET": ".st-key-credential_SH_CLIENT_SECRET",
    "CDS_API_KEY": ".st-key-credential_CDS_API_KEY",
}


def _render_missing_field_styles() -> None:
    selectors = [FIELD_SELECTORS[name] for name in st.session_state.get("_missing_fields", []) if name in FIELD_SELECTORS]
    if not selectors:
        return
    joined = ",\n".join(selectors)
    focused = ",\n".join(f"{selector}:focus-within" for selector in selectors)
    st.markdown(
        f"""
        <style>
        {joined} {{
            outline: 2px solid #ef4444 !important;
            outline-offset: 4px;
            border-radius: 12px;
        }}
        {focused} {{ outline: none !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _clear_missing_field(name: str) -> None:
    missing = set(st.session_state.get("_missing_fields", []))
    missing.discard(name)
    st.session_state._missing_fields = sorted(missing)


def _apply_purpose_preset() -> None:
    if st.session_state.get("editor_purpose") != DatasetPurpose.LST_DOWNSCALING.value:
        return
    st.session_state.editor_variables = list(TFM_GNN_VARIABLES)
    st.session_state.editor_resolution = 100.0
    st.session_state.editor_date_mode = DateSelectionMode.RECURRING_WINDOW.value
    st.session_state.editor_season_start_month = 5
    st.session_state.editor_season_start_day = 1
    st.session_state.editor_season_end_month = 10
    st.session_state.editor_season_end_day = 31
    st.session_state.editor_season_start_year = 2013
    st.session_state.editor_season_end_year = 2025
    missing = set(st.session_state.get("_missing_fields", []))
    missing.difference_update({"variables", "resolution", "start_date", "end_date"})
    st.session_state._missing_fields = sorted(missing)


def _credential_form() -> None:
    st.subheader("Credenciales para esta sesión")
    st.caption("Se guardan solo en memoria. No aparecen en prompts, metadatos ni mensajes.")
    with st.form("session_credentials", clear_on_submit=True):
        ee_project = st.text_input("EE_PROJECT", help="Proyecto de Google Cloud habilitado para Earth Engine.", key="credential_EE_PROJECT")
        sh_client_id = st.text_input(
            "SH_CLIENT_ID", help="Identificador del cliente OAuth creado en el panel de Sentinel Hub.",
            key="credential_SH_CLIENT_ID",
        )
        sh_client_secret = st.text_input(
            "SH_CLIENT_SECRET", type="password",
            help="Secreto del cliente OAuth de Sentinel Hub; solo se conserva durante esta sesión.",
            key="credential_SH_CLIENT_SECRET",
        )
        with st.expander("Fallback ERA5-Land mediante CDS (opcional)"):
            st.caption("Solo se utiliza si Earth Engine no puede completar ERA5-Land.")
            cds_api_key = st.text_input(
                "CDS_API_KEY", type="password", key="credential_CDS_API_KEY",
                help="Token opcional de Climate Data Store para recuperación meteorológica.",
            )
        submitted = st.form_submit_button("Guardar en esta sesión")
    if submitted:
        configured = set_session_credentials({
            "EE_PROJECT": ee_project, "SH_CLIENT_ID": sh_client_id,
            "SH_CLIENT_SECRET": sh_client_secret, "CDS_API_KEY": cds_api_key,
        })
        st.success("Configuración actualizada para esta sesión.") if configured else st.info("No se introdujo ningún valor nuevo.")
    for name, configured in credential_status().items():
        if name == "CDS_API_KEY":
            st.write(f"{'✅' if configured else 'ℹ️'} {name} · fallback opcional")
        else:
            st.write(f"{'✅' if configured else '⚠️'} {name}")


def _model_status_panel() -> None:
    st.subheader("Modelo conversacional local")
    status = ollama_status()
    if status["running"] and status["model_available"]:
        st.success(f"Ollama activo · {status['model']}")
    elif status["running"]:
        st.warning(f"Ollama está activo, pero falta descargar {status['model']}.")
    else:
        st.warning("Ollama no está activo. Se usará el parser simbólico.")


def _append_chat(role: str, content: str, *, elapsed_seconds: float | None = None) -> None:
    message: dict[str, object] = {"role": role, "content": content}
    if elapsed_seconds is not None:
        message["elapsed_seconds"] = elapsed_seconds
    st.session_state.chat_messages.append(message)


def _render_response_time(elapsed_seconds: float) -> None:
    st.markdown(
        f'<span class="response-time">{elapsed_seconds:.1f} s</span>',
        unsafe_allow_html=True,
    )


def _render_chat() -> None:
    if not st.session_state.chat_messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Cuéntame **qué variables**, **qué fechas**, **qué resolución** y **qué zona** necesitas. "
                "Puedes dar toda la información en una sola frase o completarla poco a poco."
            )
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("elapsed_seconds") is not None:
                _render_response_time(float(message["elapsed_seconds"]))


def _editor_signature(request: DatasetRequest) -> tuple[object, ...]:
    return (
        request.start_date, request.end_date, tuple(request.variables),
        request.target_resolution_m, request.dataset_purpose, request.date_selection_mode,
        tuple((window.start_date, window.end_date) for window in request.date_windows),
    )


def _prime_editor_widgets(seed: DatasetRequest) -> None:
    signature = _editor_signature(seed)
    widget_keys = ("editor_date_mode", "editor_variables", "editor_resolution", "editor_purpose")
    if st.session_state.get("_editor_request_signature") == signature and all(key in st.session_state for key in widget_keys):
        return
    st.session_state.editor_start_date = seed.start_date
    st.session_state.editor_end_date = seed.end_date
    st.session_state.editor_date_mode = seed.date_selection_mode.value
    if seed.date_windows:
        first = seed.date_windows[0]
        st.session_state.editor_season_start_month = first.start_date.month
        st.session_state.editor_season_start_day = first.start_date.day
        st.session_state.editor_season_end_month = first.end_date.month
        st.session_state.editor_season_end_day = first.end_date.day
        st.session_state.editor_season_start_year = seed.date_windows[0].start_date.year
        st.session_state.editor_season_end_year = seed.date_windows[-1].start_date.year
    else:
        start = seed.start_date or date(date.today().year, 5, 1)
        end = seed.end_date or date(date.today().year, 10, 31)
        st.session_state.editor_season_start_month = start.month
        st.session_state.editor_season_start_day = start.day
        st.session_state.editor_season_end_month = end.month
        st.session_state.editor_season_end_day = end.day
        st.session_state.editor_season_start_year = start.year
        st.session_state.editor_season_end_year = end.year
    st.session_state.editor_variables = list(seed.variables)
    st.session_state.editor_resolution = seed.target_resolution_m
    st.session_state.editor_purpose = seed.dataset_purpose.value
    st.session_state._editor_request_signature = signature


def _request_form(seed: DatasetRequest) -> DatasetRequest:
    _prime_editor_widgets(seed)
    with st.container(border=True, key="request_fields_card"):
        st.markdown('<div class="section-kicker">Configuración</div>', unsafe_allow_html=True)
        st.subheader("Petición editable")
        with st.container(key="date_selection_card"):
            date_mode = st.selectbox(
                "Selección temporal",
                list(DATE_MODE_LABELS),
                key="editor_date_mode",
                format_func=lambda value: DATE_MODE_LABELS[value],
                help="La ventana anual descarga únicamente esos días en cada año seleccionado.",
            )
            if date_mode == DateSelectionMode.CONTINUOUS.value:
                left, right = st.columns(2)
                start = left.date_input(
                    "Fecha inicial", key="editor_start_date",
                    on_change=_clear_missing_field, args=("start_date",),
                )
                end = right.date_input(
                    "Fecha final", key="editor_end_date",
                    on_change=_clear_missing_field, args=("end_date",),
                )
                date_windows = []
            else:
                month_names = {
                    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
                    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
                    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
                }
                start_month_col, start_day_col, end_month_col, end_day_col = st.columns(4)
                start_month = start_month_col.selectbox(
                    "Mes inicial", list(month_names), key="editor_season_start_month",
                    format_func=lambda value: month_names[value],
                    on_change=_clear_missing_field, args=("start_date",),
                )
                max_start_day = calendar.monthrange(2000, start_month)[1]
                if st.session_state.editor_season_start_day > max_start_day:
                    st.session_state.editor_season_start_day = max_start_day
                start_day = start_day_col.selectbox(
                    "Día inicial", list(range(1, max_start_day + 1)), key="editor_season_start_day",
                    on_change=_clear_missing_field, args=("start_date",),
                )
                end_month = end_month_col.selectbox(
                    "Mes final", list(month_names), key="editor_season_end_month",
                    format_func=lambda value: month_names[value],
                    on_change=_clear_missing_field, args=("end_date",),
                )
                max_end_day = calendar.monthrange(2000, end_month)[1]
                if st.session_state.editor_season_end_day > max_end_day:
                    st.session_state.editor_season_end_day = max_end_day
                end_day = end_day_col.selectbox(
                    "Día final", list(range(1, max_end_day + 1)), key="editor_season_end_day",
                    on_change=_clear_missing_field, args=("end_date",),
                )
                latest_year = max(date.today().year, seed.end_date.year if seed.end_date else date.today().year)
                start_year_col, end_year_col = st.columns(2)
                start_year = int(start_year_col.number_input(
                    "Año inicial", min_value=2013, max_value=latest_year, step=1,
                    key="editor_season_start_year",
                    help="Escribe el primer año que se descargará.",
                    on_change=_clear_missing_field, args=("start_date",),
                ))
                end_year = int(end_year_col.number_input(
                    "Año final", min_value=2013, max_value=latest_year, step=1,
                    key="editor_season_end_year",
                    help="Escribe el último año. Se incluirán todos los años intermedios.",
                    on_change=_clear_missing_field, args=("end_date",),
                ))
                if start_year > end_year:
                    st.error("El año inicial no puede ser posterior al año final.")
                    years = []
                else:
                    years = list(range(start_year, end_year + 1))
                try:
                    date_windows = build_recurring_windows(
                        date(2000, start_month, start_day), date(2000, end_month, end_day), years,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    date_windows = []
                start = min((window.start_date for window in date_windows), default=None)
                end = max((window.end_date for window in date_windows), default=None)
                if date_windows:
                    st.caption(f"Se consultarán {len(date_windows)} ventanas y no los periodos intermedios.")
        variables = st.multiselect(
            "Variables",
            selectable_variables(),
            key="editor_variables",
            format_func=lambda variable: variable_label(variable, include_category=True),
            help="Puedes buscar por nombre científico, nombre legible o categoría. Solo se descargan las variables seleccionadas.",
            on_change=_clear_missing_field,
            args=("variables",),
        )
        resolution = st.number_input(
            "Resolución objetivo (m)",
            min_value=1.0,
            placeholder="Ej.: 100",
            key="editor_resolution",
            help=(
                "Separación, en metros, entre los nodos de la malla de salida. "
                "Una cifra menor crea más nodos, aumenta el tiempo de proceso y no mejora la resolución nativa de las fuentes."
            ),
            on_change=_clear_missing_field,
            args=("resolution",),
        )
        purpose = st.selectbox(
            "Propósito del dataset",
            list(PURPOSE_LABELS),
            key="editor_purpose",
            format_func=lambda value: PURPOSE_LABELS[value],
            help=(
                "Propósito general respeta la configuración manual. Proyecto TFM-GNN aplica las variables, "
                "resolución y temporadas cálidas empleadas por el proyecto; después puedes ajustarlas."
            ),
            on_change=_apply_purpose_preset,
        )
        if purpose == DatasetPurpose.LST_DOWNSCALING.value:
            st.caption("Preset aplicado: mayo–octubre de 2013 a 2025, 100 m y predictores del TFM-GNN.")
        st.caption("Los cambios se guardan automáticamente en la solicitud.")
    request = seed.model_copy(deep=True)
    request.start_date = start
    request.end_date = end
    request.date_selection_mode = DateSelectionMode(date_mode)
    request.date_windows = date_windows
    request.variables = list(variables)
    request.target_resolution_m = resolution
    request.dataset_purpose = DatasetPurpose(purpose)
    st.session_state._editor_request_signature = _editor_signature(request)
    return request


def _render_variable_catalog(selected: list[str]) -> None:
    with st.expander("Diccionario y fichas técnicas de variables"):
        categories = variables_by_category(include_planned=True)
        category = st.selectbox("Categoría de la ficha", list(categories), key="variable_help_category")
        options = categories[category]
        default = next((name for name in selected if name in options), options[0])
        variable = st.selectbox(
            "Variable",
            options,
            index=options.index(default),
            format_func=lambda name: variable_label(name),
            key="variable_help_name",
        )
        st.markdown(describe_variable(variable, technical=True))
        if VARIABLE_CATALOG[variable]["availability"] != "available":
            st.warning("Esta variable está documentada y el bot puede explicarla, pero todavía no puede ejecutarse.")


def _sync_manual_request(state: ConversationState, request: DatasetRequest) -> None:
    state.request = request
    if state.active_roi:
        state.request.roi = as_request(state.active_roi)
        return
    roi = request.roi
    if not roi or roi.center_lat is None or roi.center_lon is None or roi.half_side_m is None:
        return
    if state.active_roi:
        state.roi_history.append(state.active_roi.model_copy(deep=True))
    state.active_roi = ROIState(
        center_lat=roi.center_lat, center_lon=roi.center_lon, half_side_m=roi.half_side_m,
        rotation_deg=roi.rotation_deg, place_name=roi.place_name, confirmed=False,
        provenance="manual_form",
    )


def _replace_active_roi(state: ConversationState, roi: ROIState) -> None:
    if state.active_roi:
        state.roi_history.append(state.active_roi.model_copy(deep=True))
    state.active_roi = roi
    state.request.roi = as_request(roi)
    st.session_state.conversation_state = state
    _clear_missing_field("roi")


def _roi_state_from_request(roi: ROIRequest) -> ROIState:
    return ROIState(
        center_lat=roi.center_lat,
        center_lon=roi.center_lon,
        half_side_m=roi.half_side_m,
        rotation_deg=roi.rotation_deg,
        place_name=roi.place_name,
        display_name=roi.area_name or roi.place_name,
        confirmed=True,
        provenance="saved_area",
    )


def _area_slug(name: str, existing: set[str]) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    base = re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_") or "area"
    candidate = base
    suffix = 2
    while candidate in existing:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _render_saved_area_manager(state: ConversationState) -> None:
    st.markdown("#### Zonas de la descarga")
    default_name = (
        state.active_roi.display_name or state.active_roi.place_name or f"Zona {len(state.request.areas) + 1}"
        if state.active_roi else f"Zona {len(state.request.areas) + 1}"
    )
    active_signature = (
        round(state.active_roi.center_lat, 5), round(state.active_roi.center_lon, 5), len(state.request.areas)
    ) if state.active_roi else (None, None, len(state.request.areas))
    if st.session_state.get("_roi_area_name_signature") != active_signature:
        st.session_state.roi_area_name = default_name
        st.session_state._roi_area_name_signature = active_signature
    area_name = st.text_input("Nombre de esta zona", key="roi_area_name")
    add_col, new_col = st.columns(2)
    if add_col.button(
        "Añadir ROI a la descarga",
        key="add_roi_to_batch",
        type="primary",
        disabled=not state.active_roi or not state.active_roi.confirmed,
        width="stretch",
    ):
        existing = {area.area_id for area in state.request.areas if area.area_id}
        area_id = _area_slug(area_name, existing)
        saved = as_request(state.active_roi).model_copy(
            update={"area_id": area_id, "area_name": area_name.strip() or area_id},
        )
        state.request.areas.append(saved)
        st.session_state.conversation_state = state
        _clear_missing_field("roi")
        st.rerun()
    if new_col.button("Preparar otra zona", key="new_roi_for_batch", width="stretch"):
        state.active_roi = None
        state.request.roi = None
        st.session_state.conversation_state = state
        st.rerun()

    if not state.request.areas:
        st.caption("Todavía no hay zonas guardadas. Si no añades ninguna, se descargará únicamente el ROI activo.")
        return
    st.caption(f"{len(state.request.areas)} ROI guardados · se procesarán como máximo 2 simultáneamente.")
    for index, area in enumerate(list(state.request.areas)):
        label_col, remove_col = st.columns([5, 1], vertical_alignment="center")
        label_col.markdown(
            f"**{index + 1}. {area.area_name or area.place_name or area.area_id}**  \n"
            f"{area.center_lat:.5f}, {area.center_lon:.5f} · lado {2 * area.half_side_m:.0f} m · giro {area.rotation_deg:.1f}°"
        )
        if remove_col.button("Quitar", key=f"remove_saved_roi_{area.area_id}_{index}"):
            state.request.areas.pop(index)
            st.session_state.conversation_state = state
            st.rerun()


def _map_event_token(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _render_interactive_roi_status(state: ConversationState, *, button_key: str) -> None:
    if not state.active_roi:
        st.info("Selecciona un resultado de búsqueda o pulsa el mapa para crear el ROI.")
        return
    roi = state.active_roi
    metrics = st.columns(4)
    metrics[0].metric("Superficie", f"{roi.area_km2:.3f} km²")
    metrics[1].metric("Lado", f"{2 * roi.half_side_m:.1f} m")
    metrics[2].metric("Giro", f"{roi.rotation_deg:.1f}°")
    metrics[3].metric("Centro", f"{roi.center_lat:.5f}, {roi.center_lon:.5f}")
    if roi.confirmed:
        st.success("ROI confirmado")
    elif st.button("Confirmar ROI mostrado", key=button_key, type="primary"):
        state.active_roi.confirmed = True
        state.request.roi = as_request(state.active_roi)
        _clear_missing_field("roi_confirmation")
        _append_chat("assistant", "ROI confirmado desde el mapa. Se utilizará esta geometría en la ejecución.")
        st.rerun()


def _render_interactive_roi_editor(state: ConversationState) -> None:
    with st.container(key="roi_editor_card"):
        st.subheader("Área de estudio")
        st.caption(
            "Busca un lugar o pulsa un punto para centrar el cuadrado. "
            "Puedes arrastrarlo y girarlo desde el control ↻ de su borde."
        )

    current_half_side = state.active_roi.half_side_m if state.active_roi else 750.0
    synced_half_side = st.session_state.get("_roi_half_side_synced")
    if "roi_side_m_editor" not in st.session_state:
        st.session_state.roi_side_m_editor = st.session_state.get("_saved_roi_side_m", 2 * current_half_side)
        st.session_state._roi_half_side_synced = current_half_side
    elif synced_half_side != current_half_side:
        st.session_state.roi_side_m_editor = 2 * current_half_side
        st.session_state._roi_half_side_synced = current_half_side
    side_m = st.number_input(
        "Lado del cuadrado (m)",
        min_value=2.0,
        step=50.0,
        key="roi_side_m_editor",
        help="La superficie será lado × lado. La resolución de los nodos se configura por separado.",
    )
    st.session_state._saved_roi_side_m = side_m
    st.caption(f"Superficie prevista: {(side_m ** 2) / 1_000_000:.3f} km²")
    if st.button("Aplicar dimensiones al cuadrado", key="apply_roi_side"):
        if state.active_roi:
            resized = resize_roi(state.active_roi, (side_m / 2) / state.active_roi.half_side_m)
            _replace_active_roi(state, resized)
            st.session_state._roi_half_side_synced = resized.half_side_m
            st.rerun()
        else:
            st.info("Ahora busca una ubicación o pulsa el mapa para colocar el cuadrado.")

    if "show_roi_map" not in st.session_state:
        st.session_state.show_roi_map = st.session_state.get("_saved_show_roi_map", True)
    show_map = st.toggle("Mostrar mapa", key="show_roi_map")
    st.session_state._saved_show_roi_map = show_map
    if not show_map:
        st.caption("El mapa está oculto; el área seleccionada se conserva sin cambios.")
        _render_interactive_roi_status(state, button_key="confirm_hidden_roi")
        _render_saved_area_manager(state)
        return

    saved_rois = [
        (area.area_name or area.place_name or area.area_id or f"Zona {index}", _roi_state_from_request(area))
        for index, area in enumerate(state.request.areas, start=1)
    ]
    map_data = st_folium(
        render_editable_roi_map(state.active_roi, saved_rois),
        key="interactive_roi_map",
        height=620,
        use_container_width=True,
        returned_objects=["last_clicked", "last_geocoder_result", "last_active_drawing"],
    ) or {}

    geocoder_result = map_data.get("last_geocoder_result")
    if geocoder_result:
        token = _map_event_token(geocoder_result)
        if token != st.session_state.get("_last_roi_geocoder_event"):
            st.session_state._last_roi_geocoder_event = token
            roi = create_roi(
                float(geocoder_result["lat"]),
                float(geocoder_result["lng"]),
                half_side_m=side_m / 2,
                place_name=geocoder_result.get("name") or "Ubicación buscada",
                display_name=geocoder_result.get("name"),
                provenance="map_search",
            )
            _replace_active_roi(state, roi)
            st.rerun()

    edited_geometry = map_data.get("last_active_drawing")
    if edited_geometry and state.active_roi:
        token = _map_event_token(edited_geometry)
        if token != st.session_state.get("_last_roi_geometry_event"):
            st.session_state._last_roi_geometry_event = token
            try:
                edited_roi = update_roi_from_geojson(state.active_roi, edited_geometry)
            except (TypeError, ValueError) as exc:
                st.error(f"No se pudo aplicar la edición del mapa: {exc}")
            else:
                _replace_active_roi(state, edited_roi)
                st.rerun()

    clicked = map_data.get("last_clicked")
    if clicked:
        token = _map_event_token(clicked)
        if token != st.session_state.get("_last_roi_click_event"):
            st.session_state._last_roi_click_event = token
            roi = create_roi(
                float(clicked["lat"]),
                float(clicked["lng"]),
                half_side_m=side_m / 2,
                place_name="Punto seleccionado en el mapa",
                provenance="map_click",
            )
            _replace_active_roi(state, roi)
            st.rerun()

    _render_interactive_roi_status(state, button_key="confirm_interactive_roi")
    _render_saved_area_manager(state)


def _summary_item(label: str, value: str | None, *, ready: bool) -> None:
    icon = "✅" if ready else "○"
    shown = value if value else "Pendiente"
    st.markdown(f"{icon} **{label}**")
    st.caption(shown)


def _render_live_summary(state: ConversationState) -> None:
    request = state.request
    roi = state.active_roi
    area_count = len(request.areas)
    values = [
        bool(request.variables), bool(request.start_date and request.end_date),
        request.target_resolution_m is not None, bool(area_count or roi is not None), bool(request.dataset_purpose),
    ]
    with st.container(border=True):
        st.subheader("Solicitud en curso")
        st.progress(sum(values) / len(values), text=f"{sum(values)} de {len(values)} campos principales")
        shown_variables = ", ".join(VARIABLE_LABELS.get(variable, variable) for variable in request.variables)
        _summary_item("Variables", shown_variables, ready=bool(request.variables))
        _summary_item(
            "Periodo", describe_date_selection(request),
            ready=bool(request.start_date and request.end_date),
        )
        _summary_item(
            "Resolución",
            f"{request.target_resolution_m:g} m" if request.target_resolution_m else None,
            ready=request.target_resolution_m is not None,
        )
        _summary_item(
            "Área",
            f"{area_count} ROI guardados" if area_count else (
                f"{roi.area_km2:.3f} km² · {roi.display_name or roi.place_name or 'ROI'}" if roi else None
            ),
            ready=bool(area_count or roi is not None),
        )
        _summary_item("Propósito", PURPOSE_LABELS[request.dataset_purpose.value], ready=True)
        if roi and not area_count:
            st.caption("Confirmada" if roi.confirmed else "Pendiente de confirmar")
            if st.session_state.get("show_roi_map", True):
                st.iframe(render_roi_map(roi, request.target_resolution_m).get_root().render(), height=285)
            if not roi.confirmed and st.button("Confirmar esta área", key="confirm_roi_summary", type="primary", width="stretch"):
                state.active_roi.confirmed = True
                state.request.roi = as_request(state.active_roi)
                _clear_missing_field("roi_confirmation")
                _append_chat("assistant", "ROI confirmado desde el mapa. Se utilizará esta geometría en la ejecución.")
                st.rerun()


def _render_roi_panel(state: ConversationState) -> None:
    if not state.active_roi:
        st.info("El ROI todavía no está definido. Puedes describirlo en la conversación o introducirlo manualmente.")
        return
    roi = state.active_roi
    st.subheader("Área de estudio")
    columns = st.columns(4)
    columns[0].metric("Superficie", f"{roi.area_km2:.3f} km²")
    columns[1].metric("Semilado", f"{roi.half_side_m:.1f} m")
    columns[2].metric("Latitud", f"{roi.center_lat:.6f}")
    columns[3].metric("Longitud", f"{roi.center_lon:.6f}")
    if roi.display_name:
        st.caption(f"Referencia geocodificada: {roi.display_name}")
    if roi.coverage_percent is not None:
        st.info(f"El cuadrado cubre aproximadamente el {roi.coverage_percent:.1f}% del límite geocodificado.")
    st.iframe(render_roi_map(roi, state.request.target_resolution_m).get_root().render(), height=560)
    if roi.confirmed:
        st.success("ROI confirmado")
    elif st.button("Confirmar ROI mostrado", key="confirm_roi_editor", type="primary"):
        state.active_roi.confirmed = True
        state.request.roi = as_request(state.active_roi)
        _append_chat("assistant", "ROI confirmado desde el mapa. Se utilizará esta geometría en la ejecución.")
        st.rerun()


def _go_to_view(target: str) -> None:
    st.session_state.active_view = target


def _handle_prompt(prompt: str, state: ConversationState) -> None:
    previous_history = list(st.session_state.chat_messages)
    _append_chat("user", prompt)
    # La conversación ya se había pintado antes de recibir el chat_input. Mostramos
    # el nuevo turno ahora para que no permanezca oculto durante la inferencia local.
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.markdown(
            '<div class="typing-indicator" aria-label="El asistente está pensando">'
            '<span></span><span></span><span></span></div>',
            unsafe_allow_html=True,
        )
        started_at = time.perf_counter()
        try:
            result = process_message(prompt, state, previous_history)
            st.session_state.conversation_state = result.state
            answer = result.assistant_message
        except Exception as exc:
            answer = explain_exception(exc, stage="interpretation").as_chat_message()
        elapsed_seconds = time.perf_counter() - started_at
        thinking.empty()
        st.markdown(answer)
        _render_response_time(elapsed_seconds)
    _append_chat("assistant", answer, elapsed_seconds=elapsed_seconds)
    st.rerun()


def _render_conversation_view(state: ConversationState) -> None:
    chat_column, summary_column = st.columns([2, 1], gap="large")
    with chat_column:
        st.subheader("Conversación")
        _render_chat()
        # Este contenedor conserva su posición aunque se rellene después de que
        # chat_input entregue el mensaje, por lo que el cuadro queda siempre debajo.
        pending_turn = st.container()
        prompt = st.chat_input(
            "Describe la zona, variables, fechas o cambios que quieras realizar…",
            key="conversation_prompt",
        )
        if prompt:
            with pending_turn:
                _handle_prompt(prompt, state)
    with summary_column:
        with st.container(key="live_summary_panel"):
            _render_live_summary(state)
            st.button(
                "Abrir petición editable →",
                key="open_editor",
                width="stretch",
                on_click=_go_to_view,
                args=(EDITOR_VIEW,),
            )


MISSING_FIELD_LABELS = {
    "start_date": "fecha inicial",
    "end_date": "fecha final",
    "variables": "variables",
    "resolution": "resolución objetivo",
    "roi": "área de estudio",
    "roi_confirmation": "confirmación del área",
    "review_confirmation": "revisión final",
    "EE_PROJECT": "EE_PROJECT",
    "SH_CLIENT_ID": "SH_CLIENT_ID",
    "SH_CLIENT_SECRET": "SH_CLIENT_SECRET",
    "CDS_API_KEY": "CDS_API_KEY",
}


def _request_missing_fields(state: ConversationState, reviewed: bool) -> list[str]:
    request = state.request
    missing = []
    if request.start_date is None:
        missing.append("start_date")
    if request.end_date is None:
        missing.append("end_date")
    if not request.variables:
        missing.append("variables")
    if request.target_resolution_m is None:
        missing.append("resolution")
    if not request.areas and not state.active_roi:
        missing.append("roi")
    elif not request.areas and not state.active_roi.confirmed:
        missing.append("roi_confirmation")
    if not reviewed:
        missing.append("review_confirmation")
    return missing


def _plan_error_fields(plan) -> list[str]:
    by_code = {
        "MISSING_ROI": ["roi"],
        "MISSING_START_DATE": ["start_date"],
        "MISSING_END_DATE": ["end_date"],
        "INVALID_DATE_RANGE": ["start_date", "end_date"],
        "MISSING_DATE_WINDOWS": ["start_date", "end_date"],
        "MISSING_VARIABLES": ["variables"],
        "MISSING_RESOLUTION": ["resolution"],
        "UNKNOWN_VARIABLE": ["variables"],
        "INCOMPATIBLE_SOURCE": ["variables"],
        "NOT_IMPLEMENTED": ["variables"],
    }
    fields = []
    for issue in plan.issues:
        if issue.severity.value == "ERROR":
            fields.extend(by_code.get(issue.code, []))
    return list(dict.fromkeys(fields))


def _render_plan_preview(plan) -> None:
    request = plan.request
    st.markdown(
        f"Se procesará **{describe_date_selection(request)}** "
        f"sobre una malla de **{request.target_resolution_m:g} m**. "
        f"Propósito: **{PURPOSE_LABELS[request.dataset_purpose.value]}**."
    )
    if request.areas:
        st.info(
            f"Descarga multizona: {len(request.areas)} ROI, hasta "
            f"{request.parameters.max_parallel_areas} áreas simultáneas y caché compartida."
        )
    for item in plan.variable_plans:
        source = item.dataset_or_product or item.source
        transformations = ", ".join(item.transformations) or "sin transformación adicional"
        st.markdown(f"- **{variable_label(item.variable)}** · {source} · {transformations}")
    st.caption("Al finalizar se guardará esta explicación ampliada en data_downloads/informe_plan.txt.")


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    return f"{minutes} min {remainder:02d} s"


def _start_download(plan, include_plan_report: bool) -> None:
    data_root = get_data_root()
    area_count = len(plan.request.areas) or 1
    run_dir = new_run_directory(area_count, data_root)
    st.session_state._download_job = start_background_download(
        plan,
        run_dir,
        include_plan_report=include_plan_report,
        cache_dir=shared_cache_directory(data_root),
    )
    st.rerun()


def _download_elapsed(status: dict) -> float:
    if status.get("state") in {"complete", "partial", "error"} and status.get("elapsed_seconds") is not None:
        return float(status["elapsed_seconds"])
    try:
        started = datetime.fromisoformat(status["started_at"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())


def _download_is_running() -> bool:
    job = st.session_state.get("_download_job")
    if not job:
        return False
    return load_download_status(job["status_path"]).get("state") in {"starting", "running"}


def _quality_from_status(status: dict) -> dict | None:
    path = status.get("outputs", {}).get("quality_report")
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _render_quality_summary(quality: dict) -> None:
    st.markdown("#### Calidad de la descarga")
    renderer = {
        "complete": st.success,
        "with_gaps": st.warning,
        "insufficient": st.error,
        "partial": st.warning,
    }.get(quality.get("status"), st.info)
    renderer(quality.get("summary", "No se pudo resumir la calidad."))
    metrics = st.columns(3)
    if quality.get("per_area") is not None:
        metrics[0].metric("Áreas completadas", f"{quality.get('areas_completed', 0)}/{quality.get('areas_requested', 0)}")
        metrics[1].metric("Filas", f"{quality.get('rows', 0):,}")
        metrics[2].metric("Áreas fallidas", f"{len(quality.get('areas_failed', []))}")
        with st.expander("Calidad por zona"):
            for item in quality.get("per_area", []):
                st.write(f"**{item['area_name']}** · {item.get('rows', 0):,} filas · {item.get('nodes', 0):,} nodos")
                st.caption(item.get("quality", {}).get("summary", "Sin resumen de calidad."))
        return
    metrics[0].metric("Completitud", f"{quality.get('scientific_completeness_percent', 0):.1f} %")
    metrics[1].metric("Filas", f"{quality.get('rows', 0):,}")
    metrics[2].metric("Fechas", f"{quality.get('dates', 0):,}")
    missing = quality.get("missing_by_column", [])
    if missing:
        shown = ", ".join(f"{item['column']} ({item['percent']:.1f} %)" for item in missing[:5])
        suffix = f" y {len(missing) - 5} columnas más" if len(missing) > 5 else ""
        st.caption(f"Valores vacíos: {shown}{suffix}.")
    if quality.get("training_ready_rows") is not None:
        st.caption(
            f"Filas utilizables para downscaling: {quality.get('training_ready_rows', 0):,} "
            f"({quality.get('training_ready_percent', 0):.1f} %)."
        )
    source_quality = quality.get("source_quality", {})
    if source_quality:
        with st.expander("Cobertura por fuente"):
            sentinel = source_quality.get("sentinel3", {})
            if sentinel:
                st.write(
                    "Sentinel-3: "
                    f"{sentinel.get('catalog_dates', 0)} fechas encontradas, "
                    f"{sentinel.get('valid_dates', 0)} válidas, "
                    f"{sentinel.get('cloudy_dates', 0)} con rechazo por nube y "
                    f"{sentinel.get('missing_dates', 0)} sin respuesta."
                )
            era5 = source_quality.get("era5_land", {})
            if era5:
                st.write(
                    "ERA5-Land: "
                    f"fuente {era5.get('source', 'desconocida')}, "
                    f"{era5.get('complete_target_hours', 0)}/{era5.get('target_hours', 0)} horas objetivo completas."
                )


@st.fragment(run_every=1.0)
def _render_download_monitor() -> None:
    job = st.session_state.get("_download_job")
    if not job:
        return
    status = load_download_status(job["status_path"])
    state = status.get("state", "starting")
    elapsed = _download_elapsed(status)
    percent = int(status.get("percent", 0))
    with st.container(border=True, key="persistent_download_status"):
        heading, timer = st.columns([3, 1], vertical_alignment="center")
        heading.subheader(
            "Descarga en curso" if state in {"starting", "running"}
            else "Descarga completada" if state in {"complete", "partial"}
            else "Descarga detenida"
        )
        timer.markdown(f'<span class="download-time">⏱ {_format_elapsed(elapsed)}</span>', unsafe_allow_html=True)
        st.progress(percent / 100, text=f"{percent} % · {status.get('message', 'Preparando…')}")
        areas = status.get("areas", [])
        if areas:
            with st.expander("Progreso por zona", expanded=True):
                for area in areas:
                    icon = {"queued": "○", "running": "◔", "complete": "✓", "error": "⚠"}.get(area.get("state"), "○")
                    st.write(
                        f"{icon} **{area.get('area_name', area.get('area_id'))}** · "
                        f"{area.get('percent', 0)} % · {area.get('message', '')}"
                    )
        events = status.get("events", [])
        for event in events[-5:]:
            st.caption(f"✓ {event['message']}")

        if state in {"complete", "partial"}:
            quality = _quality_from_status(status)
            if not job.get("announced"):
                quality_message = f"\n\n{quality['summary']}" if quality else ""
                _append_chat(
                    "assistant",
                    ("**Dataset generado con incidencias parciales.**" if state == "partial" else "**Dataset generado correctamente.**")
                    + "\n\nEl CSV unificado, las parejas Landsat–Sentinel-3 y sus metadatos están disponibles."
                    + quality_message,
                )
                job["announced"] = True
                st.session_state._download_job = job
            (st.warning if state == "partial" else st.success)(f"Proceso finalizado en {_format_elapsed(elapsed)}.")
            if quality:
                _render_quality_summary(quality)
            with st.expander("Archivos generados"):
                st.json(status.get("outputs", {}))
        elif state == "error":
            if not job.get("announced"):
                explanation = explain_exception(RuntimeError(status.get("error", "Error desconocido"))).as_chat_message()
                _append_chat("assistant", explanation)
                job["announced"] = True
                st.session_state._download_job = job
            st.error(status.get("error", "La descarga se detuvo. Consulta descarga.log."))

        if state in {"complete", "partial", "error"} and st.button("Cerrar estado de descarga", key="close_download_status"):
            st.session_state.pop("_download_job", None)
            st.rerun()


def _render_editor_view(state: ConversationState) -> None:
    st.button("← Volver a la conversación", key="back_to_chat", on_click=_go_to_view, args=(CHAT_VIEW,))
    request = _request_form(state.request)
    if request != state.request:
        _sync_manual_request(state, request)
        st.session_state.conversation_state = state

    _render_interactive_roi_editor(state)
    _render_variable_catalog(state.request.variables)
    plan = build_plan(state.request)
    with st.expander("Comprobaciones de la solicitud", expanded=False):
        st.caption(
            "Verifican que no falten campos, que cada variable tenga una fuente ejecutable y que fechas, "
            "resoluciones y transformaciones sean científicamente coherentes. Solo los errores impiden descargar."
        )
        if not plan.issues:
            st.success("La solicitud es coherente y puede ejecutarse.")
        for issue in plan.issues:
            renderer = {"ERROR": st.error, "WARNING": st.warning, "INFO": st.info}[issue.severity.value]
            renderer(issue.message)

    with st.container(border=True, key="execution_card"):
        st.markdown('<div class="section-kicker">Salida</div>', unsafe_allow_html=True)
        st.subheader("Generar dataset")
        if "include_plan_report" not in st.session_state:
            st.session_state.include_plan_report = st.session_state.get("_saved_include_plan_report", False)
        include_plan_report = st.toggle(
            "Incluir informe del plan propuesto (.txt)",
            key="include_plan_report",
            help="Guarda una explicación legible del área, fuentes, variables, transformaciones, avisos y salidas.",
        )
        st.session_state._saved_include_plan_report = include_plan_report
        if include_plan_report and plan.request.start_date and plan.request.end_date and plan.request.target_resolution_m:
            with st.expander("Vista previa del plan propuesto", expanded=False):
                _render_plan_preview(plan)

        if "review_confirmation" not in st.session_state:
            st.session_state.review_confirmation = st.session_state.get("_saved_review_confirmation", False)
        reviewed = st.checkbox(
            "He revisado el área, fechas, variables, resolución, fuentes y avisos.",
            key="review_confirmation",
            on_change=_clear_missing_field,
            args=("review_confirmation",),
        )
        st.session_state._saved_review_confirmation = reviewed
        download_running = _download_is_running()
        generate = st.button("Generar dataset", type="primary", width="stretch", disabled=download_running)
        if download_running:
            st.caption("Ya hay una descarga activa. Puedes cambiar de pestaña sin interrumpirla.")
        validation_notice = st.empty()

        missing_now = st.session_state.get("_missing_fields", [])
        if missing_now:
            labels = [MISSING_FIELD_LABELS.get(name, name) for name in missing_now]
            validation_notice.error("Faltan: " + ", ".join(labels) + ".")

        if generate:
            missing = _request_missing_fields(state, reviewed)
            credential_missing = missing_required_credentials()
            error_fields = _plan_error_fields(plan)
            st.session_state._missing_fields = list(dict.fromkeys([*missing, *credential_missing, *error_fields]))
            if missing or credential_missing:
                labels = [MISSING_FIELD_LABELS.get(name, name) for name in [*missing, *credential_missing]]
                validation_notice.error("Faltan: " + ", ".join(labels) + ".")
                if credential_missing:
                    _append_chat("assistant", explain_missing_credentials(credential_missing).as_chat_message())
            elif not plan.executable:
                validation_notice.error("Revisa los campos marcados y las comprobaciones de la solicitud.")
            else:
                st.session_state._missing_fields = []
                validation_notice.empty()
                _start_download(plan, include_plan_report)

    if st.session_state.get("_download_job"):
        _render_download_monitor()
    _render_missing_field_styles()


def main() -> None:
    _load_streamlit_secrets_into_environment()
    st.set_page_config(page_title="TFM · adquisición híbrida", page_icon="🌍", layout="wide")
    _apply_styles()
    st.title("Planificador conversacional de datos")
    st.markdown('<p class="app-subtitle">Configura, revisa y descarga datos geoespaciales desde una única interfaz.</p>', unsafe_allow_html=True)

    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = ConversationState()
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "show_roi_map" not in st.session_state:
        st.session_state.show_roi_map = st.session_state.get("_saved_show_roi_map", True)
    if "_missing_fields" not in st.session_state:
        st.session_state._missing_fields = []
    if "_next_view" in st.session_state:
        st.session_state.active_view = st.session_state.pop("_next_view")

    with st.sidebar:
        _model_status_panel()
        st.divider()
        _credential_form()

    view = st.segmented_control(
        "Sección",
        [CHAT_VIEW, EDITOR_VIEW],
        default=CHAT_VIEW,
        key="active_view",
        label_visibility="collapsed",
        width="stretch",
    )
    state: ConversationState = st.session_state.conversation_state
    if view == EDITOR_VIEW:
        _render_editor_view(state)
    else:
        _render_conversation_view(state)


if __name__ == "__main__":
    main()
