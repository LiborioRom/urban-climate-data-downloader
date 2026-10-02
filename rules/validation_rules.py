from __future__ import annotations

from schemas.request import DateSelectionMode, DatasetRequest
from schemas.validation import Severity, ValidationIssue, ValidationReport

from .variable_catalog import VARIABLE_CATALOG, canonicalize_variable, choose_sources


def normalize_request(request: DatasetRequest) -> DatasetRequest:
    normalized = request.model_copy(deep=True)
    if normalized.areas and normalized.roi is None:
        normalized.roi = normalized.areas[0].model_copy(deep=True)
    if normalized.date_selection_mode == DateSelectionMode.RECURRING_WINDOW and normalized.date_windows:
        normalized.start_date = min(window.start_date for window in normalized.date_windows)
        normalized.end_date = max(window.end_date for window in normalized.date_windows)
    elif normalized.date_selection_mode == DateSelectionMode.CONTINUOUS:
        normalized.date_windows = []
    canonical = []
    for variable in request.variables:
        resolved = canonicalize_variable(variable)
        canonical.append(resolved or variable)
    normalized.variables = list(dict.fromkeys(canonical))

    preferred = []
    for item in request.preferred_sources:
        preferred.append(item.model_copy(update={"variable": canonicalize_variable(item.variable) or item.variable}))
    normalized.preferred_sources = preferred
    return normalized


def validate_request(request: DatasetRequest) -> ValidationReport:
    request = normalize_request(request)
    issues: list[ValidationIssue] = []

    def add(severity: Severity, code: str, message: str, variable: str | None = None, source: str | None = None):
        issues.append(ValidationIssue(severity=severity, code=code, message=message, variable=variable, source=source))

    if request.roi is None and not request.areas:
        add(Severity.ERROR, "MISSING_ROI", "Falta definir el ROI o área de estudio.")
    if request.areas:
        area_ids = [area.area_id for area in request.areas if area.area_id]
        if len(area_ids) != len(set(area_ids)):
            add(Severity.ERROR, "DUPLICATE_AREA_ID", "Cada ROI debe tener un identificador único.")
    if request.start_date is None:
        add(Severity.ERROR, "MISSING_START_DATE", "Falta la fecha inicial.")
    if request.end_date is None:
        add(Severity.ERROR, "MISSING_END_DATE", "Falta la fecha final.")
    if request.start_date and request.end_date and request.start_date > request.end_date:
        add(Severity.ERROR, "INVALID_DATE_RANGE", "La fecha inicial es posterior a la final.")
    if request.date_selection_mode == DateSelectionMode.RECURRING_WINDOW and not request.date_windows:
        add(Severity.ERROR, "MISSING_DATE_WINDOWS", "Falta seleccionar al menos un año para la ventana estacional.")
    if not request.variables:
        add(Severity.ERROR, "MISSING_VARIABLES", "Falta indicar al menos una variable.")
    if request.target_resolution_m is None:
        add(Severity.ERROR, "MISSING_RESOLUTION", "Falta indicar la resolución espacial objetivo.")

    for variable in request.variables:
        if variable not in VARIABLE_CATALOG:
            add(Severity.ERROR, "UNKNOWN_VARIABLE", f"La variable '{variable}' no está soportada por el catálogo actual.", variable)
            continue
        if request.target_resolution_m is None:
            continue
        preferred = next((item.source for item in request.preferred_sources if item.variable == variable), None)
        sources = choose_sources(variable, request.target_resolution_m, preferred)
        if preferred and not sources:
            add(Severity.ERROR, "INCOMPATIBLE_SOURCE", f"La fuente preferida '{preferred}' no está disponible para {variable}.", variable, preferred)
            continue
        for source in sources:
            source_id = source["id"]
            if not source.get("execution_supported", False):
                add(Severity.ERROR, "NOT_IMPLEMENTED", f"{variable} se reconoce y se asocia a {source_id}, pero su adquisición no está implementada en el notebook actual.", variable, source_id)
            native = source.get("native_resolution_m")
            if native and request.target_resolution_m < native:
                add(Severity.WARNING, "FINER_THAN_NATIVE",
                    f"{variable} procede de {source_id} (~{native:g} m). Llevarlo a {request.target_resolution_m:g} m requiere resampling y no crea observaciones nuevas a esa escala.",
                    variable, source_id)
            if source.get("data_model") == "vector":
                add(Severity.INFO, "VECTOR_RASTERIZATION", f"{variable} es vectorial en {source_id}; debe intersectarse/rasterizarse sobre las celdas objetivo.", variable, source_id)
            transformations = " ".join(source.get("transformations", [])).lower()
            if "temporal alignment" in transformations or "nearest-hour" in transformations or "nearest-date" in transformations:
                add(Severity.INFO, "TEMPORAL_ALIGNMENT", f"{variable} requiere alineamiento temporal según la lógica existente del pipeline.", variable, source_id)

    if request.crs != "EPSG:25830":
        add(Severity.INFO, "REPROJECTION", f"El pipeline trabaja internamente en EPSG:25830; se reproyectará desde/hacia {request.crs} cuando corresponda.")
    if any(any(source["id"] == "era5_land" for source in VARIABLE_CATALOG[v]["sources"])
           for v in request.variables if v in VARIABLE_CATALOG):
        add(Severity.INFO, "ERA5_REGIONAL_CONTEXT", "ERA5-Land aporta contexto meteorológico regional; la malla objetivo no cambia su soporte espacial original.", source="era5_land")
    if "sky_view_factor" in request.variables:
        add(Severity.INFO, "APPROXIMATE_SVF", "El Sky View Factor se aproxima mediante horizontes radiales sobre alturas CNIG; no equivale a un cálculo fotogramétrico 3D completo.", "sky_view_factor", "osm_cnig_derived")

    return ValidationReport(request=request, issues=issues)
