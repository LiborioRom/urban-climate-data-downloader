from __future__ import annotations

from schemas.request import DatasetRequest
from schemas.validation import AcquisitionPlan, PlanStep, VariablePlan
from rules.validation_rules import normalize_request, validate_request
from rules.variable_catalog import VARIABLE_CATALOG, choose_sources


def build_plan(raw_request: DatasetRequest) -> AcquisitionPlan:
    request = normalize_request(raw_request)
    report = validate_request(request)
    variable_plans: list[VariablePlan] = []
    if request.target_resolution_m is not None:
        for variable in request.variables:
            if variable not in VARIABLE_CATALOG:
                continue
            preferred = next((item.source for item in request.preferred_sources if item.variable == variable), None)
            sources = choose_sources(variable, request.target_resolution_m, preferred)
            for source in sources:
                transformations = list(source.get("transformations", []))
                native = source.get("native_resolution_m")
                if native and native != request.target_resolution_m:
                    transformations.append(f"resampling/agregación {native:g} m -> {request.target_resolution_m:g} m")
                variable_plans.append(VariablePlan(
                    variable=variable,
                    source=source["id"],
                    tool=source["tool"],
                    dataset_or_product=source.get("dataset_or_product"),
                    bands=source.get("bands", []),
                    native_resolution_m=native,
                    target_resolution_m=request.target_resolution_m,
                    transformations=transformations,
                    execution_supported=source.get("execution_supported", False),
                ))

    steps = [PlanStep(order=1, tool="validate_request", action="Validar campos, compatibilidad y reglas científicas")]
    grouped: dict[str, list[str]] = {}
    for item in variable_plans:
        grouped.setdefault(item.tool, []).append(item.variable)
    for tool, variables in grouped.items():
        steps.append(PlanStep(order=len(steps) + 1, tool=tool, action="Adquirir/procesar con la lógica existente", variables=list(dict.fromkeys(variables))))
    steps.append(PlanStep(order=len(steps) + 1, tool="run_current_notebook_pipeline", action="Ejecutar una copia temporal parametrizada del notebook"))
    steps.append(PlanStep(order=len(steps) + 1, tool="write_metadata", action="Armonizar el CSV único, evaluar calidad y registrar trazabilidad"))

    harmonization = [
        "CRS interno EPSG:25830 y coordenadas geográficas en las salidas",
        "alineamiento temporal existente por fecha/hora más cercana",
        "agregación o muestreo a la malla objetivo sin presentar resampling como nueva información",
    ]
    return AcquisitionPlan(
        request=request,
        variable_plans=variable_plans,
        steps=steps,
        issues=report.issues,
        harmonization_steps=harmonization,
    )
