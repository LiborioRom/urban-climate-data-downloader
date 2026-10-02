from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    notebook_scope: str


TOOL_REGISTRY = {
    "get_landsat_data": ToolSpec("get_landsat_data", "Landsat 8/9, índices y estáticas", "bloques Landsat/GEE"),
    "get_sentinel3_data": ToolSpec("get_sentinel3_data", "Sentinel-3 L2 LST y agregados Sentinel-2", "bloques Sentinel-3/Sentinel-2"),
    "get_era5_land_data": ToolSpec("get_era5_land_data", "Variables ERA5-Land y alineamiento horario", "bloque ERA5-Land"),
    "get_osm_data": ToolSpec("get_osm_data", "Edificios OSM", "bloque morfología urbana"),
    "get_cnig_data": ToolSpec("get_cnig_data", "Altura de edificios CNIG WCS", "bloque morfología urbana"),
    "get_gee_data": ToolSpec("get_gee_data", "SRTM y productos Earth Engine", "bloques GEE"),
    "run_current_notebook_pipeline": ToolSpec("run_current_notebook_pipeline", "Ejecución única del notebook completo", "notebook completo"),
}


def get_registered_tool(name: str) -> ToolSpec:
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Tool no registrado: {name}")
    return TOOL_REGISTRY[name]

