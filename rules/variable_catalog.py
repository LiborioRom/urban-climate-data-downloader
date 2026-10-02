from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text.strip().lower())


def _source(
    source_id: str,
    product: str,
    bands: list[str],
    tool: str,
    *,
    resolution_m: float | None,
    transformations: list[str] | None = None,
    supported: bool = True,
    data_model: str = "raster",
) -> dict[str, Any]:
    return {
        "id": source_id,
        "dataset_or_product": product,
        "bands": bands,
        "native_resolution_m": resolution_m,
        "transformations": transformations or [],
        "tool": tool,
        "execution_supported": supported,
        "data_model": data_model,
    }


def _spec(
    display_name: str,
    category: str,
    unit: str,
    description: str,
    interpretation: str,
    aliases: list[str],
    sources: list[dict[str, Any]],
    *,
    variable_type: str,
    temporal_kind: str,
    output_columns: dict[str, list[str]],
    formula: str | None = None,
    limitations: str = "",
    calculation_cost: str = "low",
    availability: str = "available",
    recommended: bool = False,
) -> dict[str, Any]:
    return {
        "display_name": display_name,
        "category": category,
        "unit": unit,
        "short_description": description,
        "interpretation": interpretation,
        "aliases": aliases,
        "sources": sources,
        "variable_type": variable_type,
        "temporal_kind": temporal_kind,
        "output_columns": output_columns,
        "formula": formula,
        "limitations": limitations,
        "calculation_cost": calculation_cost,
        "availability": availability,
        "recommended": recommended,
    }


LANDSAT_PRODUCT = "LANDSAT/LC08/C02/T1_L2 + LANDSAT/LC09/C02/T1_L2"
S2_PRODUCT = "COPERNICUS/S2_SR_HARMONIZED"
S3_PRODUCT = "Sentinel-3 SLSTR SL_2_LST___ mediante Sentinel Hub Process API"
ERA5_PRODUCT = "reanalysis-era5-land"


def _landsat(bands: list[str], transformations: list[str] | None = None) -> dict[str, Any]:
    return _source("landsat_8_9", LANDSAT_PRODUCT, bands, "get_landsat_data", resolution_m=30,
                   transformations=transformations)


def _s2(bands: list[str], transformations: list[str] | None = None) -> dict[str, Any]:
    return _source("sentinel_2_l2a", S2_PRODUCT, bands, "get_sentinel3_data", resolution_m=20,
                   transformations=(transformations or []) + ["nearest-date temporal alignment", "aggregation to SLSTR pixel"])


def _s3(
    bands: list[str],
    transformations: list[str] | None = None,
    *,
    supported: bool = True,
) -> dict[str, Any]:
    return _source("sentinel_3_slstr_l2", S3_PRODUCT, bands, "get_sentinel3_data", resolution_m=1000,
                   transformations=(transformations or []) + ["server-side ROI subset via Sentinel Hub Process API"],
                   supported=supported)


def _era5(bands: list[str], transformations: list[str] | None = None) -> dict[str, Any]:
    return _source("era5_land", ERA5_PRODUCT, bands, "get_era5_land_data", resolution_m=9000,
                   transformations=(transformations or []) + ["nearest-hour temporal alignment"])


VARIABLE_CATALOG: dict[str, dict[str, Any]] = {}


def _add(key: str, **kwargs: Any) -> None:
    VARIABLE_CATALOG[key] = _spec(**kwargs)


# Núcleo existente.
_add("LST", display_name="Temperatura superficial terrestre", category="Temperatura y calidad", unit="K",
     description="Temperatura radiométrica de la superficie terrestre.",
     interpretation="Los valores altos indican superficies más calientes en el instante del paso del satélite.",
     aliases=["lst", "temperatura superficial", "temperatura de superficie", "land surface temperature"],
     sources=[_landsat(["ST_B10"], ["DN * 0.00341802 + 149.0", "QA_PIXEL mask"]), _s3(["LST"], ["cloud and quality mask"])],
     variable_type="direct_satellite_product", temporal_kind="dynamic",
     output_columns={"landsat": ["LST_K"], "sentinel3": ["LST_1km"]}, recommended=True)

for key, display, aliases, formula, bands_l8, bands_s2, output in [
    ("NDVI", "Índice de vegetación normalizado", ["ndvi", "indice de vegetacion", "índice de vegetación"],
     "(NIR - RED) / (NIR + RED)", ["SR_B5", "SR_B4"], ["B8", "B4"], "NDVI"),
    ("NDBI", "Índice normalizado de edificación", ["ndbi", "indice de edificacion", "índice de edificación"],
     "(SWIR1 - NIR) / (SWIR1 + NIR)", ["SR_B6", "SR_B5"], ["B11", "B8"], "NDBI"),
    ("ALBEDO", "Albedo de banda ancha aproximado", ["albedo", "reflectividad superficial"],
     "mean(BLUE, GREEN, RED, NIR)", ["SR_B2", "SR_B3", "SR_B4", "SR_B5"], ["B2", "B3", "B4", "B8"], "ALBEDO"),
]:
    _add(key, display_name=display, category="Vegetación y superficie", unit="1",
         description={"NDVI": "Contraste espectral asociado a vegetación verde.", "NDBI": "Contraste espectral sensible a superficies construidas.", "ALBEDO": "Fracción aproximada de radiación solar reflejada."}[key],
         interpretation={"NDVI": "Valores positivos altos suelen indicar vegetación densa.", "NDBI": "Valores altos suelen asociarse a suelo construido o seco.", "ALBEDO": "Valores altos indican superficies más reflectantes."}[key],
         aliases=aliases, sources=[_landsat(bands_l8, [formula]), _s2(bands_s2, [formula])],
         variable_type="derived_index", temporal_kind="dynamic",
         output_columns={"landsat": [output], "sentinel3": [output]}, formula=formula, recommended=True)


# Reflectancias de superficie disponibles en Landsat L2 y Sentinel-2 L2A.
REFLECTANCE_SPECS = {
    "reflectance_coastal": ("Reflectancia costera/aerosoles", "SR_B1", "B1", "SR_coastal", "coastal aerosol reflectance"),
    "reflectance_blue": ("Reflectancia azul", "SR_B2", "B2", "SR_blue", "blue reflectance"),
    "reflectance_green": ("Reflectancia verde", "SR_B3", "B3", "SR_green", "green reflectance"),
    "reflectance_red": ("Reflectancia roja", "SR_B4", "B4", "SR_red", "red reflectance"),
    "reflectance_nir": ("Reflectancia infrarrojo cercano", "SR_B5", "B8", "SR_nir", "nir reflectance"),
    "reflectance_swir1": ("Reflectancia SWIR 1", "SR_B6", "B11", "SR_swir1", "swir 1 reflectance"),
    "reflectance_swir2": ("Reflectancia SWIR 2", "SR_B7", "B12", "SR_swir2", "swir 2 reflectance"),
}
for key, (display, l8_band, s2_band, column, alias) in REFLECTANCE_SPECS.items():
    _add(key, display_name=display, category="Bandas espectrales", unit="1",
         description="Reflectancia superficial corregida atmosféricamente.",
         interpretation="Representa la proporción de energía reflejada en esta región espectral.",
         aliases=[alias, column], sources=[_landsat([l8_band], ["surface reflectance scale and offset"]), _s2([s2_band], ["DN / 10000"])],
         variable_type="direct_reflectance", temporal_kind="dynamic",
         output_columns={"landsat": [column], "sentinel3": [column]}, limitations="La resolución nativa depende de la banda y el sensor.")

for number, band, wavelength in [(1, "B5", "703 nm"), (2, "B6", "740 nm"), (3, "B7", "783 nm"), (4, "B8A", "865 nm")]:
    key, column = f"reflectance_red_edge_{number}", f"SR_red_edge_{number}"
    _add(key, display_name=f"Reflectancia red-edge {number}", category="Bandas espectrales", unit="1",
         description=f"Reflectancia Sentinel-2 en la región red-edge ({wavelength}).",
         interpretation="Es sensible a clorofila, estructura y estado de la vegetación.",
         aliases=[f"red edge {number}", column], sources=[_s2([band], ["DN / 10000"])],
         variable_type="direct_reflectance", temporal_kind="dynamic", output_columns={"sentinel3": [column]})


# Índices espectrales adicionales.
INDEX_SPECS = {
    "NDWI": ("Índice de agua NDWI", "(GREEN - NIR) / (GREEN + NIR)", ["SR_B3", "SR_B5"], ["B3", "B8"], "Valores altos suelen indicar agua o superficies húmedas."),
    "MNDWI": ("Índice de agua modificado", "(GREEN - SWIR1) / (GREEN + SWIR1)", ["SR_B3", "SR_B6"], ["B3", "B11"], "Realza masas de agua y reduce parte de la respuesta urbana."),
    "NDMI": ("Índice de humedad normalizado", "(NIR - SWIR1) / (NIR + SWIR1)", ["SR_B5", "SR_B6"], ["B8", "B11"], "Valores altos suelen indicar vegetación o suelo con más humedad."),
    "EVI": ("Índice de vegetación mejorado", "2.5*(NIR-RED)/(NIR+6*RED-7.5*BLUE+1)", ["SR_B5", "SR_B4", "SR_B2"], ["B8", "B4", "B2"], "Aumenta la sensibilidad en vegetación densa y corrige parcialmente suelo y atmósfera."),
    "SAVI": ("Índice de vegetación ajustado al suelo", "1.5*(NIR-RED)/(NIR+RED+0.5)", ["SR_B5", "SR_B4"], ["B8", "B4"], "Reduce la influencia del fondo del suelo en vegetación dispersa."),
    "BSI": ("Índice de suelo desnudo", "((SWIR1+RED)-(NIR+BLUE))/((SWIR1+RED)+(NIR+BLUE))", ["SR_B6", "SR_B4", "SR_B5", "SR_B2"], ["B11", "B4", "B8", "B2"], "Valores altos suelen indicar suelo desnudo o seco."),
    "urban_index": ("Índice urbano espectral", "(SWIR2-NIR)/(SWIR2+NIR)", ["SR_B7", "SR_B5"], ["B12", "B8"], "Valores altos pueden asociarse a superficies urbanas o secas."),
}
for key, (display, formula, l8bands, s2bands, interpretation) in INDEX_SPECS.items():
    _add(key, display_name=display, category="Vegetación y superficie", unit="1",
         description="Índice espectral derivado de reflectancias superficiales.", interpretation=interpretation,
         aliases=[key.lower(), display.lower()], sources=[_landsat(l8bands, [formula]), _s2(s2bands, [formula])],
         variable_type="derived_index", temporal_kind="dynamic", output_columns={"landsat": [key], "sentinel3": [key]},
         formula=formula, recommended=key in {"NDWI", "NDMI", "BSI"})


# Variables auxiliares Landsat L2.
for key, display, band, column, unit, description, interpretation in [
    ("surface_emissivity", "Emisividad superficial", "ST_EMIS", "surface_emissivity", "1", "Emisividad usada en la recuperación de LST.", "Valores próximos a 1 corresponden a emisores térmicos eficientes."),
    ("emissivity_uncertainty", "Incertidumbre de emisividad", "ST_EMSD", "emissivity_uncertainty", "1", "Desviación estándar estimada de la emisividad.", "Valores bajos indican mayor confianza en la emisividad."),
    ("atmospheric_transmittance", "Transmitancia atmosférica", "ST_ATRAN", "atmospheric_transmittance", "1", "Fracción de radiación térmica transmitida por la atmósfera.", "Valores altos representan una atmósfera más transparente."),
    ("cloud_distance", "Distancia a nubes", "ST_CDIST", "cloud_distance_km", "km", "Distancia al píxel nuboso más cercano.", "Distancias pequeñas pueden señalar mayor riesgo de contaminación térmica por nubes."),
    ("LST_uncertainty", "Incertidumbre de LST Landsat", "ST_QA", "LST_uncertainty_K", "K", "Incertidumbre estimada de la temperatura superficial.", "Valores bajos indican una observación térmica más fiable."),
]:
    _add(key, display_name=display, category="Temperatura y calidad", unit=unit,
         description=description, interpretation=interpretation, aliases=[display.lower(), column],
         sources=[_landsat([band], ["official scale factor"])], variable_type="direct_quality",
         temporal_kind="dynamic", output_columns={"landsat": [column]}, recommended=key in {"surface_emissivity", "LST_uncertainty"})


# Sentinel-2 L2A auxiliares.
for key, display, band, column, unit, description, interpretation in [
    ("aerosol_optical_thickness", "Espesor óptico de aerosoles", "AOT", "AOT", "1", "Carga óptica de aerosoles estimada por Sen2Cor.", "Valores altos indican mayor presencia de aerosoles en la columna atmosférica."),
    ("water_vapour", "Vapor de agua Sentinel-2", "WVP", "water_vapour_cm", "cm", "Altura equivalente del vapor de agua condensado.", "Valores altos indican una columna atmosférica más húmeda."),
]:
    _add(key, display_name=display, category="Atmósfera satelital", unit=unit, description=description,
         interpretation=interpretation, aliases=[display.lower(), column], sources=[_s2([band], ["official scale factor"])],
         variable_type="direct_satellite_product", temporal_kind="dynamic", output_columns={"sentinel3": [column]})


# ERA5-Land: parámetros disponibles en el producto horario y derivados físicos.
ERA5_SPECS = [
    ("air_temperature", "Temperatura del aire", ["temperatura del aire", "tair", "tair_c"], "degC", ["2m_temperature"], "Tair_C", "Temperatura del aire a dos metros.", "Valores altos indican aire más cálido."),
    ("dewpoint_temperature", "Temperatura de rocío", ["punto de rocio", "temperatura de rocío", "dewpoint"], "degC", ["2m_dewpoint_temperature"], "dewpoint_C", "Temperatura a la que el aire alcanzaría saturación.", "Cuanto más se aproxima a Tair, más húmedo está el aire."),
    ("relative_humidity", "Humedad relativa", ["humedad relativa", "rh"], "%", ["2m_temperature", "2m_dewpoint_temperature"], "RH", "Proporción de humedad respecto a la saturación.", "Valores altos indican aire próximo a saturación."),
    ("solar_radiation", "Radiación solar descendente", ["radiacion solar", "rsol", "rsol_wm2"], "W/m²", ["surface_solar_radiation_downwards"], "Rsol_Wm2", "Radiación solar que alcanza la superficie.", "Valores altos indican mayor aporte radiativo solar."),
    ("thermal_radiation_down", "Radiación térmica descendente", ["radiacion termica descendente", "longwave down"], "W/m²", ["surface_thermal_radiation_downwards"], "Rthermal_Wm2", "Radiación infrarroja emitida hacia la superficie por atmósfera y nubes.", "Valores altos reducen el enfriamiento radiativo de la superficie."),
    ("wind_u10", "Componente este-oeste del viento", ["viento u", "u10"], "m/s", ["10m_u_component_of_wind"], "wind_u10", "Componente zonal del viento a 10 m.", "Positiva hacia el este y negativa hacia el oeste."),
    ("wind_v10", "Componente norte-sur del viento", ["viento v", "v10"], "m/s", ["10m_v_component_of_wind"], "wind_v10", "Componente meridional del viento a 10 m.", "Positiva hacia el norte y negativa hacia el sur."),
    ("wind_speed", "Velocidad del viento", ["velocidad del viento", "wind speed", "viento"], "m/s", ["10m_u_component_of_wind", "10m_v_component_of_wind"], "wind_speed", "Módulo del viento horizontal a 10 m.", "Valores altos indican mayor ventilación regional."),
    ("wind_direction", "Dirección del viento", ["direccion del viento", "dirección del viento"], "degree", ["10m_u_component_of_wind", "10m_v_component_of_wind"], "wind_direction_deg", "Dirección meteorológica de procedencia del viento.", "0° es norte, 90° este, 180° sur y 270° oeste."),
    ("rain_1d", "Precipitación diaria", ["lluvia diaria", "rain 1d"], "mm", ["total_precipitation"], "rain_1d", "Precipitación acumulada durante el día.", "Valores altos indican mayor aporte reciente de agua."),
    ("rain_3d", "Precipitación acumulada 3 días", ["lluvia acumulada", "precipitacion", "rain 3d"], "mm", ["total_precipitation"], "rain_3d", "Precipitación acumulada en tres días.", "Representa humedad antecedente a corto plazo."),
    ("rain_3d_log", "Precipitación acumulada 3 días logarítmica", ["rain 3d log", "lluvia logarítmica", "precipitacion logaritmica"], "1", ["total_precipitation"], "rain_3d_log", "Transformación logarítmica de la precipitación acumulada durante tres días.", "Reduce la asimetría y la influencia de episodios de lluvia extremos."),
    ("is_rainy", "Indicador de lluvia reciente", ["is rainy", "día lluvioso", "indicador lluvia"], "0/1", ["total_precipitation"], "is_rainy", "Indicador binario calculado a partir de la precipitación acumulada durante tres días.", "Vale 1 si rain_3d es mayor que cero y 0 en otro caso."),
    ("skin_temperature", "Temperatura de piel ERA5-Land", ["temperatura de piel", "skin temperature"], "degC", ["skin_temperature"], "skin_temperature_C", "Temperatura teórica de la interfaz superficial del modelo.", "Aporta contexto térmico regional, no una medición urbana a 100 m."),
    ("surface_pressure", "Presión superficial", ["presion superficial", "surface pressure"], "hPa", ["surface_pressure"], "surface_pressure_hPa", "Presión atmosférica en la superficie del modelo.", "Disminuye generalmente con la altitud y cambia con la situación meteorológica."),
    ("snow_cover", "Cobertura de nieve", ["cobertura de nieve", "snow cover"], "%", ["snow_cover"], "snow_cover_pct", "Fracción de la celda cubierta por nieve.", "Cerca de 100 indica cobertura nival extensa."),
    ("snow_depth", "Espesor de nieve", ["espesor de nieve", "snow depth"], "m", ["snow_depth"], "snow_depth_m", "Espesor medio de nieve en la celda.", "Valores altos indican mayor almacenamiento nival."),
]
for key, display, aliases, unit, bands, column, description, interpretation in ERA5_SPECS:
    formula = None
    if key == "relative_humidity": formula = "Magnus formula from t2m and d2m"
    elif key == "wind_speed": formula = "sqrt(u10² + v10²)"
    elif key == "wind_direction": formula = "meteorological direction from atan2(u10, v10)"
    elif key == "rain_3d_log": formula = "ln(rain_3d + 1)"
    elif key == "is_rainy": formula = "1 if rain_3d > 0 else 0"
    _add(key, display_name=display, category="Meteorología ERA5-Land", unit=unit,
         description=description, interpretation=interpretation, aliases=aliases,
         sources=[_era5(bands, [formula] if formula else [])], variable_type="derived_meteorological" if formula else "direct_meteorological",
         temporal_kind="dynamic", output_columns={"landsat": [column], "sentinel3": [column]}, formula=formula,
         limitations="ERA5-Land conserva un soporte espacial aproximado de 9 km aunque se asigne a nodos más finos.",
         recommended=key in {"air_temperature", "relative_humidity", "solar_radiation", "wind_speed", "rain_3d"})

for level, depth in [(1, "0–7 cm"), (2, "7–28 cm"), (3, "28–100 cm"), (4, "100–289 cm")]:
    for family, display_family, band_prefix, column_prefix, unit, explanation in [
        ("soil_temperature", "Temperatura del suelo", "soil_temperature_level", "soil_temperature", "degC", "temperatura del suelo"),
        ("soil_moisture", "Humedad volumétrica del suelo", "volumetric_soil_water_layer", "soil_moisture", "m³/m³", "contenido volumétrico de agua"),
    ]:
        key, band, column = f"{family}_level_{level}", f"{band_prefix}_{level}", f"{column_prefix}_l{level}"
        _add(key, display_name=f"{display_family} {depth}", category="Meteorología ERA5-Land", unit=unit,
             description=f"{explanation.capitalize()} en la capa {depth} de ERA5-Land.",
             interpretation="Valores altos representan un suelo más cálido." if family == "soil_temperature" else "Valores altos representan mayor contenido de agua en el suelo.",
             aliases=[key.replace("_", " "), f"{display_family.lower()} {depth}"], sources=[_era5([band])],
             variable_type="direct_meteorological", temporal_kind="dynamic",
             output_columns={"landsat": [column], "sentinel3": [column]},
             limitations="Representa la celda regional de ERA5-Land, no variación urbana a 100 m.")

_add("vapour_pressure_deficit", display_name="Déficit de presión de vapor", category="Meteorología ERA5-Land", unit="kPa",
     description="Diferencia entre la presión de vapor de saturación y la real.",
     interpretation="Valores altos indican aire seco y mayor demanda evaporativa.",
     aliases=["vpd", "deficit de presion de vapor", "déficit de presión de vapor"],
     sources=[_era5(["2m_temperature", "2m_dewpoint_temperature"], ["saturation vapour pressure difference"])],
     variable_type="derived_meteorological", temporal_kind="dynamic",
     output_columns={"landsat": ["VPD_kPa"], "sentinel3": ["VPD_kPa"]},
     formula="es(Tair) - es(Tdew)", limitations="Conserva el soporte espacial aproximado de 9 km de ERA5-Land.", recommended=True)


# Resto de campos científicos horarios de ERA5-Land útiles como predictores.
# Se excluyen únicamente identificadores y metadatos administrativos del fichero.
ERA5_EXTRA_SPECS = [
    ("lake_bottom_temperature", "Temperatura del fondo del lago", "lake_bottom_temperature", "lake_bottom_temperature_C", "degC", "Temperatura del estrato inferior de masas de agua interiores.", "Describe el estado térmico profundo del lago."),
    ("lake_ice_depth", "Espesor de hielo lacustre", "lake_ice_depth", "lake_ice_depth_m", "m", "Espesor modelado del hielo sobre lagos.", "Valores altos indican una cubierta de hielo más gruesa."),
    ("lake_ice_temperature", "Temperatura del hielo lacustre", "lake_ice_temperature", "lake_ice_temperature_C", "degC", "Temperatura modelada de la capa de hielo de lagos.", "Caracteriza el estado térmico del hielo."),
    ("lake_mix_layer_depth", "Profundidad de mezcla lacustre", "lake_mix_layer_depth", "lake_mix_layer_depth_m", "m", "Profundidad de la capa superior bien mezclada del lago.", "Valores altos indican mezcla hasta mayor profundidad."),
    ("lake_mix_layer_temperature", "Temperatura de mezcla lacustre", "lake_mix_layer_temperature", "lake_mix_layer_temperature_C", "degC", "Temperatura de la capa superior mezclada del lago.", "Representa la temperatura del agua superficial modelada."),
    ("lake_shape_factor", "Factor de forma lacustre", "lake_shape_factor", "lake_shape_factor", "1", "Parámetro geométrico del perfil térmico vertical del lago.", "Modula la distribución de temperatura con la profundidad."),
    ("lake_total_layer_temperature", "Temperatura total del lago", "lake_total_layer_temperature", "lake_total_layer_temperature_C", "degC", "Temperatura media de la columna de agua interior.", "Resume el estado térmico global del lago."),
    ("leaf_area_index_high", "LAI de vegetación alta", "leaf_area_index_high_vegetation", "LAI_high", "m²/m²", "Índice de área foliar de la vegetación alta.", "Valores altos representan más superficie de hojas por suelo."),
    ("leaf_area_index_low", "LAI de vegetación baja", "leaf_area_index_low_vegetation", "LAI_low", "m²/m²", "Índice de área foliar de la vegetación baja.", "Valores altos representan mayor densidad foliar baja."),
    ("potential_evaporation", "Evaporación potencial", "potential_evaporation", "potential_evaporation_mm", "mm/h", "Evaporación potencial acumulada durante la hora.", "Aproxima la demanda evaporativa sin limitación de agua."),
    ("runoff", "Escorrentía total", "runoff", "runoff_mm", "mm/h", "Suma de escorrentía superficial y subsuperficial.", "Valores altos indican mayor drenaje de agua desde el suelo."),
    ("skin_reservoir_content", "Agua interceptada superficial", "skin_reservoir_content", "skin_reservoir_mm", "mm", "Agua almacenada sobre vegetación y la capa superficial fina.", "Representa lluvia interceptada y rocío disponible."),
    ("snow_albedo", "Albedo de nieve", "snow_albedo", "snow_albedo", "1", "Fracción de radiación solar reflejada por la nieve.", "Valores altos indican nieve muy reflectante."),
    ("snow_density", "Densidad de nieve", "snow_density", "snow_density_kgm3", "kg/m³", "Masa de nieve por unidad de volumen.", "Valores altos suelen indicar nieve más compactada."),
    ("snow_evaporation", "Evaporación/sublimación de nieve", "snow_evaporation", "snow_evaporation_mm", "mm/h", "Pérdida horaria de agua equivalente desde la nieve.", "Cuantifica sublimación o evaporación del manto."),
    ("snowfall", "Nevada", "snowfall", "snowfall_mm", "mm/h", "Agua equivalente de nieve caída durante la hora.", "Valores altos indican mayor aporte nival."),
    ("snowmelt", "Fusión de nieve", "snowmelt", "snowmelt_mm", "mm/h", "Agua equivalente producida por fusión del manto.", "Valores altos indican deshielo intenso."),
    ("snow_temperature", "Temperatura de la capa de nieve", "temperature_of_snow_layer", "snow_temperature_C", "degC", "Temperatura modelada del manto nival.", "Valores próximos a 0 °C favorecen la fusión."),
    ("total_evaporation", "Evaporación total", "total_evaporation", "total_evaporation_mm", "mm/h", "Flujo horario total de agua entre superficie y atmósfera.", "Por la convención ECMWF, la evaporación suele tener signo negativo."),
    ("evaporation_bare_soil", "Evaporación de suelo desnudo", "evaporation_from_bare_soil", "evaporation_bare_soil_mm", "mm/h", "Componente de evaporación procedente del suelo sin vegetación.", "Su magnitud indica pérdida de agua desde suelo expuesto."),
    ("evaporation_open_water", "Evaporación de aguas interiores", "evaporation_from_open_water_surfaces_excluding_oceans", "evaporation_open_water_mm", "mm/h", "Evaporación desde lagos y otras aguas no oceánicas.", "Su magnitud indica pérdida de agua desde superficies acuáticas."),
    ("evaporation_canopy", "Evaporación del dosel", "evaporation_from_the_top_of_canopy", "evaporation_canopy_mm", "mm/h", "Evaporación del agua interceptada por la vegetación.", "Su magnitud refleja el secado del dosel vegetal."),
    ("vegetation_transpiration", "Transpiración vegetal", "evaporation_from_vegetation_transpiration", "vegetation_transpiration_mm", "mm/h", "Agua extraída del suelo y transpirada por la vegetación.", "Su magnitud aproxima actividad evapotranspirativa vegetal."),
    ("subsurface_runoff", "Escorrentía subsuperficial", "sub_surface_runoff", "subsurface_runoff_mm", "mm/h", "Drenaje horario de agua por debajo de la superficie.", "Valores altos indican mayor descarga profunda del suelo."),
    ("surface_runoff", "Escorrentía superficial", "surface_runoff", "surface_runoff_mm", "mm/h", "Agua que escurre sobre la superficie durante la hora.", "Valores altos pueden asociarse a saturación o impermeabilidad regional."),
    ("surface_latent_heat_flux", "Flujo de calor latente", "surface_latent_heat_flux", "surface_latent_heat_flux_Wm2", "W/m²", "Intercambio energético asociado a cambios de fase del agua.", "Su signo sigue la convención descendente positiva de ECMWF."),
    ("surface_net_solar_radiation", "Radiación solar neta", "surface_net_solar_radiation", "surface_net_solar_Wm2", "W/m²", "Radiación solar absorbida por la superficie tras restar la reflejada.", "Valores altos aportan más energía neta de onda corta."),
    ("surface_net_thermal_radiation", "Radiación térmica neta", "surface_net_thermal_radiation", "surface_net_thermal_Wm2", "W/m²", "Balance neto de radiación térmica en la superficie.", "Su signo indica la dirección neta según la convención ECMWF."),
    ("surface_sensible_heat_flux", "Flujo de calor sensible", "surface_sensible_heat_flux", "surface_sensible_heat_flux_Wm2", "W/m²", "Intercambio turbulento de calor sin cambio de fase.", "Relaciona el contraste térmico superficie-aire con la turbulencia."),
]
for key, display, band, column, unit, description, interpretation in ERA5_EXTRA_SPECS:
    _add(key, display_name=display, category="Meteorología ERA5-Land", unit=unit,
         description=description, interpretation=interpretation,
         aliases=[key.replace("_", " "), display.lower(), column], sources=[_era5([band])],
         variable_type="direct_meteorological", temporal_kind="dynamic",
         output_columns={"landsat": [column], "sentinel3": [column]},
         limitations="Mantiene el soporte espacial aproximado de 9 km; los acumulados corresponden a la hora del producto.")


# Sentinel-3 SLSTR L2 LST y campos auxiliares oficiales.
S3_SPECS = [
    ("sentinel3_lst_uncertainty", "Incertidumbre LST Sentinel-3", ["LST_uncertainty"], "LST_uncertainty_1km", "K", "Incertidumbre total de la LST SLSTR.", "Valores bajos indican mayor confianza.", True),
    ("sentinel3_ndvi", "NDVI auxiliar Sentinel-3", ["NDVI"], "NDVI_S3", "1", "NDVI incluido en el producto auxiliar SLSTR.", "Valores altos suelen indicar vegetación densa.", True),
    ("sentinel3_vegetation_fraction", "Fracción vegetal Sentinel-3", ["fraction"], "vegetation_fraction_1km", "1", "Fracción de cobertura vegetal usada por el algoritmo LST.", "Cero indica ausencia y uno cobertura vegetal completa.", False),
    ("sentinel3_tcwv", "Vapor de agua total Sentinel-3", ["TCWV"], "TCWV_1km", "kg/m²", "Vapor de agua total de la columna atmosférica.", "Valores altos representan una columna más húmeda.", False),
    ("sentinel3_biome", "Bioma Sentinel-3", ["biome"], "biome_class_1km", "class", "Clase de cubierta usada en la recuperación de LST.", "Es una categoría, no una magnitud continua.", False),
    ("sentinel3_quality", "Indicador de calidad Sentinel-3", ["CLOUD", "BAYES", "POINTING", "CONFIDENCE"], "quality_flag_1km", "flag", "Indicadores oficiales de nube, geometría y confianza.", "Los píxeles con summary_cloud se descartan; el valor conservado corresponde a CONFIDENCE.", True),
]
for key, display, bands, column, unit, description, interpretation, supported in S3_SPECS:
    _add(key, display_name=display, category="Sentinel-3 y calidad", unit=unit, description=description,
         interpretation=interpretation, aliases=[display.lower(), column], sources=[_s3(bands, supported=supported)],
         variable_type="direct_satellite_product", temporal_kind="dynamic", output_columns={"sentinel3": [column]},
         recommended=key == "sentinel3_lst_uncertainty",
         availability="available" if supported else "not_exposed_by_process_api",
         limitations="Requiere producto completo; la ruta eficiente Process API no publica esta banda." if not supported else "")

for suffix, display in [
    ("random", "aleatoria"), ("locT", "calibración local"), ("locATM", "atmosférica local"),
    ("locSF", "emisividad superficial"), ("locGEO", "geolocalización"), ("sys", "sistemática"),
]:
    key, column = f"sentinel3_uncertainty_{suffix.lower()}", f"LST_uncertainty_{suffix}_1km"
    _add(key, display_name=f"Incertidumbre SLSTR {display}", category="Sentinel-3 y calidad", unit="K",
         description=f"Componente {display} de la incertidumbre de LST Sentinel-3.",
         interpretation="Valores bajos indican una contribución menor a la incertidumbre total.",
         aliases=[key.replace("_", " "), column], sources=[_s3([f"LST_uncertainty_{suffix}"], supported=False)],
         variable_type="direct_quality", temporal_kind="dynamic", output_columns={"sentinel3": [column]},
         availability="not_exposed_by_process_api",
         limitations="La Process API publica la incertidumbre total, pero no esta componente separada.")


# Variables estáticas existentes y SVF aproximado.
STATIC_SPECS = [
    ("elevation", "Elevación", ["elevacion", "elevación", "dem", "altitud"], "m", "DEM", "Altura del terreno sobre el nivel de referencia.", "Valores altos representan terreno más elevado."),
    ("buildings", "Presencia de edificios", ["edificios", "buildings", "morfologia urbana"], "0/1", "has_buildings", "Indica si la celda contiene huellas de edificios.", "Uno indica presencia de al menos un edificio."),
    ("building_height", "Altura media de edificios", ["altura de edificios", "building height"], "m", "building_height_m", "Altura media del MDSnE2,5 del CNIG dentro de la celda.", "Valores altos representan un tejido construido más elevado."),
    ("building_footprint", "Superficie edificada", ["huella de edificios", "superficie edificada"], "m²", "bld_footprint_m2", "Área de huellas OSM que intersecta la celda.", "Valores altos indican más suelo ocupado por edificios."),
    ("aspect_ratio", "Relación altura-anchura urbana", ["aspect ratio", "aspect_ratio", "cañon urbano"], "1", "aspect_ratio", "Proxy de altura de edificio dividida por anchura libre.", "Valores altos representan cañones urbanos más cerrados."),
    ("sky_view_factor", "Factor de visión del cielo", ["sky view factor", "svf", "factor de vision del cielo"], "0–1", "sky_view_factor", "Fracción aproximada del hemisferio celeste visible desde el nodo.", "Uno representa cielo abierto; valores bajos indican mayor obstrucción por edificios."),
]
for key, display, aliases, unit, column, description, interpretation in STATIC_SPECS:
    if key == "elevation":
        sources = [_source("srtm", "USGS/SRTMGL1_003", ["elevation"], "get_gee_data", resolution_m=30)]
    elif key == "buildings":
        # Conserva el identificador histórico usado por el plan y las pruebas;
        # la implementación combina el MDSnE2,5 con huellas OSM cuando existen.
        sources = [_source("cnig_wcs", "CNIG MDSnE2,5 + OSM building footprints", [], "get_osm_data", resolution_m=2.5,
                           transformations=["EPSG:25830 cell aggregation", "static cache"], data_model="mixed")]
    else:
        sources = [_source("osm_cnig_derived", "OSM building footprints + CNIG MDSnE2,5", [], "get_osm_data", resolution_m=2.5,
                           transformations=["EPSG:25830 cell aggregation", "static cache"], data_model="mixed")]
    _add(key, display_name=display, category="Morfología urbana", unit=unit, description=description,
         interpretation=interpretation, aliases=aliases, sources=sources,
         variable_type="derived_urban" if key != "elevation" else "direct_raster", temporal_kind="static",
         output_columns={"landsat": [column], "sentinel3": [column]},
         formula="mean(cos(horizon_angle)^2) over azimuths" if key == "sky_view_factor" else None,
         limitations="El SVF es una aproximación basada en alturas CNIG y muestreo radial." if key == "sky_view_factor" else "",
         calculation_cost="high" if key == "sky_view_factor" else "medium", recommended=key in {"elevation", "aspect_ratio", "sky_view_factor"})


# Variables urbanas estudiadas pero aún no ejecutables: el bot puede explicarlas y la UI las muestra como planificadas.
PLANNED_URBAN = {
    "building_coverage_ratio": ("Fracción de suelo edificada", "Área de huellas de edificios dividida por el área de la celda.", "Valores altos indican mayor ocupación del suelo."),
    "building_count": ("Número de edificios", "Edificios que intersectan la celda.", "Valores altos indican un parcelario más fragmentado o denso."),
    "building_height_std": ("Variabilidad de alturas", "Desviación estándar de las alturas de edificios.", "Valores altos indican un perfil urbano vertical heterogéneo."),
    "building_height_p90": ("Percentil 90 de altura", "Altura por debajo de la cual queda el 90 % del raster edificado.", "Representa edificios altos sin depender del máximo absoluto."),
    "building_volume_density": ("Densidad de volumen construido", "Suma aproximada de huella por altura dividida por el área.", "Valores altos indican más masa construida."),
    "facade_density": ("Densidad de fachada", "Perímetro edificado por unidad de superficie.", "Valores altos indican más interfaz edificio-calle."),
    "floor_area_ratio": ("Floor Area Ratio", "Superficie estimada de plantas dividida por la superficie de suelo.", "Valores altos indican mayor intensidad edificatoria."),
    "shadow_fraction": ("Fracción de sombra", "Proporción de suelo sombreado para una fecha y hora.", "Valores altos implican menor exposición solar directa."),
    "solar_exposure_fraction": ("Fracción de exposición solar", "Proporción de suelo visible desde el Sol.", "Valores altos indican mayor insolación potencial."),
    "frontal_area_index": ("Índice de área frontal", "Área de fachada enfrentada al viento dividida por el área de suelo.", "Valores altos implican más resistencia aerodinámica."),
    "urban_roughness_proxy": ("Rugosidad urbana", "Proxy derivado de altura, densidad y variabilidad edificatoria.", "Valores altos indican un tejido con mayor fricción aerodinámica."),
    "street_orientation": ("Orientación dominante de calles", "Dirección principal de la red viaria dentro de la celda.", "Permite relacionar ventilación y radiación con la trama urbana."),
    "road_length_density": ("Densidad de red viaria", "Longitud total de vías por unidad de superficie.", "Valores altos indican una malla viaria más densa."),
    "road_area_fraction": ("Fracción ocupada por calzadas", "Área estimada de vías dividida por el área de la celda.", "Aproxima superficie viaria e impermeable."),
    "major_road_distance": ("Distancia a vía principal", "Distancia desde el nodo a la carretera principal más cercana.", "Valores bajos indican mayor proximidad a corredores viarios."),
    "intersection_density": ("Densidad de intersecciones", "Número de cruces de la red por unidad de superficie.", "Valores altos indican mayor conectividad urbana."),
    "mean_street_width": ("Anchura media de calle", "Anchura media procedente de etiquetas OSM o estimación geométrica.", "Calles anchas suelen favorecer exposición solar y ventilación."),
    "paved_road_fraction": ("Fracción viaria pavimentada", "Proporción de vías con superficie pavimentada.", "Aproxima presencia de materiales impermeables."),
    "green_space_fraction": ("Fracción de zonas verdes OSM", "Área de parques, jardines y coberturas verdes cartografiadas.", "Valores altos indican mayor presencia de infraestructura verde."),
    "residential_fraction": ("Fracción residencial", "Proporción de suelo OSM clasificado como residencial.", "Describe la función urbana dominante."),
    "commercial_fraction": ("Fracción comercial", "Proporción de suelo OSM clasificado como comercial.", "Describe intensidad de uso comercial."),
    "industrial_fraction": ("Fracción industrial", "Proporción de suelo OSM clasificado como industrial.", "Valores altos suelen asociarse a grandes superficies impermeables."),
}
for key, (display, description, interpretation) in PLANNED_URBAN.items():
    _add(key, display_name=display, category="Urbanas planificadas", unit="various", description=description,
         interpretation=interpretation, aliases=[key.replace("_", " "), display.lower()],
         sources=[_source("openstreetmap_cnig_planned", "OpenStreetMap + CNIG MDSnE2,5", [], "get_osm_data",
                          resolution_m=None, transformations=["planned geometric derivation"], supported=False, data_model="vector")],
         variable_type="derived_urban", temporal_kind="static", output_columns={}, availability="planned",
         calculation_cost="high", limitations="Requiere un módulo geométrico adicional y validación de completitud OSM.")

# Alias histórico de carreteras.
_add("roads", display_name="Red viaria", category="Urbanas planificadas", unit="vector",
     description="Geometrías de calles y carreteras de OpenStreetMap.",
     interpretation="Es una fuente intermedia para obtener densidad, orientación, anchura y proximidad viaria.",
     aliases=["carreteras", "calles", "roads", "red viaria"],
     sources=[_source("openstreetmap", "OSM highway=*", [], "get_osm_data", resolution_m=None,
                      transformations=["rasterization and network metrics"], supported=False, data_model="vector")],
     variable_type="vector", temporal_kind="static", output_columns={}, availability="planned", calculation_cost="medium")


_ALIASES = {
    _normalize(alias): canonical
    for canonical, spec in VARIABLE_CATALOG.items()
    for alias in [canonical, spec["display_name"], *spec["aliases"]]
}


def canonicalize_variable(name: str) -> str | None:
    return _ALIASES.get(_normalize(name))


def selectable_variables() -> list[str]:
    return [name for name, spec in VARIABLE_CATALOG.items() if spec["availability"] == "available"]


def variables_by_category(*, include_planned: bool = False) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name, spec in VARIABLE_CATALOG.items():
        if include_planned or spec["availability"] == "available":
            grouped[spec["category"]].append(name)
    return dict(grouped)


def variable_label(name: str, *, include_category: bool = False) -> str:
    spec = VARIABLE_CATALOG[name]
    label = f"{spec['display_name']} [{spec['unit']}]"
    return f"{spec['category']} · {label}" if include_category else label


def output_columns_for(variables: list[str], stream: str) -> list[str]:
    columns: list[str] = []
    for variable in variables:
        spec = VARIABLE_CATALOG.get(variable)
        if spec:
            columns.extend(spec["output_columns"].get(stream, []))
    return list(dict.fromkeys(columns))


def describe_variable(name: str, *, technical: bool = False) -> str:
    canonical = canonicalize_variable(name) or name
    spec = VARIABLE_CATALOG.get(canonical)
    if not spec:
        return f"No encuentro una ficha técnica para `{name}`."
    status = "Disponible" if spec["availability"] == "available" else "Planificada; todavía no ejecutable"
    source_names = ", ".join(dict.fromkeys(source["id"] for source in spec["sources"]))
    text = (
        f"**{spec['display_name']} (`{canonical}`)** — {spec['short_description']} "
        f"{spec['interpretation']} Unidad: **{spec['unit']}**. Fuente: **{source_names}**. Estado: **{status}**."
    )
    if technical:
        details = [f"Tipo: {spec['variable_type']}; comportamiento temporal: {spec['temporal_kind']}; coste: {spec['calculation_cost']}"]
        if spec.get("formula"):
            details.append(f"Fórmula: `{spec['formula']}`")
        if spec.get("limitations"):
            details.append(f"Limitación: {spec['limitations']}")
        text += "\n\n" + ". ".join(details) + "."
    return text


def mentioned_variables(text: str, *, include_planned: bool = True) -> list[str]:
    plain = _normalize(text)
    matches: list[tuple[int, str]] = []
    for alias, canonical in sorted(_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if not include_planned and VARIABLE_CATALOG[canonical]["availability"] != "available":
            continue
        if re.search(rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])", plain):
            matches.append((plain.find(alias), canonical))
    return list(dict.fromkeys(name for _, name in sorted(matches)))


def choose_sources(variable: str, target_resolution_m: float, preferred_source: str | None = None) -> list[dict[str, Any]]:
    sources = VARIABLE_CATALOG[variable]["sources"]
    if preferred_source:
        preferred = _normalize(preferred_source).replace(" ", "_").replace("-", "_")
        return [source for source in sources if _normalize(source["id"]).replace(" ", "_").replace("-", "_") == preferred]
    if variable in {"buildings", "building_height", "building_footprint", "aspect_ratio", "sky_view_factor"}:
        return sources
    if any(source["id"] == "landsat_8_9" for source in sources):
        return [next(source for source in sources if source["id"] == "landsat_8_9")]
    return sources[:1]
