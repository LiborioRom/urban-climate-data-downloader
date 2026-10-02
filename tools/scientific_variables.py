from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


CORE_LANDSAT_BANDS = ["LST_K", "NDVI", "NDBI", "ALBEDO"]
CORE_S2_BANDS = ["NDVI_S2", "NDBI_S2", "ALBEDO_S2"]
CORE_ERA5_REQUEST = {
    "2m_temperature", "2m_dewpoint_temperature", "surface_solar_radiation_downwards",
    "10m_u_component_of_wind", "10m_v_component_of_wind", "total_precipitation",
}


ERA5_DEPENDENCIES = {
    "air_temperature": {"2m_temperature"},
    "dewpoint_temperature": {"2m_dewpoint_temperature"},
    "relative_humidity": {"2m_temperature", "2m_dewpoint_temperature"},
    "solar_radiation": {"surface_solar_radiation_downwards"},
    "thermal_radiation_down": {"surface_thermal_radiation_downwards"},
    "wind_u10": {"10m_u_component_of_wind"},
    "wind_v10": {"10m_v_component_of_wind"},
    "wind_speed": {"10m_u_component_of_wind", "10m_v_component_of_wind"},
    "wind_direction": {"10m_u_component_of_wind", "10m_v_component_of_wind"},
    "rain_1d": {"total_precipitation"},
    "rain_3d": {"total_precipitation"},
    "rain_3d_log": {"total_precipitation"},
    "is_rainy": {"total_precipitation"},
    "skin_temperature": {"skin_temperature"},
    "surface_pressure": {"surface_pressure"},
    "snow_cover": {"snow_cover"},
    "snow_depth": {"snow_depth"},
    "vapour_pressure_deficit": {"2m_temperature", "2m_dewpoint_temperature"},
    **{f"soil_temperature_level_{i}": {f"soil_temperature_level_{i}"} for i in range(1, 5)},
    **{f"soil_moisture_level_{i}": {f"volumetric_soil_water_layer_{i}"} for i in range(1, 5)},
}

ERA5_EXTRA_RAW = {
    "lake_bottom_temperature": ("lake_bottom_temperature", "lblt", "lake_bottom_temperature_C", "temperature"),
    "lake_ice_depth": ("lake_ice_depth", "licd", "lake_ice_depth_m", "identity"),
    "lake_ice_temperature": ("lake_ice_temperature", "lict", "lake_ice_temperature_C", "temperature"),
    "lake_mix_layer_depth": ("lake_mix_layer_depth", "lmld", "lake_mix_layer_depth_m", "identity"),
    "lake_mix_layer_temperature": ("lake_mix_layer_temperature", "lmlt", "lake_mix_layer_temperature_C", "temperature"),
    "lake_shape_factor": ("lake_shape_factor", "lshf", "lake_shape_factor", "identity"),
    "lake_total_layer_temperature": ("lake_total_layer_temperature", "llt", "lake_total_layer_temperature_C", "temperature"),
    "leaf_area_index_high": ("leaf_area_index_high_vegetation", "lai_hv", "LAI_high", "identity"),
    "leaf_area_index_low": ("leaf_area_index_low_vegetation", "lai_lv", "LAI_low", "identity"),
    "potential_evaporation": ("potential_evaporation", "pev", "potential_evaporation_mm", "water_mm"),
    "runoff": ("runoff", "ro", "runoff_mm", "water_mm"),
    "skin_reservoir_content": ("skin_reservoir_content", "src", "skin_reservoir_mm", "water_mm"),
    "snow_albedo": ("snow_albedo", "asn", "snow_albedo", "identity"),
    "snow_density": ("snow_density", "rsn", "snow_density_kgm3", "identity"),
    "snow_evaporation": ("snow_evaporation", "es", "snow_evaporation_mm", "water_mm"),
    "snowfall": ("snowfall", "sf", "snowfall_mm", "water_mm"),
    "snowmelt": ("snowmelt", "smlt", "snowmelt_mm", "water_mm"),
    "snow_temperature": ("temperature_of_snow_layer", "tsn", "snow_temperature_C", "temperature"),
    "total_evaporation": ("total_evaporation", "e", "total_evaporation_mm", "water_mm"),
    "evaporation_bare_soil": ("evaporation_from_bare_soil", "evabs", "evaporation_bare_soil_mm", "water_mm"),
    "evaporation_open_water": ("evaporation_from_open_water_surfaces_excluding_oceans", "evaow", "evaporation_open_water_mm", "water_mm"),
    "evaporation_canopy": ("evaporation_from_the_top_of_canopy", "evatc", "evaporation_canopy_mm", "water_mm"),
    "vegetation_transpiration": ("evaporation_from_vegetation_transpiration", "evavt", "vegetation_transpiration_mm", "water_mm"),
    "subsurface_runoff": ("sub_surface_runoff", "ssro", "subsurface_runoff_mm", "water_mm"),
    "surface_runoff": ("surface_runoff", "sro", "surface_runoff_mm", "water_mm"),
    "surface_latent_heat_flux": ("surface_latent_heat_flux", "slhf", "surface_latent_heat_flux_Wm2", "energy_flux"),
    "surface_net_solar_radiation": ("surface_net_solar_radiation", "ssr", "surface_net_solar_Wm2", "energy_flux"),
    "surface_net_thermal_radiation": ("surface_net_thermal_radiation", "str", "surface_net_thermal_Wm2", "energy_flux"),
    "surface_sensible_heat_flux": ("surface_sensible_heat_flux", "sshf", "surface_sensible_heat_flux_Wm2", "energy_flux"),
}
ERA5_DEPENDENCIES.update({key: {raw} for key, (raw, _, _, _) in ERA5_EXTRA_RAW.items()})

ERA5_SHORT_NAMES = {
    "2m_temperature": "t2m", "2m_dewpoint_temperature": "d2m",
    "surface_solar_radiation_downwards": "ssrd", "surface_thermal_radiation_downwards": "strd",
    "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10",
    "total_precipitation": "tp", "skin_temperature": "skt", "surface_pressure": "sp",
    "snow_cover": "snowc", "snow_depth": "sd",
    **{f"soil_temperature_level_{i}": f"stl{i}" for i in range(1, 5)},
    **{f"volumetric_soil_water_layer_{i}": f"swvl{i}" for i in range(1, 5)},
}
ERA5_SHORT_NAMES.update({raw: short for raw, short, _, _ in ERA5_EXTRA_RAW.values()})

ERA5_ACCUMULATED_RAW = {
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "total_precipitation",
    "potential_evaporation",
    "runoff",
    "snow_evaporation",
    "snowfall",
    "snowmelt",
    "total_evaporation",
    "evaporation_from_bare_soil",
    "evaporation_from_open_water_surfaces_excluding_oceans",
    "evaporation_from_the_top_of_canopy",
    "evaporation_from_vegetation_transpiration",
    "sub_surface_runoff",
    "surface_runoff",
    "surface_latent_heat_flux",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "surface_sensible_heat_flux",
}

# Earth Engine expone las variables de flujo ya convertidas a incrementos horarios
# mediante el sufijo ``_hourly``. Las variables de estado conservan su nombre.
ERA5_GEE_BAND_NAMES = {name: name for name in ERA5_SHORT_NAMES}
ERA5_GEE_BAND_NAMES.update({
    "2m_temperature": "temperature_2m",
    "2m_dewpoint_temperature": "dewpoint_temperature_2m",
    "10m_u_component_of_wind": "u_component_of_wind_10m",
    "10m_v_component_of_wind": "v_component_of_wind_10m",
})
ERA5_GEE_BAND_NAMES.update({name: f"{name}_hourly" for name in ERA5_ACCUMULATED_RAW})

ERA5_OUTPUT_COLUMNS = {
    "air_temperature": "Tair_C", "dewpoint_temperature": "dewpoint_C", "relative_humidity": "RH",
    "solar_radiation": "Rsol_Wm2", "thermal_radiation_down": "Rthermal_Wm2",
    "wind_u10": "wind_u10", "wind_v10": "wind_v10", "wind_speed": "wind_speed",
    "wind_direction": "wind_direction_deg", "rain_1d": "rain_1d", "rain_3d": "rain_3d",
    "rain_3d_log": "rain_3d_log", "is_rainy": "is_rainy",
    "skin_temperature": "skin_temperature_C", "surface_pressure": "surface_pressure_hPa",
    "snow_cover": "snow_cover_pct", "snow_depth": "snow_depth_m",
    "vapour_pressure_deficit": "VPD_kPa",
    **{f"soil_temperature_level_{i}": f"soil_temperature_l{i}" for i in range(1, 5)},
    **{f"soil_moisture_level_{i}": f"soil_moisture_l{i}" for i in range(1, 5)},
}
ERA5_OUTPUT_COLUMNS.update({key: output for key, (_, _, output, _) in ERA5_EXTRA_RAW.items()})


def era5_request_variables(selected: Iterable[str]) -> list[str]:
    requested = set(CORE_ERA5_REQUEST)
    for variable in selected:
        requested.update(ERA5_DEPENDENCIES.get(variable, set()))
    return sorted(requested)


def era5_short_names(selected: Iterable[str]) -> list[str]:
    return [ERA5_SHORT_NAMES[name] for name in era5_request_variables(selected)]


def era5_accumulated_short_names(selected: Iterable[str]) -> list[str]:
    requested = set(era5_request_variables(selected))
    return [ERA5_SHORT_NAMES[name] for name in ERA5_ACCUMULATED_RAW if name in requested]


def era5_gee_band_names(selected: Iterable[str]) -> list[str]:
    """Bandas Earth Engine equivalentes, en el mismo orden que los nombres cortos."""
    return [ERA5_GEE_BAND_NAMES[name] for name in era5_request_variables(selected)]


def era5_output_columns(selected: Iterable[str]) -> list[str]:
    core = ["Tair_C", "RH", "Rsol_Wm2", "wind_speed", "rain_3d"]
    optional = [ERA5_OUTPUT_COLUMNS[name] for name in selected if name in ERA5_OUTPUT_COLUMNS]
    return list(dict.fromkeys([*core, *optional]))


def deaccumulate_era5_land(
    frame: pd.DataFrame,
    accumulated_columns: Iterable[str],
    *,
    time_column: str = "era5_dt_utc",
) -> pd.DataFrame:
    """Convierte acumulaciones ERA5-Land 00→paso en incrementos horarios.

    ERA5-Land reinicia el pronóstico acumulado después de la validez 00 UTC.
    Por ello, 01 UTC ya representa el primer incremento del nuevo día; el resto
    se obtiene restando el paso inmediatamente anterior. Un salto temporal no se
    rellena silenciosamente porque impediría calcular una diferencia fiable.
    """
    if time_column not in frame:
        raise KeyError(f"Falta la columna temporal {time_column!r}.")
    out = frame.copy()
    out[time_column] = pd.to_datetime(out[time_column], utc=True)
    out = out.sort_values(time_column).reset_index(drop=True)
    times = out[time_column]
    consecutive = times.diff().eq(pd.Timedelta(hours=1))
    first_forecast_step = times.dt.hour.eq(1)

    for column in dict.fromkeys(accumulated_columns):
        if column not in out:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        increments = values.diff().where(consecutive)
        increments = increments.where(~first_forecast_step, values)
        if column in {"tp", "ssrd"}:
            increments = increments.clip(lower=0)
        out[column] = increments
    return out


def derive_era5(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convierte unidades ERA5-Land y calcula derivados sin inventar resolución espacial."""
    out = frame.copy()
    if "t2m" in out:
        out["Tair_C"] = out["t2m"] - 273.15
    if "d2m" in out:
        out["dewpoint_C"] = out["d2m"] - 273.15
    if {"Tair_C", "dewpoint_C"}.issubset(out):
        tair, dew = out["Tair_C"], out["dewpoint_C"]
        out["RH"] = (100 * np.exp((17.625 * dew) / (243.04 + dew) - (17.625 * tair) / (243.04 + tair))).clip(0, 100)
        es_t = 0.6108 * np.exp(17.27 * tair / (tair + 237.3))
        es_d = 0.6108 * np.exp(17.27 * dew / (dew + 237.3))
        out["VPD_kPa"] = (es_t - es_d).clip(lower=0)
    if "ssrd" in out:
        out["Rsol_Wm2"] = out["ssrd"] / 3600.0
    if "strd" in out:
        out["Rthermal_Wm2"] = out["strd"] / 3600.0
    if "u10" in out:
        out["wind_u10"] = out["u10"]
    if "v10" in out:
        out["wind_v10"] = out["v10"]
    if {"u10", "v10"}.issubset(out):
        out["wind_speed"] = np.hypot(out["u10"], out["v10"])
        out["wind_direction_deg"] = (180 + np.degrees(np.arctan2(out["u10"], out["v10"]))) % 360
    if "skt" in out:
        out["skin_temperature_C"] = out["skt"] - 273.15
    if "sp" in out:
        out["surface_pressure_hPa"] = out["sp"] / 100.0
    if "snowc" in out:
        values = out["snowc"]
        out["snow_cover_pct"] = np.where(values <= 1.0, values * 100.0, values)
    if "sd" in out:
        out["snow_depth_m"] = out["sd"]
    for level in range(1, 5):
        if f"stl{level}" in out:
            out[f"soil_temperature_l{level}"] = out[f"stl{level}"] - 273.15
        if f"swvl{level}" in out:
            out[f"soil_moisture_l{level}"] = out[f"swvl{level}"]
    for _, short, output, conversion in ERA5_EXTRA_RAW.values():
        if short not in out:
            continue
        if conversion == "temperature":
            out[output] = out[short] - 273.15
        elif conversion == "water_mm":
            out[output] = out[short] * 1000.0
        elif conversion == "energy_flux":
            out[output] = out[short] / 3600.0
        else:
            out[output] = out[short]

    if "tp" not in out:
        return out, pd.DataFrame(columns=["day", "rain_1d", "rain_3d", "rain_3d_log", "is_rainy"])
    out["precip_mm"] = out["tp"] * 1000.0
    # Cada valor desacumulado representa la hora que termina en su sello temporal.
    # La validez 00 UTC cierra el día anterior, no el día que acaba de empezar.
    precipitation_day = (out["era5_dt_utc"] - pd.Timedelta(nanoseconds=1)).dt.floor("D")
    daily = out.assign(day=precipitation_day).groupby("day", as_index=False).precip_mm.sum()
    daily = daily.sort_values("day").set_index("day").asfreq("D", fill_value=0).reset_index()
    daily["rain_1d"] = daily["precip_mm"]
    daily["rain_3d"] = daily["precip_mm"].rolling(3, min_periods=1).sum()
    daily["rain_3d_log"] = np.log1p(daily["rain_3d"].clip(lower=0))
    daily["is_rainy"] = (daily["rain_3d"] > 0).astype("int8")
    return out, daily[["day", "rain_1d", "rain_3d", "rain_3d_log", "is_rainy"]]


LANDSAT_OPTIONAL_BANDS = {
    "reflectance_coastal": "SR_coastal", "reflectance_blue": "SR_blue",
    "reflectance_green": "SR_green", "reflectance_red": "SR_red", "reflectance_nir": "SR_nir",
    "reflectance_swir1": "SR_swir1", "reflectance_swir2": "SR_swir2",
    "NDWI": "NDWI", "MNDWI": "MNDWI", "NDMI": "NDMI", "EVI": "EVI", "SAVI": "SAVI",
    "BSI": "BSI", "urban_index": "urban_index", "surface_emissivity": "surface_emissivity",
    "emissivity_uncertainty": "emissivity_uncertainty", "atmospheric_transmittance": "atmospheric_transmittance",
    "cloud_distance": "cloud_distance_km", "LST_uncertainty": "LST_uncertainty_K",
}


def landsat_band_names(selected: Iterable[str]) -> list[str]:
    return [*CORE_LANDSAT_BANDS, *[column for name, column in LANDSAT_OPTIONAL_BANDS.items() if name in set(selected)]]


def add_landsat_variables(image, selected: Iterable[str]):
    selected = set(selected)
    sr = {name: image.select(band).multiply(2.75e-5).add(-0.2) for name, band in {
        "coastal": "SR_B1", "blue": "SR_B2", "green": "SR_B3", "red": "SR_B4",
        "nir": "SR_B5", "swir1": "SR_B6", "swir2": "SR_B7",
    }.items()}
    safe = lambda numerator, denominator: numerator.divide(denominator.where(denominator.eq(0), 1))
    derived = {
        "LST_K": image.select("ST_B10").multiply(0.00341802).add(149.0),
        "NDVI": safe(sr["nir"].subtract(sr["red"]), sr["nir"].add(sr["red"])),
        "NDBI": safe(sr["swir1"].subtract(sr["nir"]), sr["swir1"].add(sr["nir"])),
        "ALBEDO": sr["blue"].add(sr["green"]).add(sr["red"]).add(sr["nir"]).divide(4),
        "NDWI": safe(sr["green"].subtract(sr["nir"]), sr["green"].add(sr["nir"])),
        "MNDWI": safe(sr["green"].subtract(sr["swir1"]), sr["green"].add(sr["swir1"])),
        "NDMI": safe(sr["nir"].subtract(sr["swir1"]), sr["nir"].add(sr["swir1"])),
        "EVI": sr["nir"].subtract(sr["red"]).multiply(2.5).divide(sr["nir"].add(sr["red"].multiply(6)).subtract(sr["blue"].multiply(7.5)).add(1)),
        "SAVI": sr["nir"].subtract(sr["red"]).multiply(1.5).divide(sr["nir"].add(sr["red"]).add(0.5)),
        "BSI": safe(sr["swir1"].add(sr["red"]).subtract(sr["nir"].add(sr["blue"])), sr["swir1"].add(sr["red"]).add(sr["nir"]).add(sr["blue"])),
        "urban_index": safe(sr["swir2"].subtract(sr["nir"]), sr["swir2"].add(sr["nir"])),
    }
    bands = [derived[name].rename(name) for name in CORE_LANDSAT_BANDS]
    for variable, output in LANDSAT_OPTIONAL_BANDS.items():
        if variable not in selected:
            continue
        if output.startswith("SR_"):
            bands.append(sr[output.removeprefix("SR_")].rename(output))
        elif output in derived:
            bands.append(derived[output].rename(output))
        else:
            source_band, scale = {
                "surface_emissivity": ("ST_EMIS", 0.0001),
                "emissivity_uncertainty": ("ST_EMSD", 0.0001),
                "atmospheric_transmittance": ("ST_ATRAN", 0.0001),
                "cloud_distance_km": ("ST_CDIST", 0.01),
                "LST_uncertainty_K": ("ST_QA", 0.01),
            }[output]
            bands.append(image.select(source_band).multiply(scale).rename(output))
    import ee
    return ee.Image.cat(bands).copyProperties(image, image.propertyNames())


S2_OPTIONAL_BANDS = {
    "reflectance_coastal": "SR_coastal", "reflectance_blue": "SR_blue", "reflectance_green": "SR_green",
    "reflectance_red": "SR_red", "reflectance_nir": "SR_nir", "reflectance_swir1": "SR_swir1",
    "reflectance_swir2": "SR_swir2", "reflectance_red_edge_1": "SR_red_edge_1",
    "reflectance_red_edge_2": "SR_red_edge_2", "reflectance_red_edge_3": "SR_red_edge_3",
    "reflectance_red_edge_4": "SR_red_edge_4", "NDWI": "NDWI", "MNDWI": "MNDWI", "NDMI": "NDMI",
    "EVI": "EVI", "SAVI": "SAVI", "BSI": "BSI", "urban_index": "urban_index",
    "aerosol_optical_thickness": "AOT", "water_vapour": "water_vapour_cm",
}


def s2_band_names(selected: Iterable[str]) -> list[str]:
    return [*CORE_S2_BANDS, *[f"{column}_S2" for name, column in S2_OPTIONAL_BANDS.items() if name in set(selected)]]


def add_s2_variables(image, selected: Iterable[str]):
    selected = set(selected)
    source = {"coastal": "B1", "blue": "B2", "green": "B3", "red": "B4", "red_edge_1": "B5",
              "red_edge_2": "B6", "red_edge_3": "B7", "nir": "B8", "red_edge_4": "B8A",
              "swir1": "B11", "swir2": "B12"}
    sr = {name: image.select(band).divide(10000) for name, band in source.items()}
    safe = lambda numerator, denominator: numerator.divide(denominator.where(denominator.eq(0), 1))
    derived = {
        "NDVI": safe(sr["nir"].subtract(sr["red"]), sr["nir"].add(sr["red"])),
        "NDBI": safe(sr["swir1"].subtract(sr["nir"]), sr["swir1"].add(sr["nir"])),
        "ALBEDO": sr["blue"].add(sr["green"]).add(sr["red"]).add(sr["nir"]).divide(4),
        "NDWI": safe(sr["green"].subtract(sr["nir"]), sr["green"].add(sr["nir"])),
        "MNDWI": safe(sr["green"].subtract(sr["swir1"]), sr["green"].add(sr["swir1"])),
        "NDMI": safe(sr["nir"].subtract(sr["swir1"]), sr["nir"].add(sr["swir1"])),
        "EVI": sr["nir"].subtract(sr["red"]).multiply(2.5).divide(sr["nir"].add(sr["red"].multiply(6)).subtract(sr["blue"].multiply(7.5)).add(1)),
        "SAVI": sr["nir"].subtract(sr["red"]).multiply(1.5).divide(sr["nir"].add(sr["red"]).add(0.5)),
        "BSI": safe(sr["swir1"].add(sr["red"]).subtract(sr["nir"].add(sr["blue"])), sr["swir1"].add(sr["red"]).add(sr["nir"]).add(sr["blue"])),
        "urban_index": safe(sr["swir2"].subtract(sr["nir"]), sr["swir2"].add(sr["nir"])),
    }
    bands = [derived["NDVI"].rename("NDVI_S2"), derived["NDBI"].rename("NDBI_S2"), derived["ALBEDO"].rename("ALBEDO_S2")]
    for variable, output in S2_OPTIONAL_BANDS.items():
        if variable not in selected:
            continue
        if output.startswith("SR_"):
            bands.append(sr[output.removeprefix("SR_")].rename(f"{output}_S2"))
        elif output in derived:
            bands.append(derived[output].rename(f"{output}_S2"))
        elif output == "AOT":
            bands.append(image.select("AOT").multiply(0.001).rename("AOT_S2"))
        elif output == "water_vapour_cm":
            bands.append(image.select("WVP").multiply(0.001).rename("water_vapour_cm_S2"))
    import ee
    return ee.Image.cat(bands).copyProperties(image, ["system:time_start"])


S3_VARIABLE_CANDIDATES = {
    "LST_uncertainty_1km": ["LST_uncertainty", "lst_uncertainty"],
    "NDVI_S3": ["NDVI", "ndvi"],
    "vegetation_fraction_1km": ["fraction", "vegetation_fraction"],
    "TCWV_1km": ["TCWV", "tcwv"],
    "biome_class_1km": ["biome"],
    "quality_flag_1km": ["confidence_in", "quality_flags", "exception"],
    "LST_uncertainty_random_1km": ["LST_uncertainty_random"],
    "LST_uncertainty_locT_1km": ["LST_uncertainty_locT"],
    "LST_uncertainty_locATM_1km": ["LST_uncertainty_locATM"],
    "LST_uncertainty_locSF_1km": ["LST_uncertainty_locSF"],
    "LST_uncertainty_locGEO_1km": ["LST_uncertainty_locGEO"],
    "LST_uncertainty_sys_1km": ["LST_uncertainty_sys"],
}


def extract_matching_s3_arrays(datasets, lst_shape: tuple[int, ...]) -> dict[str, np.ndarray | None]:
    result: dict[str, np.ndarray | None] = {}
    for output, candidates in S3_VARIABLE_CANDIDATES.items():
        found = None
        for dataset in datasets:
            for candidate in candidates:
                if candidate in dataset.variables:
                    values = np.asarray(dataset[candidate].values)
                    if values.shape == lst_shape:
                        found = values
                        break
            if found is not None:
                break
        result[output] = found
    return result
