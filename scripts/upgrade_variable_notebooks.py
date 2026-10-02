from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [ROOT / "01_descarga_datos_local.ipynb", ROOT / "01_descarga_datos_local_documentado.ipynb"]


def replace_between(source: str, start: str, end: str, replacement: str) -> str:
    if start not in source or end not in source:
        raise RuntimeError(f"No se encontró el bloque entre {start!r} y {end!r}")
    head, rest = source.split(start, 1)
    _, tail = rest.split(end, 1)
    return head + replacement.rstrip() + "\n\n" + end + tail


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Se esperó una aparición y se encontraron {source.count(old)}: {old[:80]!r}")
    return source.replace(old, new, 1)


def find_cell(notebook, marker: str) -> int:
    matches = [i for i, cell in enumerate(notebook.cells) if marker in cell.source]
    if len(matches) != 1:
        raise RuntimeError(f"Se esperaba una celda con {marker!r} y se encontraron {matches}")
    return matches[0]


ERA5_CELL = '''from tools.scientific_variables import (
    derive_era5,
    era5_output_columns,
    era5_request_variables,
    era5_short_names,
)

ERA5_VARIABLES = era5_request_variables(REQUESTED_VARIABLES)
ERA5_SHORT_VARIABLES = era5_short_names(REQUESTED_VARIABLES)
ERA5_AREA = [max_lat + 0.15, min_lon - 0.15, min_lat - 0.15, max_lon + 0.15]  # N,W,S,E

def fix_zip_disguised_as_nc(path):
    if not zipfile.is_zipfile(path):
        return
    with zipfile.ZipFile(path) as zf:
        nc_names = [n for n in zf.namelist() if n.lower().endswith(".nc")]
        if not nc_names:
            raise RuntimeError(f"{path.name} es ZIP pero no contiene NetCDF.")
        with zf.open(nc_names[0]) as src, open(str(path) + ".tmp", "wb") as dst:
            shutil.copyfileobj(src, dst)
    Path(str(path) + ".tmp").replace(path)

def expanded_era5_days(target_times):
    days = set()
    for ts in pd.to_datetime(target_times, utc=True):
        day = ts.normalize()
        for delta in range(3):
            days.add(day - pd.Timedelta(days=delta))
    return sorted(days)

def download_era5_for_targets(target_times, label):
    import hashlib
    days = expanded_era5_days(target_times)
    groups = defaultdict(list)
    for day in days:
        groups[(day.year, day.month)].append(day)
    signature = hashlib.sha1("|".join(ERA5_VARIABLES).encode()).hexdigest()[:10]
    paths = []
    for (year, month), month_days in tqdm(sorted(groups.items()), desc=f"ERA5 {label}"):
        target = CACHE_DIR / "era5" / f"era5_{label}_{year}_{month:02d}_{signature}.nc"
        if not target.exists() or OVERWRITE_OUTPUTS:
            request = {"variable": ERA5_VARIABLES, "year": str(year), "month": f"{month:02d}",
                       "day": [f"{d.day:02d}" for d in month_days],
                       "time": [f"{h:02d}:00" for h in range(24)], "area": ERA5_AREA,
                       "data_format": "netcdf", "download_format": "unarchived"}
            cds_client.retrieve("reanalysis-era5-land", request, str(target))
        fix_zip_disguised_as_nc(target)
        paths.append(target)
    return paths

def read_era5(paths):
    frames = []
    for path in paths:
        ds = xr.open_dataset(path)
        available = [name for name in ERA5_SHORT_VARIABLES if name in ds.variables]
        missing = [name for name in ERA5_SHORT_VARIABLES if name not in ds.variables]
        if missing:
            raise RuntimeError(f"ERA5-Land no devolvió las variables esperadas: {missing}")
        frame = ds[available].to_dataframe().reset_index()
        time_col = next(c for c in ["valid_time", "time", "datetime", "date"] if c in frame)
        frame = frame.rename(columns={time_col: "era5_dt_utc"})
        frame["era5_dt_utc"] = pd.to_datetime(frame["era5_dt_utc"], utc=True)
        frames.append(frame)
        ds.close()
    df = pd.concat(frames, ignore_index=True)
    df = df.groupby("era5_dt_utc", as_index=False)[ERA5_SHORT_VARIABLES].mean()
    return derive_era5(df)

def attach_era5(data, datetime_col, label):
    target_times = pd.to_datetime(data[datetime_col].dropna().unique(), utc=True)
    paths = download_era5_for_targets(target_times, label)
    hourly, daily = read_era5(paths)
    out = data.copy()
    out["era5_dt_utc"] = pd.to_datetime(out[datetime_col], utc=True).dt.round("h")
    hourly_columns = [c for c in era5_output_columns(REQUESTED_VARIABLES) if c in hourly.columns]
    out = out.merge(hourly[["era5_dt_utc", *hourly_columns]], on="era5_dt_utc", how="left")
    out["day"] = pd.to_datetime(out[datetime_col], utc=True).dt.floor("D")
    daily_columns = [c for c in ["rain_1d", "rain_3d"] if c in daily.columns]
    return out.merge(daily[["day", *daily_columns]], on="day", how="left").drop(columns="day")

landsat_df = attach_era5(landsat_df, "acquisition_dt_utc", "landsat")
'''


S2_AGGREGATION_CELL = '''from tools.scientific_variables import s2_band_names

def static_predictors_by_node():
    drop = ["row", "col", "row_norm", "col_norm", "latitude", "longitude", "x", "y"]
    predictors = static_df.drop(columns=[c for c in drop if c in static_df], errors="ignore")
    return nodes[["node_id", "x", "y"]].merge(predictors, on="node_id", how="left")

node_static = static_predictors_by_node()
s2_requested_bands = s2_band_names(REQUESTED_VARIABLES)
s2_frames = []
for acquisition in tqdm(pd.to_datetime(s3_pixels.acquisition_dt_utc.unique(), utc=True), desc="Sentinel-2 predictors"):
    frame, source_time = sentinel2_predictors_nearest(acquisition)
    if len(frame):
        frame["acquisition_dt_utc"] = acquisition
        frame["s2_source_dt_utc"] = source_time
        keep = ["node_id", *[c for c in s2_requested_bands if c in frame], "acquisition_dt_utc", "s2_source_dt_utc"]
        s2_frames.append(frame[keep])
s2_nodes = pd.concat(s2_frames, ignore_index=True) if s2_frames else pd.DataFrame(
    columns=["node_id", *s2_requested_bands, "acquisition_dt_utc", "s2_source_dt_utc"]
)

def aggregate_nodes_for_s3_pixel(pixel, fine):
    px, py = to_projected.transform(pixel.longitude, pixel.latitude)
    dist = np.sqrt((fine.x-px)**2 + (fine.y-py)**2)
    selected = fine[dist <= S3_NODE_AGG_RADIUS_M]
    if selected.empty:
        selected = fine.loc[[dist.idxmin()]]
    excluded = {"node_id", "x", "y", "row", "col", "acquisition_dt_utc", "s2_source_dt_utc"}
    numeric = [c for c in selected.select_dtypes(include=[np.number]).columns if c not in excluded]
    values = {f"{c}_mean_1km": selected[c].mean() for c in numeric}
    values["n_nodes_aggregated"] = len(selected)
    if "s2_source_dt_utc" in selected:
        values["s2_source_dt_utc"] = selected.s2_source_dt_utc.dropna().iloc[0] if selected.s2_source_dt_utc.notna().any() else pd.NaT
    return pd.Series(values)

aggregates = []
for acquisition, group in tqdm(s3_pixels.groupby("acquisition_dt_utc"), desc="Agregación 1 km"):
    fine = node_static.merge(s2_nodes[s2_nodes.acquisition_dt_utc == acquisition], on="node_id", how="left")
    part = group.apply(lambda pixel: aggregate_nodes_for_s3_pixel(pixel, fine), axis=1)
    part.index = group.index
    aggregates.append(part)
s3_pixels = s3_pixels.join(pd.concat(aggregates).sort_index())
display(s3_pixels.head())
'''


def upgrade(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)

    config_index = find_cell(notebook, "# -------------------- MODO --------------------")
    config = notebook.cells[config_index].source
    if "REQUESTED_VARIABLES =" not in config:
        marker = '# -------------------- FUENTES Y CALIDAD --------------------'
        defaults = '''# -------------------- VARIABLES --------------------
# El núcleo mantiene exactamente las 23 columnas históricas. Añade aquí
# variables del catálogo para incorporarlas como columnas opcionales.
REQUESTED_VARIABLES = [
    "LST", "NDVI", "NDBI", "ALBEDO", "air_temperature",
    "relative_humidity", "solar_radiation", "wind_speed", "rain_3d",
    "elevation", "aspect_ratio", "buildings",
]

'''
        config = config.replace(marker, defaults + marker, 1)
    notebook.cells[config_index].source = config

    ee_index = find_cell(notebook, "def nodes_to_ee_feature_collection")
    cell = notebook.cells[ee_index].source
    imports = '''from tools.scientific_variables import (
    add_landsat_variables as add_landsat_variable_bands,
    add_s2_variables,
    landsat_band_names,
    s2_band_names,
)

'''
    if "add_landsat_variable_bands" not in cell:
        cell = imports + cell
    if "def scale_sr(img, band):" in cell:
        cell = replace_between(
            cell, "def scale_sr(img, band):", "def landsat_collection(start, end):",
            '''def add_landsat_variables(img):
    return add_landsat_variable_bands(img, REQUESTED_VARIABLES)''',
        )
    if "    def add_s2_predictors(img):" in cell and "return add_s2_variables" not in cell:
        cell = replace_between(
            cell, "    def add_s2_predictors(img):", "    collection =",
            '''    def add_s2_predictors(img):
        return add_s2_variables(img, REQUESTED_VARIABLES)''',
        )
    old_sample = '    df = sample_image_at_nodes(image, ["NDVI_S2", "NDBI_S2", "ALBEDO_S2"], NODE_SPACING_M)'
    if old_sample in cell:
        cell = replace_once(cell, old_sample, '    df = sample_image_at_nodes(image, s2_band_names(REQUESTED_VARIABLES), NODE_SPACING_M)')
    old_static = '    required_static = {"node_id", "DEM", "aspect_ratio", "has_buildings"}'
    if old_static in cell and 'required_static = {"node_id", "DEM", "aspect_ratio", "has_buildings"} |' not in cell:
        cell = replace_once(cell, old_static, '    required_static = {"node_id", "DEM", "aspect_ratio", "has_buildings"} | ({"sky_view_factor"} if "sky_view_factor" in REQUESTED_VARIABLES else set())')
    duplicate_svf = ' | ({"sky_view_factor"} if "sky_view_factor" in REQUESTED_VARIABLES else set())'
    while duplicate_svf + duplicate_svf in cell:
        cell = cell.replace(duplicate_svf + duplicate_svf, duplicate_svf)
    notebook.cells[ee_index].source = cell

    morphology_index = find_cell(notebook, "def build_node_cells")
    cell = notebook.cells[morphology_index].source
    if "approximate_sky_view_factor" not in cell:
        cell = "from tools.urban_morphology import approximate_sky_view_factor\n\n" + cell
    if '    result["sky_view_factor"] = 1.0' not in cell:
        cell = replace_once(cell, '    result["aspect_ratio"] = 0.0', '    result["aspect_ratio"] = 0.0\n    result["sky_view_factor"] = 1.0')
    if 'result["sky_view_factor"] = approximate_sky_view_factor' not in cell:
        cell = replace_once(
            cell,
        '    tif = download_cnig_building_height(CACHE_DIR / "cnig" / f"building_height_cnig_{STATIC_CACHE_KEY}.tif")',
            '    tif = download_cnig_building_height(CACHE_DIR / "cnig" / f"building_height_cnig_{STATIC_CACHE_KEY}.tif")\n    if "sky_view_factor" in REQUESTED_VARIABLES:\n        result["sky_view_factor"] = approximate_sky_view_factor(nodes_df, tif, node_crs=PROJECTED_CRS)',
        )
    notebook.cells[morphology_index].source = cell
    notebook.cells[morphology_index].source = notebook.cells[morphology_index].source.replace(
        'f"x({xmin-100},{xmax+100})"), ("subset", f"y({ymin-100},{ymax+100})")',
        'f"x({xmin-250},{xmax+250})"), ("subset", f"y({ymin-250},{ymax+250})")',
    )

    landsat_index = find_cell(notebook, "def get_landsat_scenes")
    cell = notebook.cells[landsat_index].source
    old_landsat_sample = '                frame = sample_image_at_nodes(image, ["LST_K", "NDVI", "NDBI", "ALBEDO"], 30)'
    if old_landsat_sample in cell:
        cell = replace_once(cell, old_landsat_sample, '                frame = sample_image_at_nodes(image, landsat_band_names(REQUESTED_VARIABLES), 30)')
    notebook.cells[landsat_index].source = cell

    notebook.cells[find_cell(notebook, "ERA5_VARIABLES =")].source = ERA5_CELL

    landsat_final_index = find_cell(notebook, "build_landsat_final")
    cell = notebook.cells[landsat_final_index].source
    if "landsat_df = build_landsat_final(landsat_df)" in cell:
        cell = replace_once(cell, "landsat_df = build_landsat_final(landsat_df)", "landsat_df = build_landsat_final(landsat_df, REQUESTED_VARIABLES)")
    notebook.cells[landsat_final_index].source = cell

    s3_index = find_cell(notebook, "def extract_s3")
    cell = notebook.cells[s3_index].source
    if "extract_matching_s3_arrays" not in cell:
        cell = "from tools.scientific_variables import extract_matching_s3_arrays\n\n" + cell
    old_s3 = '        out["LST_S3_uncertainty_K"] = unc[rr, cc] if unc is not None and unc.shape == lst.shape else np.nan\n        out["NDVI_S3_L2"] = ndvi_l2[rr, cc] if ndvi_l2 is not None and ndvi_l2.shape == lst.shape else np.nan'
    if old_s3 in cell:
        cell = replace_once(
            cell, old_s3,
            '''        out["LST_S3_uncertainty_K"] = unc[rr, cc] if unc is not None and unc.shape == lst.shape else np.nan
        out["LST_uncertainty_1km"] = out["LST_S3_uncertainty_K"]
        out["NDVI_S3_L2"] = ndvi_l2[rr, cc] if ndvi_l2 is not None and ndvi_l2.shape == lst.shape else np.nan
        ancillary = extract_matching_s3_arrays(datasets, lst.shape)
        for column, values in ancillary.items():
            if values is not None:
                out[column] = values[rr, cc]
        if "quality_flag_1km" not in out:
            out["quality_flag_1km"] = np.nan''',
        )
    cell = cell.replace('        out["quality_flag_1km"] = 0', '        if "quality_flag_1km" not in out:\n            out["quality_flag_1km"] = np.nan')
    notebook.cells[s3_index].source = cell

    notebook.cells[find_cell(notebook, "def static_predictors_by_node")].source = S2_AGGREGATION_CELL

    sentinel_final_index = find_cell(notebook, "build_sentinel_final")
    cell = notebook.cells[sentinel_final_index].source
    if "s3_pixels, ndvi_filled_count = build_sentinel_final(s3_pixels)" in cell:
        cell = replace_once(cell, "s3_pixels, ndvi_filled_count = build_sentinel_final(s3_pixels)", "s3_pixels, ndvi_filled_count = build_sentinel_final(s3_pixels, REQUESTED_VARIABLES)")
    notebook.cells[sentinel_final_index].source = cell

    validation_index = find_cell(notebook, "validate_output(")
    cell = notebook.cells[validation_index].source
    if "output_columns_for" not in cell:
        cell = cell.replace(
            "from tools.dataset_outputs import LANDSAT_PREFERRED_COLS, SENTINEL_PREFERRED_COLS",
            "from tools.dataset_outputs import LANDSAT_PREFERRED_COLS, SENTINEL_PREFERRED_COLS\nfrom rules.variable_catalog import output_columns_for",
        )
    cell = cell.replace(
        'validate_output(LANDSAT_OUTPUT, LANDSAT_PREFERRED_COLS, "Landsat sin downscaling")',
        'validate_output(LANDSAT_OUTPUT, list(dict.fromkeys([*LANDSAT_PREFERRED_COLS, *output_columns_for(REQUESTED_VARIABLES, "landsat")])), "Landsat sin downscaling")',
    )
    cell = cell.replace(
        'validate_output(SENTINEL3_OUTPUT, SENTINEL_PREFERRED_COLS, "Sentinel-3 L2 LST a 1 km")',
        'validate_output(SENTINEL3_OUTPUT, list(dict.fromkeys([*SENTINEL_PREFERRED_COLS, *output_columns_for(REQUESTED_VARIABLES, "sentinel3")])), "Sentinel-3 L2 LST a 1 km")',
    )
    notebook.cells[validation_index].source = cell

    nbformat.write(notebook, path)


if __name__ == "__main__":
    for notebook_path in NOTEBOOKS:
        upgrade(notebook_path)
        print(f"Actualizado: {notebook_path.name}")
