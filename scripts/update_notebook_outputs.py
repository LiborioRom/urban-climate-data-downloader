from __future__ import annotations

import json
from pathlib import Path


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(keepends=True)


def find_code_cell(notebook: dict, marker: str) -> dict:
    matches = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code" and marker in source(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"Se esperaban 1 celda para {marker!r}; encontradas: {len(matches)}")
    return matches[0]


def update(notebook_path: Path) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    static_cell = find_code_cell(notebook, 'dem_image = ee.Image("USGS/SRTMGL1_003")')
    prefix = source(static_cell).split('dem_image = ee.Image("USGS/SRTMGL1_003")', 1)[0]
    static_block = '''node_hash = int(pd.util.hash_pandas_object(
    nodes[["node_id", "latitude", "longitude"]].astype({"node_id": str}), index=False
).sum())
STATIC_CACHE_KEY = (
    f"{CENTER_LAT:.5f}_{CENTER_LON:.5f}_{HALF_SIDE_M:.0f}_"
    f"{NODE_SPACING_M:.0f}_{int(INCLUDE_URBAN_MORPHOLOGY)}_{node_hash}"
).replace("-", "m").replace(".", "p")
STATIC_CACHE_FILE = CACHE_DIR / "static" / f"node_static_{STATIC_CACHE_KEY}.csv"
STATIC_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

STATIC_CACHE_HIT = False
if STATIC_CACHE_FILE.exists() and not OVERWRITE_OUTPUTS:
    cached_static = pd.read_csv(STATIC_CACHE_FILE)
    expected_ids = set(nodes["node_id"].astype(str))
    cached_ids = set(cached_static.get("node_id", pd.Series(dtype=str)).astype(str))
    required_static = {"node_id", "DEM", "aspect_ratio", "has_buildings"}
    if expected_ids == cached_ids and required_static.issubset(cached_static.columns):
        static_df = cached_static
        static_df["node_id"] = static_df["node_id"].astype(str)
        STATIC_CACHE_HIT = True
        print("✓ Variables estáticas reutilizadas desde:", STATIC_CACHE_FILE)

if not STATIC_CACHE_HIT:
    dem_image = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("DEM")
    dem_sample = sample_image_at_nodes(dem_image, ["DEM"], NODE_SPACING_M)
    if "first" in dem_sample.columns:
        dem_sample = dem_sample.rename(columns={"first": "DEM"})
    nodes["node_id"] = nodes["node_id"].astype(str)
    dem_sample["node_id"] = dem_sample["node_id"].astype(str)
    static_df = nodes.merge(
        dem_sample[[c for c in ["node_id", "DEM"] if c in dem_sample.columns]],
        on="node_id", how="left"
    )
    print("Variables estáticas GEE calculadas:", static_df.shape)

display(static_df.head())
'''
    set_source(static_cell, prefix + static_block)

    urban_cell = find_code_cell(notebook, "def compute_urban_morphology")
    urban_source = source(urban_cell).replace(
        'CACHE_DIR / "cnig" / "building_height_cnig.tif"',
        'CACHE_DIR / "cnig" / f"building_height_cnig_{STATIC_CACHE_KEY}.tif"',
    )
    old_tail = '''urban_df = compute_urban_morphology(nodes)
static_df = static_df.merge(urban_df, on="node_id", how="left")
display(static_df.head())'''
    new_tail = '''if STATIC_CACHE_HIT:
    print("✓ DEM y morfología urbana ya disponibles; no se repiten descargas estáticas.")
else:
    urban_df = compute_urban_morphology(nodes)
    static_df = static_df.merge(urban_df, on="node_id", how="left")
    static_df.to_csv(STATIC_CACHE_FILE, index=False)
    print("✓ Variables estáticas guardadas para reutilización:", STATIC_CACHE_FILE)
display(static_df.head())'''
    if old_tail not in urban_source:
        raise RuntimeError("No se encontró el bloque final de morfología urbana")
    set_source(urban_cell, urban_source.replace(old_tail, new_tail))

    landsat_cell = find_code_cell(notebook, 'landsat_df.to_csv(LANDSAT_OUTPUT')
    old_landsat = '''landsat_df = add_temporal_features(landsat_df, "acquisition_dt_utc")
landsat_df = landsat_df.sort_values(["acquisition_dt_utc", "row", "col"]).reset_index(drop=True)
landsat_df.to_csv(LANDSAT_OUTPUT, index=False)'''
    new_landsat = '''from tools.dataset_outputs import build_landsat_final

landsat_df = add_temporal_features(landsat_df, "acquisition_dt_utc")
landsat_df = landsat_df.sort_values(["acquisition_dt_utc", "row", "col"]).reset_index(drop=True)
landsat_df = build_landsat_final(landsat_df)
landsat_df.to_csv(LANDSAT_OUTPUT, index=False)'''
    landsat_source = source(landsat_cell)
    if old_landsat not in landsat_source:
        raise RuntimeError("No se encontró el bloque de salida Landsat")
    set_source(landsat_cell, landsat_source.replace(old_landsat, new_landsat))

    sentinel_cell = find_code_cell(notebook, 's3_pixels.to_csv(SENTINEL3_OUTPUT')
    set_source(sentinel_cell, '''from tools.dataset_outputs import build_sentinel_final

s3_pixels = attach_era5(s3_pixels, "acquisition_dt_utc", "sentinel3")
s3_pixels = add_temporal_features(s3_pixels, "acquisition_dt_utc")
s3_pixels = s3_pixels.sort_values(["acquisition_dt_utc", "s3_row", "s3_col"]).reset_index(drop=True)
s3_pixels, ndvi_filled_count = build_sentinel_final(s3_pixels)
s3_pixels.to_csv(SENTINEL3_OUTPUT, index=False)
print("✓ CSV Sentinel-3:", SENTINEL3_OUTPUT.resolve())
print("Shape:", s3_pixels.shape, "| valores NDVI completados temporalmente:", ndvi_filled_count)
display(s3_pixels.groupby("date").size().describe())
display(s3_pixels.head())
''')

    validation_cell = find_code_cell(notebook, "def validate_output")
    set_source(validation_cell, '''from tools.dataset_outputs import LANDSAT_PREFERRED_COLS, SENTINEL_PREFERRED_COLS

def validate_output(path, required, label):
    if not path.exists():
        raise AssertionError(f"No se creó {path}")
    df = pd.read_csv(path)
    if list(df.columns) != required:
        missing = [c for c in required if c not in df.columns]
        unexpected = [c for c in df.columns if c not in required]
        raise AssertionError(f"{label}: esquema incorrecto; faltan={missing}; sobran={unexpected}")
    if df.empty:
        raise AssertionError(f"{label}: el CSV está vacío")
    print(f"✓ {label}: {len(df):,} filas, {len(df.columns)} columnas")
    print("  Fechas:", df["date"].nunique(), "| rango:", df["date"].min(), "→", df["date"].max())
    print("  Nulos principales:")
    print(df[required].isna().sum().to_string())

validate_output(LANDSAT_OUTPUT, LANDSAT_PREFERRED_COLS, "Landsat sin downscaling")
validate_output(SENTINEL3_OUTPUT, SENTINEL_PREFERRED_COLS, "Sentinel-3 L2 LST a 1 km")

print("\\nProceso terminado. No se ha aplicado downscaling.")
print("Landsat:", LANDSAT_OUTPUT.resolve())
print("Sentinel-3:", SENTINEL3_OUTPUT.resolve())
print("Nodos:", NODES_OUTPUT.resolve())
''')

    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    update(Path("01_descarga_datos_local.ipynb"))
