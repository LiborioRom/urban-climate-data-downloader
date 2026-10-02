import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _code_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")


def test_canonical_notebook_uses_strict_final_builders_and_static_cache():
    code = _code_text(ROOT / "01_descarga_datos_local.ipynb")
    assert "build_landsat_final" in code
    assert "build_sentinel_final" in code
    assert "STATIC_CACHE_FILE" in code
    assert "variables NDVI completados" not in code  # evita un texto engañoso
    assert "Valores NDVI completados temporalmente" in code
    assert "REQUESTED_VARIABLES" in code
    assert "era5_request_variables(REQUESTED_VARIABLES)" in code
    assert "approximate_sky_view_factor" in code
    assert "MODE =" not in code
    assert "TEST_DATE" not in code
    assert "START_DATE" in code and "END_DATE" in code
    assert "DATE_WINDOWS" in code
    assert "normalized_date_windows" in code
    assert "search_s3_products(normalized_date_windows)" in code
    assert "S3_MATCH_LANDSAT_ONLY" not in code
    assert "sentinel-3-slstr-l2" not in code  # la colección se centraliza en tools.sentinelhub_s3
    assert "aligned_output_grid" in code
    assert "statistics_request_payload" in code
    assert "S3_STATISTICS_URL" in code
    assert "sentinel3_daily_" in code
    assert "download_s3_product" not in code
    assert "ee.data.computeFeatures" in code
    assert '"fileFormat": "PANDAS_DATAFRAME"' in code
    assert "collection.map(sample_scene).flatten()" in code
    assert "landsat_batch_cache_file" in code
    assert "for scene_id" not in code
    assert "load_shared_era5" in code
    assert "download_era5_land_gee" in code
    assert "era5_gee" in code
    assert 'ERA5_SOURCE = "earth_engine"' in code
    assert "attach_era5(landsat_df" not in code
    assert "s2_nodes = download_sentinel2_batch" in code
    assert "lst-tfm-sentinel2-batch" in code
    assert "S3_GRID_CACHE_KEY" in code
    assert "build_landsat_sentinel_pairs" in code
    assert "dataset_pares_landsat_sentinel3.csv" in code
    assert 'S3_REQUEST_CRS = "EPSG:32630"' in code
    assert "cell[\"bounds\"], S3_REQUEST_CRS" in code
    assert "s3_to_wgs84.transform" in code
    assert ".json.gz" in code and ".csv.gz" in code


def test_documented_notebook_has_the_same_output_contract():
    canonical = _code_text(ROOT / "01_descarga_datos_local.ipynb")
    documented = _code_text(ROOT / "01_descarga_datos_local_documentado.ipynb")
    assert documented == canonical


def test_all_non_magic_code_cells_compile():
    for filename in (
        "01_descarga_datos_local.ipynb",
        "01_descarga_datos_local_documentado.ipynb",
        "02_downscaling_xgboost.ipynb",
        "03_validacion_era5_cds_vs_gee.ipynb",
        "04_diagnostico_sentinel3.ipynb",
    ):
        notebook = json.loads((ROOT / filename).read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
                continue
            compile(source, f"{filename}:cell{index}", "exec")


def test_downscaling_notebook_matches_current_tfm_contract():
    notebook = json.loads((ROOT / "02_downscaling_xgboost.ipynb").read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    markdown = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )

    assert "find_latest_downscaling_dataset" in code
    assert "EXPECTED_YEAR_RANGE = (2013, 2025)" in code
    assert "add_rain_features" in code
    assert "metricas_test.csv" in code and "importancia_variables.csv" in code
    assert "Proyecto TFM-GNN" in markdown
    assert "no pueden entrenar el downscaling" in markdown


def test_downscaling_notebook_has_no_mojibake():
    text = (ROOT / "02_downscaling_xgboost.ipynb").read_text(encoding="utf-8")
    for corrupted_fragment in ("Ã", "Â", "â€", "�"):
        assert corrupted_fragment not in text


def test_era5_comparison_notebook_is_short_reproducible_and_decisive():
    code = _code_text(ROOT / "03_validacion_era5_cds_vs_gee.ipynb")
    assert 'N_HOURS = 60' in code
    assert "EE_PROJECT =" in code
    assert "CDS_API_KEY =" in code
    assert "ee.Authenticate()" in code
    assert "all_samples = all_samples.merge(samples)" not in code
    assert "ee.Image.cat(*stacked_images)" in code
    assert "gee_wide = compute_gee_features(samples)" in code
    assert "Too many concurrent aggregations" in code
    assert "sleep(wait_seconds)" in code
    assert "stacked.reduceRegions" in code
    assert 'ECMWF/ERA5_LAND/HOURLY' in code
    assert 'total_precipitation_hourly' in code
    assert 'deaccumulate_era5_land' in code
    assert 'comparacion_detallada.csv' in code
    assert 'resumen_variables_finales.csv' in code
    assert 'rendimiento_fuentes.csv' in code
    assert 'perf_counter()' in code
    assert 'rows_per_second' in code
    assert 'informe_comparacion.txt' in code
    assert 'se recomienda usar Earth Engine' in code


def test_sentinel3_diagnostic_uses_the_native_request_crs():
    code = _code_text(ROOT / "04_diagnostico_sentinel3.ipynb")
    assert 'PROJECTED_CRS = "EPSG:25830"' in code
    assert 'S3_REQUEST_CRS = "EPSG:32630"' in code
    assert "Transformer.from_crs(PROJECTED_CRS, S3_REQUEST_CRS" in code
    assert "process_request_payload(process_bounds, S3_REQUEST_CRS" in code
    assert 'cell["bounds"], S3_REQUEST_CRS' in code
