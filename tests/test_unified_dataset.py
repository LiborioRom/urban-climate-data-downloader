import pandas as pd
import zipfile

from schemas.request import DatasetPurpose, DatasetRequest, PipelineParameters
from tools.notebook_pipeline import _write_package
from tools.unified_dataset import build_quality_report, build_unified_dataset


def _nodes():
    return pd.DataFrame([
        {"node_id": "n0", "latitude": 37.4000, "longitude": -5.9800, "x": 1.0, "y": 1.0, "row": 0, "col": 0, "row_norm": 0.0, "col_norm": 0.0},
        {"node_id": "n1", "latitude": 37.4010, "longitude": -5.9790, "x": 2.0, "y": 1.0, "row": 0, "col": 1, "row_norm": 0.0, "col_norm": 1.0},
    ])


def _landsat():
    return pd.DataFrame([
        {"node_id": "n0", "date": "2024-07-14", "acquisition_dt_utc": "2024-07-14T10:50:00Z", "LST_K": 310.0, "NDVI": 0.2},
        {"node_id": "n1", "date": "2024-07-14", "acquisition_dt_utc": "2024-07-14T10:50:00Z", "LST_K": 312.0, "NDVI": None},
    ])


def _sentinel():
    return pd.DataFrame([
        {"node_id": "s0", "date": "2024-07-15", "acquisition_dt_utc": "2024-07-15T10:15:00Z", "latitude": 37.4002, "longitude": -5.9801, "LST_1km": 306.0, "NDVI": 0.3},
        {"node_id": "s1", "date": "2024-07-15", "acquisition_dt_utc": "2024-07-15T10:15:00Z", "latitude": 37.4100, "longitude": -5.9700, "LST_1km": 307.0, "NDVI": 0.4},
    ])


def test_downscaling_output_pairs_coarse_lst_with_fine_nodes():
    request = DatasetRequest(
        variables=["LST", "NDVI"],
        target_resolution_m=100,
        dataset_purpose=DatasetPurpose.LST_DOWNSCALING,
        parameters=PipelineParameters(s3_max_day_difference=2),
    )

    result = build_unified_dataset(_landsat(), _sentinel(), _nodes(), request)

    assert len(result) == 4
    observed = result.loc[result["landsat_valid"]]
    assert observed["LST_K"].tolist() == [310.0, 312.0]
    assert result["LST_1km"].notna().all()
    assert result["training_ready"].sum() == 2
    assert observed["s3_day_difference"].eq(1).all()
    assert result.loc[~result["landsat_valid"], "s3_day_difference"].eq(0).all()
    assert "NDVI_1km" in result


def test_general_output_only_exposes_requested_scientific_variables():
    request = DatasetRequest(
        variables=["NDVI"],
        target_resolution_m=100,
        dataset_purpose=DatasetPurpose.GENERAL_DOWNLOAD,
    )

    result = build_unified_dataset(_landsat(), _sentinel(), _nodes(), request)

    assert "NDVI" in result and "NDVI_1km" in result
    assert "LST_K" not in result and "LST_1km" not in result
    assert set(result["date"]) == {"2024-07-14", "2024-07-15"}


def test_quality_report_summarizes_gaps_and_training_coverage():
    request = DatasetRequest(
        variables=["LST", "NDVI"],
        target_resolution_m=100,
        dataset_purpose=DatasetPurpose.LST_DOWNSCALING,
    )
    frame = build_unified_dataset(_landsat(), _sentinel(), _nodes(), request)

    report = build_quality_report(frame, request, {"landsat": 1, "sentinel3": 0})

    assert report["status"] == "with_gaps"
    assert report["training_ready_rows"] == 2
    assert any(item["column"] == "NDVI" for item in report["missing_by_column"])


def test_result_package_contains_one_csv_and_its_reports(tmp_path):
    csv_path = tmp_path / "dataset_unificado.csv"
    metadata_path = tmp_path / "metadata.json"
    quality_path = tmp_path / "informe_calidad.json"
    csv_path.write_text("node_id,date\nn0,2024-07-15\n", encoding="utf-8")
    metadata_path.write_text("{}", encoding="utf-8")
    quality_path.write_text("{}", encoding="utf-8")

    package = _write_package(tmp_path / "descarga_datos.zip", [csv_path, metadata_path, quality_path])

    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
    assert names == ["dataset_unificado.csv", "metadata.json", "informe_calidad.json"]
