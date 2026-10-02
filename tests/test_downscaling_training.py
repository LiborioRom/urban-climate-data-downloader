import pandas as pd
import json
import pytest

from tools.downscaling_training import (
    ORIGINAL_DOWNSCALING_FEATURES,
    add_rain_features,
    assign_temporal_splits,
    build_grid_edges,
    find_latest_downscaling_dataset,
    prepare_training_rows,
    prepare_coincident_training_rows,
    select_numeric_features,
)


def _frame():
    rows = []
    for day in range(1, 7):
        for node_id, row, col, lat, lon in (
            ("n0", 0, 0, 37.40, -5.98),
            ("n1", 0, 1, 37.40, -5.979),
        ):
            rows.append({
                "node_id": node_id, "date": f"2024-07-{day:02d}", "row": row, "col": col,
                "latitude": lat, "longitude": lon, "LST_K": 305 + day,
                "LST_1km": 303 + day, "NDVI": 0.3, "landsat_valid": True,
            })
    return pd.DataFrame(rows)


def test_preparation_features_and_split_are_model_ready():
    frame = prepare_training_rows(_frame(), warm_months=[5, 6, 7, 8, 9, 10])
    features = select_numeric_features(frame)
    frame["split"] = assign_temporal_splits(frame)

    assert "LST_1km" in features
    assert "LST_K" not in features
    assert set(frame["split"]) == {"train", "validation", "test"}
    assert frame.groupby("date")["split"].nunique().max() == 1


def test_grid_edges_are_bidirectional():
    edges = build_grid_edges(_frame())

    assert len(edges) == 2
    assert set(map(tuple, edges[["source_node_id", "target_node_id"]].to_numpy())) == {
        ("n0", "n1"), ("n1", "n0")
    }


def test_logarithmic_rain_replaces_raw_accumulation_as_model_feature():
    frame = _frame().assign(rain_3d=4.0, rain_3d_log=1.609, is_rainy=1)

    features = select_numeric_features(frame)

    assert "rain_3d_log" in features
    assert "is_rainy" in features
    assert "rain_3d" not in features


def test_rain_fallback_preserves_missing_values():
    frame = add_rain_features(pd.DataFrame({"rain_3d": [0.0, 4.0, None]}))

    assert frame["is_rainy"].iloc[:2].tolist() == [0.0, 1.0]
    assert pd.isna(frame["is_rainy"].iloc[2])
    assert pd.isna(frame["rain_3d_log"].iloc[2])


def test_latest_dataset_ignores_general_and_incomplete_runs(tmp_path):
    general = tmp_path / "runs" / "2024-01-01" / "data_downloads"
    tfm = tmp_path / "runs" / "2024-01-02" / "data_downloads"
    incomplete = tmp_path / "runs" / "2024-01-03" / "data_downloads"
    for folder in (general, tfm, incomplete):
        folder.mkdir(parents=True)
        (folder / "dataset_unificado.csv").write_text("dataset_purpose\nlst_downscaling\n", encoding="utf-8")
    for folder, purpose in ((general, "general_download"), (tfm, "lst_downscaling")):
        (folder / "metadata.json").write_text(
            json.dumps({"request": {"dataset_purpose": purpose}}), encoding="utf-8"
        )
        (folder / "informe_calidad.json").write_text("{}", encoding="utf-8")

    assert find_latest_downscaling_dataset(tmp_path) == tfm / "dataset_unificado.csv"


def test_general_dataset_is_rejected_before_training():
    frame = _frame().assign(dataset_purpose="general_download")

    try:
        prepare_training_rows(frame)
    except ValueError as exc:
        assert "Proyecto TFM-GNN" in str(exc)
    else:
        raise AssertionError("Se aceptó un dataset de propósito general.")


def test_coincident_preparation_reproduces_original_residual_features():
    frame = _frame().assign(
        s3_day_difference=0,
        NDVI_1km=0.2,
        NDBI=0.4,
        NDBI_1km=0.1,
        ALBEDO=0.25,
        ALBEDO_1km=0.20,
        aspect_ratio=0.6,
        aspect_ratio_1km=0.4,
    )

    prepared = prepare_coincident_training_rows(frame, warm_months=[7])

    assert prepared["training_ready"].all()
    assert ORIGINAL_DOWNSCALING_FEATURES == [
        "LST_1km", "NDVI_1km", "NDVI_delta", "NDBI_delta",
        "ALBEDO_delta", "aspect_ratio_delta",
    ]
    assert prepared["Delta_LST"].iloc[0] == 2.0
    assert prepared["NDVI_delta"].iloc[0] == pytest.approx(0.1)
    assert prepared["NDBI_delta"].iloc[0] == pytest.approx(0.3)


def test_coincident_preparation_rejects_nearby_dates_and_incomplete_auxiliaries():
    frame = _frame().assign(
        s3_day_difference=0,
        NDVI_1km=0.2,
        NDBI=0.4,
        NDBI_1km=0.1,
        ALBEDO=0.25,
        ALBEDO_1km=0.20,
        aspect_ratio=0.6,
        aspect_ratio_1km=0.4,
    )
    frame.loc[0, "s3_day_difference"] = 1
    frame.loc[1, "ALBEDO"] = None

    prepared = prepare_coincident_training_rows(frame)

    assert not prepared.loc[0, "training_ready"]
    assert not prepared.loc[1, "training_ready"]
