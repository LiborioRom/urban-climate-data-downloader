import numpy as np
import pandas as pd

from tools.multizone_downscaling import build_gnn_ready_dataset, choose_model_scope
from tools.sensor_translation import engineer_translation_features


def test_translation_features_do_not_mix_equal_pixel_ids_between_areas():
    frame = pd.DataFrame({
        "area_id": ["a", "a", "b", "b"],
        "date": pd.to_datetime(["2025-07-01"] * 4),
        "sentinel3_pixel_id": ["s3_0"] * 4,
        "node_id": ["n0", "n1", "n0", "n1"],
        "LST_K": [300.0, 302.0, 310.0, 312.0],
        "NDVI_S2_100m": [0.1, 0.3, 0.7, 0.9],
        "NDBI_S2_100m": [0.0, 0.2, 0.6, 0.8],
        "ALBEDO_S2_100m": [0.1, 0.2, 0.4, 0.5],
        "DEM": [10.0, 12.0, 30.0, 32.0],
        "aspect_ratio": [0.1, 0.3, 0.7, 0.9],
        "has_buildings": [0.0, 1.0, 0.0, 1.0],
    })

    result = engineer_translation_features(frame, require_target=True)

    assert np.allclose(result.groupby("area_id")["LST_spatial_anomaly_K"].mean(), 0)
    assert np.allclose(result.groupby("area_id")["NDVI_S2_delta"].mean(), 0)


def test_scope_requires_a_meaningful_local_improvement():
    assert choose_model_scope(2.0, None) == "general"
    assert choose_model_scope(2.0, 1.99, minimum_local_improvement=0.02) == "general"
    assert choose_model_scope(2.0, 1.90, minimum_local_improvement=0.02) == "local"


def test_gnn_ready_target_prioritizes_observed_landsat():
    unified = pd.DataFrame({
        "node_id": ["n0", "n0"],
        "date": ["2025-07-01", "2025-07-02"],
        "row": [0, 0], "col": [0, 0],
        "LST_K": [305.0, np.nan],
    })
    prediction_rows = pd.DataFrame({
        "node_id": ["n0", "n0"],
        "date": pd.to_datetime(["2025-07-01", "2025-07-02"]),
    })

    result = build_gnn_ready_dataset(
        unified, prediction_rows, np.array([304.0, 306.0]),
        area_id="zona", model_name="modelo", model_scope="local", test_mae_k=1.2,
    )

    assert result["LST_target_K"].tolist() == [305.0, 306.0]
    assert result["target_source"].tolist() == ["landsat_observed", "sentinel3_downscaled_local"]
    assert result["global_node_id"].tolist() == ["zona__n0", "zona__n0"]
    assert result["target_training_eligible"].tolist() == [1, 1]
