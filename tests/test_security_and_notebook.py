import json
from pathlib import Path

from agent.conversation import SYSTEM_PROMPT
from schemas.request import DatasetPurpose, DatasetRequest, ROIRequest, ROISpecification
from agent.planner import build_plan
from tools.notebook_pipeline import prepare_notebook
from schemas.request import DateSelectionMode, DateWindow


ROOT = Path(__file__).resolve().parents[1]


def test_prompt_has_no_credential_values_or_names():
    lowered = SYSTEM_PROMPT.lower()
    assert "password" not in lowered
    assert "api_key" not in lowered
    assert "cdsapi" not in lowered


def test_prepared_notebook_injects_only_controlled_parameters(tmp_path):
    request = DatasetRequest(
        roi=ROIRequest(specification=ROISpecification.PROJECT_DEFAULT, place_name="Sevilla", center_lat=37.4035, center_lon=-5.981, half_side_m=750),
        start_date="2024-07-15",
        end_date="2024-07-15",
        variables=["LST"],
        target_resolution_m=100,
    )
    target = prepare_notebook(build_plan(request), tmp_path)
    notebook = json.loads(target.read_text(encoding="utf-8"))
    injected = "".join(notebook["cells"][5]["source"])
    assert "injected-parameters" in notebook["cells"][5]["metadata"]["tags"]
    assert "exec(" not in injected
    assert "eval(" not in injected
    assert "SH_CLIENT_SECRET" not in injected
    assert "START_DATE = '2024-07-15'" in injected
    assert "END_DATE = '2024-07-15'" in injected
    assert "MODE =" not in injected
    assert "TEST_DATE" not in injected
    assert "S3_MATCH_LANDSAT_ONLY" not in injected
    assert "S3_REFERENCE_TOLERANCE_DAYS = 3" in injected
    assert repr(str(ROOT / "data_downloads" / "cache")) in injected
    assert "CACHE_DIR = OUTPUT_DIR / 'cache'" not in injected


def test_tfm_notebook_keeps_daily_sentinel3_between_landsat_dates(tmp_path):
    request = DatasetRequest(
        roi=ROIRequest(
            specification=ROISpecification.CENTER_AND_HALF_SIDE,
            center_lat=37.4,
            center_lon=-5.98,
            half_side_m=750,
        ),
        start_date="2024-05-01",
        end_date="2024-10-31",
        variables=["LST"],
        target_resolution_m=100,
        dataset_purpose=DatasetPurpose.LST_DOWNSCALING,
    )

    target = prepare_notebook(build_plan(request), tmp_path)
    notebook = json.loads(target.read_text(encoding="utf-8"))
    injected = "".join(notebook["cells"][5]["source"])

    assert "S3_MATCH_LANDSAT_ONLY" not in injected
    assert "PAIRED_OUTPUT =" in injected


def test_prepared_notebook_injects_roi_rotation(tmp_path):
    request = DatasetRequest(
        roi=ROIRequest(
            specification=ROISpecification.CENTER_AND_HALF_SIDE,
            center_lat=37.4,
            center_lon=-5.98,
            half_side_m=750,
            rotation_deg=25,
        ),
        start_date="2024-07-15",
        end_date="2024-07-15",
        variables=["LST"],
        target_resolution_m=100,
    )
    target = prepare_notebook(build_plan(request), tmp_path)
    notebook = json.loads(target.read_text(encoding="utf-8"))
    injected = "".join(notebook["cells"][5]["source"])

    assert "ROI_ROTATION_DEG = 25.0" in injected


def test_prepared_notebook_injects_exact_recurring_windows(tmp_path):
    request = DatasetRequest(
        roi=ROIRequest(
            specification=ROISpecification.CENTER_AND_HALF_SIDE,
            center_lat=37.4,
            center_lon=-5.98,
            half_side_m=750,
        ),
        start_date="2020-05-15",
        end_date="2022-10-20",
        date_selection_mode=DateSelectionMode.RECURRING_WINDOW,
        date_windows=[
            DateWindow(start_date="2020-05-15", end_date="2020-10-20"),
            DateWindow(start_date="2022-05-15", end_date="2022-10-20"),
        ],
        variables=["LST"],
        target_resolution_m=100,
    )

    target = prepare_notebook(build_plan(request), tmp_path)
    notebook = json.loads(target.read_text(encoding="utf-8"))
    injected = "".join(notebook["cells"][5]["source"])

    assert "DATE_WINDOWS = [('2020-05-15', '2020-10-20'), ('2022-05-15', '2022-10-20')]" in injected
    assert "ALLOWED_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]" in injected


def test_canonical_notebook_rotation_cell_is_valid_python():
    notebook = json.loads((ROOT / "01_descarga_datos_local.ipynb").read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "roi_projected = rotate_geometry" in "".join(cell.get("source", []))
    )

    compile(source, "roi_rotation_cell", "exec")
    assert "rotated_x" in source
    assert "rotated_y" in source
