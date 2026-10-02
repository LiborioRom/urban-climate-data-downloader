from datetime import date

from schemas.request import DateSelectionMode, DatasetRequest
from tools.date_windows import build_recurring_windows, describe_date_selection, effective_date_windows


def test_recurring_window_builds_only_selected_years():
    windows = build_recurring_windows(date(2000, 5, 15), date(2000, 10, 20), [2019, 2021, 2024])

    assert [(str(item.start_date), str(item.end_date)) for item in windows] == [
        ("2019-05-15", "2019-10-20"),
        ("2021-05-15", "2021-10-20"),
        ("2024-05-15", "2024-10-20"),
    ]


def test_recurring_window_can_cross_new_year():
    windows = build_recurring_windows(date(2000, 11, 1), date(2000, 2, 28), [2022])

    assert str(windows[0].start_date) == "2022-11-01"
    assert str(windows[0].end_date) == "2023-02-28"


def test_continuous_request_exposes_one_effective_window():
    request = DatasetRequest(start_date="2024-06-01", end_date="2024-06-30")

    windows = effective_date_windows(request)

    assert len(windows) == 1
    assert describe_date_selection(request) == "2024-06-01 → 2024-06-30"


def test_recurring_description_lists_selected_years():
    windows = build_recurring_windows(date(2000, 5, 1), date(2000, 10, 31), [2020, 2022])
    request = DatasetRequest(
        date_selection_mode=DateSelectionMode.RECURRING_WINDOW,
        date_windows=windows,
    )

    assert describe_date_selection(request) == "01/05 → 31/10 en 2020, 2022"

