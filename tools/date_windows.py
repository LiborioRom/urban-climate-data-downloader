from __future__ import annotations

from datetime import date

from schemas.request import DateSelectionMode, DateWindow, DatasetRequest


def build_recurring_windows(start_template: date, end_template: date, years: list[int]) -> list[DateWindow]:
    """Construye ventanas exactas; los años identifican el comienzo de cada temporada."""
    crosses_year = (end_template.month, end_template.day) < (start_template.month, start_template.day)
    windows = []
    for year in sorted(set(years)):
        try:
            start = date(year, start_template.month, start_template.day)
            end_year = year + 1 if crosses_year else year
            end = date(end_year, end_template.month, end_template.day)
        except ValueError as exc:
            raise ValueError(f"La ventana no es válida para {year}: {exc}") from exc
        windows.append(DateWindow(start_date=start, end_date=end))
    return windows


def effective_date_windows(request: DatasetRequest) -> list[DateWindow]:
    if request.date_selection_mode == DateSelectionMode.RECURRING_WINDOW:
        return list(request.date_windows)
    if request.start_date and request.end_date:
        return [DateWindow(start_date=request.start_date, end_date=request.end_date)]
    return []


def describe_date_selection(request: DatasetRequest) -> str:
    windows = effective_date_windows(request)
    if not windows:
        return "Pendiente"
    if request.date_selection_mode == DateSelectionMode.CONTINUOUS:
        return f"{windows[0].start_date} → {windows[0].end_date}"
    first = windows[0]
    start_md = first.start_date.strftime("%d/%m")
    end_md = first.end_date.strftime("%d/%m")
    years = ", ".join(str(window.start_date.year) for window in windows)
    return f"{start_md} → {end_md} en {years}"

