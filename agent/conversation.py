from __future__ import annotations

import re
import unicodedata
from datetime import date

from schemas.request import DatasetRequest, ROIRequest, ROISpecification
from rules.variable_catalog import mentioned_variables


SYSTEM_PROMPT = """Convierte la petición del usuario en DatasetRequest.
Devuelve solo campos expresados o inequívocamente inferibles. No inventes fechas,
variables, resolución, coordenadas ni fuentes. Sevilla se refiere al ROI por defecto
del proyecto. Las credenciales nunca forman parte de esta conversación.
"""


def _plain(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def _local_parse(user_text: str) -> DatasetRequest:
    """Parser conservador para uso sin API y para mantener la UI operativa en local."""
    text = _plain(user_text)
    # Las variables planificadas también se reconocen para que el plan pueda
    # explicar que aún no son ejecutables, en vez de ignorarlas silenciosamente.
    variables = mentioned_variables(user_text, include_planned=True)

    roi = None
    if "sevilla" in text:
        roi = ROIRequest(
            specification=ROISpecification.PROJECT_DEFAULT,
            place_name="Sevilla",
            center_lat=37.4035,
            center_lon=-5.9810,
            half_side_m=750,
        )

    start_date = end_date = None
    full_dates = re.findall(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text)
    if full_dates:
        parsed = [date(*map(int, item)) for item in full_dates]
        start_date, end_date = parsed[0], parsed[-1]
    else:
        years = re.findall(r"\b(20\d{2})\b", text)
        if years and "mayo" in text and "octubre" in text:
            year = int(years[0])
            start_date, end_date = date(year, 5, 1), date(year, 10, 31)

    resolution_match = re.search(r"(?:a|resolucion(?: de)?)\s*(\d+(?:[.,]\d+)?)\s*(?:m|metros?)\b", text)
    resolution = float(resolution_match.group(1).replace(",", ".")) if resolution_match else None
    return DatasetRequest(
        roi=roi,
        start_date=start_date,
        end_date=end_date,
        variables=variables,
        target_resolution_m=resolution,
        user_notes=user_text,
    )


def interpret_request(user_text: str) -> DatasetRequest:
    """Interpreta una petición con el parser local conservador."""
    return _local_parse(user_text)
