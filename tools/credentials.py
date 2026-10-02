from __future__ import annotations

import os
from collections.abc import Mapping


DATA_CREDENTIAL_NAMES = ("EE_PROJECT", "SH_CLIENT_ID", "SH_CLIENT_SECRET", "CDS_API_KEY")

# Earth Engine cubre Landsat, Sentinel-2 y ERA5-Land. CDS queda como recuperación opcional.
REQUIRED_PIPELINE_CREDENTIALS = ("EE_PROJECT", "SH_CLIENT_ID", "SH_CLIENT_SECRET")


def credential_status() -> dict[str, bool]:
    return {
        name: bool(os.getenv(name))
        for name in DATA_CREDENTIAL_NAMES
    }


def missing_required_credentials() -> list[str]:
    return [name for name in REQUIRED_PIPELINE_CREDENTIALS if not os.getenv(name)]


def set_session_credentials(values: Mapping[str, str]) -> list[str]:
    """Carga secretos en el proceso y en kernels hijos; no los persiste ni registra."""
    accepted = set(DATA_CREDENTIAL_NAMES)
    configured: list[str] = []
    for name, raw_value in values.items():
        value = raw_value.strip()
        if name in accepted and value:
            os.environ[name] = value
            configured.append(name)
    return configured
