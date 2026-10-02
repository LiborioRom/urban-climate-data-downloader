from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path


def write_json_resilient(
    path: str | Path,
    payload: dict,
    *,
    retries: int = 8,
    initial_delay_seconds: float = 0.05,
) -> None:
    """Escribe JSON tolerando bloqueos breves de OneDrive y Windows Defender."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    last_error: OSError | None = None

    for attempt in range(max(1, retries)):
        temporary = target.with_name(
            f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(serialized, encoding="utf-8")
            os.replace(temporary, target)
            return
        except OSError as exc:
            last_error = exc
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt + 1 < max(1, retries):
                delay = min(initial_delay_seconds * (2 ** attempt), 0.8)
                time.sleep(delay)

    # OneDrive puede permitir reescribir el destino mientras bloquea
    # temporalmente MoveFileEx/os.replace. El lector tolera JSON incompleto.
    try:
        target.write_text(serialized, encoding="utf-8")
        return
    except OSError:
        if last_error is not None:
            raise last_error
        raise
