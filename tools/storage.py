from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_data_root() -> Path:
    """Raíz configurable para ejecuciones y caché; por defecto permanece en el proyecto."""
    configured = os.getenv("LST_DATA_ROOT", "").strip()
    return Path(configured).expanduser().resolve() if configured else PROJECT_ROOT


def new_run_directory(area_count: int, root: Path | None = None) -> Path:
    base = (root or get_data_root()) / "runs"
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")[:-3]
    label = f"multi_{area_count}" if area_count > 1 else "single"
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label)
    return base / f"{stamp}_{safe_label}"


def shared_cache_directory(root: Path | None = None) -> Path:
    selected = (root or get_data_root()).resolve()
    if selected == PROJECT_ROOT.resolve():
        return PROJECT_ROOT / "data_downloads" / "cache"
    return selected / "cache"
