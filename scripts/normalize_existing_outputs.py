from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.dataset_outputs import (
    LANDSAT_PREFERRED_COLS,
    SENTINEL_PREFERRED_COLS,
    build_landsat_final,
    build_sentinel_final,
)


def _replace_with_backup(path: Path, normalized: pd.DataFrame, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / path.name
    if not backup.exists():
        shutil.copy2(path, backup)
    temporary = path.with_suffix(".normalized.tmp.csv")
    normalized.to_csv(temporary, index=False)
    temporary.replace(path)


def normalize_outputs(output_dir: Path) -> dict[str, tuple[int, int]]:
    backup_dir = output_dir / "cache" / "legacy_full_outputs"
    results: dict[str, tuple[int, int]] = {}

    landsat_path = output_dir / "dataset_landsat_sin_downscaling.csv"
    if landsat_path.exists():
        landsat_raw = pd.read_csv(landsat_path)
        landsat = landsat_raw.loc[:, LANDSAT_PREFERRED_COLS] if list(landsat_raw.columns) == LANDSAT_PREFERRED_COLS else build_landsat_final(landsat_raw)
        _replace_with_backup(landsat_path, landsat, backup_dir)
        results[landsat_path.name] = landsat.shape

    sentinel_path = output_dir / "dataset_sentinel3_1km_sin_downscaling.csv"
    if sentinel_path.exists():
        sentinel_raw = pd.read_csv(sentinel_path)
        if list(sentinel_raw.columns) == SENTINEL_PREFERRED_COLS:
            sentinel, filled = sentinel_raw, 0
        else:
            sentinel, filled = build_sentinel_final(sentinel_raw)
        _replace_with_backup(sentinel_path, sentinel, backup_dir)
        results[sentinel_path.name] = sentinel.shape
        print("NDVI Sentinel completados temporalmente:", filled)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normaliza los CSV ya existentes al esquema final acordado.")
    parser.add_argument("output_dir", nargs="?", type=Path, default=Path("data_downloads"))
    for name, shape in normalize_outputs(parser.parse_args().output_dir).items():
        print(name, shape)
