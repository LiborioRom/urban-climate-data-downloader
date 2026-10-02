from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SECRET_ENV_NAMES = ("SH_CLIENT_ID", "SH_CLIENT_SECRET", "CDS_API_KEY")


def exposed_defaults(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [
        name
        for name in SECRET_ENV_NAMES
        if re.search(rf'os\.getenv\("{name}",\s*".+?"\)', text)
    ]


def sanitize(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        changed = False
        for name in SECRET_ENV_NAMES:
            pattern = rf'os\.getenv\("{name}",\s*"[^"]*"\)'
            source, count = re.subn(pattern, f'os.getenv("{name}", "")', source)
            changed = changed or bool(count)
        if changed:
            cell["source"] = source.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    path = parser.parse_args().notebook
    sanitize(path)
    remaining = exposed_defaults(path)
    if remaining:
        raise SystemExit(f"Siguen existiendo defaults no vacíos: {remaining}")
