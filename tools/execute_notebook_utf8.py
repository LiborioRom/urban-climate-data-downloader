"""Ejecuta un notebook y lo guarda explícitamente como UTF-8 en Windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def execute_notebook(path: Path, timeout: int = 900) -> None:
    path = path.resolve()
    notebook = nbformat.reads(path.read_text(encoding="utf-8"), as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=notebook.metadata.get("kernelspec", {}).get("name", "python3"),
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    path.write_text(
        nbformat.writes(notebook, version=nbformat.NO_CONVERT),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    execute_notebook(args.notebook, timeout=args.timeout)


if __name__ == "__main__":
    main()
