from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def ensure_dir(p: Path) -> None:
    """note: Ensures a directory exists (mkdir -p semantics)."""
    p.mkdir(parents=True, exist_ok=True)


def write_csv_header_if_needed(path: Path, header: list[str]) -> None:
    """note: Writes a CSV header only if the file does not already exist."""
    if path.exists():
        return
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)


def append_csv_row(path: Path, row: Iterable[str]) -> None:
    """note: Appends a row to a CSV file, creating parent directories as needed."""
    ensure_dir(path.parent)
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(list(row))
