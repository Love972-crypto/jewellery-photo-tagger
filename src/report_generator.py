from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .models import REPORT_COLUMNS


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append({column: row.get(column, "") for column in REPORT_COLUMNS})
    return normalized


def write_report(rows: list[dict[str, Any]], report_path: Path) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(normalize_rows(rows), columns=REPORT_COLUMNS)
    frame.to_csv(report_path, index=False)
    return report_path


def read_report(report_path: Path) -> pd.DataFrame:
    if not report_path.exists():
        return pd.DataFrame(columns=REPORT_COLUMNS)
    frame = pd.read_csv(report_path, dtype=str).fillna("")
    for column in REPORT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[REPORT_COLUMNS]


def update_report_row(report_path: Path, original_filename: str, updates: dict[str, Any]) -> pd.DataFrame:
    frame = read_report(report_path)
    if frame.empty:
        return frame
    mask = frame["original_filename"] == original_filename
    if not mask.any():
        return frame
    for key, value in updates.items():
        if key in frame.columns:
            frame.loc[mask, key] = str(value)
    frame.to_csv(report_path, index=False)
    return frame

