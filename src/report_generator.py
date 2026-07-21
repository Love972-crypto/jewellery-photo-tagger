from __future__ import annotations

from pathlib import Path
from threading import RLock, get_ident
from typing import Any

import pandas as pd

from .models import REPORT_COLUMNS

_REPORT_LOCK = RLock()


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append({column: row.get(column, "") for column in REPORT_COLUMNS})
    return normalized


def write_report(rows: list[dict[str, Any]], report_path: Path) -> Path:
    with _REPORT_LOCK:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(normalize_rows(rows), columns=REPORT_COLUMNS)
        _write_frame_atomic(frame, report_path)
    return report_path


def read_report(report_path: Path) -> pd.DataFrame:
    with _REPORT_LOCK:
        if not report_path.exists():
            return pd.DataFrame(columns=REPORT_COLUMNS)
        frame = pd.read_csv(report_path, dtype=str).fillna("")
        for column in REPORT_COLUMNS:
            if column not in frame.columns:
                frame[column] = ""
        return frame[REPORT_COLUMNS]


def update_report_row(report_path: Path, item_selector: str, updates: dict[str, Any]) -> pd.DataFrame:
    with _REPORT_LOCK:
        frame = read_report(report_path)
        if frame.empty:
            return frame
        selector = str(item_selector).strip()
        mask = frame["item_id"] == selector if selector else frame["item_id"] == "__missing__"
        if not mask.any():
            mask = frame["original_filename"] == selector
        if int(mask.sum()) != 1:
            return frame
        for key, value in updates.items():
            if key in frame.columns:
                frame.loc[mask, key] = str(value)
        _write_frame_atomic(frame, report_path)
        return frame


def _write_frame_atomic(frame: pd.DataFrame, report_path: Path) -> None:
    temporary = report_path.with_name(f".{report_path.name}.{get_ident()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(report_path)
    finally:
        temporary.unlink(missing_ok=True)
