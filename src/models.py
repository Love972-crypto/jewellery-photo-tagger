from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


STATUS_OK = "OK"
STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"
STATUS_DUPLICATE_TAG = "DUPLICATE_TAG"
STATUS_OCR_FAILED = "OCR_FAILED"
STATUS_TAG_NOT_FOUND = "TAG_NOT_FOUND"
STATUS_ERROR = "ERROR"

CORRECTION_CACHE_READY = "READY"
CORRECTION_CACHE_REVIEW_REQUIRED = "REVIEW_REQUIRED"
CORRECTION_CACHE_FAILED = "FAILED"

REPORT_COLUMNS = [
    "item_id",
    "original_filename",
    "detected_tag_number",
    "ocr_text_raw",
    "confidence_score",
    "final_filename",
    "output_folder",
    "status",
    "notes",
    "background_status",
    "background_mode",
    "transparent_filename",
    "background_notes",
    "correction_cache_status",
    "correction_cache_white_filename",
    "correction_cache_transparent_filename",
    "background_processing_seconds",
    "correction_finalize_seconds",
]


@dataclass(frozen=True)
class ProcessingSettings:
    enhance_enabled: bool = True
    save_debug_crops: bool = True
    confidence_threshold: float = 0.45
    output_format: str = "png"
    enhancement_mode: str = "fast"
    ocr_attempt_mode: str = "fast"
    hd_output_enabled: bool = False
    hd_scale: int = 2
    remove_background: bool = True
    background_output_mode: str = "white_and_transparent"
    ai_background_fallback_enabled: bool = True
    catalogue_layout_enabled: bool = True
    catalogue_canvas_width: int = 1200
    catalogue_canvas_height: int = 1500


@dataclass
class OCRTextBox:
    text: str
    confidence: float
    bbox: list[Any] | None = None
    source_rotation: str = "original"
    source_crop: str = "tag_crop"


@dataclass
class ParsedTag:
    tag_number: str = ""
    confidence: float = 0.0
    raw_text: str = ""
    status: str = STATUS_OCR_FAILED
    notes: str = ""
    evidence_count: int = 0
    strong_evidence_count: int = 0
    best_ocr_confidence: float = 0.0
    score_margin: float = 0.0
    best_source_rotation: str = ""
    best_source_crop: str = ""


@dataclass
class ProcessingResult:
    original_filename: str
    item_id: str = ""
    detected_tag_number: str = ""
    ocr_text_raw: str = ""
    confidence_score: float | str = 0.0
    final_filename: str = ""
    output_folder: str = ""
    status: str = STATUS_ERROR
    notes: str = ""
    background_status: str = ""
    background_mode: str = ""
    transparent_filename: str = ""
    background_notes: str = ""
    correction_cache_status: str = ""
    correction_cache_white_filename: str = ""
    correction_cache_transparent_filename: str = ""
    background_processing_seconds: float | str = ""
    correction_finalize_seconds: float | str = ""

    def to_report_row(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "original_filename": self.original_filename,
            "detected_tag_number": self.detected_tag_number,
            "ocr_text_raw": self.ocr_text_raw,
            "confidence_score": self.confidence_score,
            "final_filename": self.final_filename,
            "output_folder": self.output_folder,
            "status": self.status,
            "notes": self.notes,
            "background_status": self.background_status,
            "background_mode": self.background_mode,
            "transparent_filename": self.transparent_filename,
            "background_notes": self.background_notes,
            "correction_cache_status": self.correction_cache_status,
            "correction_cache_white_filename": self.correction_cache_white_filename,
            "correction_cache_transparent_filename": self.correction_cache_transparent_filename,
            "background_processing_seconds": self.background_processing_seconds,
            "correction_finalize_seconds": self.correction_finalize_seconds,
        }


@dataclass
class OutputPaths:
    root: Path
    processed_images: Path
    transparent_images: Path
    compressed_images_20kb: Path
    review_required: Path
    ai_review: Path
    background_review: Path
    correction_cache: Path
    debug_crops: Path
    report_csv: Path
    full_zip: Path
    processed_zip: Path
    transparent_zip: Path
    compressed_images_20kb_zip: Path
    debug_zip: Path


@dataclass
class BatchSummary:
    total: int = 0
    processed: int = 0
    ok: int = 0
    review_required: int = 0
    duplicate_tags: int = 0
    ocr_failed: int = 0
    tag_not_found: int = 0
    errors: int = 0
    elapsed_seconds: float = 0.0
    rows: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]], elapsed_seconds: float = 0.0) -> "BatchSummary":
        summary = cls(total=len(rows), processed=len(rows), elapsed_seconds=elapsed_seconds, rows=rows)
        for row in rows:
            status = str(row.get("status", ""))
            if status == STATUS_OK:
                summary.ok += 1
            elif status == STATUS_DUPLICATE_TAG:
                summary.duplicate_tags += 1
            elif status == STATUS_REVIEW_REQUIRED:
                summary.review_required += 1
            elif status == STATUS_OCR_FAILED:
                summary.ocr_failed += 1
                summary.review_required += 1
            elif status == STATUS_TAG_NOT_FOUND:
                summary.tag_not_found += 1
                summary.review_required += 1
            elif status == STATUS_ERROR:
                summary.errors += 1
        return summary
