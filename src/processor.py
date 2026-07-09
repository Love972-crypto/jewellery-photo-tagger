from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Callable

from .ai_enhancement import upscale_safely
from .background_processor import (
    BACKGROUND_DISABLED,
    BACKGROUND_FAILED,
    BACKGROUND_OK,
    BACKGROUND_REVIEW_REQUIRED,
    remove_background,
    save_rgba_png,
)
from .file_manager import create_zip_from_folder, setup_output_paths, unique_png_path
from .image_enhancement import enhance_image, load_image, preprocess_for_ocr, save_png
from .models import (
    BatchSummary,
    OCRTextBox,
    OutputPaths,
    ProcessingResult,
    ProcessingSettings,
    STATUS_DUPLICATE_TAG,
    STATUS_OK,
    STATUS_REVIEW_REQUIRED,
)
from .ocr_engine import OCREngine
from .report_generator import read_report, update_report_row, write_report
from .tag_detection import detect_tag_crops, rotated_versions, save_debug_crops
from .tag_parser import is_valid_manual_tag, parse_tag_from_ocr

ProgressCallback = Callable[[int, int, str, dict[str, int]], None]


class BatchProcessor:
    def __init__(self, output_root: Path, settings: ProcessingSettings, ocr_engine: OCREngine, project_root: Path | None = None) -> None:
        self.output_paths: OutputPaths = setup_output_paths(output_root)
        self.settings = settings
        self.ocr_engine = ocr_engine
        self.project_root = project_root
        self.seen_tags: dict[str, int] = {}

    def process_images(self, image_paths: list[Path], progress_callback: ProgressCallback | None = None) -> BatchSummary:
        rows: list[dict] = []
        started_at = time.perf_counter()
        counters = {"ok": 0, "review": 0, "errors": 0, "duplicates": 0}

        for index, image_path in enumerate(image_paths, start=1):
            if progress_callback:
                progress_callback(index - 1, len(image_paths), image_path.name, counters)

            if hasattr(self.ocr_engine, "set_current_image"):
                self.ocr_engine.set_current_image(image_path)
            result = self._process_one(image_path)
            rows.append(result.to_report_row())

            if result.status == STATUS_OK:
                counters["ok"] += 1
            elif result.status == STATUS_DUPLICATE_TAG:
                counters["duplicates"] += 1
            elif result.status == "ERROR":
                counters["errors"] += 1
            else:
                counters["review"] += 1

            write_report(rows, self.output_paths.report_csv)

        elapsed = time.perf_counter() - started_at
        write_report(rows, self.output_paths.report_csv)
        self._create_output_zips()
        if progress_callback:
            progress_callback(len(image_paths), len(image_paths), "Complete", counters)
        return BatchSummary.from_rows(rows, elapsed)

    def _process_one(self, image_path: Path) -> ProcessingResult:
        original_filename = image_path.name
        original_stem = image_path.stem
        try:
            image = load_image(image_path)
            output_image = enhance_image(image, mode=self.settings.enhancement_mode) if self.settings.enhance_enabled else image.copy()
            crops = detect_tag_crops(output_image)
            if self.settings.save_debug_crops:
                save_debug_crops(crops, original_stem, self.output_paths.debug_crops)

            boxes, ocr_notes = self._run_ocr(crops)
            parsed = parse_tag_from_ocr(boxes, self.settings.confidence_threshold)
            if ocr_notes and parsed.notes:
                parsed.notes = f"{parsed.notes} {' '.join(ocr_notes[:2])}"
            elif ocr_notes:
                parsed.notes = " ".join(ocr_notes[:2])

            if parsed.status == STATUS_OK:
                final_path, duplicate_by_file = unique_png_path(self.output_paths.processed_images, parsed.tag_number)
                duplicate_by_batch = parsed.tag_number in self.seen_tags
                self.seen_tags[parsed.tag_number] = self.seen_tags.get(parsed.tag_number, 0) + 1
                status = STATUS_DUPLICATE_TAG if duplicate_by_batch or duplicate_by_file else STATUS_OK
                notes = parsed.notes
                background_status = BACKGROUND_DISABLED
                background_mode = "none"
                background_notes = "Background removal disabled."
                transparent_filename = ""
                output_folder_name = "processed_images"
                final_rgba_output = None

                if self.settings.hd_output_enabled and self.project_root:
                    output_image, hd_note = upscale_safely(output_image, scale=self.settings.hd_scale)
                    notes = f"{notes} {hd_note}"

                if self.settings.remove_background:
                    background_mode = self.settings.background_output_mode
                    background_result = remove_background(
                        output_image,
                        alpha_matting=self._background_alpha_matting_enabled(),
                        catalogue_layout=self.settings.catalogue_layout_enabled,
                        canvas_size=(self.settings.catalogue_canvas_width, self.settings.catalogue_canvas_height),
                        max_side=self.settings.background_max_side,
                        model_name=self.settings.background_model_name,
                    )
                    background_status = background_result.status
                    background_notes = background_result.notes
                    if background_result.status != BACKGROUND_OK:
                        review_path, _ = unique_png_path(self.output_paths.background_review, f"BG_REVIEW_{original_stem}")
                        save_png(output_image, review_path)
                        if background_result.transparent_rgba is not None:
                            transparent_review_path = self.output_paths.background_review / f"{review_path.stem}_transparent.png"
                            save_rgba_png(background_result.transparent_rgba, transparent_review_path)
                        return ProcessingResult(
                            original_filename=original_filename,
                            detected_tag_number=parsed.tag_number,
                            ocr_text_raw=parsed.raw_text,
                            confidence_score=parsed.confidence,
                            final_filename=review_path.name,
                            output_folder="background_review",
                            status=STATUS_REVIEW_REQUIRED,
                            notes=f"{notes} Background removal needs review.",
                            background_status=background_status,
                            background_mode=background_mode,
                            background_notes=background_notes,
                        )

                    if background_result.white_bgr is not None and background_mode in {"white_and_transparent", "white_only"}:
                        output_image = background_result.white_bgr
                    if background_result.transparent_rgba is not None and background_mode == "white_and_transparent":
                        transparent_path, _ = unique_png_path(self.output_paths.transparent_images, parsed.tag_number)
                        save_rgba_png(background_result.transparent_rgba, transparent_path)
                        transparent_filename = transparent_path.name
                    if background_mode == "transparent_only" and background_result.transparent_rgba is not None:
                        final_path, duplicate_by_file = unique_png_path(self.output_paths.transparent_images, parsed.tag_number)
                        status = STATUS_DUPLICATE_TAG if duplicate_by_batch or duplicate_by_file else STATUS_OK
                        output_folder_name = "transparent_images"
                        final_rgba_output = background_result.transparent_rgba
                        transparent_filename = final_path.name

                if final_rgba_output is not None:
                    save_rgba_png(final_rgba_output, final_path)
                else:
                    save_png(output_image, final_path)
                if status == STATUS_DUPLICATE_TAG:
                    notes = f"{notes} Duplicate tag saved safely with suffix."
                return ProcessingResult(
                    original_filename=original_filename,
                    detected_tag_number=parsed.tag_number,
                    ocr_text_raw=parsed.raw_text,
                    confidence_score=parsed.confidence,
                    final_filename=final_path.name,
                    output_folder=output_folder_name,
                    status=status,
                    notes=notes,
                    background_status=background_status,
                    background_mode=background_mode,
                    transparent_filename=transparent_filename,
                    background_notes=background_notes,
                )

            review_path, _ = unique_png_path(self.output_paths.review_required, f"REVIEW_{original_stem}")
            save_png(output_image, review_path)
            return ProcessingResult(
                original_filename=original_filename,
                detected_tag_number=parsed.tag_number,
                ocr_text_raw=parsed.raw_text,
                confidence_score=parsed.confidence,
                final_filename=review_path.name,
                output_folder="review_required",
                status=parsed.status,
                notes=parsed.notes or "Tag number is unclear. Please review this photo.",
            )
        except Exception as exc:
            return ProcessingResult(
                original_filename=original_filename,
                status="ERROR",
                notes=f"This photo could not be processed. Please check it manually. {exc}",
            )

    def _background_alpha_matting_enabled(self) -> bool:
        return (
            self.settings.enhancement_mode == "quality"
            and self.settings.background_model_name == "u2net"
            and self.settings.background_max_side >= 2000
        )

    def _run_ocr(self, crops) -> tuple[list[OCRTextBox], list[str]]:
        all_boxes: list[OCRTextBox] = []
        notes: list[str] = []
        if not crops:
            return all_boxes, ["No tag crop was found."]

        attempts_done = 0
        fast_mode = self.settings.ocr_attempt_mode == "fast"
        max_attempts = 3 if fast_mode else 8
        crop_limit = 3 if fast_mode else 6
        for crop in crops[:crop_limit]:
            for rotation_label, rotated in self._rotation_attempts(crop):
                if attempts_done >= max_attempts:
                    notes.append("OCR stopped after fast attempt limit. Use manual review if the tag is still unclear.")
                    return all_boxes, notes
                try:
                    attempts_done += 1
                    prepared = preprocess_for_ocr(rotated, upscale=2, max_side=900)
                    boxes = self.ocr_engine.read_text(prepared, source_rotation=rotation_label, source_crop=crop.label)
                    all_boxes.extend(boxes)
                    parsed = parse_tag_from_ocr(all_boxes, self.settings.confidence_threshold)
                    if parsed.status == STATUS_OK:
                        return all_boxes, notes
                except Exception as exc:
                    notes.append(f"OCR skipped {crop.label}/{rotation_label}: {exc}")
        return all_boxes, notes

    def _rotation_attempts(self, crop) -> list[tuple[str, object]]:
        height, width = crop.image.shape[:2]
        if crop.label.startswith("fallback"):
            return [("original", crop.image)]

        rotations = rotated_versions(crop.image)
        rotation_map = {label: image for label, image in rotations}
        if height > width * 1.2:
            order = ["ccw90", "cw90", "original"]
        elif width > height * 1.2:
            order = ["original", "rot180"]
        else:
            order = ["original", "ccw90", "cw90", "rot180"]
        return [(label, rotation_map[label]) for label in order if label in rotation_map]

    def _create_output_zips(self) -> None:
        create_zip_from_folder(self.output_paths.processed_images, self.output_paths.processed_zip)
        create_zip_from_folder(self.output_paths.transparent_images, self.output_paths.transparent_zip)
        if self.settings.save_debug_crops:
            create_zip_from_folder(self.output_paths.debug_crops, self.output_paths.debug_zip)
        create_zip_from_folder(self.output_paths.root, self.output_paths.full_zip, exclude_zip_files=True)


def apply_manual_correction(output_root: Path, original_filename: str, corrected_tag: str) -> tuple[bool, str]:
    corrected_tag = corrected_tag.strip()
    if not is_valid_manual_tag(corrected_tag):
        return False, "Enter a numeric tag number with 5 to 8 digits."

    paths = setup_output_paths(output_root)
    report = read_report(paths.report_csv)
    if report.empty:
        return False, "No report is available yet."

    matches = report[report["original_filename"] == original_filename]
    if matches.empty:
        return False, "This item was not found in the report."

    row = matches.iloc[0]
    current_folder = str(row.get("output_folder", ""))
    current_filename = str(row.get("final_filename", ""))
    if not current_filename:
        return False, "No review image is available for this item."

    source = paths.root / current_folder / current_filename
    if not source.exists():
        source = paths.review_required / current_filename
    if not source.exists():
        return False, "The review image file could not be found."

    destination, duplicate = unique_png_path(paths.processed_images, corrected_tag)
    shutil.move(str(source), str(destination))
    status = STATUS_DUPLICATE_TAG if duplicate else STATUS_OK
    update_report_row(
        paths.report_csv,
        original_filename,
        {
            "detected_tag_number": corrected_tag,
            "confidence_score": "manual",
            "final_filename": destination.name,
            "output_folder": "processed_images",
            "status": status,
            "notes": "Manual correction saved.",
        },
    )
    create_zip_from_folder(paths.processed_images, paths.processed_zip)
    create_zip_from_folder(paths.transparent_images, paths.transparent_zip)
    create_zip_from_folder(paths.root, paths.full_zip, exclude_zip_files=True)
    return True, "Correction saved and image moved to processed images."
