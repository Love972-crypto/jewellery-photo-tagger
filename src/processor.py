from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .ai_enhancement import upscale_safely
from .background_processor import (
    AI_FALLBACK_MODEL,
    BACKGROUND_AI_ACCEPTED,
    BACKGROUND_AI_MANUAL_REVIEW,
    BACKGROUND_AI_REVIEW_READY,
    BACKGROUND_DISABLED,
    BACKGROUND_FAILED,
    BACKGROUND_HYBRID_OK,
    BACKGROUND_MANUAL_ACCEPTED,
    BACKGROUND_OK,
    BACKGROUND_ORIGINAL_KEPT,
    BACKGROUND_REVIEW_REQUIRED,
    BackgroundResult,
    fuse_u2net_preservation_with_ai,
    remove_background,
    save_rgba_png,
)
from .compressed_export import invalidate_compressed_export
from .file_manager import (
    create_zip_from_folder,
    mark_output_archives_stale,
    safe_stem,
    setup_output_paths,
    unique_png_path,
)
from .image_enhancement import enhance_image, load_image, preprocess_for_ocr, save_png
from .models import (
    BatchSummary,
    CORRECTION_CACHE_FAILED,
    CORRECTION_CACHE_READY,
    CORRECTION_CACHE_REVIEW_REQUIRED,
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
from .tag_detection import (
    build_detected_tag_preservation_mask,
    build_tag_preservation_mask,
    detect_tag_crops,
    rotated_versions,
    save_debug_crops,
)
from .tag_parser import is_valid_manual_tag, parse_tag_from_ocr

ProgressCallback = Callable[[int, int, str, dict[str, int]], None]


@dataclass(frozen=True)
class _CorrectionCacheResult:
    cache_status: str
    background_status: str
    background_mode: str
    background_notes: str
    white_filename: str = ""
    transparent_filename: str = ""
    elapsed_seconds: float = 0.0


def _run_background_pipeline(
    image: np.ndarray,
    settings: ProcessingSettings,
    preserve_mask: np.ndarray | None = None,
) -> tuple[BackgroundResult, str, str]:
    common = {
        "catalogue_layout": settings.catalogue_layout_enabled,
        "canvas_size": (settings.catalogue_canvas_width, settings.catalogue_canvas_height),
    }
    if not settings.ai_background_fallback_enabled:
        result = remove_background(
            image,
            alpha_matting=settings.enhancement_mode == "quality",
            preserve_mask=preserve_mask,
            **common,
        )
        return result, result.status, result.notes

    birefnet_result = remove_background(
        image,
        alpha_matting=False,
        model_name=AI_FALLBACK_MODEL,
        # BiRefNet is the clean final base. Applying the OCR preservation mask
        # here can force the rectangular floor around a tag into the output.
        preserve_mask=None,
        **common,
    )
    u2net_result = remove_background(
        image,
        alpha_matting=settings.enhancement_mode == "quality",
        preserve_mask=preserve_mask,
        **common,
    )
    if birefnet_result.source_rgba is None:
        u2net_safe = u2net_result.status == BACKGROUND_OK and not u2net_result.ai_refinement_reasons
        if u2net_safe:
            fallback_result = u2net_result
            status = BACKGROUND_HYBRID_OK
        else:
            fallback_result = BackgroundResult(
                status=BACKGROUND_REVIEW_REQUIRED,
                transparent_rgba=u2net_result.transparent_rgba,
                white_bgr=u2net_result.white_bgr,
                notes=u2net_result.notes,
                safety_metrics=u2net_result.safety_metrics,
                ai_refinement_reasons=u2net_result.ai_refinement_reasons,
                source_rgba=u2net_result.source_rgba,
            )
            status = BACKGROUND_AI_MANUAL_REVIEW
        notes = (
            "Always Hybrid ran BiRefNet and U2Net. BiRefNet did not return a usable matte, "
            "so the U2Net result was used only when its residue checks were safe. "
            f"{u2net_result.notes}"
        )
        return fallback_result, status, notes
    if u2net_result.source_rgba is None:
        review_result = BackgroundResult(
            status=BACKGROUND_REVIEW_REQUIRED,
            transparent_rgba=birefnet_result.transparent_rgba,
            white_bgr=birefnet_result.white_bgr,
            notes="U2Net preservation was required but did not return a usable matte.",
            safety_metrics=birefnet_result.safety_metrics,
            source_rgba=birefnet_result.source_rgba,
        )
        notes = f"Always Hybrid ran BiRefNet and U2Net. {review_result.notes}"
        return review_result, BACKGROUND_AI_MANUAL_REVIEW, notes

    hybrid_result = fuse_u2net_preservation_with_ai(
        image,
        u2net_result,
        birefnet_result,
        tag_preserve_mask=preserve_mask,
        **common,
    )
    if hybrid_result.status == BACKGROUND_OK:
        status = BACKGROUND_HYBRID_OK
        notes = (
            f"Always Hybrid used U2Net jewellery preservation and {AI_FALLBACK_MODEL} cleanup. "
            f"U2Net preservation: {u2net_result.notes} "
            f"{AI_FALLBACK_MODEL} finish: {hybrid_result.notes}"
        )
    else:
        status = BACKGROUND_AI_MANUAL_REVIEW
        notes = (
            f"Always Hybrid used U2Net jewellery preservation and {AI_FALLBACK_MODEL} cleanup. "
            f"U2Net preservation: {u2net_result.notes} "
            f"{AI_FALLBACK_MODEL} hybrid is not confident enough: {hybrid_result.notes}"
        )
    return hybrid_result, status, notes


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
            item_id = f"item_{index:06d}"
            if progress_callback:
                progress_callback(index - 1, len(image_paths), image_path.name, counters)

            if hasattr(self.ocr_engine, "set_current_image"):
                self.ocr_engine.set_current_image(image_path)
            result = self._process_one(image_path, item_id=item_id)
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

    def _process_one(self, image_path: Path, item_id: str = "item_000001") -> ProcessingResult:
        original_filename = image_path.name
        original_stem = image_path.stem
        try:
            image = load_image(image_path)
            output_image = enhance_image(image, mode=self.settings.enhancement_mode) if self.settings.enhance_enabled else image.copy()
            crops = detect_tag_crops(output_image)
            if self.settings.save_debug_crops:
                save_debug_crops(crops, item_id, self.output_paths.debug_crops)

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
                    preserve_mask = build_tag_preservation_mask(
                        output_image.shape[:2],
                        boxes,
                        parsed.tag_number,
                        crops=crops,
                        source_image=output_image,
                    )
                    best_crop_label = self._base_ocr_crop_label(parsed.best_source_crop)
                    if preserve_mask is None and best_crop_label and not best_crop_label.startswith("fallback"):
                        preserve_mask = build_detected_tag_preservation_mask(
                            output_image.shape[:2],
                            crops,
                            source_image=output_image,
                            preferred_label=best_crop_label,
                        )
                    background_result, background_status, background_notes = _run_background_pipeline(
                        output_image,
                        self.settings,
                        preserve_mask=preserve_mask,
                    )
                    if background_result.status != BACKGROUND_OK:
                        review_path, _ = unique_png_path(self.output_paths.background_review, f"BG_REVIEW_{original_stem}")
                        save_png(output_image, review_path)
                        white_review_path = _background_white_preview_path(review_path)
                        if background_result.white_bgr is not None:
                            save_png(background_result.white_bgr, white_review_path)
                        transparent_filename = ""
                        if background_result.transparent_rgba is not None:
                            transparent_review_path = self.output_paths.background_review / f"{review_path.stem}_transparent.png"
                            save_rgba_png(background_result.transparent_rgba, transparent_review_path)
                            transparent_filename = transparent_review_path.name
                        return ProcessingResult(
                            original_filename=original_filename,
                            item_id=item_id,
                            detected_tag_number=parsed.tag_number,
                            ocr_text_raw=parsed.raw_text,
                            confidence_score=parsed.confidence,
                            final_filename=review_path.name,
                            output_folder="background_review",
                            status=STATUS_REVIEW_REQUIRED,
                            notes=f"{notes} Background removal needs review.",
                            background_status=background_status,
                            background_mode=background_mode,
                            transparent_filename=transparent_filename,
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
                    item_id=item_id,
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
            cache_result = _CorrectionCacheResult(
                cache_status="",
                background_status=BACKGROUND_DISABLED,
                background_mode="none",
                background_notes="Background removal disabled.",
            )
            if self.settings.remove_background:
                cache_result = self._prepare_review_background_cache(
                    output_image,
                    item_id=item_id,
                )

            review_notes = parsed.notes or "Tag number is unclear. Please review this photo."
            if cache_result.cache_status == CORRECTION_CACHE_READY:
                review_notes = f"{review_notes} Full-quality background output is prepared for fast correction."
            elif cache_result.cache_status == CORRECTION_CACHE_REVIEW_REQUIRED:
                review_notes = f"{review_notes} The background candidate also needs visual review after tag correction."
            elif cache_result.cache_status == CORRECTION_CACHE_FAILED:
                review_notes = f"{review_notes} Background preparation will retry after tag correction."
            return ProcessingResult(
                original_filename=original_filename,
                item_id=item_id,
                detected_tag_number=parsed.tag_number,
                ocr_text_raw=parsed.raw_text,
                confidence_score=parsed.confidence,
                final_filename=review_path.name,
                output_folder="review_required",
                status=parsed.status,
                notes=review_notes,
                background_status=cache_result.background_status,
                background_mode=cache_result.background_mode,
                background_notes=cache_result.background_notes,
                correction_cache_status=cache_result.cache_status,
                correction_cache_white_filename=cache_result.white_filename,
                correction_cache_transparent_filename=cache_result.transparent_filename,
                background_processing_seconds=(
                    f"{cache_result.elapsed_seconds:.3f}" if self.settings.remove_background else ""
                ),
            )
        except Exception as exc:
            return ProcessingResult(
                original_filename=original_filename,
                item_id=item_id,
                status="ERROR",
                notes=f"This photo could not be processed. Please check it manually. {exc}",
            )

    def _prepare_review_background_cache(
        self,
        output_image: np.ndarray,
        item_id: str,
    ) -> _CorrectionCacheResult:
        started_at = time.perf_counter()
        background_mode = self.settings.background_output_mode
        white_path, transparent_path = _correction_cache_paths(self.output_paths, item_id)
        white_path.unlink(missing_ok=True)
        transparent_path.unlink(missing_ok=True)

        try:
            background_result, background_status, background_notes = _run_background_pipeline(
                output_image,
                self.settings,
                # OCR-failed photos do not have trustworthy tag geometry. The old
                # rectangular fallback preserved floor/background blocks as foreground.
                # Both models still see the complete enhanced original here.
                preserve_mask=None,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            return _CorrectionCacheResult(
                cache_status=CORRECTION_CACHE_FAILED,
                background_status=BACKGROUND_FAILED,
                background_mode=background_mode,
                background_notes=f"Background preparation failed safely: {exc}",
                elapsed_seconds=elapsed,
            )

        if background_result.white_bgr is not None:
            save_png(background_result.white_bgr, white_path)
        if background_result.transparent_rgba is not None:
            save_rgba_png(background_result.transparent_rgba, transparent_path)

        white_filename = white_path.name if white_path.exists() else ""
        transparent_filename = transparent_path.name if transparent_path.exists() else ""
        if background_result.status == BACKGROUND_OK and _correction_cache_is_complete(
            background_mode,
            white_path,
            transparent_path,
        ):
            cache_status = CORRECTION_CACHE_READY
        elif background_result.status != BACKGROUND_OK:
            cache_status = CORRECTION_CACHE_REVIEW_REQUIRED
        else:
            cache_status = CORRECTION_CACHE_FAILED
            background_notes = f"{background_notes} Required cached output was not created."

        return _CorrectionCacheResult(
            cache_status=cache_status,
            background_status=background_status,
            background_mode=background_mode,
            background_notes=background_notes,
            white_filename=white_filename,
            transparent_filename=transparent_filename,
            elapsed_seconds=time.perf_counter() - started_at,
        )

    def _run_ocr(self, crops) -> tuple[list[OCRTextBox], list[str]]:
        all_boxes: list[OCRTextBox] = []
        notes: list[str] = []
        if not crops:
            return all_boxes, ["No tag crop was found."]

        attempts_done = 0
        fast_mode = self.settings.ocr_attempt_mode == "fast"
        max_attempts = 16 if fast_mode else 28
        prioritized_crops = self._prioritize_ocr_crops(crops)
        crop_limit = 7 if fast_mode else len(prioritized_crops)
        for crop in prioritized_crops[:crop_limit]:
            rotation_groups = self._rotation_groups(crop)
            crop_attempts = [attempt for group in rotation_groups for attempt in group]
            if attempts_done + len(crop_attempts) > max_attempts:
                notes.append("OCR completed the priority search but could not confirm a tag.")
                return all_boxes, notes

            optimized_boxes: list[OCRTextBox] = []
            optimized_reader = getattr(self.ocr_engine, "read_text_with_rotations", None)
            if crop.label == "fallback_full" and callable(optimized_reader) and len(crop_attempts) == 4:
                try:
                    boxes = optimized_reader(crop.image, source_crop=crop.label)
                    optimized_boxes = list(boxes)
                    all_boxes.extend(boxes)
                    attempts_done += len(crop_attempts)
                except Exception as exc:
                    notes.append(f"Optimized OCR skipped {crop.label}: {exc}")
                    attempts_done += len(crop_attempts)
            else:
                for rotation_label, rotated in crop_attempts:
                    try:
                        attempts_done += 1
                        # EasyOCR reads small printed digits more reliably from
                        # the real crop. Aggressive threshold/upscale variants
                        # can split a clear value such as 121995 into single digits.
                        prepared = (
                            rotated
                            if crop.label == "fallback_full" or not crop.label.startswith("fallback")
                            else preprocess_for_ocr(rotated, upscale=2, max_side=1600)
                        )
                        boxes = self.ocr_engine.read_text(
                            prepared,
                            source_rotation=rotation_label,
                            source_crop=crop.label,
                        )
                        all_boxes.extend(boxes)
                    except Exception as exc:
                        notes.append(f"OCR skipped {crop.label}/{rotation_label}: {exc}")

            parsed = parse_tag_from_ocr(all_boxes, self.settings.confidence_threshold)
            if self._has_safe_ocr_consensus(parsed):
                return all_boxes, notes

            if crop.label == "fallback_full" and optimized_boxes and attempts_done + 4 <= max_attempts:
                focused_crop = self._focused_ocr_recheck_crop(
                    crop.image,
                    optimized_boxes,
                    parsed.tag_number,
                )
                if focused_crop is not None:
                    for rotation_label, rotated in rotated_versions(focused_crop):
                        try:
                            attempts_done += 1
                            boxes = self.ocr_engine.read_text(
                                rotated,
                                source_rotation=rotation_label,
                                source_crop="fallback_full_recheck",
                            )
                            all_boxes.extend(boxes)
                        except Exception as exc:
                            notes.append(f"Focused OCR skipped {rotation_label}: {exc}")
                    parsed = parse_tag_from_ocr(all_boxes, self.settings.confidence_threshold)
                    if self._has_safe_ocr_consensus(parsed):
                        return all_boxes, notes

            # Only after all four raw directions fail do we try the enhanced
            # variant. Keeping it secondary preserves clear glyph shapes while
            # still recovering faint/low-contrast tags.
            if (
                not crop.label.startswith("fallback")
                and parsed.tag_number
                and attempts_done + len(crop_attempts) <= max_attempts
            ):
                for rotation_label, rotated in crop_attempts:
                    try:
                        attempts_done += 1
                        prepared = preprocess_for_ocr(rotated, upscale=3, max_side=1200)
                        boxes = self.ocr_engine.read_text(
                            prepared,
                            source_rotation=rotation_label,
                            source_crop=f"{crop.label}_enhanced",
                        )
                        all_boxes.extend(boxes)
                    except Exception as exc:
                        notes.append(f"Enhanced OCR skipped {crop.label}/{rotation_label}: {exc}")

                parsed = parse_tag_from_ocr(all_boxes, self.settings.confidence_threshold)
                if self._has_safe_ocr_consensus(parsed):
                    return all_boxes, notes

            verifier = getattr(self.ocr_engine, "read_text_verification", None)
            if callable(verifier) and parsed.tag_number and attempts_done < max_attempts:
                best_crop = self._base_ocr_crop_label(parsed.best_source_crop)
                if best_crop == self._base_ocr_crop_label(crop.label):
                    matching_rotation = next(
                        (
                            (rotation_label, rotated)
                            for rotation_label, rotated in crop_attempts
                            if rotation_label == parsed.best_source_rotation
                        ),
                        None,
                    )
                    if matching_rotation is not None:
                        rotation_label, rotated = matching_rotation
                        try:
                            attempts_done += 1
                            prepared = preprocess_for_ocr(rotated, upscale=3, max_side=1400)
                            boxes = verifier(
                                prepared,
                                source_rotation=rotation_label,
                                source_crop=f"{crop.label}_verification",
                            )
                            all_boxes.extend(boxes)
                        except Exception as exc:
                            notes.append(f"OCR verification skipped {crop.label}/{rotation_label}: {exc}")

                        parsed = parse_tag_from_ocr(all_boxes, self.settings.confidence_threshold)
                        if self._has_safe_ocr_consensus(parsed):
                            return all_boxes, notes
        if attempts_done >= max_attempts:
            notes.append("OCR reached the directional-attempt limit without confirming a tag.")
        elif not all_boxes:
            notes.append("OCR searched all available tag crops but found no readable number.")
        return all_boxes, notes

    def _has_safe_ocr_consensus(self, parsed) -> bool:
        """Early exit only after independent OCR directions agree on a value."""
        return (
            parsed.status == STATUS_OK
            and parsed.strong_evidence_count >= 2
            and parsed.confidence >= max(self.settings.confidence_threshold, 0.78)
        )

    @staticmethod
    def _base_ocr_crop_label(label: str) -> str:
        if label.endswith("_enhanced"):
            return label[:-9]
        if label.endswith("_verification"):
            return label[:-13]
        return label

    @staticmethod
    def _focused_ocr_recheck_crop(
        image: np.ndarray,
        boxes: list[OCRTextBox],
        tag_number: str,
    ) -> np.ndarray | None:
        if image is None or image.size == 0 or not tag_number:
            return None
        matches = []
        for box in boxes:
            digits = "".join(character for character in box.text if character.isdigit())
            if box.bbox and tag_number in digits:
                matches.append(box)
        if not matches:
            return None

        box = max(matches, key=lambda item: item.confidence)
        try:
            points = np.asarray(box.bbox, dtype=np.float32)
            xs = points[:, 0]
            ys = points[:, 1]
        except Exception:
            return None

        image_height, image_width = image.shape[:2]
        text_width = max(float(np.ptp(xs)), 1.0)
        text_height = max(float(np.ptp(ys)), 1.0)
        padding = max(int(max(text_width, text_height) * 1.15), 28)
        x0 = max(0, int(np.floor(xs.min())) - padding)
        y0 = max(0, int(np.floor(ys.min())) - padding)
        x1 = min(image_width, int(np.ceil(xs.max())) + padding)
        y1 = min(image_height, int(np.ceil(ys.max())) + padding)
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None
        if (x1 - x0) * (y1 - y0) > image_width * image_height * 0.45:
            return None
        return image[y0:y1, x0:x1].copy()

    @staticmethod
    def _prioritize_ocr_crops(crops):
        full_frame = [crop for crop in crops if crop.label == "fallback_full"]
        detected = [crop for crop in crops if not crop.label.startswith("fallback")]
        regional_fallbacks = [
            crop for crop in crops if crop.label.startswith("fallback") and crop.label != "fallback_full"
        ]
        # The strongest detected rectangle is usually the physical tag and is
        # much sharper than the full frame. Keep full-frame OCR immediately
        # behind it in case pearls or white stones produced a false rectangle.
        return detected[:1] + full_frame + detected[1:] + regional_fallbacks

    def _rotation_groups(self, crop) -> list[list[tuple[str, object]]]:
        height, width = crop.image.shape[:2]
        rotations = rotated_versions(crop.image)
        rotation_map = {label: image for label, image in rotations}
        if crop.label == "fallback_full":
            orders = [["original", "rot180"], ["cw90", "ccw90"]]
        elif crop.label.startswith("fallback"):
            orders = [["original", "rot180"], ["cw90", "ccw90"]]
        elif height > width * 1.2:
            orders = [["ccw90", "cw90"], ["original", "rot180"]]
        else:
            orders = [["original", "rot180"], ["cw90", "ccw90"]]
        return [
            [(label, rotation_map[label]) for label in order if label in rotation_map]
            for order in orders
        ]

    def _rotation_attempts(self, crop) -> list[tuple[str, object]]:
        return [attempt for group in self._rotation_groups(crop) for attempt in group]

    def _create_output_zips(self) -> None:
        create_zip_from_folder(self.output_paths.processed_images, self.output_paths.processed_zip)
        create_zip_from_folder(self.output_paths.transparent_images, self.output_paths.transparent_zip)
        if self.settings.save_debug_crops:
            create_zip_from_folder(self.output_paths.debug_crops, self.output_paths.debug_zip)
        create_zip_from_folder(self.output_paths.root, self.output_paths.full_zip, exclude_zip_files=True)


def apply_manual_correction(
    output_root: Path,
    item_selector: str,
    corrected_tag: str,
    settings: ProcessingSettings | None = None,
) -> tuple[bool, str]:
    correction_started_at = time.perf_counter()
    corrected_tag = corrected_tag.strip()
    if not is_valid_manual_tag(corrected_tag):
        return False, "Enter a numeric tag number with 5 to 8 digits."

    paths = setup_output_paths(output_root)
    report = read_report(paths.report_csv)
    if report.empty:
        return False, "No report is available yet."

    matches = _select_report_rows(report, item_selector)
    if matches.empty:
        return False, "This item was not found in the report."
    if len(matches) != 1:
        return False, "More than one photo has this filename. Reopen Review and select the exact item."

    row = matches.iloc[0]
    item_id = str(row.get("item_id", "")).strip() or str(row.get("original_filename", ""))
    current_folder = str(row.get("output_folder", ""))
    current_filename = str(row.get("final_filename", ""))
    if not current_filename:
        return False, "No review image is available for this item."

    source = paths.root / current_folder / current_filename
    if not source.exists():
        source = paths.review_required / current_filename
    if not source.exists():
        return False, "The review image file could not be found."

    background_mode = str(row.get("background_mode", "")).strip()
    if not background_mode and settings is not None and settings.remove_background:
        background_mode = settings.background_output_mode
    if background_mode not in {"white_and_transparent", "white_only", "transparent_only"}:
        background_mode = "none"

    final_filename = ""
    output_folder = "processed_images"
    transparent_filename = ""
    background_status = str(row.get("background_status", "")).strip() or BACKGROUND_DISABLED
    background_notes = str(row.get("background_notes", "")).strip() or "Background removal disabled."
    background_processing_seconds = str(row.get("background_processing_seconds", "")).strip()
    duplicate = False
    used_precomputed_output = False

    cache_status = str(row.get("correction_cache_status", "")).strip()
    cache_white_path = _reported_cache_path(
        paths.correction_cache,
        str(row.get("correction_cache_white_filename", "")),
    )
    cache_transparent_path = _reported_cache_path(
        paths.correction_cache,
        str(row.get("correction_cache_transparent_filename", "")),
    )

    if background_mode == "none":
        destination, duplicate = unique_png_path(paths.processed_images, corrected_tag)
        shutil.move(str(source), str(destination))
        final_filename = destination.name
    elif cache_status == CORRECTION_CACHE_REVIEW_REQUIRED:
        review_path, _ = unique_png_path(paths.background_review, f"BG_REVIEW_{Path(current_filename).stem}")
        created_paths: list[Path] = []
        try:
            shutil.copy2(source, review_path)
            created_paths.append(review_path)
            white_review_path = _background_white_preview_path(review_path)
            if cache_white_path is not None and cache_white_path.is_file():
                shutil.copy2(cache_white_path, white_review_path)
                created_paths.append(white_review_path)
            if cache_transparent_path is not None and cache_transparent_path.is_file():
                transparent_review_path = paths.background_review / f"{review_path.stem}_transparent.png"
                shutil.copy2(cache_transparent_path, transparent_review_path)
                created_paths.append(transparent_review_path)
                transparent_filename = transparent_review_path.name
        except Exception as exc:
            _remove_cache_files(*created_paths)
            return False, f"Prepared background review could not be saved: {exc}"

        source.unlink(missing_ok=True)
        _remove_cache_files(cache_white_path, cache_transparent_path)
        correction_seconds = time.perf_counter() - correction_started_at
        update_report_row(
            paths.report_csv,
            item_id,
            {
                "detected_tag_number": corrected_tag,
                "confidence_score": "manual",
                "final_filename": review_path.name,
                "output_folder": "background_review",
                "status": STATUS_REVIEW_REQUIRED,
                "notes": "Manual tag correction saved; prepared background needs visual review.",
                "background_status": background_status,
                "background_mode": background_mode,
                "transparent_filename": transparent_filename,
                "background_notes": background_notes,
                "correction_cache_status": "",
                "correction_cache_white_filename": "",
                "correction_cache_transparent_filename": "",
                "correction_finalize_seconds": f"{correction_seconds:.3f}",
            },
        )
        _mark_outputs_changed(paths)
        return True, (
            f"Tag correction saved in {_format_elapsed(correction_seconds)}. "
            "Background preview needs visual review."
        )
    elif cache_status == CORRECTION_CACHE_READY and _correction_cache_is_complete(
        background_mode,
        cache_white_path or paths.correction_cache / "__missing_white__",
        cache_transparent_path or paths.correction_cache / "__missing_transparent__",
    ):
        created_paths = []
        try:
            if background_mode == "transparent_only":
                destination, duplicate = unique_png_path(paths.transparent_images, corrected_tag)
                shutil.copy2(cache_transparent_path, destination)
                created_paths.append(destination)
                final_filename = destination.name
                output_folder = "transparent_images"
                transparent_filename = destination.name
            else:
                destination, duplicate = unique_png_path(paths.processed_images, corrected_tag)
                shutil.copy2(cache_white_path, destination)
                created_paths.append(destination)
                final_filename = destination.name
                if background_mode == "white_and_transparent":
                    transparent_path, transparent_duplicate = unique_png_path(
                        paths.transparent_images,
                        destination.stem,
                    )
                    shutil.copy2(cache_transparent_path, transparent_path)
                    created_paths.append(transparent_path)
                    transparent_filename = transparent_path.name
                    duplicate = duplicate or transparent_duplicate
        except Exception as exc:
            _remove_cache_files(*created_paths)
            return False, f"Prepared full-quality output could not be saved: {exc}"

        source.unlink(missing_ok=True)
        _remove_cache_files(cache_white_path, cache_transparent_path)
        used_precomputed_output = True
    else:
        effective_settings = settings or ProcessingSettings(background_output_mode=background_mode)
        image = load_image(source)
        background_started_at = time.perf_counter()
        background_result, background_status, background_notes = _run_background_pipeline(
            image,
            effective_settings,
        )
        background_processing_seconds = f"{time.perf_counter() - background_started_at:.3f}"

        if background_result.status != BACKGROUND_OK:
            review_path, _ = unique_png_path(paths.background_review, f"BG_REVIEW_{Path(current_filename).stem}")
            shutil.copy2(source, review_path)
            white_review_path = _background_white_preview_path(review_path)
            if background_result.white_bgr is not None:
                save_png(background_result.white_bgr, white_review_path)
            if background_result.transparent_rgba is not None:
                transparent_review_path = paths.background_review / f"{review_path.stem}_transparent.png"
                save_rgba_png(background_result.transparent_rgba, transparent_review_path)
                transparent_filename = transparent_review_path.name
            source.unlink(missing_ok=True)
            _remove_cache_files(cache_white_path, cache_transparent_path)
            update_report_row(
                paths.report_csv,
                item_id,
                {
                    "detected_tag_number": corrected_tag,
                    "confidence_score": "manual",
                    "final_filename": review_path.name,
                    "output_folder": "background_review",
                    "status": STATUS_REVIEW_REQUIRED,
                    "notes": "Manual tag correction saved; background needs visual review.",
                    "background_status": background_status,
                    "background_mode": background_mode,
                    "transparent_filename": transparent_filename,
                    "background_notes": background_notes,
                    "correction_cache_status": "",
                    "correction_cache_white_filename": "",
                    "correction_cache_transparent_filename": "",
                    "background_processing_seconds": background_processing_seconds,
                    "correction_finalize_seconds": f"{time.perf_counter() - correction_started_at:.3f}",
                },
            )
            _mark_outputs_changed(paths)
            return True, "Tag correction saved. Background preview needs visual review before download."

        if background_mode == "transparent_only":
            destination, duplicate = unique_png_path(paths.transparent_images, corrected_tag)
            if background_result.transparent_rgba is None:
                return False, "Background processor did not return a transparent image."
            save_rgba_png(background_result.transparent_rgba, destination)
            final_filename = destination.name
            output_folder = "transparent_images"
            transparent_filename = destination.name
        else:
            destination, duplicate = unique_png_path(paths.processed_images, corrected_tag)
            if background_result.white_bgr is None:
                return False, "Background processor did not return a white-background image."
            save_png(background_result.white_bgr, destination)
            final_filename = destination.name
            if background_mode == "white_and_transparent" and background_result.transparent_rgba is not None:
                transparent_path, _ = unique_png_path(paths.transparent_images, corrected_tag)
                save_rgba_png(background_result.transparent_rgba, transparent_path)
                transparent_filename = transparent_path.name
        source.unlink(missing_ok=True)
        _remove_cache_files(cache_white_path, cache_transparent_path)

    status = STATUS_DUPLICATE_TAG if duplicate else STATUS_OK
    correction_seconds = time.perf_counter() - correction_started_at
    update_report_row(
        paths.report_csv,
        item_id,
        {
            "detected_tag_number": corrected_tag,
            "confidence_score": "manual",
            "final_filename": final_filename,
            "output_folder": output_folder,
            "status": status,
            "notes": "Manual correction saved.",
            "background_status": background_status,
            "background_mode": background_mode,
            "transparent_filename": transparent_filename,
            "background_notes": background_notes,
            "correction_cache_status": "",
            "correction_cache_white_filename": "",
            "correction_cache_transparent_filename": "",
            "background_processing_seconds": background_processing_seconds,
            "correction_finalize_seconds": f"{correction_seconds:.3f}",
        },
    )
    _mark_outputs_changed(paths)
    if used_precomputed_output:
        return True, (
            "Correction saved from the prepared full-quality output in "
            f"{_format_elapsed(correction_seconds)}."
        )
    return True, (
        "Correction saved and final background outputs created safely in "
        f"{_format_elapsed(correction_seconds)}."
    )


def resolve_ai_review(output_root: Path, item_selector: str, action: str) -> tuple[bool, str]:
    if action not in {"accept_ai", "send_manual"}:
        return False, "Choose a valid AI review action."

    paths = setup_output_paths(output_root)
    report = read_report(paths.report_csv)
    if report.empty:
        return False, "No report is available yet."

    matches = _select_report_rows(report, item_selector)
    if matches.empty:
        return False, "This AI review item was not found in the report."
    if len(matches) != 1:
        return False, "More than one photo has this filename. Reopen AI Review and select the exact item."

    row = matches.iloc[0]
    item_id = str(row.get("item_id", "")).strip() or str(row.get("original_filename", ""))
    if str(row.get("output_folder", "")) != "ai_review":
        return False, "This item is not waiting for AI review."

    tag_number = str(row.get("detected_tag_number", "")).strip()
    if not is_valid_manual_tag(tag_number):
        return False, "The detected tag must be corrected before resolving AI review."

    review_path = paths.ai_review / str(row.get("final_filename", ""))
    if not review_path.is_file():
        return False, "The complete enhanced original could not be found."

    ai_white_path = _ai_white_preview_path(review_path)
    ai_transparent_path = _ai_transparent_preview_path(review_path)
    u2net_white_path = _u2net_white_preview_path(review_path)
    background_mode = str(row.get("background_mode", "white_and_transparent"))

    if action == "send_manual":
        manual_stem = review_path.stem.replace("AI_REVIEW_", "BG_REVIEW_", 1)
        manual_path, _ = unique_png_path(paths.background_review, manual_stem)
        shutil.copy2(review_path, manual_path)
        candidate_source = ai_white_path if ai_white_path.is_file() else u2net_white_path
        if candidate_source.is_file():
            shutil.copy2(candidate_source, _background_white_preview_path(manual_path))
        transparent_filename = ""
        if ai_transparent_path.is_file():
            transparent_path = paths.background_review / f"{manual_path.stem}_transparent.png"
            shutil.copy2(ai_transparent_path, transparent_path)
            transparent_filename = transparent_path.name
        update_report_row(
            paths.report_csv,
            item_id,
            {
                "final_filename": manual_path.name,
                "output_folder": "background_review",
                "status": STATUS_REVIEW_REQUIRED,
                "notes": "AI candidate sent to manual jewellery review.",
                "background_status": BACKGROUND_AI_MANUAL_REVIEW,
                "transparent_filename": transparent_filename,
                "background_notes": (
                    f"{row.get('background_notes', '')} AI result was not approved and now needs manual review."
                ).strip(),
            },
        )
        _refresh_output_archives(paths)
        return True, "Photo moved to Manual Review with the AI candidate and complete original."

    output_folder = "processed_images"
    final_transparent_filename = ""
    if background_mode == "transparent_only":
        if not ai_transparent_path.is_file():
            return False, "The AI transparent candidate could not be found."
        destination, duplicate = unique_png_path(paths.transparent_images, tag_number)
        shutil.copy2(ai_transparent_path, destination)
        output_folder = "transparent_images"
        final_filename = destination.name
        final_transparent_filename = destination.name
    else:
        if not ai_white_path.is_file():
            return False, "The AI white-background candidate could not be found."
        destination, duplicate = unique_png_path(paths.processed_images, tag_number)
        shutil.copy2(ai_white_path, destination)
        final_filename = destination.name
        if background_mode == "white_and_transparent" and ai_transparent_path.is_file():
            transparent_destination, _ = unique_png_path(paths.transparent_images, destination.stem)
            shutil.copy2(ai_transparent_path, transparent_destination)
            final_transparent_filename = transparent_destination.name

    status = STATUS_DUPLICATE_TAG if duplicate else STATUS_OK
    update_report_row(
        paths.report_csv,
        item_id,
        {
            "final_filename": final_filename,
            "output_folder": output_folder,
            "status": status,
            "notes": "Local AI background repair approved.",
            "background_status": BACKGROUND_AI_ACCEPTED,
            "transparent_filename": final_transparent_filename,
            "background_notes": (
                f"{row.get('background_notes', '')} AI repair approved after visual review."
            ).strip(),
        },
    )
    _refresh_output_archives(paths)
    return True, "AI-cleaned jewellery image moved to final output."


def resolve_background_review(output_root: Path, item_selector: str, action: str) -> tuple[bool, str]:
    if action not in {"accept_preview", "keep_original"}:
        return False, "Choose a valid background review action."

    paths = setup_output_paths(output_root)
    report = read_report(paths.report_csv)
    if report.empty:
        return False, "No report is available yet."

    matches = _select_report_rows(report, item_selector)
    if matches.empty:
        return False, "This review item was not found in the report."
    if len(matches) != 1:
        return False, "More than one photo has this filename. Reopen Review and select the exact item."

    row = matches.iloc[0]
    item_id = str(row.get("item_id", "")).strip() or str(row.get("original_filename", ""))
    if str(row.get("output_folder", "")) != "background_review":
        return False, "This item is not waiting for background review."

    tag_number = str(row.get("detected_tag_number", "")).strip()
    if not is_valid_manual_tag(tag_number):
        return False, "The detected tag must be corrected before resolving the background."

    review_path = paths.background_review / str(row.get("final_filename", ""))
    white_preview_path = _background_white_preview_path(review_path)
    transparent_name = str(row.get("transparent_filename", "")).strip()
    transparent_preview_path = paths.background_review / transparent_name if transparent_name else None
    background_mode = str(row.get("background_mode", "white_and_transparent"))

    if action == "keep_original":
        if not review_path.is_file():
            return False, "The complete original review photo could not be found."
        destination, duplicate = unique_png_path(paths.processed_images, tag_number)
        shutil.copy2(review_path, destination)
        output_folder = "processed_images"
        final_filename = destination.name
        final_transparent_filename = ""
        background_status = BACKGROUND_ORIGINAL_KEPT
        background_notes = "Original enhanced photo kept after manual review; no jewellery was removed."
    else:
        if background_mode == "transparent_only":
            if transparent_preview_path is None or not transparent_preview_path.is_file():
                return False, "The transparent review preview could not be found."
            destination, duplicate = unique_png_path(paths.transparent_images, tag_number)
            shutil.copy2(transparent_preview_path, destination)
            output_folder = "transparent_images"
            final_filename = destination.name
            final_transparent_filename = destination.name
        else:
            if not white_preview_path.is_file():
                return False, "The white-background review preview could not be found."
            destination, duplicate = unique_png_path(paths.processed_images, tag_number)
            shutil.copy2(white_preview_path, destination)
            output_folder = "processed_images"
            final_filename = destination.name
            final_transparent_filename = ""
            if background_mode == "white_and_transparent" and transparent_preview_path is not None and transparent_preview_path.is_file():
                transparent_destination, _ = unique_png_path(paths.transparent_images, destination.stem)
                shutil.copy2(transparent_preview_path, transparent_destination)
                final_transparent_filename = transparent_destination.name
        background_status = BACKGROUND_MANUAL_ACCEPTED
        background_notes = "Background preview accepted after visual jewellery-completeness review."

    status = STATUS_DUPLICATE_TAG if duplicate else STATUS_OK
    update_report_row(
        paths.report_csv,
        item_id,
        {
            "final_filename": final_filename,
            "output_folder": output_folder,
            "status": status,
            "notes": "Background review resolved manually.",
            "background_status": background_status,
            "transparent_filename": final_transparent_filename,
            "background_notes": background_notes,
        },
    )
    _mark_outputs_changed(paths)
    if action == "keep_original":
        return True, "Complete original photo saved safely. No jewellery part was removed."
    return True, "Reviewed background preview moved to final output."


def _correction_cache_paths(paths: OutputPaths, item_id: str) -> tuple[Path, Path]:
    stem = safe_stem(item_id or "review_item")
    return (
        paths.correction_cache / f"{stem}_white.png",
        paths.correction_cache / f"{stem}_transparent.png",
    )


def _format_elapsed(seconds: float) -> str:
    return f"{seconds:.2f}s" if seconds < 1.0 else f"{seconds:.1f}s"


def _correction_cache_is_complete(background_mode: str, white_path: Path, transparent_path: Path) -> bool:
    if background_mode == "white_only":
        return white_path.is_file()
    if background_mode == "transparent_only":
        return transparent_path.is_file()
    if background_mode == "white_and_transparent":
        return white_path.is_file() and transparent_path.is_file()
    return False


def _reported_cache_path(cache_root: Path, filename: str) -> Path | None:
    clean_name = Path(str(filename).strip()).name
    if not clean_name:
        return None
    return cache_root / clean_name


def _remove_cache_files(*paths: Path | None) -> None:
    for path in paths:
        if path is not None:
            path.unlink(missing_ok=True)


def _background_white_preview_path(review_path: Path) -> Path:
    return review_path.with_name(f"{review_path.stem}_candidate_white.png")


def _u2net_white_preview_path(review_path: Path) -> Path:
    return review_path.with_name(f"{review_path.stem}_u2net_white.png")


def _ai_white_preview_path(review_path: Path) -> Path:
    return review_path.with_name(f"{review_path.stem}_ai_white.png")


def _ai_transparent_preview_path(review_path: Path) -> Path:
    return review_path.with_name(f"{review_path.stem}_ai_transparent.png")


def _refresh_output_archives(paths: OutputPaths) -> None:
    _mark_outputs_changed(paths)


def _mark_outputs_changed(paths: OutputPaths) -> None:
    invalidate_compressed_export(paths.compressed_images_20kb, paths.compressed_images_20kb_zip)
    mark_output_archives_stale(paths.root)


def _select_report_rows(report, item_selector: str):
    selector = str(item_selector).strip()
    if selector and "item_id" in report.columns:
        matches = report[report["item_id"] == selector]
        if not matches.empty:
            return matches
    return report[report["original_filename"] == selector]
