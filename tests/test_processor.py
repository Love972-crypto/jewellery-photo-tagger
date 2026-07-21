from pathlib import Path

from PIL import Image, ImageDraw

from src.background_processor import (
    AI_FALLBACK_MODEL,
    BACKGROUND_AI_MANUAL_REVIEW,
    BACKGROUND_HYBRID_OK,
    BackgroundResult,
    BACKGROUND_OK,
    BACKGROUND_REVIEW_REQUIRED,
    MaskSafetyMetrics,
)
from src.models import (
    CORRECTION_CACHE_READY,
    CORRECTION_CACHE_REVIEW_REQUIRED,
    OCRTextBox,
    ProcessingSettings,
    STATUS_DUPLICATE_TAG,
    STATUS_OK,
    STATUS_OCR_FAILED,
    STATUS_REVIEW_REQUIRED,
)
from src.ocr_engine import StaticOCREngine
import src.processor as processor_module
from src.processor import BatchProcessor, apply_manual_correction, resolve_background_review
from src.report_generator import read_report
from src.tag_detection import CandidateCrop
from src.tag_parser import parse_tag_from_ocr


def make_photo(path: Path) -> None:
    image = Image.new("RGB", (600, 420), "#e9e0cf")
    draw = ImageDraw.Draw(image)
    draw.rectangle((360, 50, 560, 145), fill="#fbfaf6", outline="#d0c8b8", width=3)
    draw.text((398, 78), "121134", fill="#111111")
    image.save(path)


def test_processor_success_and_report(tmp_path):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    engine = StaticOCREngine([OCRTextBox("121134", 0.95, bbox=[[0, 0], [200, 0], [200, 80], [0, 80]])])
    processor = BatchProcessor(tmp_path / "Jewellery_Output", ProcessingSettings(remove_background=False), engine)
    summary = processor.process_images([photo])
    assert summary.ok == 1
    assert (tmp_path / "Jewellery_Output" / "processed_images" / "121134.png").exists()
    report = read_report(tmp_path / "Jewellery_Output" / "report.csv")
    assert report.iloc[0]["status"] == STATUS_OK


def test_processor_duplicate_tag_suffix(tmp_path):
    first = tmp_path / "IMG_001.jpg"
    second = tmp_path / "IMG_002.jpg"
    make_photo(first)
    make_photo(second)
    engine = StaticOCREngine([OCRTextBox("121134", 0.95, bbox=[[0, 0], [200, 0], [200, 80], [0, 80]])])
    processor = BatchProcessor(tmp_path / "Jewellery_Output", ProcessingSettings(remove_background=False), engine)
    summary = processor.process_images([first, second])
    assert summary.ok == 1
    assert summary.duplicate_tags == 1
    assert (tmp_path / "Jewellery_Output" / "processed_images" / "121134_2.png").exists()
    report = read_report(tmp_path / "Jewellery_Output" / "report.csv")
    assert STATUS_DUPLICATE_TAG in set(report["status"])


def test_processor_ocr_failure_goes_to_review(tmp_path):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    engine = StaticOCREngine([])
    processor = BatchProcessor(tmp_path / "Jewellery_Output", ProcessingSettings(remove_background=False), engine)
    summary = processor.process_images([photo])
    assert summary.ocr_failed == 1
    report = read_report(tmp_path / "Jewellery_Output" / "report.csv")
    assert report.iloc[0]["status"] == STATUS_OCR_FAILED
    assert (tmp_path / "Jewellery_Output" / "review_required" / report.iloc[0]["final_filename"]).exists()


def test_fast_ocr_tries_clear_detected_crop_then_recovers_from_full_frame(tmp_path):
    class RotationAwareEngine:
        name = "Rotation-aware test OCR"

        def __init__(self):
            self.calls = []

        def read_text(self, image, source_rotation="original", source_crop="tag_crop"):
            self.calls.append((source_crop, source_rotation, image.shape[:2]))
            if source_crop == "fallback_full" and source_rotation == "original":
                return [OCRTextBox("82722", 0.58, bbox=[[0, 0], [70, 0], [70, 18], [0, 18]])]
            if source_crop == "fallback_full" and source_rotation == "rot180":
                return [OCRTextBox("122278", 0.93, bbox=[[0, 0], [190, 0], [190, 55], [0, 55]])]
            return []

    full_image = __import__("numpy").zeros((1500, 1080, 3), dtype="uint8")
    wrong_crop = __import__("numpy").zeros((55, 62, 3), dtype="uint8")
    crops = [
        CandidateCrop("tag_crop_1_strict_white", wrong_crop, 0.9),
        CandidateCrop("fallback_full", full_image, 0.2),
    ]
    engine = RotationAwareEngine()
    processor = BatchProcessor(
        tmp_path / "Jewellery_Output",
        ProcessingSettings(remove_background=False, ocr_attempt_mode="fast"),
        engine,
    )

    boxes, notes = processor._run_ocr(crops)
    parsed = parse_tag_from_ocr(boxes, confidence_threshold=0.45)

    assert parsed.tag_number == "122278"
    assert parsed.status == STATUS_REVIEW_REQUIRED
    assert [call[0] for call in engine.calls[:4]] == ["tag_crop_1_strict_white"] * 4
    assert [call[1] for call in engine.calls[-4:]] == ["original", "rot180", "cw90", "ccw90"]
    assert engine.calls[-4][0] == "fallback_full"
    assert not notes


def test_ocr_completes_cw_and_ccw_pair_before_accepting_reversed_tag(tmp_path):
    class ReversedTagEngine:
        name = "Reversed tag test OCR"

        def __init__(self):
            self.calls = []

        def read_text(self, image, source_rotation="original", source_crop="tag_crop"):
            self.calls.append((source_crop, source_rotation))
            if source_crop == "fallback_full" and source_rotation == "cw90":
                return [OCRTextBox("690221", 0.8326, bbox=[[0, 0], [190, 0], [190, 55], [0, 55]])]
            if source_crop == "fallback_full" and source_rotation == "ccw90":
                return [OCRTextBox("122069", 0.9804, bbox=[[0, 0], [190, 0], [190, 55], [0, 55]])]
            return []

    full_image = __import__("numpy").zeros((1500, 1080, 3), dtype="uint8")
    engine = ReversedTagEngine()
    processor = BatchProcessor(
        tmp_path / "Jewellery_Output",
        ProcessingSettings(remove_background=False, ocr_attempt_mode="fast"),
        engine,
    )

    boxes, _ = processor._run_ocr([CandidateCrop("fallback_full", full_image, 0.2)])
    parsed = parse_tag_from_ocr(boxes, confidence_threshold=0.45)

    assert engine.calls == [
        ("fallback_full", "original"),
        ("fallback_full", "rot180"),
        ("fallback_full", "cw90"),
        ("fallback_full", "ccw90"),
    ]
    assert parsed.status == STATUS_OK
    assert parsed.tag_number == "122069"


def test_ocr_does_not_accept_confident_reversed_number_before_all_rotations(tmp_path):
    class ConflictingOrientationEngine:
        name = "Conflicting orientation test OCR"

        def __init__(self):
            self.calls = []

        def read_text(self, image, source_rotation="original", source_crop="tag_crop"):
            self.calls.append((source_crop, source_rotation))
            if source_rotation in {"original", "rot180"}:
                return [OCRTextBox("690221", 0.8326, bbox=[[0, 0], [190, 0], [190, 55], [0, 55]])]
            if source_rotation in {"cw90", "ccw90"}:
                return [OCRTextBox("122069", 0.9804, bbox=[[0, 0], [190, 0], [190, 55], [0, 55]])]
            return []

    engine = ConflictingOrientationEngine()
    full_image = __import__("numpy").zeros((1500, 1080, 3), dtype="uint8")
    processor = BatchProcessor(
        tmp_path / "Jewellery_Output",
        ProcessingSettings(remove_background=False, ocr_attempt_mode="fast"),
        engine,
    )

    boxes, _ = processor._run_ocr([CandidateCrop("fallback_full", full_image, 0.2)])
    parsed = parse_tag_from_ocr(boxes, confidence_threshold=0.45)

    assert [rotation for _, rotation in engine.calls] == ["original", "rot180", "cw90", "ccw90"]
    assert parsed.status == STATUS_OK
    assert parsed.tag_number == "122069"


def test_fast_ocr_stops_at_sixteen_directional_attempts(tmp_path):
    class EmptyEngine:
        name = "Empty test OCR"

        def __init__(self):
            self.calls = []

        def read_text(self, image, source_rotation="original", source_crop="tag_crop"):
            self.calls.append((source_crop, source_rotation))
            return []

    full_image = __import__("numpy").zeros((1500, 1080, 3), dtype="uint8")
    detected = __import__("numpy").zeros((180, 420, 3), dtype="uint8")
    engine = EmptyEngine()
    processor = BatchProcessor(
        tmp_path / "Jewellery_Output",
        ProcessingSettings(remove_background=False, ocr_attempt_mode="fast"),
        engine,
    )

    _, notes = processor._run_ocr(
        [
            CandidateCrop("fallback_full", full_image, 0.2),
            CandidateCrop("tag_crop_1_strict_white", detected, 0.9),
            CandidateCrop("fallback_lower", full_image, 0.18),
            CandidateCrop("fallback_center", full_image, 0.16),
            CandidateCrop("fallback_right", full_image, 0.14),
        ]
    )

    assert len(engine.calls) == 16
    assert notes


def test_detected_tag_uses_discrete_physical_rotations_instead_of_auto_rotation(tmp_path):
    class DiscreteEngine:
        name = "Discrete test OCR"

        def __init__(self):
            self.calls = []

        def read_text_with_rotations(self, image, source_crop="tag_crop", rotations=(90, 180, 270)):
            raise AssertionError("detected crops must not use EasyOCR auto-rotation")

        def read_text(self, image, source_rotation="original", source_crop="tag_crop"):
            self.calls.append((source_crop, source_rotation, image.shape[:2]))
            if source_rotation == "ccw90":
                return [OCRTextBox("122444", 0.96, source_rotation=source_rotation, source_crop=source_crop)]
            return []

    vertical_crop = __import__("numpy").zeros((360, 120, 3), dtype="uint8")
    engine = DiscreteEngine()
    processor = BatchProcessor(
        tmp_path / "Jewellery_Output",
        ProcessingSettings(remove_background=False, ocr_attempt_mode="fast"),
        engine,
    )

    boxes, notes = processor._run_ocr([CandidateCrop("tag_crop_1_strict_white", vertical_crop, 0.9)])
    parsed = parse_tag_from_ocr(boxes, confidence_threshold=0.45)

    assert parsed.tag_number == "122444"
    assert [rotation for _, rotation, _ in engine.calls] == [
        "ccw90",
        "cw90",
        "original",
        "rot180",
        "ccw90",
        "cw90",
        "original",
        "rot180",
    ]
    assert engine.calls[0][2][1] > engine.calls[0][2][0]
    assert not notes


def test_uncertain_single_reading_is_cross_checked_with_independent_decoder(tmp_path):
    class VerificationEngine:
        name = "Verification test OCR"

        def __init__(self):
            self.verification_calls = []

        def read_text(self, image, source_rotation="original", source_crop="tag_crop"):
            if source_rotation == "cw90":
                return [
                    OCRTextBox(
                        "121995",
                        0.80,
                        bbox=[[0, 0], [250, 0], [250, 70], [0, 70]],
                        source_rotation=source_rotation,
                        source_crop=source_crop,
                    )
                ]
            return []

        def read_text_verification(self, image, source_rotation="original", source_crop="tag_crop_verification"):
            self.verification_calls.append((source_crop, source_rotation))
            return [
                OCRTextBox(
                    "121995",
                    0.82,
                    bbox=[[0, 0], [250, 0], [250, 70], [0, 70]],
                    source_rotation=source_rotation,
                    source_crop=source_crop,
                )
            ]

    crop = __import__("numpy").zeros((180, 420, 3), dtype="uint8")
    engine = VerificationEngine()
    processor = BatchProcessor(
        tmp_path / "Jewellery_Output",
        ProcessingSettings(remove_background=False, ocr_attempt_mode="fast"),
        engine,
    )

    boxes, _ = processor._run_ocr([CandidateCrop("tag_crop_1_strict_white", crop, 0.9)])
    parsed = parse_tag_from_ocr(boxes, confidence_threshold=0.45)

    assert engine.verification_calls == [("tag_crop_1_strict_white_verification", "cw90")]
    assert parsed.status == STATUS_OK
    assert parsed.tag_number == "121995"
    assert parsed.evidence_count == 2


def test_full_frame_candidate_is_rechecked_on_a_focused_crop(tmp_path):
    class FocusedRecheckEngine:
        name = "Focused recheck test OCR"

        def __init__(self):
            self.focused_calls = []

        def read_text_with_rotations(self, image, source_crop="tag_crop", rotations=(90, 180, 270)):
            return [
                OCRTextBox(
                    "122358",
                    0.99,
                    bbox=[[380, 160], [520, 160], [520, 220], [380, 220]],
                    source_rotation="auto_best",
                    source_crop=source_crop,
                )
            ]

        def read_text(self, image, source_rotation="original", source_crop="tag_crop"):
            if source_crop == "fallback_full_recheck":
                self.focused_calls.append((source_rotation, image.shape[:2]))
                if source_rotation in {"original", "rot180"}:
                    return [
                        OCRTextBox(
                            "122350",
                            0.96,
                            bbox=[[0, 0], [240, 0], [240, 70], [0, 70]],
                            source_rotation=source_rotation,
                            source_crop=source_crop,
                        )
                    ]
            return []

    image = __import__("numpy").zeros((900, 700, 3), dtype="uint8")
    engine = FocusedRecheckEngine()
    processor = BatchProcessor(
        tmp_path / "Jewellery_Output",
        ProcessingSettings(remove_background=False, ocr_attempt_mode="fast"),
        engine,
    )

    boxes, _ = processor._run_ocr([CandidateCrop("fallback_full", image, 0.2)])
    parsed = parse_tag_from_ocr(boxes, confidence_threshold=0.45)

    assert len(engine.focused_calls) == 4
    assert all(height < 900 and width < 700 for _, (height, width) in engine.focused_calls)
    assert parsed.status == STATUS_OK
    assert parsed.tag_number == "122350"


def test_corrupt_image_is_logged_and_batch_continues(tmp_path):
    bad = tmp_path / "bad.jpg"
    good = tmp_path / "good.jpg"
    bad.write_bytes(b"not an image")
    make_photo(good)
    engine = StaticOCREngine([OCRTextBox("121134", 0.95, bbox=[[0, 0], [200, 0], [200, 80], [0, 80]])])
    processor = BatchProcessor(tmp_path / "Jewellery_Output", ProcessingSettings(remove_background=False), engine)
    summary = processor.process_images([bad, good])
    assert summary.errors == 1
    assert summary.ok == 1


def test_manual_correction_moves_review_item_to_processed(tmp_path):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    engine = StaticOCREngine([])
    processor = BatchProcessor(tmp_path / "Jewellery_Output", ProcessingSettings(remove_background=False), engine)
    processor.process_images([photo])

    ok, message = apply_manual_correction(tmp_path / "Jewellery_Output", "IMG_001.jpg", "121999")

    assert ok, message
    assert (tmp_path / "Jewellery_Output" / "processed_images" / "121999.png").exists()
    report = read_report(tmp_path / "Jewellery_Output" / "report.csv")
    assert report.iloc[0]["status"] == STATUS_OK
    assert report.iloc[0]["detected_tag_number"] == "121999"


def test_manual_correction_targets_only_one_of_two_same_named_photos(tmp_path):
    first = tmp_path / "folder_a" / "IMG_001.jpg"
    second = tmp_path / "folder_b" / "IMG_001.jpg"
    first.parent.mkdir()
    second.parent.mkdir()
    make_photo(first)
    make_photo(second)
    output_root = tmp_path / "Jewellery_Output"
    BatchProcessor(output_root, ProcessingSettings(remove_background=False), StaticOCREngine([])).process_images(
        [first, second]
    )
    before = read_report(output_root / "report.csv")
    assert before["item_id"].is_unique

    selected_item = before.iloc[1]["item_id"]
    ok, message = apply_manual_correction(output_root, selected_item, "121999")

    assert ok, message
    after = read_report(output_root / "report.csv")
    assert after.iloc[0]["status"] == "OCR_FAILED"
    assert after.iloc[1]["status"] == STATUS_OK
    assert after.iloc[1]["detected_tag_number"] == "121999"
    assert len(list((output_root / "review_required").glob("*.png"))) == 1


def test_manual_correction_reuses_precomputed_background_outputs_without_model_rerun(tmp_path, monkeypatch):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    output_root = tmp_path / "Jewellery_Output"
    settings = ProcessingSettings(remove_background=True, background_output_mode="white_and_transparent")

    pipeline_calls = []

    def fake_background_pipeline(image, settings, preserve_mask=None):
        pipeline_calls.append(preserve_mask)
        rgba = __import__("numpy").dstack(
            [image[:, :, ::-1], __import__("numpy").full(image.shape[:2], 255, dtype="uint8")]
        )
        result = BackgroundResult(BACKGROUND_OK, transparent_rgba=rgba, white_bgr=image, notes="Safe mock mask.")
        return result, BACKGROUND_HYBRID_OK, "Safe precomputed background."

    monkeypatch.setattr(processor_module, "_run_background_pipeline", fake_background_pipeline)
    BatchProcessor(output_root, settings, StaticOCREngine([])).process_images([photo])

    before = read_report(output_root / "report.csv")
    assert before.iloc[0]["correction_cache_status"] == CORRECTION_CACHE_READY
    white_cache = output_root / ".correction_cache" / before.iloc[0]["correction_cache_white_filename"]
    transparent_cache = output_root / ".correction_cache" / before.iloc[0]["correction_cache_transparent_filename"]
    expected_white_bytes = white_cache.read_bytes()
    expected_transparent_bytes = transparent_cache.read_bytes()
    assert pipeline_calls == [None]

    def unexpected_pipeline(*args, **kwargs):
        raise AssertionError("manual correction must not rerun the background models")

    monkeypatch.setattr(processor_module, "_run_background_pipeline", unexpected_pipeline)
    ok, message = apply_manual_correction(output_root, "IMG_001.jpg", "121999", settings=settings)

    assert ok, message
    assert (output_root / "processed_images" / "121999.png").read_bytes() == expected_white_bytes
    assert (output_root / "transparent_images" / "121999.png").read_bytes() == expected_transparent_bytes
    report = read_report(output_root / "report.csv")
    assert report.iloc[0]["background_status"] == BACKGROUND_HYBRID_OK
    assert report.iloc[0]["transparent_filename"] == "121999.png"
    assert report.iloc[0]["correction_cache_status"] == ""
    assert float(report.iloc[0]["correction_finalize_seconds"]) < 5.0
    assert not white_cache.exists()
    assert not transparent_cache.exists()


def test_manual_correction_routes_unsafe_background_to_review(tmp_path, monkeypatch):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    output_root = tmp_path / "Jewellery_Output"
    settings = ProcessingSettings(remove_background=True, background_output_mode="white_and_transparent")
    def fake_background_pipeline(image, settings, preserve_mask=None):
        rgba = __import__("numpy").dstack(
            [image[:, :, ::-1], __import__("numpy").full(image.shape[:2], 255, dtype="uint8")]
        )
        result = BackgroundResult(
            BACKGROUND_REVIEW_REQUIRED,
            transparent_rgba=rgba,
            white_bgr=image,
            notes="Unsafe mock mask.",
        )
        return result, BACKGROUND_AI_MANUAL_REVIEW, "Unsafe precomputed background."

    monkeypatch.setattr(processor_module, "_run_background_pipeline", fake_background_pipeline)
    BatchProcessor(output_root, settings, StaticOCREngine([])).process_images([photo])
    before = read_report(output_root / "report.csv")
    assert before.iloc[0]["correction_cache_status"] == CORRECTION_CACHE_REVIEW_REQUIRED

    def unexpected_pipeline(*args, **kwargs):
        raise AssertionError("unsafe precomputed output must route to review without a model rerun")

    monkeypatch.setattr(processor_module, "_run_background_pipeline", unexpected_pipeline)
    ok, message = apply_manual_correction(output_root, "IMG_001.jpg", "121999", settings=settings)

    assert ok, message
    report = read_report(output_root / "report.csv")
    assert report.iloc[0]["status"] == STATUS_REVIEW_REQUIRED
    assert report.iloc[0]["output_folder"] == "background_review"
    assert not list((output_root / "processed_images").glob("121999*.png"))


def test_processor_background_success_writes_white_and_transparent(tmp_path, monkeypatch):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    engine = StaticOCREngine([OCRTextBox("121134", 0.95, bbox=[[0, 0], [200, 0], [200, 80], [0, 80]])])

    background_kwargs = {}

    def fake_remove_background(image, **kwargs):
        background_kwargs.update(kwargs)
        alpha = Image.new("L", (image.shape[1], image.shape[0]), 255)
        rgba = Image.fromarray(image[:, :, ::-1]).convert("RGBA")
        rgba.putalpha(alpha)
        return BackgroundResult(BACKGROUND_OK, transparent_rgba=__import__("numpy").array(rgba), white_bgr=image, notes="Mock background OK.")

    monkeypatch.setattr(processor_module, "remove_background", fake_remove_background)
    settings = ProcessingSettings(remove_background=True, background_output_mode="white_and_transparent")
    processor = BatchProcessor(tmp_path / "Jewellery_Output", settings, engine)
    summary = processor.process_images([photo])

    assert summary.ok == 1
    assert (tmp_path / "Jewellery_Output" / "processed_images" / "121134.png").exists()
    assert (tmp_path / "Jewellery_Output" / "transparent_images" / "121134.png").exists()
    report = read_report(tmp_path / "Jewellery_Output" / "report.csv")
    assert report.iloc[0]["background_status"] == BACKGROUND_HYBRID_OK
    assert report.iloc[0]["transparent_filename"] == "121134.png"
    assert background_kwargs["preserve_mask"] is not None


def test_processor_background_review_routes_to_background_review(tmp_path, monkeypatch):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    engine = StaticOCREngine([OCRTextBox("121134", 0.95, bbox=[[0, 0], [200, 0], [200, 80], [0, 80]])])

    def fake_remove_background(image, **kwargs):
        rgba = __import__("numpy").dstack(
            [image[:, :, ::-1], __import__("numpy").full(image.shape[:2], 255, dtype="uint8")]
        )
        return BackgroundResult(
            BACKGROUND_REVIEW_REQUIRED,
            transparent_rgba=rgba,
            white_bgr=image,
            notes="Mock unsafe mask.",
        )

    monkeypatch.setattr(processor_module, "remove_background", fake_remove_background)
    settings = ProcessingSettings(remove_background=True, background_output_mode="white_and_transparent")
    processor = BatchProcessor(tmp_path / "Jewellery_Output", settings, engine)
    summary = processor.process_images([photo])

    assert summary.review_required == 1
    report = read_report(tmp_path / "Jewellery_Output" / "report.csv")
    assert report.iloc[0]["status"] == "REVIEW_REQUIRED"
    assert report.iloc[0]["output_folder"] == "background_review"
    assert report.iloc[0]["background_status"] == BACKGROUND_AI_MANUAL_REVIEW
    review_path = tmp_path / "Jewellery_Output" / "background_review" / report.iloc[0]["final_filename"]
    assert review_path.exists()
    assert review_path.with_name(f"{review_path.stem}_candidate_white.png").exists()
    assert (review_path.parent / report.iloc[0]["transparent_filename"]).exists()


def test_background_review_preview_can_be_accepted(tmp_path, monkeypatch):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    engine = StaticOCREngine([OCRTextBox("121134", 0.95, bbox=[[0, 0], [200, 0], [200, 80], [0, 80]])])

    def fake_remove_background(image, **kwargs):
        rgba = __import__("numpy").dstack(
            [image[:, :, ::-1], __import__("numpy").full(image.shape[:2], 255, dtype="uint8")]
        )
        return BackgroundResult(BACKGROUND_REVIEW_REQUIRED, transparent_rgba=rgba, white_bgr=image, notes="Unsafe mask.")

    monkeypatch.setattr(processor_module, "remove_background", fake_remove_background)
    output_root = tmp_path / "Jewellery_Output"
    BatchProcessor(output_root, ProcessingSettings(remove_background=True), engine).process_images([photo])

    ok, message = resolve_background_review(output_root, "IMG_001.jpg", "accept_preview")

    assert ok, message
    assert (output_root / "processed_images" / "121134.png").exists()
    assert (output_root / "transparent_images" / "121134.png").exists()
    report = read_report(output_root / "report.csv")
    assert report.iloc[0]["status"] == STATUS_OK
    assert report.iloc[0]["background_status"] == "MANUAL_ACCEPTED"


def test_background_review_can_keep_complete_original(tmp_path, monkeypatch):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    engine = StaticOCREngine([OCRTextBox("121134", 0.95, bbox=[[0, 0], [200, 0], [200, 80], [0, 80]])])

    def fake_remove_background(image, **kwargs):
        return BackgroundResult(BACKGROUND_REVIEW_REQUIRED, white_bgr=image, notes="Unsafe mask.")

    monkeypatch.setattr(processor_module, "remove_background", fake_remove_background)
    output_root = tmp_path / "Jewellery_Output"
    BatchProcessor(output_root, ProcessingSettings(remove_background=True), engine).process_images([photo])

    ok, message = resolve_background_review(output_root, "IMG_001.jpg", "keep_original")

    assert ok, message
    assert (output_root / "processed_images" / "121134.png").exists()
    report = read_report(output_root / "report.csv")
    assert report.iloc[0]["background_status"] == "ORIGINAL_KEPT"
    assert report.iloc[0]["transparent_filename"] == ""


def test_always_hybrid_clean_photo_runs_both_models(tmp_path, monkeypatch):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    engine = StaticOCREngine([OCRTextBox("121134", 0.95, bbox=[[0, 0], [200, 0], [200, 80], [0, 80]])])
    calls = []

    def fake_remove_background(image, **kwargs):
        model_name = kwargs.get("model_name", "u2net")
        calls.append(model_name)
        height, width = image.shape[:2]
        alpha = __import__("numpy").zeros((height, width), dtype="uint8")
        alpha[80:340, 120:480] = 255
        rgba = __import__("numpy").dstack([image[:, :, ::-1], alpha])
        metrics = MaskSafetyMetrics(
            0.37,
            0.37,
            0.0,
            0.0,
            0.0,
            1,
            1,
            0,
            0,
            0.0001,
            0.00005,
        )
        return BackgroundResult(
            BACKGROUND_OK,
            transparent_rgba=rgba,
            white_bgr=image,
            notes="Clean BiRefNet result.",
            safety_metrics=metrics,
            source_rgba=rgba,
        )

    monkeypatch.setattr(processor_module, "remove_background", fake_remove_background)
    output_root = tmp_path / "Jewellery_Output"
    summary = BatchProcessor(output_root, ProcessingSettings(remove_background=True), engine).process_images([photo])

    assert summary.ok == 1
    assert calls == [AI_FALLBACK_MODEL, "u2net"]
    report = read_report(output_root / "report.csv")
    assert report.iloc[0]["background_status"] == BACKGROUND_HYBRID_OK
    assert "Always Hybrid" in report.iloc[0]["background_notes"]


def test_u2net_and_birefnet_safe_hybrid_is_saved_automatically(tmp_path, monkeypatch):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    engine = StaticOCREngine([OCRTextBox("121134", 0.95, bbox=[[0, 0], [200, 0], [200, 80], [0, 80]])])
    calls = []

    def fake_remove_background(image, **kwargs):
        model_name = kwargs.get("model_name", "u2net")
        calls.append((model_name, kwargs.get("preserve_mask")))
        alpha = __import__("numpy").zeros(image.shape[:2], dtype="uint8")
        alpha[70:350, 100:500] = 255
        rgba = __import__("numpy").dstack([image[:, :, ::-1], alpha])
        status = BACKGROUND_REVIEW_REQUIRED if model_name == "u2net" else BACKGROUND_OK
        return BackgroundResult(
            status,
            transparent_rgba=rgba,
            white_bgr=image,
            notes=f"{model_name} result.",
            source_rgba=rgba,
        )

    monkeypatch.setattr(processor_module, "remove_background", fake_remove_background)
    output_root = tmp_path / "Jewellery_Output"
    summary = BatchProcessor(output_root, ProcessingSettings(remove_background=True), engine).process_images([photo])

    assert summary.ok == 1
    assert [model_name for model_name, _ in calls] == [AI_FALLBACK_MODEL, "u2net"]
    assert calls[0][1] is None
    assert calls[1][1] is not None
    report = read_report(output_root / "report.csv")
    assert report.iloc[0]["output_folder"] == "processed_images"
    assert report.iloc[0]["background_status"] == BACKGROUND_HYBRID_OK
    assert (output_root / "processed_images" / "121134.png").exists()
    assert (output_root / "transparent_images" / "121134.png").exists()


def test_uncertain_hybrid_goes_to_background_review(tmp_path, monkeypatch):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    engine = StaticOCREngine([OCRTextBox("121134", 0.95, bbox=[[0, 0], [200, 0], [200, 80], [0, 80]])])

    def fake_remove_background(image, **kwargs):
        rgba = __import__("numpy").dstack(
            [image[:, :, ::-1], __import__("numpy").full(image.shape[:2], 255, dtype="uint8")]
        )
        return BackgroundResult(
            BACKGROUND_OK,
            transparent_rgba=rgba,
            white_bgr=image,
            notes="Model result.",
            source_rgba=rgba,
        )

    def fake_fuse(image, u2net_result, ai_result, **kwargs):
        return BackgroundResult(
            BACKGROUND_REVIEW_REQUIRED,
            transparent_rgba=ai_result.transparent_rgba,
            white_bgr=ai_result.white_bgr,
            notes="Hybrid confidence check failed.",
        )

    monkeypatch.setattr(processor_module, "remove_background", fake_remove_background)
    monkeypatch.setattr(processor_module, "fuse_u2net_preservation_with_ai", fake_fuse)
    output_root = tmp_path / "Jewellery_Output"
    summary = BatchProcessor(output_root, ProcessingSettings(remove_background=True), engine).process_images([photo])

    assert summary.review_required == 1
    report = read_report(output_root / "report.csv")
    assert report.iloc[0]["output_folder"] == "background_review"
    assert report.iloc[0]["background_status"] == BACKGROUND_AI_MANUAL_REVIEW
    assert not list((output_root / "processed_images").glob("121134*.png"))


def test_manual_tag_correction_uses_same_hybrid_confidence_gate(tmp_path, monkeypatch):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    output_root = tmp_path / "Jewellery_Output"
    settings = ProcessingSettings(remove_background=True)
    def fake_remove_background(image, **kwargs):
        rgba = __import__("numpy").dstack(
            [image[:, :, ::-1], __import__("numpy").full(image.shape[:2], 255, dtype="uint8")]
        )
        return BackgroundResult(
            BACKGROUND_OK,
            transparent_rgba=rgba,
            white_bgr=image,
            notes="Model result.",
            source_rgba=rgba,
        )

    def fake_fuse(image, u2net_result, ai_result, **kwargs):
        return BackgroundResult(
            BACKGROUND_REVIEW_REQUIRED,
            transparent_rgba=ai_result.transparent_rgba,
            white_bgr=ai_result.white_bgr,
            notes="Hybrid confidence check failed.",
        )

    monkeypatch.setattr(processor_module, "remove_background", fake_remove_background)
    monkeypatch.setattr(processor_module, "fuse_u2net_preservation_with_ai", fake_fuse)
    BatchProcessor(output_root, settings, StaticOCREngine([])).process_images([photo])

    ok, message = apply_manual_correction(output_root, "IMG_001.jpg", "121134", settings=settings)

    assert ok, message
    report = read_report(output_root / "report.csv")
    assert report.iloc[0]["output_folder"] == "background_review"
    assert report.iloc[0]["background_status"] == BACKGROUND_AI_MANUAL_REVIEW


def test_ai_fallback_can_be_disabled_for_fast_u2net_only_processing(tmp_path, monkeypatch):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    engine = StaticOCREngine([OCRTextBox("121134", 0.95, bbox=[[0, 0], [200, 0], [200, 80], [0, 80]])])
    calls = []

    def fake_remove_background(image, **kwargs):
        calls.append(kwargs.get("model_name", "u2net"))
        return BackgroundResult(BACKGROUND_REVIEW_REQUIRED, white_bgr=image, notes="Unsafe U2Net mask.")

    monkeypatch.setattr(processor_module, "remove_background", fake_remove_background)
    settings = ProcessingSettings(remove_background=True, ai_background_fallback_enabled=False)
    output_root = tmp_path / "Jewellery_Output"
    BatchProcessor(output_root, settings, engine).process_images([photo])

    assert calls == ["u2net"]
    report = read_report(output_root / "report.csv")
    assert report.iloc[0]["output_folder"] == "background_review"
    assert report.iloc[0]["background_status"] == BACKGROUND_REVIEW_REQUIRED
