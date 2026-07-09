from pathlib import Path

from PIL import Image, ImageDraw

from src.background_processor import BackgroundResult, BACKGROUND_OK, BACKGROUND_REVIEW_REQUIRED
from src.models import OCRTextBox, ProcessingSettings, STATUS_DUPLICATE_TAG, STATUS_OK, STATUS_OCR_FAILED
from src.ocr_engine import StaticOCREngine
import src.processor as processor_module
from src.processor import BatchProcessor, apply_manual_correction, generate_transparent_outputs_for_processed
from src.report_generator import read_report


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


def test_manual_correction_with_background_writes_transparent(tmp_path, monkeypatch):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    engine = StaticOCREngine([])
    processor = BatchProcessor(tmp_path / "Jewellery_Output", ProcessingSettings(remove_background=False), engine)
    processor.process_images([photo])

    def fake_remove_background(image, **kwargs):
        alpha = Image.new("L", (image.shape[1], image.shape[0]), 255)
        rgba = Image.fromarray(image[:, :, ::-1]).convert("RGBA")
        rgba.putalpha(alpha)
        return BackgroundResult(BACKGROUND_OK, transparent_rgba=__import__("numpy").array(rgba), white_bgr=image, notes="Mock background OK.")

    monkeypatch.setattr(processor_module, "remove_background", fake_remove_background)
    settings = ProcessingSettings(remove_background=True, background_output_mode="white_and_transparent")

    ok, message = apply_manual_correction(tmp_path / "Jewellery_Output", "IMG_001.jpg", "121999", settings)

    assert ok, message
    assert (tmp_path / "Jewellery_Output" / "processed_images" / "121999.png").exists()
    assert (tmp_path / "Jewellery_Output" / "transparent_images" / "121999.png").exists()
    report = read_report(tmp_path / "Jewellery_Output" / "report.csv")
    assert report.iloc[0]["transparent_filename"] == "121999.png"
    assert report.iloc[0]["background_status"] == BACKGROUND_OK


def test_processor_background_success_writes_white_and_transparent(tmp_path, monkeypatch):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    engine = StaticOCREngine([OCRTextBox("121134", 0.95, bbox=[[0, 0], [200, 0], [200, 80], [0, 80]])])

    def fake_remove_background(image, **kwargs):
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
    assert report.iloc[0]["background_status"] == BACKGROUND_OK
    assert report.iloc[0]["transparent_filename"] == "121134.png"


def test_processor_white_only_still_writes_transparent_companion(tmp_path, monkeypatch):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    engine = StaticOCREngine([OCRTextBox("121134", 0.95, bbox=[[0, 0], [200, 0], [200, 80], [0, 80]])])

    def fake_remove_background(image, **kwargs):
        alpha = Image.new("L", (image.shape[1], image.shape[0]), 255)
        rgba = Image.fromarray(image[:, :, ::-1]).convert("RGBA")
        rgba.putalpha(alpha)
        return BackgroundResult(BACKGROUND_OK, transparent_rgba=__import__("numpy").array(rgba), white_bgr=image, notes="Mock background OK.")

    monkeypatch.setattr(processor_module, "remove_background", fake_remove_background)
    settings = ProcessingSettings(remove_background=True, background_output_mode="white_only")
    processor = BatchProcessor(tmp_path / "Jewellery_Output", settings, engine)
    summary = processor.process_images([photo])

    assert summary.ok == 1
    assert (tmp_path / "Jewellery_Output" / "processed_images" / "121134.png").exists()
    assert (tmp_path / "Jewellery_Output" / "transparent_images" / "121134.png").exists()


def test_generate_transparent_outputs_for_processed_backfills_missing_zip(tmp_path, monkeypatch):
    output_root = tmp_path / "Jewellery_Output"
    processed_dir = output_root / "processed_images"
    processed_dir.mkdir(parents=True)
    make_photo(processed_dir / "121134.png")

    def fake_remove_background(image, **kwargs):
        alpha = Image.new("L", (image.shape[1], image.shape[0]), 255)
        rgba = Image.fromarray(image[:, :, ::-1]).convert("RGBA")
        rgba.putalpha(alpha)
        return BackgroundResult(BACKGROUND_OK, transparent_rgba=__import__("numpy").array(rgba), white_bgr=image, notes="Mock background OK.")

    monkeypatch.setattr(processor_module, "remove_background", fake_remove_background)

    count, message = generate_transparent_outputs_for_processed(output_root, ProcessingSettings(remove_background=True))

    assert count == 1, message
    assert (output_root / "transparent_images" / "121134.png").exists()
    assert (output_root / "transparent_images.zip").exists()


def test_processor_passes_full_quality_background_settings(tmp_path, monkeypatch):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    engine = StaticOCREngine([OCRTextBox("121134", 0.95, bbox=[[0, 0], [200, 0], [200, 80], [0, 80]])])
    captured = {}

    def fake_remove_background(image, **kwargs):
        captured.update(kwargs)
        alpha = Image.new("L", (image.shape[1], image.shape[0]), 255)
        rgba = Image.fromarray(image[:, :, ::-1]).convert("RGBA")
        rgba.putalpha(alpha)
        return BackgroundResult(BACKGROUND_OK, transparent_rgba=__import__("numpy").array(rgba), white_bgr=image, notes="Mock background OK.")

    monkeypatch.setattr(processor_module, "remove_background", fake_remove_background)
    settings = ProcessingSettings(
        remove_background=True,
        background_model_name="u2net",
        background_max_side=2200,
        enhancement_mode="quality",
    )
    processor = BatchProcessor(tmp_path / "Jewellery_Output", settings, engine)
    processor.process_images([photo])

    assert captured["model_name"] == "u2net"
    assert captured["max_side"] == 2200
    assert captured["alpha_matting"] is True


def test_processor_uses_alpha_matting_for_full_quality_u2net(tmp_path, monkeypatch):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    engine = StaticOCREngine([OCRTextBox("121134", 0.95, bbox=[[0, 0], [200, 0], [200, 80], [0, 80]])])
    captured = {}

    def fake_remove_background(image, **kwargs):
        captured.update(kwargs)
        alpha = Image.new("L", (image.shape[1], image.shape[0]), 255)
        rgba = Image.fromarray(image[:, :, ::-1]).convert("RGBA")
        rgba.putalpha(alpha)
        return BackgroundResult(BACKGROUND_OK, transparent_rgba=__import__("numpy").array(rgba), white_bgr=image, notes="Mock background OK.")

    monkeypatch.setattr(processor_module, "remove_background", fake_remove_background)
    settings = ProcessingSettings(
        remove_background=True,
        background_model_name="u2net",
        background_max_side=2200,
        enhancement_mode="quality",
    )
    processor = BatchProcessor(tmp_path / "Jewellery_Output", settings, engine)
    processor.process_images([photo])

    assert captured["alpha_matting"] is True


def test_processor_background_review_routes_to_background_review(tmp_path, monkeypatch):
    photo = tmp_path / "IMG_001.jpg"
    make_photo(photo)
    engine = StaticOCREngine([OCRTextBox("121134", 0.95, bbox=[[0, 0], [200, 0], [200, 80], [0, 80]])])

    def fake_remove_background(image, **kwargs):
        return BackgroundResult(BACKGROUND_REVIEW_REQUIRED, white_bgr=image, notes="Mock unsafe mask.")

    monkeypatch.setattr(processor_module, "remove_background", fake_remove_background)
    settings = ProcessingSettings(remove_background=True, background_output_mode="white_and_transparent")
    processor = BatchProcessor(tmp_path / "Jewellery_Output", settings, engine)
    summary = processor.process_images([photo])

    assert summary.review_required == 1
    report = read_report(tmp_path / "Jewellery_Output" / "report.csv")
    assert report.iloc[0]["status"] == "REVIEW_REQUIRED"
    assert report.iloc[0]["output_folder"] == "background_review"
    assert report.iloc[0]["background_status"] == BACKGROUND_REVIEW_REQUIRED
