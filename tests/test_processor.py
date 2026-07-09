from pathlib import Path

from PIL import Image, ImageDraw

from src.background_processor import BackgroundResult, BACKGROUND_OK, BACKGROUND_REVIEW_REQUIRED
from src.models import OCRTextBox, ProcessingSettings, STATUS_DUPLICATE_TAG, STATUS_OK, STATUS_OCR_FAILED
from src.ocr_engine import StaticOCREngine
import src.processor as processor_module
from src.processor import BatchProcessor, apply_manual_correction
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
