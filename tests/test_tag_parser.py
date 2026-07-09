from src.models import OCRTextBox, STATUS_OK, STATUS_REVIEW_REQUIRED, STATUS_TAG_NOT_FOUND
from src.tag_parser import is_valid_manual_tag, parse_tag_from_ocr


def test_selects_large_numeric_tag_and_ignores_alphanumeric_product_code():
    boxes = [
        OCRTextBox("221235FLDE1ICSB000", 0.98, bbox=[[0, 0], [80, 0], [80, 20], [0, 20]]),
        OCRTextBox("121134", 0.88, bbox=[[0, 0], [250, 0], [250, 70], [0, 70]]),
    ]
    parsed = parse_tag_from_ocr(boxes, confidence_threshold=0.45)
    assert parsed.status == STATUS_OK
    assert parsed.tag_number == "121134"


def test_low_confidence_goes_to_review():
    boxes = [OCRTextBox("121134", 0.15, bbox=[[0, 0], [60, 0], [60, 15], [0, 15]])]
    parsed = parse_tag_from_ocr(boxes, confidence_threshold=0.7)
    assert parsed.status == STATUS_REVIEW_REQUIRED
    assert parsed.tag_number == "121134"


def test_no_valid_tag_is_tag_not_found():
    parsed = parse_tag_from_ocr([OCRTextBox("SUN AAR FLDE", 0.9)], confidence_threshold=0.45)
    assert parsed.status == STATUS_TAG_NOT_FOUND


def test_manual_tag_validation():
    assert is_valid_manual_tag("121134")
    assert not is_valid_manual_tag("121134ABC")
    assert not is_valid_manual_tag("1234")

