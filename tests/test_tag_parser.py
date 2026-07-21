from src.models import OCRTextBox, STATUS_OK, STATUS_REVIEW_REQUIRED, STATUS_TAG_NOT_FOUND
from src.tag_parser import is_valid_manual_tag, parse_tag_from_ocr


def test_selects_large_numeric_tag_and_ignores_alphanumeric_product_code():
    boxes = [
        OCRTextBox("221235FLDE1ICSB000", 0.98, bbox=[[0, 0], [80, 0], [80, 20], [0, 20]]),
        OCRTextBox(
            "121134",
            0.88,
            bbox=[[0, 0], [250, 0], [250, 70], [0, 70]],
            source_rotation="original",
        ),
        OCRTextBox(
            "121134",
            0.87,
            bbox=[[0, 0], [250, 0], [250, 70], [0, 70]],
            source_rotation="rot180",
        ),
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


def test_stronger_opposite_rotation_resolves_reversed_tag_number():
    boxes = [
        OCRTextBox(
            "690221",
            0.8326,
            bbox=[[0, 0], [190, 0], [190, 55], [0, 55]],
            source_rotation="cw90",
            source_crop="fallback_full",
        ),
        OCRTextBox(
            "122069",
            0.9804,
            bbox=[[0, 0], [190, 0], [190, 55], [0, 55]],
            source_rotation="ccw90",
            source_crop="fallback_full",
        ),
    ]

    parsed = parse_tag_from_ocr(boxes, confidence_threshold=0.45)

    assert parsed.status == STATUS_OK
    assert parsed.tag_number == "122069"
    assert "reverse-orientation" in parsed.notes


def test_close_reverse_orientation_conflict_goes_to_review():
    boxes = [
        OCRTextBox(
            "690221",
            0.90,
            bbox=[[0, 0], [190, 0], [190, 55], [0, 55]],
            source_rotation="cw90",
            source_crop="fallback_full",
        ),
        OCRTextBox(
            "122069",
            0.86,
            bbox=[[0, 0], [190, 0], [190, 55], [0, 55]],
            source_rotation="ccw90",
            source_crop="fallback_full",
        ),
    ]

    parsed = parse_tag_from_ocr(boxes, confidence_threshold=0.45)

    assert parsed.status == STATUS_REVIEW_REQUIRED
    assert "Opposite rotations disagree" in parsed.notes


def test_single_moderate_reading_is_not_auto_accepted():
    boxes = [
        OCRTextBox(
            "121995",
            0.85,
            bbox=[[0, 0], [250, 0], [250, 70], [0, 70]],
            source_rotation="cw90",
            source_crop="tag_crop_1",
        )
    ]

    parsed = parse_tag_from_ocr(boxes, confidence_threshold=0.45)

    assert parsed.status == STATUS_REVIEW_REQUIRED
    assert parsed.tag_number == "121995"
    assert parsed.evidence_count == 1


def test_single_high_confidence_reading_still_requires_verification():
    parsed = parse_tag_from_ocr(
        [
            OCRTextBox(
                "122350",
                0.99,
                bbox=[[0, 0], [250, 0], [250, 70], [0, 70]],
                source_rotation="auto_best",
                source_crop="fallback_full",
            )
        ],
        confidence_threshold=0.45,
    )

    assert parsed.status == STATUS_REVIEW_REQUIRED
    assert parsed.tag_number == "122350"


def test_weak_focused_repeat_does_not_validate_a_confident_wrong_digit():
    bbox = [[0, 0], [250, 0], [250, 70], [0, 70]]
    parsed = parse_tag_from_ocr(
        [
            OCRTextBox(
                "122358",
                1.0,
                bbox=bbox,
                source_rotation="auto_best",
                source_crop="fallback_full",
            ),
            OCRTextBox(
                "122358",
                0.5157,
                bbox=bbox,
                source_rotation="original",
                source_crop="fallback_full_recheck",
            ),
        ],
        confidence_threshold=0.45,
    )

    assert parsed.status == STATUS_REVIEW_REQUIRED
    assert parsed.strong_evidence_count == 1


def test_two_independent_moderate_readings_are_accepted_by_consensus():
    bbox = [[0, 0], [250, 0], [250, 70], [0, 70]]
    boxes = [
        OCRTextBox("121995", 0.65, bbox=bbox, source_rotation="original", source_crop="tag_crop_1"),
        OCRTextBox("121995", 0.65, bbox=bbox, source_rotation="rot180", source_crop="tag_crop_1"),
    ]

    parsed = parse_tag_from_ocr(boxes, confidence_threshold=0.45)

    assert parsed.status == STATUS_OK
    assert parsed.evidence_count == 2


def test_raw_and_enhanced_same_direction_do_not_fake_consensus():
    bbox = [[0, 0], [250, 0], [250, 70], [0, 70]]
    boxes = [
        OCRTextBox("121995", 0.80, bbox=bbox, source_rotation="cw90", source_crop="tag_crop_1"),
        OCRTextBox("121995", 0.82, bbox=bbox, source_rotation="cw90", source_crop="tag_crop_1_enhanced"),
    ]

    parsed = parse_tag_from_ocr(boxes, confidence_threshold=0.45)

    assert parsed.status == STATUS_REVIEW_REQUIRED
    assert parsed.evidence_count == 1


def test_close_non_reverse_candidates_go_to_review():
    bbox = [[0, 0], [250, 0], [250, 70], [0, 70]]
    boxes = [
        OCRTextBox("121995", 0.91, bbox=bbox, source_rotation="original", source_crop="tag_crop_1"),
        OCRTextBox("121985", 0.89, bbox=bbox, source_rotation="cw90", source_crop="tag_crop_1"),
    ]

    parsed = parse_tag_from_ocr(boxes, confidence_threshold=0.45)

    assert parsed.status == STATUS_REVIEW_REQUIRED
    assert "competing tag values" in parsed.notes
