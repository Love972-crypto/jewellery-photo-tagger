import numpy as np

from src.models import OCRTextBox
from src.tag_detection import (
    CandidateCrop,
    build_detected_tag_preservation_mask,
    build_tag_preservation_mask,
)


def test_tag_preservation_mask_maps_rot180_boxes_to_original_image():
    boxes = [
        OCRTextBox(
            "122278",
            0.98,
            bbox=[[70, 801], [201, 844], [186, 886], [55, 843]],
            source_rotation="rot180",
            source_crop="fallback_full",
        ),
        OCRTextBox(
            "221235188744522",
            0.44,
            bbox=[[54, 842], [283, 915], [271, 948], [41, 874]],
            source_rotation="rot180",
            source_crop="fallback_full",
        ),
    ]

    mask = build_tag_preservation_mask((1599, 899), boxes, "122278")

    assert mask is not None
    assert int(mask[755, 770]) == 255
    assert int(mask[100, 100]) == 0
    assert 0.01 < float(np.mean(mask > 0)) < 0.12


def test_tag_preservation_mask_keeps_complete_overlapping_tag_crop():
    boxes = [
        OCRTextBox(
            "122358",
            0.99,
            bbox=[[190, 115], [275, 115], [275, 145], [190, 145]],
            source_rotation="original",
            source_crop="fallback_full",
        )
    ]
    source = np.full((400, 500, 3), 90, dtype=np.uint8)
    tag_polygon = np.asarray([[150, 105], [315, 80], [330, 155], [165, 180]], dtype=np.int32)
    import cv2

    cv2.fillConvexPoly(source, tag_polygon, (242, 242, 242))
    crops = [
        CandidateCrop(
            "tag_crop_1_strict_white",
            np.zeros((100, 180, 3), dtype=np.uint8),
            0.8,
            (150, 80, 180, 100),
        )
    ]

    mask = build_tag_preservation_mask(
        (400, 500),
        boxes,
        "122358",
        crops=crops,
        source_image=source,
    )

    assert mask is not None
    assert int(mask[130, 230]) == 255
    assert int(mask[82, 150]) == 0
    assert int(mask[300, 450]) == 0


def test_ocr_fallback_is_refined_to_physical_tag_instead_of_floor_rectangle():
    import cv2

    source = np.full((420, 620, 3), (142, 151, 158), dtype=np.uint8)
    tag_polygon = np.asarray([[325, 160], [500, 190], [485, 275], [310, 245]], dtype=np.int32)
    cv2.fillConvexPoly(source, tag_polygon, (244, 244, 244))
    boxes = [
        OCRTextBox(
            "126725",
            0.97,
            bbox=[[355, 195], [460, 212], [455, 240], [350, 223]],
            source_rotation="original",
            source_crop="fallback_full",
        )
    ]

    mask = build_tag_preservation_mask(
        source.shape[:2],
        boxes,
        "126725",
        source_image=source,
    )

    assert mask is not None
    assert int(mask[220, 400]) == 255
    assert int(mask[150, 300]) == 0
    assert int(mask[290, 510]) == 0
    assert float(np.mean(mask > 0)) < 0.08


def test_ocr_fallback_returns_none_when_no_physical_tag_can_be_isolated():
    source = np.full((320, 480, 3), (170, 170, 170), dtype=np.uint8)
    boxes = [
        OCRTextBox(
            "126725",
            0.96,
            bbox=[[210, 125], [320, 125], [320, 160], [210, 160]],
            source_rotation="original",
            source_crop="fallback_full",
        )
    ]

    mask = build_tag_preservation_mask(
        source.shape[:2],
        boxes,
        "126725",
        source_image=source,
    )

    assert mask is None


def test_winning_detected_crop_builds_tight_physical_tag_mask():
    import cv2

    source = np.full((360, 520, 3), (132, 145, 156), dtype=np.uint8)
    tag_polygon = np.asarray([[285, 110], [450, 130], [440, 210], [275, 190]], dtype=np.int32)
    cv2.fillConvexPoly(source, tag_polygon, (246, 246, 246))
    crops = [
        CandidateCrop(
            "tag_crop_1_strict_white",
            source[90:230, 250:475].copy(),
            0.91,
            (250, 90, 225, 140),
        ),
        CandidateCrop(
            "tag_crop_2_soft_tag",
            source[20:100, 20:120].copy(),
            0.72,
            (20, 20, 100, 80),
        ),
    ]

    mask = build_detected_tag_preservation_mask(
        source.shape[:2],
        crops,
        source_image=source,
        preferred_label="tag_crop_1_strict_white",
    )

    assert mask is not None
    assert int(mask[160, 360]) == 255
    assert int(mask[95, 255]) == 0
    assert int(mask[225, 470]) == 0
    assert build_detected_tag_preservation_mask(
        source.shape[:2],
        crops,
        source_image=source,
        preferred_label="unknown_crop",
    ) is None
