from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .image_enhancement import save_png
from .models import OCRTextBox


@dataclass
class CandidateCrop:
    label: str
    image: np.ndarray
    score: float
    bbox: tuple[int, int, int, int] | None = None


def _crop_with_padding(image: np.ndarray, x: int, y: int, w: int, h: int, padding: int = 18) -> np.ndarray:
    img_h, img_w = image.shape[:2]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(img_w, x + w + padding)
    y1 = min(img_h, y + h + padding)
    return image[y0:y1, x0:x1].copy()


def detect_tag_crops(image: np.ndarray, max_crops: int = 4) -> list[CandidateCrop]:
    """Find likely light rectangular jewellery tag regions, then add robust fallbacks."""
    if image is None or image.size == 0:
        return []

    img_h, img_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    candidates: list[CandidateCrop] = []
    image_area = float(img_w * img_h)

    mask_specs = [
        ("strict_white", 225, 85, 5, 1),
        ("white_low_sat", 205, 95, 5, 1),
        ("soft_tag", 185, 70, 7, 1),
        ("broad_light", 155, 115, 9, 2),
    ]
    candidate_index = 1
    for label, gray_min, saturation_max, kernel_size, close_iterations in mask_specs:
        light_mask = cv2.inRange(gray, gray_min, 255)
        low_sat_mask = cv2.inRange(saturation, 0, saturation_max)
        mask = cv2.bitwise_and(light_mask, low_sat_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = float(w * h)
            if rect_area < image_area * 0.0012 or rect_area > image_area * 0.14:
                continue
            if area < image_area * 0.00035 or area > image_area * 0.12:
                continue
            if w < 24 or h < 24:
                continue
            aspect = w / max(h, 1)
            if aspect < 0.18 or aspect > 7.0:
                continue

            roi_gray = gray[y : y + h, x : x + w]
            roi_sat = saturation[y : y + h, x : x + w]
            crop = _crop_with_padding(image, x, y, w, h, padding=max(14, int(min(w, h) * 0.25)))
            mean_light = float(np.mean(roi_gray)) / 255.0
            low_sat = 1.0 - min(float(np.mean(roi_sat)) / 255.0, 1.0)
            extent = min(area / max(rect_area, 1.0), 1.0)
            dark_ratio = float(np.mean(roi_gray < 95))
            text_score = min(dark_ratio / 0.07, 1.0)
            size_score = 1.0 - min(abs(rect_area - (image_area * 0.018)) / max(image_area * 0.05, 1.0), 1.0)
            border_penalty = 0.72 if x <= 3 or y <= 3 or x + w >= img_w - 3 or y + h >= img_h - 3 else 1.0
            score = (
                (mean_light * 0.24)
                + (low_sat * 0.18)
                + (extent * 0.16)
                + (text_score * 0.24)
                + (size_score * 0.18)
            ) * border_penalty
            if label == "strict_white":
                score += 0.06
            if min(w, h) >= 55 and max(w, h) >= 150:
                score += 0.03
            candidates.append(CandidateCrop(f"tag_crop_{candidate_index}_{label}", crop, score, (x, y, w, h)))
            candidate_index += 1

    candidates.sort(key=lambda item: item.score, reverse=True)
    candidates = _dedupe_overlapping_candidates(candidates)[:max_crops]
    candidates.extend(_fallback_crops(image))

    deduped: list[CandidateCrop] = []
    seen_shapes: set[tuple[int, int, str]] = set()
    for candidate in candidates:
        h, w = candidate.image.shape[:2]
        key = (round(w / 20), round(h / 20), candidate.label)
        if key not in seen_shapes and h > 20 and w > 20:
            deduped.append(candidate)
            seen_shapes.add(key)
    return deduped[: max_crops + 4]


def _dedupe_overlapping_candidates(candidates: list[CandidateCrop]) -> list[CandidateCrop]:
    deduped: list[CandidateCrop] = []
    for candidate in candidates:
        if candidate.bbox is None:
            deduped.append(candidate)
            continue
        if all(_bbox_iou(candidate.bbox, existing.bbox) < 0.42 for existing in deduped if existing.bbox is not None):
            deduped.append(candidate)
    return deduped


def _bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0, min(ay2, by2) - max(ay, by))
    intersection = float(inter_w * inter_h)
    union = float((aw * ah) + (bw * bh) - intersection)
    return 0.0 if union <= 0 else intersection / union


def _fallback_crops(image: np.ndarray) -> list[CandidateCrop]:
    img_h, img_w = image.shape[:2]
    crops = [
        ("fallback_full", image.copy(), 0.2, None),
        ("fallback_lower", image[int(img_h * 0.45) : img_h, :].copy(), 0.18, None),
        ("fallback_center", image[int(img_h * 0.18) : int(img_h * 0.82), int(img_w * 0.12) : int(img_w * 0.88)].copy(), 0.16, None),
        ("fallback_right", image[:, int(img_w * 0.48) : img_w].copy(), 0.14, None),
    ]
    return [CandidateCrop(*item) for item in crops]


def rotated_versions(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("original", image),
        ("cw90", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
        ("ccw90", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ("rot180", cv2.rotate(image, cv2.ROTATE_180)),
    ]


def save_debug_crops(crops: list[CandidateCrop], original_stem: str, debug_dir: Path) -> list[Path]:
    saved: list[Path] = []
    if not crops:
        return saved

    debug_dir.mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(crops[:2], start=1):
        suffix = "tag_crop" if index == 1 else f"tag_crop_{index}"
        path = debug_dir / f"{original_stem}_{suffix}.png"
        save_png(candidate.image, path)
        saved.append(path)
    return saved


def build_tag_preservation_mask(
    image_shape: tuple[int, int],
    boxes: list[OCRTextBox],
    tag_number: str,
    crops: list[CandidateCrop] | None = None,
    source_image: np.ndarray | None = None,
) -> np.ndarray | None:
    height, width = image_shape
    anchors = [
        box
        for box in boxes
        if box.bbox
        and box.source_crop == "fallback_full"
        and tag_number in "".join(character for character in box.text if character.isdigit())
    ]
    if not anchors:
        return None

    anchor = max(anchors, key=lambda item: item.confidence)
    anchor_points = _map_ocr_points_to_original(anchor.bbox, anchor.source_rotation, width, height)
    if anchor_points.size == 0:
        return None

    selected_points = [anchor_points]
    anchor_center = np.mean(anchor_points, axis=0)
    anchor_span = max(float(np.ptp(anchor_points[:, 0])), float(np.ptp(anchor_points[:, 1])), 1.0)
    nearby_distance = max(anchor_span * 3.0, min(width, height) * 0.28)

    for box in boxes:
        digits = "".join(character for character in box.text if character.isdigit())
        if box is anchor or not box.bbox or len(digits) < 9:
            continue
        if box.source_crop != anchor.source_crop or box.source_rotation != anchor.source_rotation:
            continue
        points = _map_ocr_points_to_original(box.bbox, box.source_rotation, width, height)
        if points.size == 0:
            continue
        if float(np.linalg.norm(np.mean(points, axis=0) - anchor_center)) <= nearby_distance:
            selected_points.append(points)

    all_points = np.vstack(selected_points).astype(np.float32)
    center, size, angle = cv2.minAreaRect(all_points)
    rect_width, rect_height = size
    has_companion = len(selected_points) > 1
    if rect_width >= rect_height:
        width_factor, height_factor = ((1.35, 1.75) if has_companion else (2.2, 3.0))
    else:
        width_factor, height_factor = ((1.75, 1.35) if has_companion else (3.0, 2.2))
    expanded_size = (
        min(rect_width * width_factor, width * 0.48),
        min(rect_height * height_factor, height * 0.28),
    )
    polygon = cv2.boxPoints((center, expanded_size, angle)).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)

    # OCR usually identifies only the printed digits. When the rectangular tag
    # detector found the physical tag, prefer that tight shape over the broad
    # OCR fallback so nearby table/background pixels are not forced into output.
    anchor_x, anchor_y = float(anchor_center[0]), float(anchor_center[1])
    containing_crops: list[CandidateCrop] = []
    for crop in crops or []:
        if crop.bbox is None or crop.label.startswith("fallback"):
            continue
        x, y, box_width, box_height = crop.bbox
        margin_x = max(4, int(box_width * 0.08))
        margin_y = max(4, int(box_height * 0.08))
        if (
            x - margin_x <= anchor_x <= x + box_width + margin_x
            and y - margin_y <= anchor_y <= y + box_height + margin_y
        ):
            containing_crops.append(crop)

    shape_candidate: CandidateCrop | None = None
    if containing_crops:
        candidate = min(
            containing_crops,
            key=lambda item: item.bbox[2] * item.bbox[3] if item.bbox else float("inf"),
        )
        shape_candidate = candidate
    elif source_image is not None and source_image.size:
        # OCR fallback boxes describe printed digits, not the physical tag. Use
        # the broad geometry only as a search window, then isolate the actual
        # bright tag shape. Never return that broad rectangle as foreground.
        fallback_x, fallback_y, fallback_width, fallback_height = cv2.boundingRect(polygon)
        if fallback_width > 0 and fallback_height > 0:
            shape_candidate = CandidateCrop(
                "fallback_refine_physical_tag",
                source_image[
                    fallback_y : fallback_y + fallback_height,
                    fallback_x : fallback_x + fallback_width,
                ].copy(),
                anchor.confidence,
                (fallback_x, fallback_y, fallback_width, fallback_height),
            )

    if shape_candidate is not None:
        detected_shape = _detected_tag_shape_mask(source_image, shape_candidate, (anchor_x, anchor_y))
        if detected_shape is not None:
            return detected_shape
    if source_image is not None:
        return None
    return mask


def _detected_tag_shape_mask(
    source_image: np.ndarray | None,
    candidate: CandidateCrop,
    anchor_center: tuple[float, float],
) -> np.ndarray | None:
    if source_image is None or source_image.size == 0 or candidate.bbox is None:
        return None

    image_height, image_width = source_image.shape[:2]
    x, y, box_width, box_height = candidate.bbox
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(image_width, x + box_width), min(image_height, y + box_height)
    if x1 <= x0 or y1 <= y0:
        return None

    roi = source_image[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1]
    if candidate.label == "fallback_refine_physical_tag":
        otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gray_min = max(180, min(235, int(round(otsu_threshold))))
        saturation_max = 105
    elif "strict_white" in candidate.label:
        gray_min, saturation_max = 218, 100
    elif "white_low_sat" in candidate.label:
        gray_min, saturation_max = 198, 112
    elif "soft_tag" in candidate.label:
        gray_min, saturation_max = 180, 105
    else:
        gray_min, saturation_max = 165, 125

    light = (gray >= gray_min) & (saturation <= saturation_max)
    light_mask = light.astype(np.uint8) * 255
    kernel_size = max(3, int(round(min(box_width, box_height) * 0.035)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    light_mask = cv2.morphologyEx(light_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    local_anchor = np.asarray([anchor_center[0] - x0, anchor_center[1] - y0], dtype=np.float32)

    def contour_rank(contour: np.ndarray) -> tuple[int, float, float]:
        inside = int(cv2.pointPolygonTest(contour, tuple(local_anchor), False) >= 0)
        moments = cv2.moments(contour)
        if moments["m00"]:
            center = np.asarray(
                [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
                dtype=np.float32,
            )
            distance = float(np.linalg.norm(center - local_anchor))
        else:
            distance = float("inf")
        return inside, -distance, float(cv2.contourArea(contour))

    contour = max(contours, key=contour_rank)
    if cv2.contourArea(contour) < max(80.0, box_width * box_height * 0.08):
        return None

    # Preserve the physical silhouette, not its rotated bounding rectangle.
    # A rectangle fills tag notches/corners with floor pixels and creates the
    # brown block seen beside otherwise clean catalogue cutouts.
    local_shape = np.zeros_like(light_mask)
    cv2.drawContours(local_shape, [contour], -1, 255, thickness=cv2.FILLED)
    edge_pad = max(2, int(round(min(box_width, box_height) * 0.015)))
    edge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (edge_pad * 2 + 1, edge_pad * 2 + 1),
    )
    local_shape = cv2.dilate(local_shape, edge_kernel, iterations=1)
    shape_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    shape_mask[y0:y1, x0:x1] = local_shape
    return shape_mask


def build_detected_tag_preservation_mask(
    image_shape: tuple[int, int],
    crops: list[CandidateCrop],
    source_image: np.ndarray | None = None,
    preferred_label: str = "",
) -> np.ndarray | None:
    """Preserve the strongest detected tag when OCR needs manual correction."""
    detected = [crop for crop in crops if crop.bbox is not None and not crop.label.startswith("fallback")]
    if not detected:
        return None

    if preferred_label:
        preferred = [crop for crop in detected if crop.label == preferred_label]
        if not preferred:
            return None
        detected = preferred

    height, width = image_shape
    candidate = max(detected, key=lambda item: item.score)
    x, y, box_width, box_height = candidate.bbox
    if source_image is not None and source_image.size:
        tight_shape = _detected_tag_shape_mask(
            source_image,
            candidate,
            (x + box_width / 2.0, y + box_height / 2.0),
        )
        if tight_shape is not None:
            return tight_shape
        return None

    pad_x = max(10, int(box_width * 0.18))
    pad_y = max(10, int(box_height * 0.24))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + box_width + pad_x)
    y1 = min(height, y + box_height + pad_y)
    if x1 <= x0 or y1 <= y0:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def _map_ocr_points_to_original(bbox, rotation: str, width: int, height: int) -> np.ndarray:
    try:
        points = np.asarray(bbox, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return np.empty((0, 2), dtype=np.float32)

    mapped = points.copy()
    if rotation == "rot180":
        mapped[:, 0] = (width - 1) - points[:, 0]
        mapped[:, 1] = (height - 1) - points[:, 1]
    elif rotation == "cw90":
        mapped[:, 0] = points[:, 1]
        mapped[:, 1] = (height - 1) - points[:, 0]
    elif rotation == "ccw90":
        mapped[:, 0] = (width - 1) - points[:, 1]
        mapped[:, 1] = points[:, 0]
    mapped[:, 0] = np.clip(mapped[:, 0], 0, width - 1)
    mapped[:, 1] = np.clip(mapped[:, 1], 0, height - 1)
    return mapped
