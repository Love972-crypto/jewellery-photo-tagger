from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .image_enhancement import save_png


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
