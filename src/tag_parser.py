from __future__ import annotations

import re
from dataclasses import dataclass

from .models import OCRTextBox, ParsedTag, STATUS_OCR_FAILED, STATUS_OK, STATUS_REVIEW_REQUIRED, STATUS_TAG_NOT_FOUND

STANDALONE_NUMERIC_RE = re.compile(r"(?<![A-Za-z0-9])(\d{5,8})(?![A-Za-z0-9])")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass
class TagCandidate:
    value: str
    score: float
    source_text: str
    confidence: float
    prominence: float


def normalize_possible_digits(text: str) -> str:
    table = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "S": "5", "s": "5", "B": "8"})
    return text.translate(table)


def _bbox_prominence(box: OCRTextBox) -> float:
    if not box.bbox:
        return 0.5
    try:
        xs = [float(point[0]) for point in box.bbox]
        ys = [float(point[1]) for point in box.bbox]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        return min(max((height / 60.0) * 0.65 + (width / 260.0) * 0.35, 0.05), 1.0)
    except Exception:
        return 0.5


def _extract_candidates_from_box(box: OCRTextBox) -> list[TagCandidate]:
    text = box.text.strip()
    normalized = normalize_possible_digits(text)
    candidates: list[TagCandidate] = []
    prominence = _bbox_prominence(box)

    for match in STANDALONE_NUMERIC_RE.finditer(normalized):
        value = match.group(1)
        length_bonus = 1.0 if 5 <= len(value) <= 8 else 0.0
        score = (box.confidence * 0.68) + (prominence * 0.22) + (length_bonus * 0.1)
        candidates.append(TagCandidate(value, score, text, box.confidence, prominence))

    if candidates:
        return candidates

    for token in TOKEN_RE.findall(normalized):
        if token.isdigit() and 5 <= len(token) <= 8:
            score = (box.confidence * 0.65) + (prominence * 0.2) + 0.08
            candidates.append(TagCandidate(token, score, text, box.confidence, prominence))
    return candidates


def parse_tag_from_ocr(boxes: list[OCRTextBox], confidence_threshold: float = 0.45) -> ParsedTag:
    raw_text = " | ".join(box.text for box in boxes if box.text.strip())
    if not boxes or not raw_text:
        return ParsedTag(status=STATUS_OCR_FAILED, raw_text="", notes="OCR returned no readable text.")

    candidates: list[TagCandidate] = []
    for box in boxes:
        candidates.extend(_extract_candidates_from_box(box))

    if not candidates:
        return ParsedTag(status=STATUS_TAG_NOT_FOUND, raw_text=raw_text, notes="No valid 5-8 digit tag number found.")

    candidates.sort(key=lambda item: item.score, reverse=True)
    best = candidates[0]
    confidence = round(min(max(best.score, 0.0), 1.0), 4)
    if confidence < confidence_threshold:
        return ParsedTag(
            tag_number=best.value,
            confidence=confidence,
            raw_text=raw_text,
            status=STATUS_REVIEW_REQUIRED,
            notes=f"Possible tag {best.value}, but confidence is below threshold.",
        )

    return ParsedTag(
        tag_number=best.value,
        confidence=confidence,
        raw_text=raw_text,
        status=STATUS_OK,
        notes=f"Selected prominent numeric tag {best.value}.",
    )


def is_valid_manual_tag(value: str) -> bool:
    return bool(re.fullmatch(r"\d{5,8}", value.strip()))

