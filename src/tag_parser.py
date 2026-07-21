from __future__ import annotations

import re
from dataclasses import dataclass

from .models import OCRTextBox, ParsedTag, STATUS_OCR_FAILED, STATUS_OK, STATUS_REVIEW_REQUIRED, STATUS_TAG_NOT_FOUND

STANDALONE_NUMERIC_RE = re.compile(r"(?<![A-Za-z0-9])(\d{5,8})(?![A-Za-z0-9])")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
REVERSE_CONFLICT_MARGIN = 0.08
GENERAL_CONFLICT_MARGIN = 0.10
CONSENSUS_BONUS_PER_SOURCE = 0.025
MAX_CONSENSUS_BONUS = 0.08
CORROBORATED_MIN_SCORE = 0.72
SINGLE_SOURCE_MIN_SCORE = 0.88
MIN_CORROBORATING_OCR_CONFIDENCE = 0.60


@dataclass
class TagCandidate:
    value: str
    score: float
    source_text: str
    confidence: float
    prominence: float
    source_rotation: str
    source_crop: str


@dataclass
class AggregatedTagCandidate:
    value: str
    score: float
    best: TagCandidate
    source_count: int
    strong_source_count: int


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
    source_adjustment = _source_score_adjustment(box)

    for match in STANDALONE_NUMERIC_RE.finditer(normalized):
        value = match.group(1)
        length_bonus = 1.0 if 5 <= len(value) <= 8 else 0.0
        score = (box.confidence * 0.68) + (prominence * 0.22) + (length_bonus * 0.1) + source_adjustment
        candidates.append(
            TagCandidate(
                value,
                score,
                text,
                box.confidence,
                prominence,
                box.source_rotation,
                box.source_crop,
            )
        )

    if candidates:
        return candidates

    for token in TOKEN_RE.findall(normalized):
        if token.isdigit() and 5 <= len(token) <= 8:
            score = (box.confidence * 0.65) + (prominence * 0.2) + 0.08 + source_adjustment
            candidates.append(
                TagCandidate(
                    token,
                    score,
                    text,
                    box.confidence,
                    prominence,
                    box.source_rotation,
                    box.source_crop,
                )
            )
    return candidates


def _source_score_adjustment(box: OCRTextBox) -> float:
    # Auto-rotation on a full frame is useful for discovery, but a focused
    # physical crop is more trustworthy for visually similar digits such as 0/8.
    if box.source_crop == "fallback_full" and box.source_rotation == "auto_best":
        return -0.06
    if box.source_crop == "fallback_full_recheck":
        return 0.015
    return 0.0


def _aggregate_candidates(candidates: list[TagCandidate]) -> list[AggregatedTagCandidate]:
    grouped: dict[str, list[TagCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.value, []).append(candidate)

    aggregated: list[AggregatedTagCandidate] = []
    for value, matches in grouped.items():
        best = max(matches, key=lambda item: item.score)
        source_confidences: dict[tuple[str, str], float] = {}
        for item in matches:
            source = (_canonical_crop_label(item.source_crop), item.source_rotation)
            source_confidences[source] = max(source_confidences.get(source, 0.0), item.confidence)
        sources = set(source_confidences)
        strong_source_count = sum(
            confidence >= MIN_CORROBORATING_OCR_CONFIDENCE
            for confidence in source_confidences.values()
        )
        consensus_bonus = min(
            max(0, strong_source_count - 1) * CONSENSUS_BONUS_PER_SOURCE,
            MAX_CONSENSUS_BONUS,
        )
        aggregated.append(
            AggregatedTagCandidate(
                value=value,
                score=min(best.score + consensus_bonus, 1.0),
                best=best,
                source_count=len(sources),
                strong_source_count=strong_source_count,
            )
        )
    return sorted(aggregated, key=lambda item: item.score, reverse=True)


def _canonical_crop_label(label: str) -> str:
    """Treat raw and enhanced variants of one crop as the same evidence source."""
    return label[:-9] if label.endswith("_enhanced") else label


def parse_tag_from_ocr(boxes: list[OCRTextBox], confidence_threshold: float = 0.45) -> ParsedTag:
    raw_text = " | ".join(box.text for box in boxes if box.text.strip())
    if not boxes or not raw_text:
        return ParsedTag(status=STATUS_OCR_FAILED, raw_text="", notes="OCR returned no readable text.")

    candidates: list[TagCandidate] = []
    for box in boxes:
        candidates.extend(_extract_candidates_from_box(box))

    if not candidates:
        return ParsedTag(status=STATUS_TAG_NOT_FOUND, raw_text=raw_text, notes="No valid 5-8 digit tag number found.")

    aggregated = _aggregate_candidates(candidates)
    best = aggregated[0]
    confidence = round(min(max(best.score, 0.0), 1.0), 4)
    score_margin = best.score - aggregated[1].score if len(aggregated) > 1 else 1.0
    evidence = {
        "evidence_count": best.source_count,
        "strong_evidence_count": best.strong_source_count,
        "best_ocr_confidence": round(best.best.confidence, 4),
        "score_margin": round(max(score_margin, 0.0), 4),
        "best_source_rotation": best.best.source_rotation,
        "best_source_crop": best.best.source_crop,
    }

    reverse_match = _strongest_opposite_orientation_candidate(best, aggregated[1:])
    if reverse_match is not None:
        score_gap = best.score - reverse_match.score
        if score_gap < REVERSE_CONFLICT_MARGIN:
            return ParsedTag(
                tag_number=best.value,
                confidence=confidence,
                raw_text=raw_text,
                status=STATUS_REVIEW_REQUIRED,
                notes=(
                    f"Opposite rotations disagree between {best.value} and {reverse_match.value}; "
                    "please confirm the tag."
                ),
                **evidence,
            )
        conflict_note = (
            f"Resolved reverse-orientation candidate {reverse_match.value} "
            f"using stronger evidence for {best.value}."
        )
    else:
        conflict_note = ""

    runner_up = aggregated[1] if len(aggregated) > 1 else None
    verified_best_over_preliminary_runner = (
        best.strong_source_count >= 2
        and runner_up is not None
        and runner_up.source_count == 1
        and runner_up.best.source_crop == "fallback_full"
        and runner_up.best.source_rotation == "auto_best"
        and best.score > runner_up.score
    )
    if (
        runner_up is not None
        and runner_up is not reverse_match
        and best.score - runner_up.score < GENERAL_CONFLICT_MARGIN
        and runner_up.score >= CORROBORATED_MIN_SCORE
        and not verified_best_over_preliminary_runner
    ):
        return ParsedTag(
            tag_number=best.value,
            confidence=confidence,
            raw_text=raw_text,
            status=STATUS_REVIEW_REQUIRED,
            notes=(
                f"OCR found competing tag values {best.value} and {runner_up.value} "
                "with nearly equal evidence; please confirm the tag."
            ),
            **evidence,
        )

    required_score = max(
        confidence_threshold,
        SINGLE_SOURCE_MIN_SCORE if best.strong_source_count < 2 else CORROBORATED_MIN_SCORE,
    )
    resolved_reverse_is_strong = (
        reverse_match is not None
        and best.score - reverse_match.score >= REVERSE_CONFLICT_MARGIN
        and best.best.confidence >= 0.94
    )
    single_source_is_unverified = best.strong_source_count < 2 and not resolved_reverse_is_strong
    if confidence < required_score or single_source_is_unverified:
        reason = (
            "only one independent OCR reading was available"
            if best.strong_source_count < 2
            else "the combined OCR confidence is below the safe threshold"
        )
        return ParsedTag(
            tag_number=best.value,
            confidence=confidence,
            raw_text=raw_text,
            status=STATUS_REVIEW_REQUIRED,
            notes=f"Possible tag {best.value}, but {reason}; please confirm it manually.",
            **evidence,
        )

    return ParsedTag(
        tag_number=best.value,
        confidence=confidence,
        raw_text=raw_text,
        status=STATUS_OK,
        notes=(
            f"Selected prominent numeric tag {best.value} from {best.strong_source_count} verified OCR source(s)."
            + (f" {conflict_note}" if conflict_note else "")
        ),
        **evidence,
    )


def _strongest_opposite_orientation_candidate(
    best: AggregatedTagCandidate,
    others: list[AggregatedTagCandidate],
) -> AggregatedTagCandidate | None:
    opposite_pairs = {
        frozenset(("cw90", "ccw90")),
        frozenset(("original", "rot180")),
    }
    conflicts = []
    for candidate in others:
        if candidate.value == best.value:
            continue
        rotations = frozenset((best.best.source_rotation, candidate.best.source_rotation))
        same_crop_opposite_rotation = (
            best.best.source_crop == candidate.best.source_crop
            and rotations in opposite_pairs
        )
        literal_reverse = candidate.value in {best.value[::-1], best.value[::-1].translate(str.maketrans("69", "96"))}
        if same_crop_opposite_rotation or literal_reverse:
            conflicts.append(candidate)
    return max(conflicts, key=lambda item: item.score, default=None)


def is_valid_manual_tag(value: str) -> bool:
    return bool(re.fullmatch(r"\d{5,8}", value.strip()))
