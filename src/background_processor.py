from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Callable

import cv2
import numpy as np
from PIL import Image


BACKGROUND_OK = "OK"
BACKGROUND_DISABLED = "DISABLED"
BACKGROUND_REVIEW_REQUIRED = "REVIEW_REQUIRED"
BACKGROUND_FAILED = "FAILED"
BACKGROUND_MANUAL_ACCEPTED = "MANUAL_ACCEPTED"
BACKGROUND_ORIGINAL_KEPT = "ORIGINAL_KEPT"
BACKGROUND_AI_REVIEW_READY = "AI_REVIEW_READY"
BACKGROUND_AI_ACCEPTED = "AI_ACCEPTED"
BACKGROUND_AI_MANUAL_REVIEW = "AI_MANUAL_REVIEW"
BACKGROUND_HYBRID_OK = "AI_HYBRID_OK"
BACKGROUND_BIREFNET_ONLY_OK = "BIREFNET_ONLY_OK"

AI_FALLBACK_MODEL = "birefnet-general-lite"
AI_REFINEMENT_RESIDUE_RATIO = 0.0008
AI_REFINEMENT_COMPONENT_RATIO = 0.0002
TAG_PRESERVE_RESIDUE_MIN_IMAGE_RATIO = 0.00035
TAG_PRESERVE_RESIDUE_MIN_REGION_RATIO = 0.08
SMART_CLEAN_BACKGROUND_MIN_COVERAGE = 0.12
SMART_CLEAN_GRADIENT_MEDIAN_MAX = 12.0
SMART_CLEAN_GRADIENT_P90_MAX = 38.0

MAX_REMOVED_FOREGROUND_RATIO = 0.02
MAX_REMOVED_SOURCE_SUPPORTED_RATIO = 0.0008
MAX_INTERIOR_SOURCE_HOLE_RATIO = 0.0004
MAX_RETAINED_RESIDUE_RATIO = 0.0012
MAX_RESIDUE_COMPONENT_RATIO = 0.00025
MAX_CATASTROPHIC_SOURCE_LOSS_RATIO = 0.008
MAX_CATASTROPHIC_INTERIOR_HOLE_RATIO = 0.008
MAX_CATASTROPHIC_LOST_COMPONENTS = 25
MAX_CATASTROPHIC_MISSING_COMPONENTS = 3

RemoveCallable = Callable[[Image.Image], Image.Image]


@dataclass
class BackgroundResult:
    status: str
    transparent_rgba: np.ndarray | None = None
    white_bgr: np.ndarray | None = None
    notes: str = ""
    safety_metrics: "MaskSafetyMetrics | None" = None
    ai_refinement_reasons: tuple[str, ...] = ()
    source_rgba: np.ndarray | None = None


@dataclass(frozen=True)
class MaskSafetyMetrics:
    raw_foreground_ratio: float
    cleaned_foreground_ratio: float
    removed_foreground_ratio: float
    removed_source_supported_ratio: float
    interior_source_hole_ratio: float
    raw_component_count: int
    cleaned_component_count: int
    lost_supported_components: int
    missing_source_components: int
    retained_background_residue_ratio: float
    largest_residue_component_ratio: float
    reasons: tuple[str, ...] = ()

    @property
    def safe(self) -> bool:
        return not self.reasons

    def summary(self) -> str:
        details = (
            f"raw foreground {self.raw_foreground_ratio:.1%}, "
            f"removed {self.removed_foreground_ratio:.1%} of raw foreground, "
            f"source-supported loss {self.removed_source_supported_ratio:.2%} of image, "
            f"retained residue {self.retained_background_residue_ratio:.2%}"
        )
        if self.reasons:
            return f"Jewellery preservation risk: {'; '.join(self.reasons)} ({details})."
        return f"Jewellery preservation check passed ({details})."


@dataclass(frozen=True)
class SmartHybridDecision:
    use_u2net_preservation: bool
    reasons: tuple[str, ...]
    background_coverage: float
    gradient_median: float
    gradient_p90: float

    @property
    def route(self) -> str:
        return "birefnet_plus_u2net" if self.use_u2net_preservation else "birefnet_only"

    def summary(self) -> str:
        signal = (
            f"background coverage {self.background_coverage:.1%}, "
            f"texture median {self.gradient_median:.1f}, p90 {self.gradient_p90:.1f}"
        )
        if self.reasons:
            return f"Smart Hybrid selected U2Net preservation: {'; '.join(self.reasons)} ({signal})."
        return f"Smart Hybrid selected BiRefNet only for a clean photo ({signal})."


def needs_ai_refinement(result: BackgroundResult) -> bool:
    """Return True when U2Net completed but left a likely quality defect."""
    if result.status != BACKGROUND_OK:
        return True
    if result.ai_refinement_reasons:
        return True
    metrics = result.safety_metrics
    if metrics is None:
        return False
    return (
        metrics.missing_source_components > 0
        or metrics.retained_background_residue_ratio > AI_REFINEMENT_RESIDUE_RATIO
        or metrics.largest_residue_component_ratio > AI_REFINEMENT_COMPONENT_RATIO
    )


def choose_smart_hybrid_route(image_bgr: np.ndarray, birefnet_result: BackgroundResult) -> SmartHybridDecision:
    """Choose the fast BiRefNet-only path only when both mask and background are clean."""
    reasons: list[str] = []
    metrics = birefnet_result.safety_metrics

    if birefnet_result.status != BACKGROUND_OK:
        reasons.append(f"BiRefNet status is {birefnet_result.status}")
    if birefnet_result.source_rgba is None:
        reasons.append("BiRefNet source matte is unavailable")
    if metrics is None:
        reasons.append("BiRefNet safety metrics are unavailable")
    else:
        if metrics.removed_source_supported_ratio > MAX_REMOVED_SOURCE_SUPPORTED_RATIO:
            reasons.append("source-supported jewellery pixels need preservation")
        if metrics.interior_source_hole_ratio > MAX_INTERIOR_SOURCE_HOLE_RATIO:
            reasons.append("possible interior jewellery holes were detected")
        if metrics.lost_supported_components > 0:
            reasons.append("a supported jewellery component may be missing")
        if metrics.missing_source_components > 0:
            reasons.append("a coloured jewellery component may be missing")
        if metrics.retained_background_residue_ratio > AI_REFINEMENT_RESIDUE_RATIO:
            reasons.append("the BiRefNet matte may retain background residue")
        if metrics.largest_residue_component_ratio > AI_REFINEMENT_COMPONENT_RATIO:
            reasons.append("a significant residue patch may remain")
    if birefnet_result.ai_refinement_reasons:
        reasons.extend(birefnet_result.ai_refinement_reasons)

    background_coverage, gradient_median, gradient_p90 = _background_texture_metrics(
        image_bgr,
        birefnet_result.source_rgba,
    )
    if background_coverage < SMART_CLEAN_BACKGROUND_MIN_COVERAGE:
        reasons.append("too little verified background is available for a clean-photo decision")
    if (
        gradient_median > SMART_CLEAN_GRADIENT_MEDIAN_MAX
        or gradient_p90 > SMART_CLEAN_GRADIENT_P90_MAX
    ):
        reasons.append("the background is textured or visually complex")

    return SmartHybridDecision(
        use_u2net_preservation=bool(reasons),
        reasons=tuple(dict.fromkeys(reasons)),
        background_coverage=background_coverage,
        gradient_median=gradient_median,
        gradient_p90=gradient_p90,
    )


def _background_texture_metrics(
    image_bgr: np.ndarray,
    source_rgba: np.ndarray | None,
) -> tuple[float, float, float]:
    if source_rgba is None or source_rgba.size == 0:
        return 0.0, float("inf"), float("inf")

    alpha = source_rgba[:, :, 3]
    foreground = alpha > 18
    radius = max(5, int(min(alpha.shape) * 0.025))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    near_foreground = cv2.dilate(foreground.astype(np.uint8), kernel, iterations=1) > 0
    background = ~near_foreground
    coverage = float(np.mean(background))
    if coverage < SMART_CLEAN_BACKGROUND_MIN_COVERAGE:
        return coverage, float("inf"), float("inf")

    source = _resize_source(image_bgr, alpha.shape)
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    values = gradient[background]
    if values.size == 0:
        return coverage, float("inf"), float("inf")
    return coverage, float(np.percentile(values, 50)), float(np.percentile(values, 90))


def remove_background(
    image_bgr: np.ndarray,
    remover: RemoveCallable | None = None,
    alpha_matting: bool = True,
    catalogue_layout: bool = True,
    canvas_size: tuple[int, int] = (1200, 1500),
    max_side: int = 2200,
    model_name: str = "u2net",
    preserve_mask: np.ndarray | None = None,
) -> BackgroundResult:
    if image_bgr is None or image_bgr.size == 0:
        return BackgroundResult(status=BACKGROUND_FAILED, notes="Input image was empty.")

    try:
        image_bgr = _limit_long_side(image_bgr, max_side=max_side)
        pil_input = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if remover:
            pil_output = remover(pil_input)
        else:
            pil_output = _remove_with_rembg(pil_input, alpha_matting=alpha_matting, model_name=model_name)
        transparent_rgba = np.array(pil_output.convert("RGBA"))
    except Exception as exc:
        return BackgroundResult(status=BACKGROUND_FAILED, notes=f"Background removal failed: {exc}")

    raw_rgba = _apply_preservation_mask(transparent_rgba, image_bgr, preserve_mask)
    ai_refinement_reasons = (
        _tag_preservation_refinement_reasons(image_bgr, raw_rgba, preserve_mask)
        if model_name == "u2net"
        else ()
    )
    # The proven pre-deploy U2Net path keeps the model matte intact and only
    # removes isolated weak alpha. Source-guided colour recovery created the
    # grey/speckled floor artifacts seen in production photos.
    preserve_raw_matte_models = {"u2net", AI_FALLBACK_MODEL}
    source_for_cleanup = image_bgr if remover is not None or model_name not in preserve_raw_matte_models else None
    transparent_rgba, cleanup_notes = _clean_alpha_matte(raw_rgba, source_bgr=source_for_cleanup)
    source_rgba = transparent_rgba.copy()
    measured_safety = _evaluate_cleanup_safety(
        raw_rgba,
        transparent_rgba,
        image_bgr,
    )
    safety_metrics = replace(measured_safety, reasons=_catastrophic_safety_reasons(measured_safety))

    check_ok, check_notes = _validate_alpha_mask(transparent_rgba[:, :, 3])
    if not check_ok:
        transparent_review, white_review = _review_outputs(
            transparent_rgba,
            catalogue_layout=catalogue_layout,
            canvas_size=canvas_size,
        )
        return BackgroundResult(
            status=BACKGROUND_REVIEW_REQUIRED,
            transparent_rgba=transparent_review,
            white_bgr=white_review,
            notes=f"{check_notes} {safety_metrics.summary()}",
            safety_metrics=safety_metrics,
            ai_refinement_reasons=ai_refinement_reasons,
            source_rgba=source_rgba,
        )

    if not safety_metrics.safe:
        transparent_review, white_review = _review_outputs(
            transparent_rgba,
            catalogue_layout=catalogue_layout,
            canvas_size=canvas_size,
        )
        notes = safety_metrics.summary()
        if cleanup_notes:
            notes = f"{notes} {cleanup_notes}"
        return BackgroundResult(
            status=BACKGROUND_REVIEW_REQUIRED,
            transparent_rgba=transparent_review,
            white_bgr=white_review,
            notes=notes,
            safety_metrics=safety_metrics,
            ai_refinement_reasons=ai_refinement_reasons,
            source_rgba=source_rgba,
        )

    if catalogue_layout:
        transparent_rgba, white_bgr = _catalogue_portrait_outputs(transparent_rgba, canvas_size)
        notes = f"Background removed safely with {model_name}. Catalogue portrait aligned."
    else:
        white_bgr = _compose_on_white(transparent_rgba)
        notes = f"Background removed safely with {model_name}."
    if cleanup_notes:
        notes = f"{notes} {cleanup_notes}"
    if ai_refinement_reasons:
        notes = f"{notes} AI refinement recommended: {'; '.join(ai_refinement_reasons)}."

    return BackgroundResult(
        status=BACKGROUND_OK,
        transparent_rgba=transparent_rgba,
        white_bgr=white_bgr,
        notes=notes,
        safety_metrics=safety_metrics,
        ai_refinement_reasons=ai_refinement_reasons,
        source_rgba=source_rgba,
    )


def fuse_u2net_preservation_with_ai(
    source_bgr: np.ndarray,
    u2net_result: BackgroundResult,
    ai_result: BackgroundResult,
    catalogue_layout: bool = True,
    canvas_size: tuple[int, int] = (1200, 1500),
    tag_preserve_mask: np.ndarray | None = None,
) -> BackgroundResult:
    """Use the AI matte as base and restore only source-supported jewellery from U2Net."""
    if u2net_result.source_rgba is None or ai_result.source_rgba is None:
        return ai_result

    u2net_rgba = u2net_result.source_rgba
    ai_rgba = ai_result.source_rgba
    if u2net_rgba.shape[:2] != ai_rgba.shape[:2]:
        u2net_rgba = cv2.resize(
            u2net_rgba,
            (ai_rgba.shape[1], ai_rgba.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )

    shape = ai_rgba.shape[:2]
    source = _resize_source(source_bgr, shape)
    u2net_alpha = u2net_rgba[:, :, 3]
    ai_alpha = ai_rgba[:, :, 3]
    restore = _hybrid_jewellery_support_mask(source, u2net_alpha, ai_alpha)

    fused = ai_rgba.copy()
    fused_alpha = np.maximum(ai_alpha, np.where(restore, u2net_alpha, 0).astype(np.uint8))
    source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    fused[restore, :3] = source_rgb[restore]
    fused[:, :, 3] = fused_alpha
    fused[fused_alpha == 0, :3] = 255
    fused, removed_tag_residue = _remove_neutral_residue_around_tag(
        fused,
        source,
        tag_preserve_mask,
    )
    fused, removed_islands = _remove_tiny_alpha_islands(fused)
    fused_alpha = fused[:, :, 3]

    check_ok, check_notes = _validate_alpha_mask(fused_alpha)
    remaining_missing = _missing_source_component_count(
        _source_colored_jewellery_mask(source, shape),
        fused_alpha > 18,
    )
    prior_reasons = ai_result.safety_metrics.reasons if ai_result.safety_metrics is not None else ()
    only_missing_component_risk = bool(prior_reasons) and all(
        "coloured jewellery component" in reason for reason in prior_reasons
    )
    can_recover_ai_review = ai_result.status == BACKGROUND_REVIEW_REQUIRED and only_missing_component_risk
    status = ai_result.status
    if ai_result.status == BACKGROUND_OK or can_recover_ai_review:
        status = BACKGROUND_OK if check_ok and remaining_missing < MAX_CATASTROPHIC_MISSING_COMPONENTS else BACKGROUND_REVIEW_REQUIRED

    restored_pixels = int(np.count_nonzero(restore))
    restored_ratio = float(np.mean(restore))
    notes = (
        f"Hybrid {AI_FALLBACK_MODEL} finish kept the AI matte as the clean base and used U2Net only "
        f"to protect {restored_pixels:,} verified jewellery-edge pixel(s) ({restored_ratio:.3%} of the image)."
    )
    if removed_islands:
        notes = f"{notes} Removed {removed_islands} tiny background island(s)."
    if removed_tag_residue:
        notes = (
            f"{notes} Removed {removed_tag_residue:,} neutral background/shadow pixel(s) "
            "immediately outside the physical tag."
        )
    if remaining_missing:
        if status == BACKGROUND_OK:
            notes = (
                f"{notes} {remaining_missing} small source-colour component(s) stayed outside the mask "
                "but remained within the automatic safety tolerance."
            )
        else:
            notes = f"{notes} {remaining_missing} coloured jewellery component(s) need review."
    if not check_ok:
        notes = f"{notes} {check_notes}"

    if status == BACKGROUND_OK:
        if catalogue_layout:
            transparent_rgba, white_bgr = _catalogue_portrait_outputs(fused, canvas_size)
            notes = f"{notes} Catalogue portrait aligned."
        else:
            transparent_rgba = fused
            white_bgr = _compose_on_white(fused)
    else:
        transparent_rgba, white_bgr = _review_outputs(
            fused,
            catalogue_layout=catalogue_layout,
            canvas_size=canvas_size,
        )

    safety_metrics = ai_result.safety_metrics
    if safety_metrics is not None:
        remaining_reasons = tuple(
            reason for reason in safety_metrics.reasons if "coloured jewellery component" not in reason
        )
        if remaining_missing >= MAX_CATASTROPHIC_MISSING_COMPONENTS:
            remaining_reasons = remaining_reasons + (
                f"{remaining_missing} coloured jewellery components are missing from the mask",
            )
        safety_metrics = replace(
            safety_metrics,
            missing_source_components=remaining_missing,
            reasons=_merge_safety_reasons(remaining_reasons),
        )

    return BackgroundResult(
        status=status,
        transparent_rgba=transparent_rgba,
        white_bgr=white_bgr,
        notes=notes,
        safety_metrics=safety_metrics,
        source_rgba=fused,
    )


def _remove_neutral_residue_around_tag(
    image_rgba: np.ndarray,
    source_bgr: np.ndarray,
    tag_preserve_mask: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    """Remove wood/shadow retained beside a tag without touching its bright body.

    The physical-tag mask is already tightened by tag detection. Only neutral,
    non-bright foreground in a narrow band outside that mask is removed; coloured
    jewellery and the tag itself remain protected.
    """
    if tag_preserve_mask is None or tag_preserve_mask.size == 0:
        return image_rgba, 0

    height, width = image_rgba.shape[:2]
    tag_mask = tag_preserve_mask
    if tag_mask.shape[:2] != (height, width):
        tag_mask = cv2.resize(tag_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    tag = tag_mask > 127
    tag_pixels = int(np.count_nonzero(tag))
    if tag_pixels < 32 or tag_pixels > int(tag.size * 0.18):
        return image_rgba, 0

    radius = max(5, int(round(min(height, width) * 0.014)))
    outside_band = _dilate_mask(tag, radius=radius) & ~tag
    foreground = image_rgba[:, :, 3] > 18

    source = _resize_source(source_bgr, (height, width))
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    coloured_jewellery = _dilate_mask(
        _source_colored_jewellery_mask(source, (height, width)),
        radius=2,
    )

    # Wood, grey floor and tag shadows are neutral and darker than the tag.
    # Bright white strings stay protected, as do saturated jewellery materials.
    residue = (
        outside_band
        & foreground
        & (saturation <= 82)
        & (value <= 218)
        & ~coloured_jewellery
    )
    removed = int(np.count_nonzero(residue))
    if not removed:
        return image_rgba, 0

    cleaned = image_rgba.copy()
    cleaned[residue, 3] = 0
    cleaned[residue, :3] = 255
    return cleaned, removed


def save_rgba_png(image_rgba: np.ndarray, output_path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise ValueError(f"Could not encode transparent PNG: {output_path.name}")
    encoded.tofile(str(output_path))


def _limit_long_side(image_bgr: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return image_bgr
    height, width = image_bgr.shape[:2]
    current_long_side = max(height, width)
    if current_long_side <= max_side:
        return image_bgr
    scale = max_side / current_long_side
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)


def _remove_with_rembg(
    pil_input: Image.Image,
    alpha_matting: bool = True,
    model_name: str = "u2net",
) -> Image.Image:
    from rembg import remove

    kwargs = {"session": _rembg_session(model_name)}
    if alpha_matting and model_name != "isnet-general-use":
        kwargs.update(
            {
                "alpha_matting": True,
                "alpha_matting_foreground_threshold": 240,
                "alpha_matting_background_threshold": 10,
                "alpha_matting_erode_size": 7,
            }
        )
    return remove(pil_input, **kwargs)


@lru_cache(maxsize=4)
def _rembg_session(model_name: str = "u2net"):
    from rembg import new_session

    return new_session(model_name)


def _compose_on_white(image_rgba: np.ndarray) -> np.ndarray:
    rgb = image_rgba[:, :, :3].astype(np.float32)
    alpha = image_rgba[:, :, 3:4].astype(np.float32) / 255.0
    white = np.full_like(rgb, 255.0)
    composited = (rgb * alpha) + (white * (1.0 - alpha))
    return cv2.cvtColor(np.clip(composited, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _apply_preservation_mask(
    image_rgba: np.ndarray,
    source_bgr: np.ndarray,
    preserve_mask: np.ndarray | None,
) -> np.ndarray:
    if preserve_mask is None or preserve_mask.size == 0:
        return image_rgba

    height, width = image_rgba.shape[:2]
    mask = preserve_mask
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.GaussianBlur(mask.astype(np.uint8), (0, 0), 2.2)

    source = source_bgr
    if source.shape[:2] != (height, width):
        source = cv2.resize(source, (width, height), interpolation=cv2.INTER_LANCZOS4)
    source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    restored = image_rgba.copy()
    active = mask > 0
    restored[active, :3] = source_rgb[active]
    restored[:, :, 3] = np.maximum(restored[:, :, 3], mask)
    return restored


def _tag_preservation_refinement_reasons(
    source_bgr: np.ndarray,
    raw_rgba: np.ndarray,
    preserve_mask: np.ndarray | None,
) -> tuple[str, ...]:
    """Flag a tag mask that also force-preserves a substantial floor patch."""
    if preserve_mask is None or preserve_mask.size == 0:
        return ()

    height, width = raw_rgba.shape[:2]
    mask = preserve_mask
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    preserved = mask > 96
    preserved_area = int(np.count_nonzero(preserved))
    if preserved_area < 120:
        return ()

    source = _resize_source(source_bgr, (height, width))
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    gold = (hue >= 10) & (hue <= 36) & (saturation >= 58) & (value >= 105)
    green = (hue >= 38) & (hue <= 100) & (saturation >= 45) & (value >= 45)
    red = ((hue <= 9) | (hue >= 165)) & (saturation >= 48) & (value >= 30)
    strong_jewellery_colour = gold | green | red

    foreground = raw_rgba[:, :, 3] > 18
    likely_neutral_background = (
        preserved
        & foreground
        & (saturation <= 96)
        & (value >= 35)
        & (value <= 225)
        & ~strong_jewellery_colour
    )

    scale = max(3, int(round(np.sqrt(preserved_area) * 0.016)))
    if scale % 2 == 0:
        scale += 1
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale))
    close_size = max(5, scale + 4)
    if close_size % 2 == 0:
        close_size += 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    neutral_regions = cv2.morphologyEx(
        likely_neutral_background.astype(np.uint8),
        cv2.MORPH_OPEN,
        open_kernel,
    )
    neutral_regions = cv2.morphologyEx(neutral_regions, cv2.MORPH_CLOSE, close_kernel)
    neutral_regions = neutral_regions.astype(bool) & preserved & foreground

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        neutral_regions.astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        return ()

    largest_area = max(int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count))
    region_ratio = largest_area / max(1, preserved_area)
    image_ratio = largest_area / max(1, height * width)
    if (
        region_ratio < TAG_PRESERVE_RESIDUE_MIN_REGION_RATIO
        or image_ratio < TAG_PRESERVE_RESIDUE_MIN_IMAGE_RATIO
    ):
        return ()

    return (
        "tag preservation includes likely background "
        f"({region_ratio:.1%} of the preserved tag region)",
    )


def _clean_alpha_matte(image_rgba: np.ndarray, source_bgr: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    alpha = image_rgba[:, :, 3]
    if alpha is None or alpha.size == 0:
        return image_rgba, ""

    foreground = alpha > 18
    solid = alpha > 225
    if not np.any(foreground) or not np.any(solid):
        return image_rgba, ""

    cleaned_alpha = alpha.copy()
    object_mask = np.zeros_like(foreground, dtype=bool)
    object_near_mask = np.zeros_like(foreground, dtype=bool)
    resized_source = None
    if source_bgr is not None:
        object_mask, neutral_residue = _source_guided_masks(source_bgr, image_rgba.shape[:2], foreground)
        cleaned_alpha[object_mask] = np.maximum(cleaned_alpha[object_mask], 235)
        cleaned_alpha[neutral_residue & (cleaned_alpha < 252)] = 0
        object_near_mask = _dilate_mask(object_mask, radius=max(5, int(min(image_rgba.shape[:2]) * 0.014)))
        resized_source = _resize_source(source_bgr, image_rgba.shape[:2])

    keep_mask = _near_solid_mask(solid | object_mask, image_rgba.shape[:2]) | object_near_mask
    foreground = cleaned_alpha > 18
    weak_background = foreground & ~keep_mask & (cleaned_alpha < 245)
    cleaned_alpha[weak_background] = 0

    cleaned_alpha, removed_linear_tails = _remove_weak_linear_edge_tails(cleaned_alpha)
    cleaned_alpha = _remove_small_alpha_components(cleaned_alpha, image_rgba.shape[:2])
    cleaned_alpha, removed_linear_residue = _remove_isolated_linear_residue(
        cleaned_alpha,
        image_rgba.shape[:2],
    )
    cleaned_alpha = _remove_lonely_alpha_pixels(cleaned_alpha)

    changed_ratio = float(np.mean(cleaned_alpha != alpha))
    cleaned = image_rgba.copy()
    cleaned[:, :, 3] = cleaned_alpha
    if resized_source is not None and np.any(object_mask):
        source_rgb = cv2.cvtColor(resized_source, cv2.COLOR_BGR2RGB)
        cleaned[object_mask, :3] = source_rgb[object_mask]
    cleaned[cleaned_alpha == 0, :3] = 255

    notes: list[str] = []
    if changed_ratio > 0.004:
        notes.append(f"Removed background residue ({changed_ratio:.1%}).")
    if removed_linear_tails or removed_linear_residue:
        notes.append("Removed isolated background line.")
    return cleaned, " ".join(notes)


def _source_guided_masks(source_bgr: np.ndarray, shape: tuple[int, int], foreground: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = _resize_source(source_bgr, shape)
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    gold = (hue >= 10) & (hue <= 36) & (saturation >= 58) & (value >= 120)
    green = (hue >= 42) & (hue <= 98) & (saturation >= 48) & (value >= 52)
    red = ((hue <= 8) | (hue >= 165)) & (saturation >= 45) & (value >= 35)
    object_color = gold | green | red

    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    local_average = cv2.GaussianBlur(gray, (0, 0), 3.2)
    local_contrast = cv2.absdiff(gray, local_average)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = cv2.magnitude(gradient_x, gradient_y)
    near_colored_object = _dilate_mask(
        object_color & foreground,
        radius=max(5, int(min(shape) * 0.012)),
    )
    pearl_or_stone = (
        (saturation <= 78)
        & (value >= 138)
        & ((local_contrast >= 7) | (edge_strength >= 24))
        & (foreground | near_colored_object)
    )

    restore_area = _dilate_mask(foreground, radius=max(14, int(min(shape) * 0.026))) | _fill_mask_holes(foreground)
    object_mask = (object_color | pearl_or_stone) & restore_area
    object_near = _dilate_mask(object_mask, radius=max(3, int(min(shape) * 0.004)))
    neutral_floor = ((saturation <= 72) & (value < 230)) | (value < 88)
    neutral_residue = foreground & neutral_floor & ~object_near
    return object_mask, neutral_residue


def _hybrid_jewellery_support_mask(
    source_bgr: np.ndarray,
    u2net_alpha: np.ndarray,
    ai_alpha: np.ndarray,
) -> np.ndarray:
    """Find missing jewellery edge pixels without restoring U2Net floor residue.

    BiRefNet remains authoritative anywhere it already found foreground. U2Net may
    only add a narrow, high-confidence material edge that BiRefNet omitted.
    """
    hsv = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    u2net_foreground = u2net_alpha > 48
    ai_foreground = ai_alpha > 18
    missing_from_ai = ai_alpha <= 18
    edge_radius = max(2, int(min(source_bgr.shape[:2]) * 0.002))
    near_ai_object = _dilate_mask(ai_foreground, radius=edge_radius)

    saturated_gem = (
        (saturation >= 75)
        & (value >= 45)
        & ((hue <= 9) | (hue >= 38))
    )
    bright_textured_gold = (
        (hue >= 10)
        & (hue <= 34)
        & (saturation >= 100)
        & (value >= 135)
    )

    gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    local_average = cv2.GaussianBlur(gray, (0, 0), 2.0)
    local_contrast = cv2.absdiff(gray, local_average)
    bright_textured_gold &= local_contrast >= 7
    strong_material = saturated_gem | bright_textured_gold
    nearby_material = strong_material & (u2net_foreground | ai_foreground)
    near_strong_material = _dilate_mask(nearby_material, radius=max(3, edge_radius + 1))
    pearl_or_bright_stone = (
        (saturation <= 70)
        & (value >= 175)
        & (local_contrast >= 9)
        & near_strong_material
    )

    support = (
        (strong_material | pearl_or_bright_stone)
        & u2net_foreground
        & missing_from_ai
        & near_ai_object
    )
    return _remove_tiny_isolated_support(support, ai_foreground, anchor_radius=edge_radius)


def _remove_tiny_isolated_support(
    support: np.ndarray,
    anchor_foreground: np.ndarray,
    anchor_radius: int = 2,
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), connectivity=8)
    if count <= 1:
        return support
    image_area = support.shape[0] * support.shape[1]
    min_area = max(2, int(image_area * 0.000001))
    anchor_area = _dilate_mask(anchor_foreground, radius=max(1, anchor_radius))
    cleaned = np.zeros_like(support, dtype=bool)
    for index in range(1, count):
        component = labels == index
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= min_area and np.any(component & anchor_area):
            cleaned |= component
    return cleaned


def _remove_tiny_alpha_islands(rgba: np.ndarray) -> tuple[np.ndarray, int]:
    alpha = rgba[:, :, 3] > 18
    count, labels, stats, _ = cv2.connectedComponentsWithStats(alpha.astype(np.uint8), connectivity=8)
    if count <= 1:
        return rgba, 0
    image_area = alpha.shape[0] * alpha.shape[1]
    min_area = max(36, int(image_area * 0.00004))
    cleaned = rgba.copy()
    removed = 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= min_area:
            continue
        component = labels == index
        cleaned[component, 3] = 0
        cleaned[component, :3] = 255
        removed += 1
    return cleaned, removed


def _resize_source(source_bgr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if source_bgr.shape[:2] == shape:
        return source_bgr
    interpolation = cv2.INTER_AREA if source_bgr.shape[0] > shape[0] or source_bgr.shape[1] > shape[1] else cv2.INTER_LANCZOS4
    return cv2.resize(source_bgr, (shape[1], shape[0]), interpolation=interpolation)


def _evaluate_cleanup_safety(
    raw_rgba: np.ndarray,
    cleaned_rgba: np.ndarray,
    source_bgr: np.ndarray,
) -> MaskSafetyMetrics:
    raw_alpha = raw_rgba[:, :, 3]
    cleaned_alpha = cleaned_rgba[:, :, 3]
    raw_foreground = raw_alpha > 18
    cleaned_foreground = cleaned_alpha > 18
    removed = raw_foreground & ~cleaned_foreground
    raw_pixels = max(1, int(np.count_nonzero(raw_foreground)))

    source_support, _ = _source_guided_masks(source_bgr, raw_alpha.shape, raw_foreground)
    removed_source_support = removed & source_support
    raw_envelope = _fill_mask_holes(raw_foreground)
    interior_source_holes = removed_source_support & raw_envelope

    removed_foreground_ratio = float(np.count_nonzero(removed) / raw_pixels)
    removed_source_supported_ratio = float(np.mean(removed_source_support))
    interior_source_hole_ratio = float(np.mean(interior_source_holes))
    lost_supported_components = _lost_supported_component_count(
        source_support & raw_foreground,
        cleaned_foreground,
    )
    raw_component_count = _foreground_component_count(raw_alpha)
    cleaned_component_count = _foreground_component_count(cleaned_alpha)
    source_colored_support = _source_colored_jewellery_mask(source_bgr, raw_alpha.shape)
    missing_source_components = _missing_source_component_count(source_colored_support, raw_foreground)
    _, retained_residue = _source_guided_masks(source_bgr, raw_alpha.shape, cleaned_foreground)
    retained_background_residue_ratio = float(np.mean(retained_residue))
    largest_residue_component_ratio = _largest_component_ratio(retained_residue)

    reasons: list[str] = []
    has_supported_loss = (
        removed_source_supported_ratio > MAX_REMOVED_SOURCE_SUPPORTED_RATIO
        or lost_supported_components > 0
    )
    if removed_foreground_ratio > MAX_REMOVED_FOREGROUND_RATIO and has_supported_loss:
        reasons.append(f"cleanup removed {removed_foreground_ratio:.1%} of the raw foreground")
    if removed_source_supported_ratio > MAX_REMOVED_SOURCE_SUPPORTED_RATIO:
        reasons.append(f"source-supported jewellery loss is {removed_source_supported_ratio:.2%} of the image")
    if lost_supported_components:
        reasons.append(f"{lost_supported_components} supported jewellery component(s) disappeared")
    if interior_source_hole_ratio > MAX_INTERIOR_SOURCE_HOLE_RATIO:
        reasons.append(f"new source-supported interior holes cover {interior_source_hole_ratio:.2%} of the image")
    if missing_source_components:
        reasons.append(f"{missing_source_components} coloured jewellery component(s) are missing from the raw mask")
    if (
        retained_background_residue_ratio > MAX_RETAINED_RESIDUE_RATIO
        or largest_residue_component_ratio > MAX_RESIDUE_COMPONENT_RATIO
    ):
        reasons.append(
            "background residue remains "
            f"({retained_background_residue_ratio:.2%} total, {largest_residue_component_ratio:.2%} largest patch)"
        )

    return MaskSafetyMetrics(
        raw_foreground_ratio=float(np.mean(raw_foreground)),
        cleaned_foreground_ratio=float(np.mean(cleaned_foreground)),
        removed_foreground_ratio=removed_foreground_ratio,
        removed_source_supported_ratio=removed_source_supported_ratio,
        interior_source_hole_ratio=interior_source_hole_ratio,
        raw_component_count=raw_component_count,
        cleaned_component_count=cleaned_component_count,
        lost_supported_components=lost_supported_components,
        missing_source_components=missing_source_components,
        retained_background_residue_ratio=retained_background_residue_ratio,
        largest_residue_component_ratio=largest_residue_component_ratio,
        reasons=tuple(reasons),
    )


def _source_colored_jewellery_mask(source_bgr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    source = _resize_source(source_bgr, shape)
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    # Wood and beige floors occupy the yellow/orange range. Missing-component
    # detection intentionally targets saturated gems, beads, and thread only.
    return (saturation >= 90) & (value >= 35) & ((hue <= 9) | (hue >= 38))


def _missing_source_component_count(source_support: np.ndarray, raw_foreground: np.ndarray) -> int:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(source_support.astype(np.uint8), connectivity=8)
    image_area = source_support.shape[0] * source_support.shape[1]
    min_area = max(24, int(image_area * 0.00003))
    max_area = int(image_area * 0.03)
    missing = 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        retained_ratio = float(np.mean(raw_foreground[labels == index]))
        if retained_ratio < 0.35:
            missing += 1
    return missing


def _largest_component_ratio(mask: np.ndarray) -> float:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return 0.0
    largest = max(int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count))
    return float(largest / mask.size)


def _merge_safety_reasons(*reason_groups: tuple[str, ...]) -> tuple[str, ...]:
    categories = (
        "cleanup removed",
        "source-supported jewellery loss",
        "supported jewellery component",
        "new source-supported interior holes",
        "coloured jewellery component",
        "background residue remains",
    )
    merged: list[str] = []
    seen: set[str] = set()
    for reason in (item for group in reason_groups for item in group):
        key = next((category for category in categories if category in reason), reason)
        if key in seen:
            continue
        merged.append(reason)
        seen.add(key)
    return tuple(merged)


def _catastrophic_safety_reasons(metrics: MaskSafetyMetrics) -> tuple[str, ...]:
    reasons: list[str] = []
    if metrics.removed_source_supported_ratio > MAX_CATASTROPHIC_SOURCE_LOSS_RATIO:
        reasons.append(
            f"source-supported jewellery loss is {metrics.removed_source_supported_ratio:.2%} of the image"
        )
    if metrics.interior_source_hole_ratio > MAX_CATASTROPHIC_INTERIOR_HOLE_RATIO:
        reasons.append(
            f"new source-supported interior holes cover {metrics.interior_source_hole_ratio:.2%} of the image"
        )
    if metrics.lost_supported_components >= MAX_CATASTROPHIC_LOST_COMPONENTS:
        reasons.append(f"{metrics.lost_supported_components} supported jewellery components disappeared")
    if metrics.missing_source_components >= MAX_CATASTROPHIC_MISSING_COMPONENTS:
        reasons.append(f"{metrics.missing_source_components} coloured jewellery components are missing from the mask")
    return tuple(reasons)


def _lost_supported_component_count(source_support: np.ndarray, cleaned_foreground: np.ndarray) -> int:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(source_support.astype(np.uint8), connectivity=8)
    if count <= 1:
        return 0

    image_area = source_support.shape[0] * source_support.shape[1]
    min_area = max(18, int(image_area * 0.00002))
    lost = 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        retained_ratio = float(np.mean(cleaned_foreground[labels == index]))
        if retained_ratio < 0.6:
            lost += 1
    return lost


def _review_outputs(
    image_rgba: np.ndarray,
    catalogue_layout: bool,
    canvas_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if catalogue_layout:
        return _catalogue_portrait_outputs(image_rgba, canvas_size)
    return image_rgba, _compose_on_white(image_rgba)


def _expanded_bbox_mask(mask: np.ndarray, shape: tuple[int, int], margin_ratio: float) -> np.ndarray:
    coords = np.argwhere(mask)
    bbox = np.zeros(shape, dtype=bool)
    if coords.size == 0:
        return bbox
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    margin = max(4, int(max(shape) * margin_ratio))
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(shape[0], y_max + margin)
    x_max = min(shape[1], x_max + margin)
    bbox[y_min:y_max, x_min:x_max] = True
    return bbox


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    inverted = (~mask).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    if count <= 1:
        return mask.copy()

    border_labels = set(labels[0, :].tolist())
    border_labels.update(labels[-1, :].tolist())
    border_labels.update(labels[:, 0].tolist())
    border_labels.update(labels[:, -1].tolist())

    filled = mask.copy()
    for index in range(1, count):
        if index not in border_labels:
            filled[labels == index] = True
    return filled


def _near_solid_mask(solid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    radius = max(9, int(min(height, width) * 0.026))
    return _dilate_mask(solid, radius)


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _remove_small_alpha_components(alpha: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = (alpha > 18).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return alpha

    image_area = shape[0] * shape[1]
    min_area = max(18, int(image_area * 0.00002))
    cleaned = alpha.copy()
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        if area < min_area:
            cleaned[labels == index] = 0
    return cleaned


def _remove_weak_linear_edge_tails(alpha: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove straight low-confidence seams protruding from a dense product."""
    cleaned, vertical_count = _remove_vertical_weak_edge_tails(alpha)
    transposed, horizontal_count = _remove_vertical_weak_edge_tails(cleaned.T)
    return transposed.T.copy(), vertical_count + horizontal_count


def _remove_vertical_weak_edge_tails(alpha: np.ndarray) -> tuple[np.ndarray, int]:
    mask = alpha > 18
    height, width = alpha.shape
    row_counts = np.count_nonzero(mask, axis=1)
    dense_threshold = max(20, int(width * 0.03))
    dense_rows = np.flatnonzero(row_counts >= dense_threshold)
    if dense_rows.size == 0:
        return alpha, 0

    cleaned = alpha.copy()
    removed = 0
    dense_start = int(dense_rows[0])
    dense_end = int(dense_rows[-1])
    regions = ((0, dense_start, "top"), (dense_end + 1, height, "bottom"))
    minimum_length = max(72, int(min(height, width) * 0.08))
    maximum_thickness = max(12, int(width * 0.04))
    border_margin = max(8, int(min(height, width) * 0.015))

    for region_start, region_end, edge in regions:
        if region_end - region_start < minimum_length:
            continue
        ys, xs = np.where(mask[region_start:region_end])
        if ys.size == 0:
            continue
        ys = ys + region_start
        occupied_rows = np.unique(ys)
        ordered_rows = occupied_rows if edge == "top" else occupied_rows[::-1]
        edge_row = int(ordered_rows[0])
        touches_edge = edge_row <= border_margin if edge == "top" else edge_row >= height - border_margin - 1
        if not touches_edge:
            continue

        connected_rows = [edge_row]
        previous_row = edge_row
        for row_value in ordered_rows[1:]:
            row = int(row_value)
            if abs(row - previous_row) > 6:
                break
            connected_rows.append(row)
            previous_row = row
        occupied_rows = np.asarray(sorted(connected_rows), dtype=np.int32)
        selected = np.isin(ys, occupied_rows)
        ys = ys[selected]
        xs = xs[selected]
        span = int(ys.max() - ys.min() + 1)
        if span < minimum_length:
            continue
        coverage = occupied_rows.size / max(1, span)
        if coverage < 0.6:
            continue
        widths = []
        centers = []
        for row in occupied_rows:
            row_xs = xs[ys == row]
            widths.append(int(row_xs.max() - row_xs.min() + 1))
            centers.append(float(np.median(row_xs)))
        if float(np.quantile(widths, 0.9)) > maximum_thickness:
            continue

        slope, intercept = np.polyfit(occupied_rows.astype(np.float64), np.asarray(centers), 1)
        expected_centers = slope * occupied_rows + intercept
        residual = np.abs(np.asarray(centers) - expected_centers)
        maximum_residual = max(5.0, width * 0.008)
        if float(np.quantile(residual, 0.9)) > maximum_residual:
            continue

        candidate_alpha = alpha[ys, xs]
        if float(np.median(candidate_alpha)) >= 190 or float(np.mean(candidate_alpha > 225)) >= 0.25:
            continue

        corridor_half_width = min(12, max(5, int(np.quantile(widths, 0.9) // 2 + 3)))
        clear_start = region_start
        clear_end = region_end
        strong_streak = 0
        extension_rows = range(region_start - 1, -1, -1) if edge == "bottom" else range(region_end, height)
        for row in extension_rows:
            center = int(round(slope * row + intercept))
            x_start = max(0, center - corridor_half_width)
            x_end = min(width, center + corridor_half_width + 1)
            row_alpha = alpha[row, x_start:x_end]
            foreground_alpha = row_alpha[row_alpha > 18]
            row_is_strong = (
                foreground_alpha.size > maximum_thickness
                or (
                    foreground_alpha.size > 0
                    and float(np.median(foreground_alpha)) >= 230
                    and float(np.mean(foreground_alpha > 225)) >= 0.5
                )
            )
            if row_is_strong:
                strong_streak += 1
                if strong_streak >= 3:
                    break
                continue
            strong_streak = 0
            if edge == "bottom":
                clear_start = row
            else:
                clear_end = row + 1

        taper_length = max(18, int(min(height, width) * 0.035))
        for row in range(clear_start, clear_end):
            center = int(round(slope * row + intercept))
            distance_from_product = row - clear_start if edge == "bottom" else clear_end - row - 1
            taper = min(1.0, (distance_from_product + 1) / taper_length)
            effective_half_width = max(1, int(round(corridor_half_width * taper)))
            x_start = max(0, center - effective_half_width)
            x_end = min(width, center + effective_half_width + 1)
            cleaned[row, x_start:x_end] = 0
        removed += 1

    return cleaned, removed


def _remove_isolated_linear_residue(alpha: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, int]:
    """Remove weak floor seams while preserving solid jewellery chains."""
    mask = (alpha > 18).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 2:
        # One foreground component means there is no isolated line to remove.
        return alpha, 0

    foreground_indices = range(1, count)
    largest_index = max(foreground_indices, key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
    largest_area = max(1, int(stats[largest_index, cv2.CC_STAT_AREA]))
    height, width = shape
    image_area = height * width
    minimum_dimension = min(height, width)
    minimum_length = max(72, int(minimum_dimension * 0.08))
    maximum_thickness = max(12, int(minimum_dimension * 0.04))
    maximum_area = max(420, int(image_area * 0.004))
    border_margin = max(8, int(minimum_dimension * 0.015))

    cleaned = alpha.copy()
    removed = 0
    for index in foreground_indices:
        if index == largest_index:
            continue

        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        component_width = int(stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        short_side = max(1, min(component_width, component_height))
        long_side = max(component_width, component_height)
        aspect_ratio = long_side / short_side
        fill_ratio = area / max(1, component_width * component_height)
        touches_border = (
            x <= border_margin
            or y <= border_margin
            or x + component_width >= width - border_margin
            or y + component_height >= height - border_margin
        )
        component_alpha = alpha[labels == index]
        solid_ratio = float(np.mean(component_alpha > 225))
        median_alpha = float(np.median(component_alpha))

        is_weak_edge_line = (
            touches_border
            and long_side >= minimum_length
            and short_side <= maximum_thickness
            and aspect_ratio >= 6.0
            and fill_ratio <= 0.5
            and area <= maximum_area
            and area <= largest_area * 0.05
            and median_alpha < 190
            and solid_ratio < 0.25
        )
        if is_weak_edge_line:
            component_mask = labels == index
            halo_radius = max(3, min(8, short_side // 3))
            line_region = _dilate_mask(component_mask, radius=halo_radius)
            cleaned[line_region] = 0
            removed += 1

    return cleaned, removed


def _remove_lonely_alpha_pixels(alpha: np.ndarray) -> np.ndarray:
    mask = (alpha > 0).astype(np.uint8)
    neighbors = cv2.filter2D(mask, cv2.CV_16S, np.ones((3, 3), dtype=np.uint8), borderType=cv2.BORDER_CONSTANT)
    cleaned = alpha.copy()
    cleaned[(mask > 0) & (neighbors <= 2)] = 0
    return cleaned


def _catalogue_portrait_outputs(image_rgba: np.ndarray, canvas_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    crop = _foreground_crop(image_rgba)
    canvas_width, canvas_height = canvas_size
    resized = _resize_for_catalogue(crop, canvas_width, canvas_height)
    transparent_canvas = _place_on_portrait_canvas(resized, canvas_width, canvas_height)
    white_bgr = _compose_on_white_with_shadow(transparent_canvas)
    return transparent_canvas, white_bgr


def _foreground_crop(image_rgba: np.ndarray) -> np.ndarray:
    alpha = image_rgba[:, :, 3]
    # Very weak matte residue must not stretch the catalogue crop or shrink
    # the real product on the portrait canvas.
    coords = np.argwhere(alpha > 18)
    if coords.size == 0:
        return image_rgba

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    pad = max(12, int(max(image_rgba.shape[:2]) * 0.018))
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(image_rgba.shape[0], y_max + pad)
    x_max = min(image_rgba.shape[1], x_max + pad)
    return image_rgba[y_min:y_max, x_min:x_max].copy()


def _resize_for_catalogue(crop_rgba: np.ndarray, canvas_width: int, canvas_height: int) -> np.ndarray:
    crop_height, crop_width = crop_rgba.shape[:2]
    fit_width = int(canvas_width * 0.84)
    fit_height = int(canvas_height * 0.72)
    scale = min(fit_width / crop_width, fit_height / crop_height)
    scale = min(scale, 1.7)
    new_width = max(1, int(crop_width * scale))
    new_height = max(1, int(crop_height * scale))
    pil_crop = Image.fromarray(crop_rgba)
    resized = pil_crop.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return np.array(resized)


def _place_on_portrait_canvas(object_rgba: np.ndarray, canvas_width: int, canvas_height: int) -> np.ndarray:
    canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    canvas[:, :, :3] = 255

    object_height, object_width = object_rgba.shape[:2]
    x = (canvas_width - object_width) // 2
    target_center_y = int(canvas_height * 0.54)
    y = target_center_y - (object_height // 2)
    vertical_margin = int(canvas_height * 0.065)
    y = max(vertical_margin, min(y, canvas_height - vertical_margin - object_height))
    x = max(0, min(x, canvas_width - object_width))

    region = canvas[y : y + object_height, x : x + object_width]
    region[:, :, :3] = object_rgba[:, :, :3]
    region[:, :, 3] = np.maximum(region[:, :, 3], object_rgba[:, :, 3])
    canvas[y : y + object_height, x : x + object_width] = region
    return canvas


def _compose_on_white_with_shadow(image_rgba: np.ndarray) -> np.ndarray:
    height, width = image_rgba.shape[:2]
    alpha = image_rgba[:, :, 3].astype(np.float32) / 255.0
    blur_size = max(31, int(min(width, height) * 0.055))
    if blur_size % 2 == 0:
        blur_size += 1
    shadow = cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)
    shadow = _shift_mask(shadow, int(height * 0.012), int(width * 0.004))
    background = np.full((height, width, 3), 255.0, dtype=np.float32)
    background -= shadow[:, :, None] * 42.0

    rgb = image_rgba[:, :, :3].astype(np.float32)
    composited = rgb * alpha[:, :, None] + background * (1.0 - alpha[:, :, None])
    return cv2.cvtColor(np.clip(composited, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _shift_mask(mask: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    shifted = np.zeros_like(mask)
    source_y1 = max(0, -shift_y)
    source_y2 = mask.shape[0] - max(0, shift_y)
    source_x1 = max(0, -shift_x)
    source_x2 = mask.shape[1] - max(0, shift_x)
    dest_y1 = max(0, shift_y)
    dest_y2 = dest_y1 + (source_y2 - source_y1)
    dest_x1 = max(0, shift_x)
    dest_x2 = dest_x1 + (source_x2 - source_x1)
    if source_y2 > source_y1 and source_x2 > source_x1:
        shifted[dest_y1:dest_y2, dest_x1:dest_x2] = mask[source_y1:source_y2, source_x1:source_x2]
    return shifted


def _validate_alpha_mask(alpha: np.ndarray) -> tuple[bool, str]:
    if alpha is None or alpha.size == 0:
        return False, "Background mask was empty."

    foreground_ratio = float(np.mean(alpha > 12))
    solid_ratio = float(np.mean(alpha > 220))
    soft_ratio = float(np.mean((alpha > 12) & (alpha < 220)))

    if foreground_ratio < 0.015:
        return False, f"Foreground mask is too small ({foreground_ratio:.1%})."
    if foreground_ratio > 0.92:
        return False, f"Foreground mask covers almost the whole image ({foreground_ratio:.1%})."
    if solid_ratio < 0.006:
        return False, f"Foreground has too few solid pixels ({solid_ratio:.1%})."
    if soft_ratio > 0.82:
        return False, f"Mask is too uncertain around edges ({soft_ratio:.1%})."

    components = _foreground_component_count(alpha)
    if components > 160:
        return False, f"Foreground mask is too fragmented ({components} components)."

    return True, f"Mask OK. Foreground {foreground_ratio:.1%}."


def _foreground_component_count(alpha: np.ndarray) -> int:
    mask = (alpha > 24).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return 0

    image_area = alpha.shape[0] * alpha.shape[1]
    min_area = max(12, int(image_area * 0.000015))
    return sum(1 for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= min_area)
