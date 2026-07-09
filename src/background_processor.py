from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import cv2
import numpy as np
from PIL import Image


BACKGROUND_OK = "OK"
BACKGROUND_DISABLED = "DISABLED"
BACKGROUND_REVIEW_REQUIRED = "REVIEW_REQUIRED"
BACKGROUND_FAILED = "FAILED"

RemoveCallable = Callable[[Image.Image], Image.Image]


@dataclass
class BackgroundResult:
    status: str
    transparent_rgba: np.ndarray | None = None
    white_bgr: np.ndarray | None = None
    notes: str = ""


def remove_background(
    image_bgr: np.ndarray,
    remover: RemoveCallable | None = None,
    alpha_matting: bool = True,
    catalogue_layout: bool = True,
    canvas_size: tuple[int, int] = (1200, 1500),
    max_side: int = 2200,
    model_name: str = "u2net",
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

    source_for_cleanup = image_bgr if remover is not None or model_name != "u2net" else None
    transparent_rgba, cleanup_notes = _clean_alpha_matte(transparent_rgba, source_bgr=source_for_cleanup)
    check_ok, check_notes = _validate_alpha_mask(transparent_rgba[:, :, 3])
    if not check_ok:
        return BackgroundResult(
            status=BACKGROUND_REVIEW_REQUIRED,
            transparent_rgba=transparent_rgba,
            white_bgr=_compose_on_white(transparent_rgba),
            notes=check_notes,
        )

    if catalogue_layout:
        transparent_rgba, white_bgr = _catalogue_portrait_outputs(transparent_rgba, canvas_size)
        notes = f"Background removed safely with {model_name}. Catalogue portrait aligned."
    else:
        white_bgr = _compose_on_white(transparent_rgba)
        notes = f"Background removed safely with {model_name}."
    if cleanup_notes:
        notes = f"{notes} {cleanup_notes}"

    return BackgroundResult(
        status=BACKGROUND_OK,
        transparent_rgba=transparent_rgba,
        white_bgr=white_bgr,
        notes=notes,
    )


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


def _remove_with_rembg(pil_input: Image.Image, alpha_matting: bool = True, model_name: str = "u2net") -> Image.Image:
    from rembg import remove

    kwargs = {"session": _rembg_session(model_name)}
    if alpha_matting:
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
    if source_bgr is not None:
        object_mask, neutral_residue = _source_guided_masks(source_bgr, image_rgba.shape[:2], foreground)
        cleaned_alpha[object_mask] = np.maximum(cleaned_alpha[object_mask], 235)
        cleaned_alpha[neutral_residue & (cleaned_alpha < 252)] = 0
        object_near_mask = _dilate_mask(object_mask, radius=max(5, int(min(image_rgba.shape[:2]) * 0.014)))

    keep_mask = _near_solid_mask(solid | object_mask, image_rgba.shape[:2]) | object_near_mask
    foreground = cleaned_alpha > 18
    weak_background = foreground & ~keep_mask & (cleaned_alpha < 245)
    cleaned_alpha[weak_background] = 0

    cleaned_alpha = _remove_small_alpha_components(cleaned_alpha, image_rgba.shape[:2])
    cleaned_alpha = _remove_lonely_alpha_pixels(cleaned_alpha)

    changed_ratio = float(np.mean(cleaned_alpha != alpha))
    cleaned = image_rgba.copy()
    cleaned[:, :, 3] = cleaned_alpha
    cleaned[cleaned_alpha == 0, :3] = 255

    if changed_ratio > 0.004:
        return cleaned, f"Removed background residue ({changed_ratio:.1%})."
    return cleaned, ""


def _source_guided_masks(source_bgr: np.ndarray, shape: tuple[int, int], foreground: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = source_bgr
    if source.shape[:2] != shape:
        source = cv2.resize(source, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    gold = (hue >= 10) & (hue <= 36) & (saturation >= 58) & (value >= 120)
    green = (hue >= 42) & (hue <= 98) & (saturation >= 48) & (value >= 52)
    red = ((hue <= 8) | (hue >= 165)) & (saturation >= 45) & (value >= 35)
    object_color = gold | green | red

    restore_area = _dilate_mask(foreground, radius=max(14, int(min(shape) * 0.026))) | _fill_mask_holes(foreground)
    object_mask = object_color & restore_area
    object_near = _dilate_mask(object_mask, radius=max(7, int(min(shape) * 0.018)))
    neutral_floor = ((saturation <= 72) & (value < 230)) | (value < 88)
    neutral_residue = foreground & neutral_floor & ~object_near
    return object_mask, neutral_residue


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
    coords = np.argwhere(alpha > 8)
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
