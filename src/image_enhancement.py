from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def _register_heif() -> None:
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except Exception:
        return


def load_image(path: Path) -> np.ndarray:
    """Load common raster formats, including HEIC when pillow-heif is installed."""
    _register_heif()
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise ValueError(f"Could not read image: {path.name}") from exc


def enhance_image(image: np.ndarray, mode: str = "fast") -> np.ndarray:
    """Apply gentle product-safe enhancement without changing the background."""
    if image is None or image.size == 0:
        raise ValueError("Empty image")

    if mode == "quality":
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 4, 4, 7, 21)
    else:
        denoised = image

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.55 if mode == "fast" else 1.7, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_channel)
    l_blend = cv2.addWeighted(l_channel, 0.62 if mode == "fast" else 0.55, l_clahe, 0.38 if mode == "fast" else 0.45, 0)
    contrast = cv2.merge((l_blend, a_channel, b_channel))
    contrast = cv2.cvtColor(contrast, cv2.COLOR_LAB2BGR)

    adjusted = cv2.convertScaleAbs(contrast, alpha=1.03 if mode == "fast" else 1.04, beta=3 if mode == "fast" else 4)
    blur = cv2.GaussianBlur(adjusted, (0, 0), 0.9 if mode == "fast" else 1.15)
    sharpened = cv2.addWeighted(adjusted, 1.26 if mode == "fast" else 1.35, blur, -0.26 if mode == "fast" else -0.35, 0)
    return sharpened


def _resize_for_ocr(crop: np.ndarray, upscale: int = 2, max_side: int = 1280) -> np.ndarray:
    """Keep OCR inputs readable without letting phone-size photos become huge."""
    height, width = crop.shape[:2]
    long_side = max(height, width)
    if long_side <= 0:
        raise ValueError("Empty OCR crop")

    scale = 1.0
    if long_side > max_side:
        scale = max_side / float(long_side)
    elif upscale > 1:
        scale = min(float(upscale), max_side / float(long_side))

    if abs(scale - 1.0) < 0.02:
        return crop

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=interpolation)
    return resized


def preprocess_for_ocr(crop: np.ndarray, upscale: int = 2, max_side: int = 1280) -> np.ndarray:
    """Improve crop readability for OCR while preserving numeric shapes."""
    if crop is None or crop.size == 0:
        raise ValueError("Empty OCR crop")

    crop = _resize_for_ocr(crop, upscale=upscale, max_side=max_side)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.bilateralFilter(gray, 5, 35, 35)
    clahe = cv2.createCLAHE(clipLimit=2.3, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 0.85)
    sharp = cv2.addWeighted(gray, 1.55, blur, -0.55, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def save_png(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise ValueError(f"Could not encode PNG: {output_path.name}")
    encoded.tofile(str(output_path))
