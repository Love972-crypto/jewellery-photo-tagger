from __future__ import annotations

import os
import sys
import types
from typing import Protocol

import cv2
import numpy as np

from .models import OCRTextBox


class OCREngine(Protocol):
    name: str

    def read_text(self, image: np.ndarray, source_rotation: str = "original", source_crop: str = "tag_crop") -> list[OCRTextBox]:
        ...


class EasyOCREngine:
    name = "EasyOCR"

    def __init__(self, languages: list[str] | None = None, gpu: bool = False) -> None:
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._reader = None

    @property
    def reader(self):
        if self._reader is None:
            try:
                import torch
                restore_torchvision = _ensure_torchvision_compatibility(torch)
                try:
                    _ensure_bidi_compatibility()
                    import easyocr
                finally:
                    restore_torchvision()
            except Exception as exc:
                raise RuntimeError("EasyOCR is not installed. Install requirements or use the fallback OCR adapter.") from exc
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
        return self._reader

    def warm_up(self) -> None:
        """Initialize the EasyOCR reader before batch processing starts."""
        _ = self.reader

    def read_text(self, image: np.ndarray, source_rotation: str = "original", source_crop: str = "tag_crop") -> list[OCRTextBox]:
        return self._read_text(
            image,
            source_rotation=source_rotation,
            source_crop=source_crop,
            rotation_info=None,
            decoder="greedy",
        )

    def read_text_with_rotations(
        self,
        image: np.ndarray,
        source_crop: str = "tag_crop",
        rotations: tuple[int, ...] = (90, 180, 270),
    ) -> list[OCRTextBox]:
        """Run EasyOCR detection once and choose the strongest recognition rotation."""
        return self._read_text(
            image,
            source_rotation="auto_best",
            source_crop=source_crop,
            rotation_info=list(rotations),
            decoder="greedy",
        )

    def read_text_verification(
        self,
        image: np.ndarray,
        source_rotation: str = "original",
        source_crop: str = "tag_crop_verification",
    ) -> list[OCRTextBox]:
        """Cross-check an uncertain numeric crop with EasyOCR's beam-search decoder."""
        return self._read_text(
            image,
            source_rotation=source_rotation,
            source_crop=source_crop,
            rotation_info=None,
            decoder="beamsearch",
        )

    def _read_text(
        self,
        image: np.ndarray,
        source_rotation: str,
        source_crop: str,
        rotation_info: list[int] | None,
        decoder: str,
    ) -> list[OCRTextBox]:
        if image is None or image.size == 0:
            return []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        read_options = {
            "detail": 1,
            "paragraph": False,
            "decoder": decoder,
            "batch_size": 1,
            "workers": 0,
            "canvas_size": 1280,
            "mag_ratio": 1.35,
            "min_size": 5,
            "text_threshold": 0.45,
            "low_text": 0.25,
            "link_threshold": 0.3,
            "contrast_ths": 0.05,
            "adjust_contrast": 0.7,
            "allowlist": "0123456789",
        }
        if rotation_info:
            read_options["rotation_info"] = rotation_info
        if decoder == "beamsearch":
            read_options["beamWidth"] = 5
        if source_crop == "fallback_full":
            # Full-frame passes recover small tags when rectangular crop detection
            # is confused by white stones, pearls, or a light background.
            read_options.update(
                canvas_size=1800,
                mag_ratio=1.2,
                min_size=8,
                text_threshold=0.45,
                low_text=0.25,
                link_threshold=0.25,
            )
        results = self.reader.readtext(rgb, **read_options)
        boxes: list[OCRTextBox] = []
        for item in results:
            if len(item) < 3:
                continue
            bbox, text, confidence = item[0], str(item[1]), float(item[2])
            if text.strip():
                boxes.append(
                    OCRTextBox(
                        text=text.strip(),
                        confidence=max(0.0, min(confidence, 1.0)),
                        bbox=bbox,
                        source_rotation=source_rotation,
                        source_crop=source_crop,
                    )
                )
        return boxes


class NullOCREngine:
    name = "Fallback OCR"

    def read_text(self, image: np.ndarray, source_rotation: str = "original", source_crop: str = "tag_crop") -> list[OCRTextBox]:
        return []

    def warm_up(self) -> None:
        return None


class StaticOCREngine:
    """Test helper that returns the same OCR boxes for every image."""

    name = "Static OCR"

    def __init__(self, boxes: list[OCRTextBox]) -> None:
        self.boxes = boxes

    def read_text(self, image: np.ndarray, source_rotation: str = "original", source_crop: str = "tag_crop") -> list[OCRTextBox]:
        return [
            OCRTextBox(
                text=box.text,
                confidence=box.confidence,
                bbox=box.bbox,
                source_rotation=source_rotation,
                source_crop=source_crop,
            )
            for box in self.boxes
        ]


def build_ocr_engine(use_easyocr: bool = True) -> OCREngine:
    return EasyOCREngine() if use_easyocr else NullOCREngine()


def _ensure_torchvision_compatibility(torch_module):
    """Skip fake registration for optional ops absent from mismatched CPU wheels."""
    original_register_fake = getattr(torch_module.library, "register_fake", None)
    if not callable(original_register_fake):
        return lambda: None

    def safe_register_fake(operator_name, *args, **kwargs):
        if isinstance(operator_name, str) and operator_name.startswith("torchvision::"):
            try:
                torch_module._C._dispatch_has_kernel_for_dispatch_key(operator_name, "Meta")
            except RuntimeError:
                if args and callable(args[0]):
                    return args[0]
                return lambda function: function
            except Exception:
                pass
        return original_register_fake(operator_name, *args, **kwargs)

    torch_module.library.register_fake = safe_register_fake

    def restore() -> None:
        torch_module.library.register_fake = original_register_fake

    return restore


def _ensure_bidi_compatibility() -> None:
    """Use a tiny left-to-right bidi fallback if the native wheel is blocked."""
    try:
        import bidi  # noqa: F401
        return
    except Exception:
        pass

    fallback = types.ModuleType("bidi")
    fallback.get_display = lambda text, *args, **kwargs: text
    fallback.get_base_level = lambda text, *args, **kwargs: 0
    fallback.VERSION = "fallback-ltr"
    sys.modules["bidi"] = fallback
