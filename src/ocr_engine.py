from __future__ import annotations

import os
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
                import easyocr
                import torch
            except Exception as exc:
                raise RuntimeError("EasyOCR is not installed. Install requirements or use the fallback OCR adapter.") from exc
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
        return self._reader

    def warm_up(self) -> None:
        """Initialize the EasyOCR reader before batch processing starts."""
        _ = self.reader

    def read_text(self, image: np.ndarray, source_rotation: str = "original", source_crop: str = "tag_crop") -> list[OCRTextBox]:
        if image is None or image.size == 0:
            return []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        results = self.reader.readtext(
            rgb,
            detail=1,
            paragraph=False,
            decoder="greedy",
            batch_size=1,
            workers=0,
            canvas_size=900,
            mag_ratio=1.0,
            allowlist="0123456789",
        )
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
