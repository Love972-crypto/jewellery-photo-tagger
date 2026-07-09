from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import OCRTextBox, ProcessingSettings
from src.ocr_engine import OCREngine
from src.processor import BatchProcessor


class FilenameOCREngine:
    name = "Filename OCR smoke"

    def __init__(self) -> None:
        self.current_tag = "121134"

    def set_current_image(self, image_path: Path) -> None:
        match = re.search(r"(\d+)", image_path.stem)
        if match:
            self.current_tag = f"{121000 + int(match.group(1)):06d}"

    def read_text(self, image, source_rotation: str = "original", source_crop: str = "tag_crop") -> list[OCRTextBox]:
        return [
            OCRTextBox(
                text=self.current_tag,
                confidence=0.92,
                bbox=[[0, 0], [220, 0], [220, 70], [0, 70]],
                source_rotation=source_rotation,
                source_crop=source_crop,
            )
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        if "smoke" not in args.output.name.lower() and "synthetic" not in args.output.name.lower():
            raise SystemExit("Refusing to clear output folder unless its name includes smoke or synthetic.")
        shutil.rmtree(args.output)
    images = sorted([path for path in args.input.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    engine: OCREngine = FilenameOCREngine()
    processor = BatchProcessor(args.output / "Jewellery_Output", ProcessingSettings(save_debug_crops=True), engine)
    summary = processor.process_images(images)
    print(f"Processed={summary.processed} OK={summary.ok} Duplicates={summary.duplicate_tags} Review={summary.review_required} Errors={summary.errors}")


if __name__ == "__main__":
    main()
