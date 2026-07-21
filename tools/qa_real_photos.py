from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_case(value: str) -> tuple[Path, str | None]:
    path_text, separator, expected = value.rpartition("=")
    if not separator or not path_text:
        raise argparse.ArgumentTypeError("Each case must use IMAGE_PATH=EXPECTED_TAG or IMAGE_PATH=-")
    path = Path(path_text)
    return path, None if expected == "-" else expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    args = parser.parse_args()

    os.environ["EASYOCR_MODULE_PATH"] = str(args.model_cache / "EasyOCR")
    os.environ["U2NET_HOME"] = str(args.model_cache / "u2net")

    from src.models import ProcessingSettings, STATUS_DUPLICATE_TAG, STATUS_OK
    from src.ocr_engine import EasyOCREngine
    from src.processor import BatchProcessor
    from src.report_generator import read_report

    cases = list(args.case)
    missing = [str(path) for path, _ in cases if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing QA images: {missing}")
    if args.output.exists():
        raise SystemExit(f"QA output must be a fresh path: {args.output}")

    settings = ProcessingSettings(
        remove_background=True,
        background_output_mode="white_and_transparent",
        catalogue_layout_enabled=True,
        ocr_attempt_mode="fast",
    )
    engine = EasyOCREngine()
    engine.warm_up()
    processor = BatchProcessor(args.output / "Jewellery_Output", settings, engine, project_root=PROJECT_ROOT)
    summary = processor.process_images([path for path, _ in cases])
    report = read_report(processor.output_paths.report_csv)
    by_name = {str(row["original_filename"]): row for _, row in report.iterrows()}

    failures: list[str] = []
    results: list[dict[str, object]] = []
    for path, expected in cases:
        row = by_name.get(path.name)
        if row is None:
            failures.append(f"Missing report row: {path.name}")
            continue
        detected = str(row.get("detected_tag_number", "")).strip()
        status = str(row.get("status", ""))
        if detected.lower() == "nan":
            detected = ""
        if expected is None:
            if detected or status in {STATUS_OK, STATUS_DUPLICATE_TAG}:
                failures.append(f"No-tag image incorrectly accepted: {path.name} -> {detected} ({status})")
        elif detected != expected:
            failures.append(f"Tag mismatch: {path.name} expected {expected}, got {detected or '<empty>'} ({status})")

        final_name = str(row.get("final_filename", "")).strip()
        output_folder = str(row.get("output_folder", "")).strip()
        if final_name and final_name.lower() != "nan" and output_folder and output_folder.lower() != "nan":
            output_path = processor.output_paths.root / output_folder / final_name
            if not output_path.is_file() or output_path.stat().st_size == 0:
                failures.append(f"Missing output file: {output_path}")
            elif output_path.suffix.lower() == ".png":
                with Image.open(output_path) as image:
                    if image.width < 100 or image.height < 100:
                        failures.append(f"Invalid output dimensions: {output_path} -> {image.size}")

        results.append(
            {
                "file": path.name,
                "expected": expected,
                "detected": detected or None,
                "status": status,
                "background_status": str(row.get("background_status", "")),
            }
        )

    required_artifacts = [
        processor.output_paths.report_csv,
        processor.output_paths.full_zip,
        processor.output_paths.debug_zip,
    ]
    for artifact in required_artifacts:
        if not artifact.is_file() or artifact.stat().st_size == 0:
            failures.append(f"Missing batch artifact: {artifact}")

    payload = {
        "summary": {
            "total": summary.total,
            "processed": summary.processed,
            "ok": summary.ok,
            "review_required": summary.review_required,
            "errors": summary.errors,
            "elapsed_seconds": round(summary.elapsed_seconds, 2),
        },
        "results": results,
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
