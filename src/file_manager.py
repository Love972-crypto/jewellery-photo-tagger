from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .models import OutputPaths

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return stem or "image"


def create_run_workspace(project_root: Path) -> tuple[Path, Path, OutputPaths]:
    runtime_root = project_root / ".runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    run_dir = runtime_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    upload_dir = run_dir / "uploads"
    output_paths = setup_output_paths(run_dir / "Jewellery_Output")
    upload_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, upload_dir, output_paths


def setup_output_paths(output_root: Path) -> OutputPaths:
    paths = OutputPaths(
        root=output_root,
        processed_images=output_root / "processed_images",
        transparent_images=output_root / "transparent_images",
        review_required=output_root / "review_required",
        background_review=output_root / "background_review",
        debug_crops=output_root / "debug_crops",
        report_csv=output_root / "report.csv",
        full_zip=output_root / "Jewellery_Output.zip",
        processed_zip=output_root / "processed_images.zip",
        transparent_zip=output_root / "transparent_images.zip",
        debug_zip=output_root / "debug_crops.zip",
    )
    for folder in (paths.processed_images, paths.transparent_images, paths.review_required, paths.background_review, paths.debug_crops):
        folder.mkdir(parents=True, exist_ok=True)
    return paths


def collect_image_files(input_dir: Path) -> tuple[list[Path], list[str]]:
    image_paths: list[Path] = []
    validation_errors: list[str] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            image_paths.append(path)
        elif suffix != ".zip":
            validation_errors.append(f"{path.name} is not a supported image format.")
    return image_paths, validation_errors


def unique_png_path(folder: Path, desired_stem: str) -> tuple[Path, bool]:
    folder.mkdir(parents=True, exist_ok=True)
    clean = safe_stem(desired_stem)
    candidate = folder / f"{clean}.png"
    counter = 2
    duplicate = candidate.exists()
    while candidate.exists():
        candidate = folder / f"{clean}_{counter}.png"
        counter += 1
    return candidate, duplicate


def extract_zip_safely(zip_path: Path, destination: Path) -> list[str]:
    errors: list[str] = []
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                member_name = member.filename.replace("\\", "/")
                target = (destination / member_name).resolve()
                if not str(target).startswith(str(destination.resolve())):
                    errors.append(f"Skipped unsafe ZIP member: {member.filename}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
    except zipfile.BadZipFile:
        errors.append(f"{zip_path.name} is not a valid ZIP file.")
    except Exception as exc:
        errors.append(f"Could not extract {zip_path.name}: {exc}")
    return errors


def materialize_uploaded_files(uploaded_files: Iterable, upload_dir: Path) -> tuple[list[Path], list[str]]:
    saved_paths: list[Path] = []
    errors: list[str] = []
    upload_dir.mkdir(parents=True, exist_ok=True)
    for uploaded in uploaded_files:
        name = Path(uploaded.name).name
        suffix = Path(name).suffix.lower()
        target = upload_dir / name
        try:
            with target.open("wb") as handle:
                handle.write(uploaded.getbuffer())
            if suffix == ".zip":
                errors.extend(extract_zip_safely(target, upload_dir / safe_stem(name)))
            elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
                saved_paths.append(target)
            else:
                errors.append(f"{name} is not supported. Please upload JPG, JPEG, PNG, WEBP, HEIC, or ZIP.")
        except Exception:
            errors.append(f"{name} could not be uploaded. Please try again.")
    discovered, discovery_errors = collect_image_files(upload_dir)
    saved_paths = sorted(set(saved_paths + discovered))
    return saved_paths, errors + discovery_errors


def create_zip_from_paths(paths: Iterable[Path], zip_path: Path, base_dir: Path | None = None) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    base = base_dir or zip_path.parent
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in paths:
            if path.is_file():
                archive.write(path, arcname=path.relative_to(base))
    return zip_path


def create_zip_from_folder(folder: Path, zip_path: Path, exclude_zip_files: bool = False) -> Path:
    files = _zip_source_files(folder, zip_path, exclude_zip_files=exclude_zip_files)
    return create_zip_from_paths(files, zip_path, folder)


def _zip_source_files(folder: Path, zip_path: Path, exclude_zip_files: bool = False) -> list[Path]:
    return [
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.resolve() != zip_path.resolve()
        and not (exclude_zip_files and path.suffix.lower() == ".zip")
    ]


def _zip_needs_rebuild(folder: Path, zip_path: Path, exclude_zip_files: bool = False) -> bool:
    if not zip_path.exists():
        return True
    files = _zip_source_files(folder, zip_path, exclude_zip_files=exclude_zip_files)
    if not files:
        return zip_path.stat().st_size == 0
    newest_source = max(path.stat().st_mtime for path in files)
    return zip_path.stat().st_mtime < newest_source


def find_latest_output_root(project_root: Path) -> Path | None:
    runtime_root = project_root / ".runtime"
    if not runtime_root.exists():
        return None

    reports = sorted(runtime_root.rglob("Jewellery_Output/report.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not reports:
        return None
    return reports[0].parent


def rebuild_output_archives(output_root: Path) -> OutputPaths:
    paths = setup_output_paths(output_root)
    if _zip_needs_rebuild(paths.processed_images, paths.processed_zip):
        create_zip_from_folder(paths.processed_images, paths.processed_zip)
    if _zip_needs_rebuild(paths.transparent_images, paths.transparent_zip):
        create_zip_from_folder(paths.transparent_images, paths.transparent_zip)
    if _zip_needs_rebuild(paths.debug_crops, paths.debug_zip):
        create_zip_from_folder(paths.debug_crops, paths.debug_zip)
    if _zip_needs_rebuild(paths.root, paths.full_zip, exclude_zip_files=True):
        create_zip_from_folder(paths.root, paths.full_zip, exclude_zip_files=True)
    return paths


def file_size_label(path: Path) -> str:
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024
    return f"{path.stat().st_size} B"


def copy_to_temp_path(source: Path) -> Path:
    tmp = Path(tempfile.mkdtemp()) / source.name
    shutil.copy2(source, tmp)
    return tmp
