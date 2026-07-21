from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .models import OutputPaths

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
MAX_ZIP_MEMBERS = 5_000
MAX_ZIP_MEMBER_BYTES = 150 * 1024 * 1024
MAX_ZIP_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
MAX_ZIP_EXPANSION_RATIO = 200.0
MIN_RATIO_CHECK_BYTES = 1024 * 1024
ARCHIVE_STALE_MARKER = ".archives_stale"
CORRECTION_CACHE_DIRNAME = ".correction_cache"


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
        compressed_images_20kb=output_root / "compressed_images_20kb",
        review_required=output_root / "review_required",
        ai_review=output_root / "ai_review",
        background_review=output_root / "background_review",
        correction_cache=output_root / CORRECTION_CACHE_DIRNAME,
        debug_crops=output_root / "debug_crops",
        report_csv=output_root / "report.csv",
        full_zip=output_root / "Jewellery_Output.zip",
        processed_zip=output_root / "processed_images.zip",
        transparent_zip=output_root / "transparent_images.zip",
        compressed_images_20kb_zip=output_root / "compressed_images_20kb.zip",
        debug_zip=output_root / "debug_crops.zip",
    )
    for folder in (
        paths.processed_images,
        paths.transparent_images,
        paths.compressed_images_20kb,
        paths.review_required,
        paths.ai_review,
        paths.background_review,
        paths.correction_cache,
        paths.debug_crops,
    ):
        folder.mkdir(parents=True, exist_ok=True)
    return paths


def collect_image_files(input_dir: Path) -> tuple[list[Path], list[str]]:
    image_paths: list[Path] = []
    validation_errors: list[str] = []
    seen_content: dict[str, Path] = {}
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            try:
                digest = _sha256_file(path)
            except OSError as exc:
                validation_errors.append(f"{path.name} could not be read: {exc}")
                continue
            original = seen_content.get(digest)
            if original is not None:
                duplicate_name = path.relative_to(input_dir).as_posix()
                original_name = original.relative_to(input_dir).as_posix()
                validation_errors.append(
                    f"Skipped exact duplicate photo {duplicate_name}; same image as {original_name}."
                )
                continue
            seen_content[digest] = path
            image_paths.append(path)
        elif suffix != ".zip":
            validation_errors.append(f"{path.name} is not a supported image format.")
    return image_paths, validation_errors


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def unique_file_path(folder: Path, filename: str) -> tuple[Path, bool]:
    folder.mkdir(parents=True, exist_ok=True)
    clean_name = Path(filename).name
    candidate = folder / clean_name
    duplicate = candidate.exists()
    counter = 2
    while candidate.exists():
        source = Path(clean_name)
        candidate = folder / f"{source.stem}_{counter}{source.suffix}"
        counter += 1
    return candidate, duplicate


def extract_zip_safely(zip_path: Path, destination: Path) -> list[str]:
    errors: list[str] = []
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if len(members) > MAX_ZIP_MEMBERS:
                return [f"{zip_path.name} contains too many files ({len(members):,})."]

            total_uncompressed = sum(max(0, int(member.file_size)) for member in members)
            if total_uncompressed > MAX_ZIP_TOTAL_BYTES:
                return [f"{zip_path.name} expands beyond the {MAX_ZIP_TOTAL_BYTES // (1024 ** 3)} GB safety limit."]

            destination_root = destination.resolve()
            used_targets: set[Path] = set()
            for member in members:
                if member.is_dir():
                    continue
                member_name = member.filename.replace("\\", "/")
                if not member_name or member_name.endswith("/"):
                    continue
                target = (destination_root / member_name).resolve()
                try:
                    target.relative_to(destination_root)
                except ValueError:
                    errors.append(f"Skipped unsafe ZIP member: {member.filename}")
                    continue

                if member.flag_bits & 0x1:
                    errors.append(f"Skipped encrypted ZIP member: {member.filename}")
                    continue
                if member.file_size > MAX_ZIP_MEMBER_BYTES:
                    errors.append(f"Skipped oversized ZIP member: {member.filename}")
                    continue
                compressed_size = max(1, int(member.compress_size))
                expansion_ratio = float(member.file_size) / compressed_size
                if member.file_size >= MIN_RATIO_CHECK_BYTES and expansion_ratio > MAX_ZIP_EXPANSION_RATIO:
                    errors.append(f"Skipped suspiciously compressed ZIP member: {member.filename}")
                    continue

                if target in used_targets or target.exists():
                    target, _ = unique_file_path(target.parent, target.name)
                    errors.append(f"Renamed duplicate ZIP member safely: {member.filename} -> {target.name}")
                used_targets.add(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
    except zipfile.BadZipFile:
        errors.append(f"{zip_path.name} is not a valid ZIP file.")
    except Exception as exc:
        errors.append(f"Could not extract {zip_path.name}: {exc}")
    return errors


def materialize_uploaded_files(uploaded_files: Iterable, upload_dir: Path) -> tuple[list[Path], list[str]]:
    errors: list[str] = []
    upload_dir.mkdir(parents=True, exist_ok=True)
    for uploaded in uploaded_files:
        name = Path(uploaded.name).name
        suffix = Path(name).suffix.lower()
        target, duplicate = unique_file_path(upload_dir, name)
        try:
            with target.open("wb") as handle:
                handle.write(uploaded.getbuffer())
            if duplicate:
                errors.append(f"{name} was uploaded more than once and saved safely as {target.name}.")
            if suffix == ".zip":
                errors.extend(extract_zip_safely(target, upload_dir / safe_stem(target.name)))
            elif suffix not in SUPPORTED_IMAGE_EXTENSIONS:
                errors.append(f"{name} is not supported. Please upload JPG, JPEG, PNG, WEBP, HEIC, or ZIP.")
        except Exception:
            errors.append(f"{name} could not be uploaded. Please try again.")
    discovered, discovery_errors = collect_image_files(upload_dir)
    return discovered, errors + discovery_errors


def create_zip_from_paths(paths: Iterable[Path], zip_path: Path, base_dir: Path | None = None) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    base = base_dir or zip_path.parent
    temporary = zip_path.with_name(f".{zip_path.name}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in paths:
                if path.is_file():
                    archive.write(path, arcname=path.relative_to(base))
        temporary.replace(zip_path)
    finally:
        temporary.unlink(missing_ok=True)
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
        and CORRECTION_CACHE_DIRNAME not in path.relative_to(folder).parts
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

    reports = sorted(
        (
            path
            for path in runtime_root.rglob("Jewellery_Output/report.csv")
            if not (path.parent / ".processing").exists()
            and not (path.parent / ".failed").exists()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not reports:
        return None
    return reports[0].parent


def rebuild_output_archives(output_root: Path) -> OutputPaths:
    paths = setup_output_paths(output_root)
    force_refresh = output_archives_are_stale(output_root)
    if force_refresh or _zip_needs_rebuild(paths.processed_images, paths.processed_zip):
        create_zip_from_folder(paths.processed_images, paths.processed_zip)
    if force_refresh or _zip_needs_rebuild(paths.transparent_images, paths.transparent_zip):
        create_zip_from_folder(paths.transparent_images, paths.transparent_zip)
    if _zip_needs_rebuild(paths.debug_crops, paths.debug_zip):
        create_zip_from_folder(paths.debug_crops, paths.debug_zip)
    if force_refresh or _zip_needs_rebuild(paths.root, paths.full_zip, exclude_zip_files=True):
        create_zip_from_folder(paths.root, paths.full_zip, exclude_zip_files=True)
    (paths.root / ARCHIVE_STALE_MARKER).unlink(missing_ok=True)
    return paths


def mark_output_archives_stale(output_root: Path) -> Path:
    """Record that review edits changed outputs without rebuilding large ZIPs inline."""
    output_root.mkdir(parents=True, exist_ok=True)
    marker = output_root / ARCHIVE_STALE_MARKER
    temporary = output_root / f"{ARCHIVE_STALE_MARKER}.tmp"
    temporary.write_text(datetime.now().isoformat(), encoding="utf-8")
    temporary.replace(marker)
    return marker


def output_archives_are_stale(output_root: Path) -> bool:
    return (Path(output_root) / ARCHIVE_STALE_MARKER).is_file()


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
