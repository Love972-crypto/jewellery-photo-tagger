import zipfile
from pathlib import Path

from src.file_manager import (
    collect_image_files,
    create_zip_from_folder,
    extract_zip_safely,
    find_latest_output_root,
    mark_output_archives_stale,
    output_archives_are_stale,
    rebuild_output_archives,
    safe_stem,
    setup_output_paths,
    unique_png_path,
)


def test_output_folders_created(tmp_path):
    paths = setup_output_paths(tmp_path / "Jewellery_Output")
    assert paths.processed_images.exists()
    assert paths.transparent_images.exists()
    assert paths.compressed_images_20kb.exists()
    assert paths.review_required.exists()
    assert paths.ai_review.exists()
    assert paths.background_review.exists()
    assert paths.correction_cache.exists()
    assert paths.debug_crops.exists()
    assert paths.compressed_images_20kb_zip.name == "compressed_images_20kb.zip"


def test_unique_png_path_suffixes_duplicates(tmp_path):
    folder = tmp_path / "processed"
    first, duplicate = unique_png_path(folder, "121134")
    assert first.name == "121134.png"
    assert duplicate is False
    first.write_bytes(b"data")
    second, duplicate = unique_png_path(folder, "121134")
    assert second.name == "121134_2.png"
    assert duplicate is True


def test_safe_stem_removes_unsafe_characters():
    assert safe_stem("../IMG 001!!.jpg") == "IMG_001"


def test_collect_image_files_skips_exact_content_duplicates(tmp_path):
    first = tmp_path / "direct" / "photo.jpg"
    duplicate = tmp_path / "from_zip" / "same-photo.jpeg"
    different = tmp_path / "from_zip" / "different.jpg"
    first.parent.mkdir()
    duplicate.parent.mkdir()
    first.write_bytes(b"same image bytes")
    duplicate.write_bytes(b"same image bytes")
    different.write_bytes(b"different image bytes")

    paths, errors = collect_image_files(tmp_path)

    assert paths == [first, different]
    assert len(errors) == 1
    assert "Skipped exact duplicate photo" in errors[0]
    assert "same-photo.jpeg" in errors[0]


def test_collect_image_files_keeps_different_photos_even_when_names_match(tmp_path):
    first = tmp_path / "one" / "photo.jpg"
    second = tmp_path / "two" / "photo.jpg"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first image")
    second.write_bytes(b"second image")

    paths, errors = collect_image_files(tmp_path)

    assert paths == [first, second]
    assert errors == []


def test_zip_creation(tmp_path):
    folder = tmp_path / "output"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"abc")
    zip_path = create_zip_from_folder(folder, tmp_path / "out.zip")
    assert zip_path.exists()
    assert zip_path.stat().st_size > 0


def test_full_output_zip_excludes_hidden_correction_cache(tmp_path):
    paths = setup_output_paths(tmp_path / "Jewellery_Output")
    (paths.processed_images / "121134.png").write_bytes(b"final")
    (paths.correction_cache / "item_000001_white.png").write_bytes(b"private-cache")

    create_zip_from_folder(paths.root, paths.full_zip, exclude_zip_files=True)

    with zipfile.ZipFile(paths.full_zip) as archive:
        names = archive.namelist()
    assert "processed_images/121134.png" in names
    assert not any(".correction_cache" in name for name in names)


def test_zip_member_cannot_escape_to_similarly_named_sibling(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    destination = tmp_path / "uploads"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../uploads_evil/escaped.txt", b"unsafe")

    errors = extract_zip_safely(archive_path, destination)

    assert any("unsafe ZIP member" in error for error in errors)
    assert not (tmp_path / "uploads_evil" / "escaped.txt").exists()


def test_zip_bomb_ratio_is_rejected(tmp_path):
    archive_path = tmp_path / "compressed.zip"
    destination = tmp_path / "uploads"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("large.jpg", b"0" * (2 * 1024 * 1024))

    errors = extract_zip_safely(archive_path, destination)

    assert any("suspiciously compressed" in error for error in errors)
    assert not (destination / "large.jpg").exists()


def test_duplicate_zip_members_are_renamed_instead_of_overwritten(tmp_path):
    archive_path = tmp_path / "duplicates.zip"
    destination = tmp_path / "uploads"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("photo.jpg", b"first")
        archive.writestr("photo.jpg", b"second")

    errors = extract_zip_safely(archive_path, destination)

    assert any("Renamed duplicate" in error for error in errors)
    assert (destination / "photo.jpg").read_bytes() == b"first"
    assert (destination / "photo_2.jpg").read_bytes() == b"second"


def test_latest_output_ignores_processing_and_failed_partial_reports(tmp_path):
    completed = tmp_path / ".runtime" / "run_complete" / "Jewellery_Output"
    processing = tmp_path / ".runtime" / "run_processing" / "Jewellery_Output"
    failed = tmp_path / ".runtime" / "run_failed" / "Jewellery_Output"
    for folder in (completed, processing, failed):
        folder.mkdir(parents=True)
        (folder / "report.csv").write_text("status\nOK\n", encoding="utf-8")
    (completed / ".complete").write_text("complete", encoding="utf-8")
    (processing / ".processing").write_text("running", encoding="utf-8")
    (failed / ".failed").write_text("failed", encoding="utf-8")

    assert find_latest_output_root(tmp_path) == completed


def test_review_change_marks_archives_stale_until_lazy_rebuild(tmp_path):
    output_root = tmp_path / "Jewellery_Output"
    paths = setup_output_paths(output_root)
    (paths.processed_images / "121995.png").write_bytes(b"png")

    mark_output_archives_stale(output_root)

    assert output_archives_are_stale(output_root)
    rebuilt = rebuild_output_archives(output_root)
    assert rebuilt.processed_zip.is_file()
    assert rebuilt.full_zip.is_file()
    assert not output_archives_are_stale(output_root)
