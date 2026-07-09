from pathlib import Path

from src.file_manager import create_zip_from_folder, safe_stem, setup_output_paths, unique_png_path


def test_output_folders_created(tmp_path):
    paths = setup_output_paths(tmp_path / "Jewellery_Output")
    assert paths.processed_images.exists()
    assert paths.transparent_images.exists()
    assert paths.review_required.exists()
    assert paths.background_review.exists()
    assert paths.debug_crops.exists()


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


def test_zip_creation(tmp_path):
    folder = tmp_path / "output"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"abc")
    zip_path = create_zip_from_folder(folder, tmp_path / "out.zip")
    assert zip_path.exists()
    assert zip_path.stat().st_size > 0
