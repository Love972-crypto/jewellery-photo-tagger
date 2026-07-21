import time
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from src.compressed_export import (
    MAX_COMPRESSED_IMAGE_BYTES,
    compressed_export_is_current,
    invalidate_compressed_export,
    prepare_compressed_export,
)


def _catalogue_image(path: Path, size: tuple[int, int], transparent: bool = False) -> None:
    mode = "RGBA" if transparent else "RGB"
    background = (255, 255, 255, 0) if transparent else (255, 255, 255)
    image = Image.new(mode, size, background)
    draw = ImageDraw.Draw(image)
    width, height = size
    padding = max(40, min(width, height) // 12)
    for index in range(180):
        x = padding + (index * 47) % max(1, width - padding * 2)
        y = padding + (index * 83) % max(1, height - padding * 2)
        radius = 5 + (index % 18)
        colour = (172 + index % 70, 104 + index % 90, 10 + index % 40, 255) if transparent else (172 + index % 70, 104 + index % 90, 10 + index % 40)
        draw.ellipse((x, y, x + radius, y + radius), fill=colour)
    draw.rectangle((width // 3, height // 3, width * 2 // 3, height * 2 // 3), fill=(188, 129, 25, 255) if transparent else (188, 129, 25))
    image.save(path)


def test_compressed_export_creates_strict_white_jpegs_and_zip(tmp_path):
    processed = tmp_path / "processed_images"
    compressed = tmp_path / "compressed_images_20kb"
    zip_path = tmp_path / "compressed_images_20kb.zip"
    processed.mkdir()
    _catalogue_image(processed / "122444.png", (1200, 1500), transparent=True)
    _catalogue_image(processed / "122445.png", (1500, 1200))

    summary = prepare_compressed_export(processed, compressed, zip_path)

    assert summary.converted == 2
    assert summary.skipped == 0
    assert compressed_export_is_current(processed, compressed, zip_path)
    for image_path in compressed.glob("*.jpg"):
        assert 0 < image_path.stat().st_size <= MAX_COMPRESSED_IMAGE_BYTES
        with Image.open(image_path) as image:
            assert image.mode == "RGB"
            corner = image.getpixel((0, 0))
            assert min(corner) >= 245

    with zipfile.ZipFile(zip_path) as archive:
        assert sorted(archive.namelist()) == ["122444.jpg", "122445.jpg"]
        assert all(0 < item.file_size <= MAX_COMPRESSED_IMAGE_BYTES for item in archive.infolist())


def test_compressed_export_reuses_current_files_and_invalidates_changed_source(tmp_path):
    processed = tmp_path / "processed"
    compressed = tmp_path / "compressed"
    zip_path = tmp_path / "compressed.zip"
    processed.mkdir()
    source = processed / "121134.png"
    _catalogue_image(source, (800, 1000))
    first = prepare_compressed_export(processed, compressed, zip_path)
    target = compressed / "121134.jpg"
    first_mtime = target.stat().st_mtime_ns

    second = prepare_compressed_export(processed, compressed, zip_path)
    assert second.converted == 0
    assert second.reused == 1
    assert target.stat().st_mtime_ns == first_mtime

    time.sleep(0.01)
    _catalogue_image(source, (900, 1200))
    assert not compressed_export_is_current(processed, compressed, zip_path)
    third = prepare_compressed_export(processed, compressed, zip_path)
    assert third.converted == 1
    assert compressed_export_is_current(processed, compressed, zip_path)

    invalidate_compressed_export(compressed, zip_path)
    assert not target.exists()
    assert not zip_path.exists()


def test_compressed_export_skips_corrupt_images(tmp_path):
    processed = tmp_path / "processed"
    compressed = tmp_path / "compressed"
    zip_path = tmp_path / "compressed.zip"
    processed.mkdir()
    (processed / "broken.png").write_bytes(b"not an image")

    summary = prepare_compressed_export(processed, compressed, zip_path)

    assert summary.ready == 0
    assert summary.skipped == 1
    assert "broken.png" in summary.errors[0]
    assert not compressed_export_is_current(processed, compressed, zip_path)


def test_mixed_export_is_not_current_when_any_source_is_missing(tmp_path):
    processed = tmp_path / "processed"
    compressed = tmp_path / "compressed"
    zip_path = tmp_path / "compressed.zip"
    processed.mkdir()
    _catalogue_image(processed / "good.png", (600, 800))
    (processed / "broken.png").write_bytes(b"not an image")

    summary = prepare_compressed_export(processed, compressed, zip_path)

    assert summary.ready == 1
    assert summary.skipped == 1
    assert not compressed_export_is_current(processed, compressed, zip_path)


def test_corrupt_cached_jpeg_is_rebuilt_instead_of_reused(tmp_path):
    processed = tmp_path / "processed"
    compressed = tmp_path / "compressed"
    zip_path = tmp_path / "compressed.zip"
    processed.mkdir()
    source = processed / "122444.png"
    _catalogue_image(source, (600, 800))
    prepare_compressed_export(processed, compressed, zip_path)
    target = compressed / "122444.jpg"
    target.write_bytes(b"not a jpeg")
    target.touch()

    assert not compressed_export_is_current(processed, compressed, zip_path)
    repaired = prepare_compressed_export(processed, compressed, zip_path)

    assert repaired.converted == 1
    assert repaired.reused == 0
    assert compressed_export_is_current(processed, compressed, zip_path)
