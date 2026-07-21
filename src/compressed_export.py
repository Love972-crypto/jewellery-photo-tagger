from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps


MAX_COMPRESSED_IMAGE_BYTES = 20_000
SOURCE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
LONG_EDGE_STEPS = (1500, 1200, 1000, 800, 640, 512, 400, 320, 256, 192, 128, 96, 64)
MINIMUM_QUALITY_STEPS = (60, 50, 40, 30, 20, 10)


@dataclass(frozen=True)
class CompressedImageInfo:
    source_name: str
    output_name: str
    byte_size: int
    width: int
    height: int
    quality: int
    reused: bool = False


@dataclass
class CompressionSummary:
    converted: int = 0
    reused: int = 0
    skipped: int = 0
    images: list[CompressedImageInfo] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ready(self) -> int:
        return self.converted + self.reused


def prepare_compressed_export(
    processed_images: Path,
    compressed_images: Path,
    zip_path: Path,
    max_bytes: int = MAX_COMPRESSED_IMAGE_BYTES,
) -> CompressionSummary:
    compressed_images.mkdir(parents=True, exist_ok=True)
    sources = _source_images(processed_images)
    summary = CompressionSummary()
    expected_names = {f"{source.stem}.jpg" for source in sources}

    for orphan in compressed_images.glob("*.jpg"):
        if orphan.name not in expected_names:
            orphan.unlink(missing_ok=True)

    for source in sources:
        target = compressed_images / f"{source.stem}.jpg"
        try:
            if _is_current(source, target, max_bytes):
                with Image.open(target) as cached:
                    width, height = cached.size
                summary.reused += 1
                summary.images.append(
                    CompressedImageInfo(
                        source_name=source.name,
                        output_name=target.name,
                        byte_size=target.stat().st_size,
                        width=width,
                        height=height,
                        quality=0,
                        reused=True,
                    )
                )
                continue

            encoded, width, height, quality = compress_image_to_jpeg(source, max_bytes=max_bytes)
            temporary = target.with_suffix(".jpg.tmp")
            temporary.write_bytes(encoded)
            temporary.replace(target)
            summary.converted += 1
            summary.images.append(
                CompressedImageInfo(
                    source_name=source.name,
                    output_name=target.name,
                    byte_size=len(encoded),
                    width=width,
                    height=height,
                    quality=quality,
                )
            )
        except Exception as exc:
            summary.skipped += 1
            summary.errors.append(f"{source.name}: {exc}")
            target.unlink(missing_ok=True)
            target.with_suffix(".jpg.tmp").unlink(missing_ok=True)

    _write_images_zip(compressed_images, zip_path, max_bytes=max_bytes)
    return summary


def compress_image_to_jpeg(source: Path, max_bytes: int = MAX_COMPRESSED_IMAGE_BYTES) -> tuple[bytes, int, int, int]:
    if max_bytes < 1_000:
        raise ValueError("The compressed image limit is too small.")

    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened)
        rgb = _flatten_on_white(image)

    original_size = rgb.size
    long_edges = _candidate_long_edges(max(original_size))
    resized_cache: dict[int, Image.Image] = {}

    for minimum_quality in MINIMUM_QUALITY_STEPS:
        for long_edge in long_edges:
            candidate = resized_cache.get(long_edge)
            if candidate is None:
                candidate = _resize_to_long_edge(rgb, long_edge)
                resized_cache[long_edge] = candidate
            minimum_encoded = _encode_jpeg(candidate, minimum_quality)
            if len(minimum_encoded) > max_bytes:
                continue
            encoded, quality = _highest_quality_under_limit(candidate, minimum_quality, 95, max_bytes)
            return encoded, candidate.width, candidate.height, quality

    for long_edge in (48, 32, 24, 16):
        candidate = _resize_to_long_edge(rgb, long_edge)
        encoded = _encode_jpeg(candidate, 1)
        if len(encoded) <= max_bytes:
            return encoded, candidate.width, candidate.height, 1

    raise ValueError(f"Could not create an image at or below {max_bytes} bytes.")


def compressed_export_is_current(
    processed_images: Path,
    compressed_images: Path,
    zip_path: Path,
    max_bytes: int = MAX_COMPRESSED_IMAGE_BYTES,
) -> bool:
    sources = _source_images(processed_images)
    if not sources or not zip_path.is_file() or zip_path.stat().st_size == 0:
        return False

    expected = {f"{source.stem}.jpg" for source in sources}
    actual = {path.name for path in compressed_images.glob("*.jpg") if path.is_file()}
    if not actual or actual != expected:
        return False
    source_by_output = {f"{source.stem}.jpg": source for source in sources}
    if any(not _is_current(source_by_output[name], compressed_images / name, max_bytes) for name in actual):
        return False
    newest_input = max(source.stat().st_mtime_ns for source in sources)
    newest_output = max(path.stat().st_mtime_ns for path in compressed_images.glob("*.jpg"))
    if zip_path.stat().st_mtime_ns < max(newest_input, newest_output):
        return False

    try:
        with zipfile.ZipFile(zip_path) as archive:
            entries = [item for item in archive.infolist() if not item.is_dir()]
            if {item.filename for item in entries} != expected:
                return False
            if any(item.file_size <= 0 or item.file_size > max_bytes for item in entries):
                return False
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def invalidate_compressed_export(compressed_images: Path, zip_path: Path) -> None:
    for image in compressed_images.glob("*.jpg"):
        image.unlink(missing_ok=True)
    zip_path.unlink(missing_ok=True)


def _source_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS)


def _is_current(source: Path, target: Path, max_bytes: int) -> bool:
    metadata_current = (
        target.is_file()
        and 0 < target.stat().st_size <= max_bytes
        and target.stat().st_mtime_ns >= source.stat().st_mtime_ns
    )
    if not metadata_current:
        return False
    try:
        with Image.open(target) as image:
            if image.format != "JPEG":
                return False
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _flatten_on_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, rgba).convert("RGB")


def _candidate_long_edges(original_long_edge: int) -> list[int]:
    values = [min(original_long_edge, edge) for edge in LONG_EDGE_STEPS]
    values.append(original_long_edge)
    return sorted(set(values), reverse=True)


def _resize_to_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    current_long_edge = max(image.size)
    if long_edge >= current_long_edge:
        return image.copy()
    scale = long_edge / current_long_edge
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _encode_jpeg(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
        subsampling="4:2:0",
    )
    return buffer.getvalue()


def _highest_quality_under_limit(image: Image.Image, low: int, high: int, max_bytes: int) -> tuple[bytes, int]:
    best = _encode_jpeg(image, low)
    best_quality = low
    while low <= high:
        quality = (low + high) // 2
        encoded = _encode_jpeg(image, quality)
        if len(encoded) <= max_bytes:
            best = encoded
            best_quality = quality
            low = quality + 1
        else:
            high = quality - 1
    return best, best_quality


def _write_images_zip(folder: Path, zip_path: Path, max_bytes: int) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = zip_path.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for image in sorted(folder.glob("*.jpg")):
                if 0 < image.stat().st_size <= max_bytes:
                    archive.write(image, arcname=image.name)
        temporary.replace(zip_path)
    finally:
        temporary.unlink(missing_ok=True)
