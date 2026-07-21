from pathlib import Path
from types import SimpleNamespace

import pytest

from src import local_export
from src.local_export import (
    LocalExportError,
    choose_output_folder,
    save_all_artifacts,
    save_artifact,
    unique_destination,
    validate_destination_folder,
)


def test_save_artifact_uses_collision_safe_suffix(tmp_path):
    source = tmp_path / "source" / "processed_images.zip"
    destination = tmp_path / "destination"
    source.parent.mkdir()
    destination.mkdir()
    source.write_bytes(b"zip-data")

    first = save_artifact(source, destination)
    second = save_artifact(source, destination)

    assert first.name == "processed_images.zip"
    assert second.name == "processed_images_2.zip"
    assert first.read_bytes() == second.read_bytes() == b"zip-data"


def test_save_all_creates_named_bundle_without_overwriting(tmp_path):
    source_a = tmp_path / "a.zip"
    source_b = tmp_path / "report.csv"
    destination = tmp_path / "destination"
    destination.mkdir()
    source_a.write_bytes(b"a")
    source_b.write_bytes(b"b")

    first_bundle, first_saved = save_all_artifacts([source_a, source_b], destination, folder_name="Sunaar_Downloads_test")
    second_bundle, second_saved = save_all_artifacts([source_a, source_b], destination, folder_name="Sunaar_Downloads_test")

    assert first_bundle.name == "Sunaar_Downloads_test"
    assert second_bundle.name == "Sunaar_Downloads_test_2"
    assert sorted(path.name for path in first_saved) == ["a.zip", "report.csv"]
    assert sorted(path.name for path in second_saved) == ["a.zip", "report.csv"]


def test_choose_output_folder_handles_cancel(monkeypatch, tmp_path):
    monkeypatch.setattr(
        local_export.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    assert choose_output_folder(tmp_path) is None


def test_validate_destination_rejects_unwritable_folder(monkeypatch, tmp_path):
    def fail_write(*args, **kwargs):
        raise PermissionError("blocked")

    monkeypatch.setattr(local_export.tempfile, "NamedTemporaryFile", fail_write)
    with pytest.raises(LocalExportError, match="cannot write"):
        validate_destination_folder(tmp_path)


def test_unique_destination_handles_files_and_directories(tmp_path):
    (tmp_path / "report.csv").write_text("one", encoding="utf-8")
    (tmp_path / "Bundle").mkdir()
    assert unique_destination(tmp_path, "report.csv").name == "report_2.csv"
    assert unique_destination(tmp_path, "Bundle", is_directory=True).name == "Bundle_2"
