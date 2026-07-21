from __future__ import annotations

import time
from pathlib import Path
from threading import Event

from src.job_manager import CorrectionJobManager, JOB_COMPLETED, JOB_FAILED, ProcessingJobManager
from src.models import BatchSummary, ProcessingSettings
from src.ocr_engine import StaticOCREngine


def wait_for_status(manager: ProcessingJobManager, job_id: str, status: str, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = manager.snapshot(job_id)
        if snapshot and snapshot.status == status:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"Job {job_id} did not reach {status}.")


def test_job_continues_without_page_polling_and_duplicate_start_is_ignored(tmp_path):
    started = Event()
    release = Event()
    processor_calls = []

    class BlockingProcessor:
        def __init__(self, output_root, settings, ocr_engine, project_root=None):
            processor_calls.append(output_root)

        def process_images(self, image_paths, progress_callback=None):
            progress_callback(0, len(image_paths), image_paths[0].name, {"ok": 0, "review": 0, "errors": 0, "duplicates": 0})
            started.set()
            assert release.wait(timeout=2)
            progress_callback(2, 2, "Complete", {"ok": 2, "review": 0, "errors": 0, "duplicates": 0})
            return BatchSummary(total=2, processed=2, ok=2)

    manager = ProcessingJobManager(processor_factory=BlockingProcessor)
    paths = [tmp_path / "one.jpg", tmp_path / "two.jpg"]
    engine = StaticOCREngine([])
    try:
        first = manager.start_job("batch-1", paths, tmp_path / "output", ProcessingSettings(), engine)
        assert first.running
        assert first.output_root == (tmp_path / "output").resolve()
        assert (tmp_path / "output" / ".processing").is_file()
        assert started.wait(timeout=2)
        assert manager.latest_running().job_id == "batch-1"

        duplicate = manager.start_job("batch-1", paths, tmp_path / "output", ProcessingSettings(), engine)
        assert duplicate.running
        assert len(processor_calls) == 1

        release.set()
        completed = wait_for_status(manager, "batch-1", JOB_COMPLETED)
        assert completed.done == 2
        assert completed.summary is not None
        assert completed.summary.ok == 2
        assert completed.counters["ok"] == 2
        assert (tmp_path / "output" / ".complete").is_file()
        assert not (tmp_path / "output" / ".processing").exists()
        assert manager.latest_running() is None
    finally:
        release.set()
        manager.shutdown()


def test_job_failure_is_persisted_for_later_page_reruns(tmp_path):
    class FailingProcessor:
        def __init__(self, output_root, settings, ocr_engine, project_root=None):
            pass

        def process_images(self, image_paths, progress_callback=None):
            raise RuntimeError("mock failure")

    manager = ProcessingJobManager(processor_factory=FailingProcessor)
    try:
        manager.start_job(
            "batch-failed",
            [Path("one.jpg")],
            tmp_path / "output",
            ProcessingSettings(),
            StaticOCREngine([]),
        )
        failed = wait_for_status(manager, "batch-failed", JOB_FAILED)
        assert not failed.running
        assert "mock failure" in failed.error
        assert manager.snapshot("batch-failed").error == failed.error
        assert (tmp_path / "output" / ".failed").is_file()
        assert not (tmp_path / "output" / ".processing").exists()
        assert manager.forget("batch-failed")
        assert manager.snapshot("batch-failed") is None
    finally:
        manager.shutdown()


def test_manual_correction_runs_without_blocking_ui_and_duplicate_click_is_ignored(tmp_path):
    started = Event()
    release = Event()
    calls = []

    def blocking_correction(output_root, item_id, corrected_tag, settings):
        calls.append((output_root, item_id, corrected_tag, settings))
        started.set()
        assert release.wait(timeout=2)
        return True, "Correction finished."

    manager = CorrectionJobManager(correction_callable=blocking_correction)
    try:
        first = manager.start_job(
            "correction-1",
            tmp_path / "output",
            "item_000001",
            "121995",
            ProcessingSettings(),
        )
        assert first.running
        assert started.wait(timeout=2)

        duplicate = manager.start_job(
            "correction-2",
            tmp_path / "output",
            "item_000001",
            "121995",
            ProcessingSettings(),
        )
        assert duplicate.job_id == "correction-1"
        assert len(calls) == 1

        release.set()
        completed = wait_for_status(manager, "correction-1", JOB_COMPLETED)
        assert completed.message == "Correction finished."
        assert manager.forget("correction-1")
    finally:
        release.set()
        manager.shutdown()
