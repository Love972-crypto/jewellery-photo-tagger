from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Callable

from .models import BatchSummary, ProcessingSettings
from .ocr_engine import OCREngine
from .processor import BatchProcessor, apply_manual_correction


JOB_QUEUED = "QUEUED"
JOB_RUNNING = "RUNNING"
JOB_COMPLETED = "COMPLETED"
JOB_FAILED = "FAILED"

ProcessorFactory = Callable[..., BatchProcessor]
CorrectionCallable = Callable[[Path, str, str, ProcessingSettings | None], tuple[bool, str]]


@dataclass(frozen=True)
class ProcessingJobSnapshot:
    job_id: str
    output_root: Path
    status: str
    done: int
    total: int
    current_file: str
    counters: dict[str, int]
    summary: BatchSummary | None = None
    error: str = ""

    @property
    def running(self) -> bool:
        return self.status in {JOB_QUEUED, JOB_RUNNING}


@dataclass
class _ProcessingJob:
    job_id: str
    output_root: Path
    status: str
    total: int
    done: int = 0
    current_file: str = "Waiting to start"
    counters: dict[str, int] = field(
        default_factory=lambda: {"ok": 0, "review": 0, "errors": 0, "duplicates": 0}
    )
    summary: BatchSummary | None = None
    error: str = ""
    future: Future | None = None


class ProcessingJobManager:
    """Runs a batch independently from Streamlit page reruns."""

    def __init__(self, max_workers: int = 1, processor_factory: ProcessorFactory = BatchProcessor) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sunaar-processing")
        self._processor_factory = processor_factory
        self._jobs: dict[str, _ProcessingJob] = {}
        self._lock = RLock()

    def start_job(
        self,
        job_id: str,
        image_paths: list[Path],
        output_root: Path,
        settings: ProcessingSettings,
        ocr_engine: OCREngine,
        project_root: Path | None = None,
    ) -> ProcessingJobSnapshot:
        normalized_job_id = str(job_id).strip()
        if not normalized_job_id:
            raise ValueError("Processing job id cannot be empty.")
        if not image_paths:
            raise ValueError("At least one image is required to start processing.")

        normalized_output_root = Path(output_root).resolve()
        normalized_output_root.mkdir(parents=True, exist_ok=True)

        with self._lock:
            existing = self._jobs.get(normalized_job_id)
            if existing is not None:
                return self._snapshot_locked(existing)

            job = _ProcessingJob(
                job_id=normalized_job_id,
                output_root=normalized_output_root,
                status=JOB_QUEUED,
                total=len(image_paths),
            )
            self._set_output_marker(normalized_output_root, ".processing", normalized_job_id)
            self._jobs[normalized_job_id] = job
            job.future = self._executor.submit(
                self._run_job,
                job,
                tuple(Path(path) for path in image_paths),
                normalized_output_root,
                settings,
                ocr_engine,
                Path(project_root) if project_root else None,
            )
            return self._snapshot_locked(job)

    def snapshot(self, job_id: str) -> ProcessingJobSnapshot | None:
        with self._lock:
            job = self._jobs.get(str(job_id).strip())
            return self._snapshot_locked(job) if job is not None else None

    def latest_running(self) -> ProcessingJobSnapshot | None:
        with self._lock:
            for job in reversed(tuple(self._jobs.values())):
                if job.status in {JOB_QUEUED, JOB_RUNNING}:
                    return self._snapshot_locked(job)
        return None

    def forget(self, job_id: str) -> bool:
        """Forget a terminal job so the UI can start a clean batch."""
        normalized_job_id = str(job_id).strip()
        with self._lock:
            job = self._jobs.get(normalized_job_id)
            if job is None:
                return False
            if job.status in {JOB_QUEUED, JOB_RUNNING}:
                raise RuntimeError("A running processing job cannot be cleared.")
            del self._jobs[normalized_job_id]
            return True

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _run_job(
        self,
        job: _ProcessingJob,
        image_paths: tuple[Path, ...],
        output_root: Path,
        settings: ProcessingSettings,
        ocr_engine: OCREngine,
        project_root: Path | None,
    ) -> None:
        with self._lock:
            job.status = JOB_RUNNING
            job.current_file = image_paths[0].name

        def progress_callback(done: int, total: int, filename: str, counters: dict[str, int]) -> None:
            with self._lock:
                job.done = max(0, min(int(done), int(total)))
                job.total = int(total)
                job.current_file = str(filename)
                job.counters = dict(counters)

        try:
            processor = self._processor_factory(
                output_root,
                settings,
                ocr_engine,
                project_root=project_root,
            )
            summary = processor.process_images(list(image_paths), progress_callback=progress_callback)
            self._set_output_marker(output_root, ".complete", job.job_id)
            with self._lock:
                job.summary = summary
                job.done = job.total
                job.current_file = "Complete"
                job.status = JOB_COMPLETED
        except Exception as exc:
            try:
                self._set_output_marker(output_root, ".failed", job.job_id)
            except Exception as marker_exc:
                exc = RuntimeError(f"{exc}; could not record failed state: {marker_exc}")
            with self._lock:
                job.error = f"Processing stopped: {exc}"
                job.status = JOB_FAILED

    @staticmethod
    def _snapshot_locked(job: _ProcessingJob) -> ProcessingJobSnapshot:
        return ProcessingJobSnapshot(
            job_id=job.job_id,
            output_root=job.output_root,
            status=job.status,
            done=job.done,
            total=job.total,
            current_file=job.current_file,
            counters=dict(job.counters),
            summary=job.summary,
            error=job.error,
        )

    @staticmethod
    def _set_output_marker(output_root: Path, marker_name: str, job_id: str) -> None:
        output_root.mkdir(parents=True, exist_ok=True)
        for name in (".processing", ".complete", ".failed"):
            (output_root / name).unlink(missing_ok=True)
        marker = output_root / marker_name
        temporary = output_root / f"{marker_name}.tmp"
        temporary.write_text(str(job_id), encoding="utf-8")
        temporary.replace(marker)


@dataclass(frozen=True)
class CorrectionJobSnapshot:
    job_id: str
    output_root: Path
    item_id: str
    corrected_tag: str
    status: str
    message: str = ""
    error: str = ""

    @property
    def running(self) -> bool:
        return self.status in {JOB_QUEUED, JOB_RUNNING}


@dataclass
class _CorrectionJob:
    job_id: str
    output_root: Path
    item_id: str
    corrected_tag: str
    status: str = JOB_QUEUED
    message: str = ""
    error: str = ""
    future: Future | None = None


class CorrectionJobManager:
    """Finalizes manual tag corrections without blocking Streamlit's UI thread."""

    def __init__(
        self,
        max_workers: int = 1,
        correction_callable: CorrectionCallable = apply_manual_correction,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sunaar-correction")
        self._correction_callable = correction_callable
        self._jobs: dict[str, _CorrectionJob] = {}
        self._lock = RLock()

    def start_job(
        self,
        job_id: str,
        output_root: Path,
        item_id: str,
        corrected_tag: str,
        settings: ProcessingSettings | None = None,
    ) -> CorrectionJobSnapshot:
        normalized_job_id = str(job_id).strip()
        normalized_item_id = str(item_id).strip()
        if not normalized_job_id:
            raise ValueError("Correction job id cannot be empty.")
        if not normalized_item_id:
            raise ValueError("Correction item id cannot be empty.")

        normalized_output_root = Path(output_root).resolve()
        with self._lock:
            existing = self._jobs.get(normalized_job_id)
            if existing is not None:
                return self._snapshot_locked(existing)
            for active in self._jobs.values():
                if (
                    active.status in {JOB_QUEUED, JOB_RUNNING}
                    and active.output_root == normalized_output_root
                    and active.item_id == normalized_item_id
                ):
                    return self._snapshot_locked(active)

            job = _CorrectionJob(
                job_id=normalized_job_id,
                output_root=normalized_output_root,
                item_id=normalized_item_id,
                corrected_tag=str(corrected_tag).strip(),
            )
            self._jobs[normalized_job_id] = job
            job.future = self._executor.submit(self._run_job, job, settings)
            return self._snapshot_locked(job)

    def snapshot(self, job_id: str) -> CorrectionJobSnapshot | None:
        with self._lock:
            job = self._jobs.get(str(job_id).strip())
            return self._snapshot_locked(job) if job is not None else None

    def forget(self, job_id: str) -> bool:
        normalized_job_id = str(job_id).strip()
        with self._lock:
            job = self._jobs.get(normalized_job_id)
            if job is None:
                return False
            if job.status in {JOB_QUEUED, JOB_RUNNING}:
                raise RuntimeError("A running correction job cannot be cleared.")
            del self._jobs[normalized_job_id]
            return True

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _run_job(self, job: _CorrectionJob, settings: ProcessingSettings | None) -> None:
        with self._lock:
            job.status = JOB_RUNNING
        try:
            ok, message = self._correction_callable(
                job.output_root,
                job.item_id,
                job.corrected_tag,
                settings,
            )
            with self._lock:
                job.message = str(message)
                if ok:
                    job.status = JOB_COMPLETED
                else:
                    job.status = JOB_FAILED
                    job.error = str(message)
        except Exception as exc:
            with self._lock:
                job.error = f"Correction could not be completed: {exc}"
                job.status = JOB_FAILED

    @staticmethod
    def _snapshot_locked(job: _CorrectionJob) -> CorrectionJobSnapshot:
        return CorrectionJobSnapshot(
            job_id=job.job_id,
            output_root=job.output_root,
            item_id=job.item_id,
            corrected_tag=job.corrected_tag,
            status=job.status,
            message=job.message,
            error=job.error,
        )
