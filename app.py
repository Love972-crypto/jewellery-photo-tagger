from __future__ import annotations

import html
from pathlib import Path
from uuid import uuid4

import streamlit as st

from src.compressed_export import (
    MAX_COMPRESSED_IMAGE_BYTES,
    compressed_export_is_current,
    prepare_compressed_export,
)
from src.download_server import download_url, start_download_server
from src.file_manager import (
    create_run_workspace,
    file_size_label,
    find_latest_output_root,
    materialize_uploaded_files,
    output_archives_are_stale,
    rebuild_output_archives,
)
from src.job_manager import (
    CorrectionJobManager,
    CorrectionJobSnapshot,
    JOB_COMPLETED,
    JOB_FAILED,
    ProcessingJobManager,
    ProcessingJobSnapshot,
)
from src.local_export import (
    LocalExportError,
    choose_output_folder,
    open_folder,
    save_all_artifacts,
    save_artifact,
)
from src.models import BatchSummary, ProcessingSettings
from src.ocr_engine import build_ocr_engine
from src.processor import resolve_ai_review, resolve_background_review
from src.report_generator import read_report
from src.tag_parser import is_valid_manual_tag
from src.ui_components import (
    FRIENDLY_STATUS,
    apply_theme,
    display_report_table,
    process_panel,
    render_dashboard,
    render_status_badge,
    render_stepper,
)

PROJECT_ROOT = Path(__file__).parent.resolve()


def init_state() -> None:
    defaults = {
        "settings": ProcessingSettings(),
        "run_dir": None,
        "upload_dir": None,
        "output_root": None,
        "image_paths": [],
        "upload_errors": [],
        "summary": None,
        "processing_done": False,
        "is_processing": False,
        "processing_job_id": "",
        "last_error": "",
        "active_page": "Dashboard",
        "save_folder": "",
        "last_saved_folder": "",
        "correction_job_id": "",
        "correction_notice": "",
        "correction_notice_error": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def page_header(title: str, subtitle: str, active_step: str | None = None) -> None:
    st.title(title)
    st.caption(subtitle)
    if active_step:
        render_stepper(active_step)


def active_output_root() -> Path | None:
    output_root = st.session_state.output_root
    if output_root and (Path(output_root) / "report.csv").exists():
        return Path(output_root)
    if output_root and st.session_state.image_paths:
        return None

    latest = find_latest_output_root(PROJECT_ROOT)
    if latest:
        st.session_state.output_root = latest
        return latest
    return None


def prepare_uploads(uploaded_files) -> None:
    run_dir, upload_dir, output_paths = create_run_workspace(PROJECT_ROOT)
    image_paths, errors = materialize_uploaded_files(uploaded_files, upload_dir)
    st.session_state.run_dir = run_dir
    st.session_state.upload_dir = upload_dir
    st.session_state.output_root = output_paths.root
    st.session_state.image_paths = image_paths
    st.session_state.upload_errors = errors
    st.session_state.summary = None
    st.session_state.processing_done = False
    st.session_state.is_processing = False
    st.session_state.processing_job_id = ""
    st.session_state.last_error = ""


@st.cache_resource(show_spinner=False)
def get_cached_ocr_engine():
    return build_ocr_engine(use_easyocr=True)


@st.cache_resource(show_spinner=False)
def get_processing_job_manager():
    return ProcessingJobManager(max_workers=1)


@st.cache_resource(show_spinner=False)
def get_correction_job_manager():
    return CorrectionJobManager(max_workers=1)


@st.cache_resource(show_spinner=False)
def get_download_server(project_root: str):
    return start_download_server(Path(project_root))


def active_job_snapshot() -> ProcessingJobSnapshot | None:
    job_id = str(st.session_state.get("processing_job_id", "")).strip()
    if not job_id:
        return None
    return get_processing_job_manager().snapshot(job_id)


def active_correction_snapshot() -> CorrectionJobSnapshot | None:
    job_id = str(st.session_state.get("correction_job_id", "")).strip()
    if not job_id:
        return None
    return get_correction_job_manager().snapshot(job_id)


def sync_correction_job_state() -> CorrectionJobSnapshot | None:
    snapshot = active_correction_snapshot()
    if snapshot is None:
        st.session_state.correction_job_id = ""
        return None
    if snapshot.running:
        return snapshot

    st.session_state.correction_job_id = ""
    st.session_state.correction_notice = snapshot.message or snapshot.error
    st.session_state.correction_notice_error = snapshot.status == JOB_FAILED
    get_correction_job_manager().forget(snapshot.job_id)
    return None


def apply_job_snapshot(snapshot: ProcessingJobSnapshot) -> None:
    st.session_state.processing_job_id = snapshot.job_id
    st.session_state.output_root = snapshot.output_root
    if snapshot.running:
        st.session_state.is_processing = True
        st.session_state.processing_done = False
        st.session_state.last_error = ""
    elif snapshot.status == JOB_COMPLETED and snapshot.summary is not None:
        st.session_state.summary = snapshot.summary
        st.session_state.is_processing = False
        st.session_state.processing_done = True
        st.session_state.last_error = ""
    elif snapshot.status == JOB_FAILED:
        st.session_state.is_processing = False
        st.session_state.processing_done = False
        st.session_state.last_error = snapshot.error or "Processing stopped unexpectedly."


def sync_active_job_state() -> ProcessingJobSnapshot | None:
    snapshot = active_job_snapshot()
    if snapshot is None and not str(st.session_state.get("processing_job_id", "")).strip():
        snapshot = get_processing_job_manager().latest_running()
    if snapshot is None:
        if st.session_state.get("is_processing"):
            st.session_state.is_processing = False
        return None
    apply_job_snapshot(snapshot)
    return snapshot


@st.fragment(run_every=1)
def render_processing_job_progress(job_id: str) -> None:
    snapshot = get_processing_job_manager().snapshot(job_id)
    if snapshot is None:
        st.error("The active processing job could not be found.")
        return

    total = max(snapshot.total, 1)
    st.progress(min(snapshot.done / total, 1.0))
    if snapshot.running:
        st.info(f"Processing in background: {snapshot.current_file}")
    elif snapshot.status == JOB_COMPLETED:
        st.success("Processing completed successfully.")
    elif snapshot.status == JOB_FAILED:
        st.error(snapshot.error or "Processing stopped unexpectedly.")

    counters = snapshot.counters
    st.markdown(
        f"""
        **Live count:** Processed `{snapshot.done}/{snapshot.total}` - Ready `{counters.get('ok', 0)}` -
        Review `{counters.get('review', 0)}` - Duplicates `{counters.get('duplicates', 0)}` - Errors `{counters.get('errors', 0)}`
        """
    )
    if snapshot.running:
        st.caption("You can safely open Dashboard, Download, or any other page. This batch will continue.")
    elif snapshot.status == JOB_COMPLETED and not st.session_state.get("processing_done"):
        apply_job_snapshot(snapshot)
        st.rerun()
    elif snapshot.status == JOB_FAILED:
        apply_job_snapshot(snapshot)


@st.fragment(run_every=1)
def render_correction_job_progress(job_id: str) -> None:
    snapshot = get_correction_job_manager().snapshot(job_id)
    if snapshot is None:
        st.session_state.correction_job_id = ""
        st.rerun()
    if snapshot.running:
        st.info(
            f"Saving corrected tag {snapshot.corrected_tag} and preparing its final image in the background..."
        )
        return

    st.session_state.correction_job_id = ""
    st.session_state.correction_notice = snapshot.message or snapshot.error
    st.session_state.correction_notice_error = snapshot.status == JOB_FAILED
    get_correction_job_manager().forget(snapshot.job_id)
    st.rerun()


def render_active_correction_gate() -> bool:
    snapshot = active_correction_snapshot()
    if snapshot is None or not snapshot.running:
        return False
    st.info("A tag correction is being finalized in the background. You can use the Dashboard meanwhile.")
    return True


def render_active_job_gate() -> bool:
    snapshot = active_job_snapshot()
    if snapshot is None:
        return False
    if snapshot.running:
        st.info("Your batch is still processing in the background. Final files will appear automatically when it completes.")
        render_processing_job_progress(snapshot.job_id)
        return True
    if snapshot.status == JOB_FAILED:
        st.error(snapshot.error or "Processing stopped unexpectedly.")
        if st.button("Clear failed batch and upload again", type="primary", width="stretch"):
            get_processing_job_manager().forget(snapshot.job_id)
            st.session_state.processing_job_id = ""
            st.session_state.image_paths = []
            st.session_state.output_root = None
            st.session_state.summary = None
            st.session_state.processing_done = False
            st.session_state.is_processing = False
            st.session_state.last_error = ""
            st.rerun()
        return True
    return False


def render_upload_page() -> None:
    page_header("Upload Photos", "Add jewellery photos or a ZIP file. Originals stay untouched.", "Upload")
    if render_active_job_gate():
        return
    if render_active_correction_gate():
        return
    uploaded = st.file_uploader(
        "Choose JPG, JPEG, PNG, WEBP, HEIC, or ZIP files",
        type=["jpg", "jpeg", "png", "webp", "heic", "zip"],
        accept_multiple_files=True,
    )
    col_a, col_b = st.columns([1, 2])
    with col_a:
        if st.button("Prepare files", type="primary", width="stretch", disabled=not uploaded):
            prepare_uploads(uploaded)
            st.success("Files prepared. Check settings next.")
    with col_b:
        st.info("For a folder, compress it as a ZIP and upload the ZIP here.")

    image_paths = st.session_state.image_paths
    if image_paths:
        st.metric("Photos ready", len(image_paths))
        st.write("First files:")
        st.write(", ".join(path.name for path in image_paths[:8]))
    for error in st.session_state.upload_errors[:10]:
        st.warning(error)


def render_settings_page() -> None:
    page_header("Processing Settings", "Simple controls for clear, safe batch processing.", "Settings")
    if render_active_job_gate():
        return
    if render_active_correction_gate():
        return
    settings: ProcessingSettings = st.session_state.settings
    speed_choice = st.radio(
        "Processing mode",
        ["Fast production", "Quality max"],
        horizontal=True,
        index=0 if settings.enhancement_mode == "fast" else 1,
    )
    enhance_enabled = st.toggle("Enhance photo quality", value=settings.enhance_enabled)
    save_debug_crops = st.toggle("Save debug tag crops", value=settings.save_debug_crops)
    confidence_threshold = st.slider("OCR confidence threshold", min_value=0.20, max_value=0.90, value=float(settings.confidence_threshold), step=0.05)
    hd_output_enabled = st.toggle(
        "Safe HD output x2 (no AI tiling)",
        value=settings.hd_output_enabled,
    )
    st.caption("Real-ESRGAN is disabled on this machine because it produced tiled/corrupt images. Safe HD uses deterministic upscale and controlled sharpening.")
    remove_background = st.toggle("Remove background", value=settings.remove_background)
    background_mode_labels = {
        "White + Transparent": "white_and_transparent",
        "White only": "white_only",
        "Transparent only": "transparent_only",
    }
    current_bg_label = next(
        (label for label, value in background_mode_labels.items() if value == settings.background_output_mode),
        "White + Transparent",
    )
    background_mode_label = st.selectbox(
        "Background output",
        list(background_mode_labels.keys()),
        index=list(background_mode_labels.keys()).index(current_bg_label),
        disabled=not remove_background,
    )
    catalogue_layout_enabled = st.toggle(
        "Portrait catalogue alignment",
        value=settings.catalogue_layout_enabled,
        disabled=not remove_background,
    )
    ai_background_fallback_enabled = st.toggle(
        "Always Hybrid (BiRefNet + U2Net)",
        value=settings.ai_background_fallback_enabled,
        disabled=not remove_background,
    )
    st.caption(
        "Every photo uses BiRefNet cleanup with U2Net jewellery preservation for consistent catalogue quality."
    )
    st.caption("Background removal runs after OCR, so tag reading stays on the original enhanced photo.")
    st.text_input("Output format", value="PNG", disabled=True)
    st.text_input("Duplicate file handling", value="Auto suffix, for example 121134_2.png", disabled=True)
    st.session_state.settings = ProcessingSettings(
        enhance_enabled=enhance_enabled,
        save_debug_crops=save_debug_crops,
        confidence_threshold=confidence_threshold,
        enhancement_mode="fast" if speed_choice == "Fast production" else "quality",
        ocr_attempt_mode="fast" if speed_choice == "Fast production" else "deep",
        hd_output_enabled=hd_output_enabled,
        hd_scale=2,
        remove_background=remove_background,
        background_output_mode=background_mode_labels[background_mode_label],
        ai_background_fallback_enabled=ai_background_fallback_enabled,
        catalogue_layout_enabled=catalogue_layout_enabled,
        catalogue_canvas_width=1200,
        catalogue_canvas_height=1500,
    )
    st.success("Settings are ready. Fast production is the recommended daily mode.")


def render_processing_page() -> None:
    page_header("Processing", "Start once. The batch keeps running even when you open another page.", "Process")
    if render_active_correction_gate():
        return
    snapshot = active_job_snapshot()
    if snapshot is not None and snapshot.running:
        process_panel(
            "Processing in background",
            "This batch is protected from page navigation. You can open Download, Dashboard, or any other page now.",
            "processing",
        )
        st.metric("Photos in this batch", snapshot.total)
        render_processing_job_progress(snapshot.job_id)
        return

    image_paths = st.session_state.image_paths
    if not image_paths:
        st.info("Upload photos first. Once files are prepared, the start button will appear here.")
        return

    if snapshot is not None and snapshot.status == JOB_FAILED:
        process_panel("Processing stopped", snapshot.error or "The batch stopped unexpectedly.", "error")
        st.error(snapshot.error or "Processing stopped unexpectedly.")
        if st.button("Clear failed batch and upload again", type="primary", width="stretch", key="clear_failed_processing"):
            get_processing_job_manager().forget(snapshot.job_id)
            st.session_state.processing_job_id = ""
            st.session_state.image_paths = []
            st.session_state.output_root = None
            st.session_state.summary = None
            st.session_state.processing_done = False
            st.session_state.is_processing = False
            st.session_state.last_error = ""
            st.session_state.active_page = "Upload Photos"
            st.rerun()
        return

    process_panel(
        "Batch ready",
        f"{len(image_paths)} photo ready. Click Start processing once. OCR and image work will continue safely in the background.",
        "ready",
    )
    st.metric("Photos in this batch", len(image_paths))

    if st.session_state.processing_done and st.session_state.summary:
        st.success("This batch has already been processed. Upload a new batch to start fresh.")
        render_summary(st.session_state.summary)
        if st.session_state.output_root:
            render_download_buttons(Path(st.session_state.output_root), compact=True)
        col_review, col_download = st.columns(2)
        with col_review:
            if st.button("Go to review", width="stretch"):
                st.session_state.active_page = "Review Required"
                st.rerun()
        with col_download:
            if st.button("Go to downloads", type="primary", width="stretch"):
                st.session_state.active_page = "Download"
                st.rerun()
        return

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    if st.button("Start processing now", type="primary", width="stretch"):
        try:
            st.session_state.last_error = ""
            st.session_state.summary = None
            st.session_state.processing_done = False
            output_root = Path(st.session_state.output_root).resolve()
            job_id = str(output_root)
            snapshot = get_processing_job_manager().start_job(
                job_id=job_id,
                image_paths=image_paths,
                output_root=output_root,
                settings=st.session_state.settings,
                ocr_engine=get_cached_ocr_engine(),
                project_root=PROJECT_ROOT,
            )
            st.session_state.processing_job_id = job_id
            apply_job_snapshot(snapshot)
            st.rerun()
        except Exception as exc:
            st.session_state.last_error = f"Processing stopped: {exc}"
            st.session_state.is_processing = False

    if st.session_state.summary:
        render_summary(st.session_state.summary)


def render_summary(summary: BatchSummary) -> None:
    st.subheader("Processing Results")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total", summary.total)
    col2.metric("Ready", summary.ok)
    col3.metric("Needs review", summary.review_required)
    col4.metric("Duplicates", summary.duplicate_tags)
    col5.metric("Errors", summary.errors)
    st.caption(f"Total processing time: {summary.elapsed_seconds:.1f} seconds")


def render_ai_review_page() -> None:
    page_header("AI Review", "Approve the stronger local AI repair or send it to manual review.", "Review")
    if render_active_job_gate():
        return
    if render_active_correction_gate():
        return
    output_root = active_output_root()
    if not output_root:
        st.info("No batch has been processed yet.")
        return

    report = read_report(Path(output_root) / "report.csv")
    ai_rows = report[report["output_folder"] == "ai_review"] if not report.empty else report
    if ai_rows.empty:
        st.success("No AI repairs are waiting for approval.")
        return

    st.info(f"{len(ai_rows)} photo repaired by the local fallback model. Check completeness before accepting.")
    for _, row in ai_rows.iterrows():
        original = row["original_filename"]
        item_id = str(row.get("item_id", "")).strip() or original
        with st.expander(f"{original} - AI repair ready", expanded=True):
            review_path = Path(output_root) / "ai_review" / row["final_filename"]
            u2net_path = review_path.with_name(f"{review_path.stem}_u2net_white.png")
            ai_white_path = review_path.with_name(f"{review_path.stem}_ai_white.png")
            original_col, u2net_col, ai_col = st.columns(3)
            with original_col:
                if review_path.exists():
                    st.image(str(review_path), caption="Complete enhanced original", width="stretch")
            with u2net_col:
                if u2net_path.exists():
                    st.image(str(u2net_path), caption="U2Net attempt", width="stretch")
                else:
                    st.caption("U2Net preview unavailable")
            with ai_col:
                if ai_white_path.exists():
                    st.image(str(ai_white_path), caption="Local AI repair", width="stretch")

            render_status_badge(row["status"])
            st.write(f"Detected tag: {row.get('detected_tag_number', '') or 'None'}")
            st.caption(row.get("background_notes", "") or "Local AI background repair completed.")
            accept_col, manual_col = st.columns(2)
            with accept_col:
                if st.button("Accept AI repair", type="primary", width="stretch", key=f"accept-ai-{item_id}"):
                    ok, message = resolve_ai_review(Path(output_root), item_id, "accept_ai")
                    if ok:
                        st.success(message)
                        st.rerun()
                    st.warning(message)
            with manual_col:
                if st.button("Send to Manual Review", width="stretch", key=f"manual-ai-{item_id}"):
                    ok, message = resolve_ai_review(Path(output_root), item_id, "send_manual")
                    if ok:
                        st.success(message)
                        st.session_state.active_page = "Review Required"
                        st.rerun()
                    st.warning(message)


def render_review_page() -> None:
    page_header("Review Required", "Confirm unclear tags or inspect a background mask before final output.", "Review")
    if render_active_job_gate():
        return
    if render_active_correction_gate():
        return
    output_root = active_output_root()
    if not output_root:
        st.info("No batch has been processed yet.")
        return

    report = read_report(Path(output_root) / "report.csv")
    review_statuses = {"REVIEW_REQUIRED", "OCR_FAILED", "TAG_NOT_FOUND"}
    review_rows = (
        report[report["status"].isin(review_statuses) & (report["output_folder"] != "ai_review")]
        if not report.empty
        else report
    )
    if review_rows.empty:
        st.success("No review items are waiting.")
        return

    for _, row in review_rows.iterrows():
        original = row["original_filename"]
        item_id = str(row.get("item_id", "")).strip() or original
        is_background_review = str(row.get("output_folder", "")) == "background_review"
        with st.expander(f"{original} - {FRIENDLY_STATUS.get(row['status'], row['status'])}", expanded=False):
            image_path = Path(output_root) / row["output_folder"] / row["final_filename"]
            crop_name = f"{item_id}_tag_crop.png" if str(row.get("item_id", "")).strip() else f"{Path(original).stem}_tag_crop.png"
            crop_path = Path(output_root) / "debug_crops" / crop_name
            if is_background_review:
                white_preview_path = image_path.with_name(f"{image_path.stem}_candidate_white.png")
                transparent_name = str(row.get("transparent_filename", "")).strip()
                transparent_preview_path = image_path.parent / transparent_name if transparent_name else None
                original_col, preview_col = st.columns(2)
                with original_col:
                    if image_path.exists():
                        st.image(str(image_path), caption="Complete enhanced original", width="stretch")
                with preview_col:
                    if white_preview_path.exists():
                        st.image(str(white_preview_path), caption="Background-removal candidate", width="stretch")
                    elif transparent_preview_path is not None and transparent_preview_path.exists():
                        st.image(str(transparent_preview_path), caption="Transparent candidate", width="stretch")

                render_status_badge(row["status"])
                st.write(f"Detected tag: {row.get('detected_tag_number', '') or 'None'}")
                st.warning(row.get("background_notes", "") or "Jewellery preservation needs visual confirmation.")
                accept_col, original_col = st.columns(2)
                with accept_col:
                    if st.button("Accept complete preview", type="primary", width="stretch", key=f"accept-bg-{item_id}"):
                        ok, message = resolve_background_review(Path(output_root), item_id, "accept_preview")
                        if ok:
                            st.success(message)
                            st.rerun()
                        st.warning(message)
                with original_col:
                    if st.button("Keep original photo", width="stretch", key=f"keep-bg-{item_id}"):
                        ok, message = resolve_background_review(Path(output_root), item_id, "keep_original")
                        if ok:
                            st.success(message)
                            st.rerun()
                        st.warning(message)
            else:
                col_img, col_info = st.columns([1, 1])
                with col_img:
                    if image_path.exists():
                        st.image(str(image_path), caption="Enhanced photo", width="stretch")
                    if crop_path.exists():
                        st.image(str(crop_path), caption="Tag crop", width="stretch")
                with col_info:
                    render_status_badge(row["status"])
                    st.write("Raw OCR text")
                    st.code(row.get("ocr_text_raw", "") or "No readable text")
                    st.write(f"Suggested tag: {row.get('detected_tag_number', '') or 'None'}")
                    st.write(f"Confidence: {row.get('confidence_score', '') or '0'}")
                    corrected = st.text_input("Correct tag number", value=row.get("detected_tag_number", ""), key=f"correct_{item_id}")
                    if st.button("Save correction", key=f"save_{item_id}"):
                        if not is_valid_manual_tag(corrected):
                            st.warning("Enter a numeric tag number with 5 to 8 digits.")
                        else:
                            job_id = f"correction-{Path(output_root).name}-{item_id}-{uuid4().hex[:8]}"
                            snapshot = get_correction_job_manager().start_job(
                                job_id,
                                Path(output_root),
                                item_id,
                                corrected,
                                st.session_state.settings,
                            )
                            st.session_state.correction_job_id = snapshot.job_id
                            st.rerun()


def render_report_page() -> None:
    page_header("Report", "Search and filter the CSV report without losing technical status values.", None)
    if render_active_job_gate():
        return
    if render_active_correction_gate():
        return
    output_root = active_output_root()
    if not output_root:
        st.info("No report is available yet.")
        return
    report = read_report(Path(output_root) / "report.csv")
    if report.empty:
        st.info("No report is available yet.")
        return

    status_options = ["All", "OK", "REVIEW_REQUIRED", "DUPLICATE_TAG", "OCR_FAILED", "TAG_NOT_FOUND", "ERROR"]
    col_filter, col_search = st.columns([1, 2])
    selected = col_filter.selectbox("Filter", status_options)
    search = col_search.text_input("Search filename or tag")
    filtered = report.copy()
    if selected != "All":
        filtered = filtered[filtered["status"] == selected]
    if search:
        text = search.lower()
        filtered = filtered[
            filtered["original_filename"].str.lower().str.contains(text, na=False)
            | filtered["detected_tag_number"].str.lower().str.contains(text, na=False)
        ]
    display_report_table(filtered)


def render_download_page() -> None:
    page_header("Download Output", "Save full-quality files or prepare a lightweight 20 KB catalogue ZIP.", "Download")
    if render_active_job_gate():
        return
    if render_active_correction_gate():
        return
    output_root = active_output_root()
    if not output_root:
        st.info("Process a batch first.")
        return

    render_download_buttons(Path(output_root))


def render_download_buttons(output_root: Path, compact: bool = False) -> None:
    if output_archives_are_stale(output_root):
        with st.spinner("Refreshing downloads after review changes..."):
            paths = rebuild_output_archives(output_root)
    else:
        paths = rebuild_output_archives(output_root)
    server = get_download_server(str(PROJECT_ROOT))
    selected_folder = _selected_save_folder()

    if not compact:
        st.subheader("Save to a folder")
        st.caption("Choose a Windows folder once. Individual saves and Save All will use it for this session.")
        choose_col, open_col = st.columns(2)
        with choose_col:
            if st.button("Choose output folder", type="primary", width="stretch", key=f"choose-folder-{output_root}"):
                try:
                    picked = choose_output_folder(selected_folder or Path.home())
                    if picked is not None:
                        st.session_state.save_folder = str(picked)
                        selected_folder = picked
                        st.success("Output folder selected.")
                except LocalExportError as exc:
                    st.error(str(exc))
        with open_col:
            if st.button(
                "Open selected folder",
                width="stretch",
                disabled=selected_folder is None,
                key=f"open-selected-{output_root}",
            ):
                try:
                    open_folder(selected_folder)
                except LocalExportError as exc:
                    st.error(str(exc))

        if selected_folder is not None:
            st.caption("Selected save folder")
            st.code(str(selected_folder))
        else:
            st.info("Choose a folder to enable Save and Save All. Browser download buttons still work without it.")

        st.caption("Current batch folder")
        st.code(str(paths.root))
        if st.button("Open current batch folder", width="stretch", key=f"open-batch-{output_root}"):
            try:
                open_folder(paths.root)
            except LocalExportError as exc:
                st.error(str(exc))

        st.divider()
        st.subheader("20 KB catalogue ZIP")
        st.caption("Creates white-background JPG copies only. Full-quality PNG and transparent files stay untouched.")
        processed_count = sum(1 for path in paths.processed_images.iterdir() if path.is_file() and path.suffix.lower() == ".png")
        if st.button(
            "Prepare compressed ZIP (each image <= 20 KB)",
            type="primary",
            width="stretch",
            disabled=processed_count == 0,
            key=f"prepare-20kb-{output_root}",
        ):
            with st.spinner("Optimising white catalogue images without touching full-quality files..."):
                summary = prepare_compressed_export(
                    paths.processed_images,
                    paths.compressed_images_20kb,
                    paths.compressed_images_20kb_zip,
                )
                paths = rebuild_output_archives(output_root)
            if summary.ready:
                st.success(
                    f"20 KB ZIP ready: {summary.ready} image(s), "
                    f"{summary.converted} newly converted and {summary.reused} reused."
                )
            if summary.errors:
                st.warning(f"{summary.skipped} image(s) could not be compressed.")
                with st.expander("Compression details"):
                    for error in summary.errors:
                        st.write(error)

    compressed_ready = compressed_export_is_current(
        paths.processed_images,
        paths.compressed_images_20kb,
        paths.compressed_images_20kb_zip,
    )

    downloads = [
        ("Download full output ZIP", paths.full_zip, paths.full_zip.exists() and paths.report_csv.exists()),
        ("Download processed images ZIP", paths.processed_zip, any(paths.processed_images.glob("*.png"))),
        ("Download transparent images ZIP", paths.transparent_zip, any(paths.transparent_images.glob("*.png"))),
        ("Download compressed images ZIP (<=20 KB each)", paths.compressed_images_20kb_zip, compressed_ready),
        ("Download report.csv", paths.report_csv, paths.report_csv.exists() and paths.report_csv.stat().st_size > 0),
        ("Download debug crops ZIP", paths.debug_zip, any(paths.debug_crops.glob("*.png"))),
    ]

    available_artifacts = [path for _, path, available in downloads if available and path.is_file() and path.stat().st_size > 0]
    if not compact and selected_folder is not None and available_artifacts:
        if st.button("Save All to selected folder", type="primary", width="stretch", key=f"save-all-{output_root}"):
            try:
                bundle, saved = save_all_artifacts(available_artifacts, selected_folder)
                st.session_state.last_saved_folder = str(bundle)
                st.success(f"Saved {len(saved)} output file(s) to {bundle}")
            except LocalExportError as exc:
                st.error(str(exc))

    if not compact:
        st.divider()
        st.subheader("Available downloads")

    for label, path, available in downloads:
        if path.exists() and path.stat().st_size > 0 and available:
            if compact:
                _render_direct_download(server, label, path)
                continue
            link_col, save_col = st.columns([3, 1])
            with link_col:
                _render_direct_download(server, label, path)
            with save_col:
                if st.button(
                    "Save",
                    width="stretch",
                    disabled=selected_folder is None,
                    key=f"save-{path.name}-{output_root}",
                ):
                    try:
                        saved = save_artifact(path, selected_folder)
                        st.session_state.last_saved_folder = str(saved.parent)
                        st.success(f"Saved as {saved.name}")
                    except LocalExportError as exc:
                        st.error(str(exc))
        else:
            st.caption(f"{path.name} is not available yet.")

    if not compact and compressed_ready:
        largest = max((path.stat().st_size for path in paths.compressed_images_20kb.glob("*.jpg")), default=0)
        st.caption(
            f"Compressed catalogue verified: every JPG is <= {MAX_COMPRESSED_IMAGE_BYTES:,} bytes. "
            f"Largest file: {largest:,} bytes."
        )


def _selected_save_folder() -> Path | None:
    value = str(st.session_state.get("save_folder", "")).strip()
    if not value:
        return None
    folder = Path(value)
    if folder.is_dir():
        return folder
    st.session_state.save_folder = ""
    return None


def _render_direct_download(server, label: str, path: Path) -> None:
    href = download_url(server, path)
    st.markdown(
        f'<a class="direct-download" href="{html.escape(href, quote=True)}" '
        f'download="{html.escape(path.name, quote=True)}">'
        f'{html.escape(label)} ({file_size_label(path)})</a>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Sunaar Photo Tagger", page_icon=None, layout="wide")
    init_state()
    job_snapshot = sync_active_job_state()
    correction_snapshot = sync_correction_job_state()
    apply_theme()
    with st.sidebar:
        st.subheader("Sunaar Tagger")
        pages = ["Dashboard", "Upload Photos", "Settings", "Processing", "AI Review", "Review Required", "Report", "Download"]
        page = st.radio(
            "Navigate",
            pages,
            index=pages.index(st.session_state.active_page) if st.session_state.active_page in pages else 0,
        )
        st.session_state.active_page = page
        st.divider()
        st.caption("Upload -> Settings -> Process -> Review -> Download")
        if job_snapshot is not None and job_snapshot.running:
            st.info(f"Processing in background: {job_snapshot.done}/{job_snapshot.total}")
        if correction_snapshot is not None and correction_snapshot.running:
            st.info(f"Saving corrected tag: {correction_snapshot.corrected_tag}")

    correction_notice = str(st.session_state.get("correction_notice", "")).strip()
    if correction_notice:
        if st.session_state.get("correction_notice_error"):
            st.error(correction_notice)
        else:
            st.success(correction_notice)
        st.session_state.correction_notice = ""
        st.session_state.correction_notice_error = False
    if correction_snapshot is not None and correction_snapshot.running:
        render_correction_job_progress(correction_snapshot.job_id)

    if page == "Dashboard":
        if job_snapshot is not None and job_snapshot.running:
            st.info("Your active batch is continuing in the background while you use the dashboard.")
            render_processing_job_progress(job_snapshot.job_id)
        render_dashboard(PROJECT_ROOT, st.session_state.summary)
        st.write("")
        col_upload, col_process, col_download = st.columns(3)
        with col_upload:
            if st.button("Go to Upload Photos", type="primary", width="stretch"):
                st.session_state.active_page = "Upload Photos"
                st.rerun()
        with col_process:
            if st.button("Go to Processing", width="stretch"):
                st.session_state.active_page = "Processing"
                st.rerun()
        with col_download:
            if st.button("Go to Download", width="stretch"):
                st.session_state.active_page = "Download"
                st.rerun()
    elif page == "Upload Photos":
        render_upload_page()
    elif page == "Settings":
        render_settings_page()
    elif page == "Processing":
        render_processing_page()
    elif page == "AI Review":
        render_ai_review_page()
    elif page == "Review Required":
        render_review_page()
    elif page == "Report":
        render_report_page()
    elif page == "Download":
        render_download_page()


if __name__ == "__main__":
    main()
