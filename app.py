from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.download_server import download_url, start_download_server
from src.file_manager import (
    create_run_workspace,
    file_size_label,
    find_latest_output_root,
    materialize_uploaded_files,
    rebuild_output_archives,
)
from src.models import BatchSummary, ProcessingSettings
from src.ocr_engine import build_ocr_engine
from src.processor import BatchProcessor, apply_manual_correction
from src.report_generator import read_report
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
        "last_error": "",
        "active_page": "Dashboard",
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
    st.session_state.last_error = ""


@st.cache_resource(show_spinner=False)
def get_cached_ocr_engine():
    engine = build_ocr_engine(use_easyocr=True)
    if hasattr(engine, "warm_up"):
        engine.warm_up()
    return engine


@st.cache_resource(show_spinner=False)
def get_download_server():
    return start_download_server(PROJECT_ROOT)


def render_upload_page() -> None:
    page_header("Upload Photos", "Add jewellery photos or a ZIP file. Originals stay untouched.", "Upload")
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
        catalogue_layout_enabled=catalogue_layout_enabled,
        catalogue_canvas_width=1200,
        catalogue_canvas_height=1500,
    )
    st.success("Settings are ready. Fast production is the recommended daily mode.")


def render_processing_page() -> None:
    page_header("Processing", "Start once, then wait here. The app will show every stage clearly.", "Process")
    image_paths = st.session_state.image_paths
    if not image_paths:
        st.info("Upload photos first. Once files are prepared, the start button will appear here.")
        return

    process_panel(
        "Batch ready",
        f"{len(image_paths)} photo ready. Click Start processing once. First run loads OCR; after that normal photos should move much faster.",
        "ready",
    )
    st.metric("Photos in this batch", len(image_paths))
    progress = st.progress(0)
    current_file = st.empty()
    counters_box = st.empty()
    stage_box = st.empty()

    def progress_callback(done: int, total: int, filename: str, counters: dict[str, int]) -> None:
        progress.progress(0 if total == 0 else min(done / total, 1.0))
        current_file.info(f"Now working on: {filename}")
        counters_box.markdown(
            f"""
            **Live count:** Processed `{done}/{total}` - Ready `{counters.get('ok', 0)}` -
            Review `{counters.get('review', 0)}` - Duplicates `{counters.get('duplicates', 0)}` - Errors `{counters.get('errors', 0)}`
            """
        )

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

    if st.session_state.is_processing:
        st.warning("Processing is already running. Please stay on this screen until it finishes.")

    if st.button("Start processing now", type="primary", width="stretch", disabled=st.session_state.is_processing):
        try:
            st.session_state.is_processing = True
            st.session_state.last_error = ""
            stage_box.info("Starting OCR engine. First time can take extra seconds. Please wait on this screen.")
            with st.spinner("Loading OCR reader and models..."):
                ocr_engine = get_cached_ocr_engine()
            stage_box.success("OCR engine ready. Enhancing photo and reading tag now.")
            processor = BatchProcessor(Path(st.session_state.output_root), st.session_state.settings, ocr_engine, project_root=PROJECT_ROOT)
            summary = processor.process_images(image_paths, progress_callback=progress_callback)
            st.session_state.summary = summary
            st.session_state.processing_done = True
            st.session_state.is_processing = False
            progress.progress(1.0)
            stage_box.success("Processing completed. Check the result below.")
            st.success("Processing completed successfully.")
            st.rerun()
        except Exception as exc:
            st.session_state.last_error = f"Processing stopped: {exc}"
            stage_box.error(st.session_state.last_error)
        finally:
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


def render_review_page() -> None:
    page_header("Review Required", "Fix unclear tag numbers and move images into processed outputs.", "Review")
    output_root = active_output_root()
    if not output_root:
        st.info("No batch has been processed yet.")
        return

    report = read_report(Path(output_root) / "report.csv")
    review_statuses = {"REVIEW_REQUIRED", "OCR_FAILED", "TAG_NOT_FOUND"}
    review_rows = report[report["status"].isin(review_statuses)] if not report.empty else report
    if review_rows.empty:
        st.success("No review items are waiting.")
        return

    for _, row in review_rows.iterrows():
        original = row["original_filename"]
        with st.expander(f"{original} - {FRIENDLY_STATUS.get(row['status'], row['status'])}", expanded=False):
            col_img, col_info = st.columns([1, 1])
            image_path = Path(output_root) / row["output_folder"] / row["final_filename"]
            crop_path = Path(output_root) / "debug_crops" / f"{Path(original).stem}_tag_crop.png"
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
                corrected = st.text_input("Correct tag number", value=row.get("detected_tag_number", ""), key=f"correct_{original}")
                if st.button("Save correction", key=f"save_{original}"):
                    ok, message = apply_manual_correction(Path(output_root), original, corrected)
                    if ok:
                        st.success(message)
                        st.rerun()
                    else:
                        st.warning(message)


def render_report_page() -> None:
    page_header("Report", "Search and filter the CSV report without losing technical status values.", None)
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
    page_header("Download Output", "Download final images, report, and debug crops.", "Download")
    output_root = active_output_root()
    if not output_root:
        st.info("Process a batch first.")
        return

    render_download_buttons(Path(output_root))


def render_download_buttons(output_root: Path, compact: bool = False) -> None:
    paths = rebuild_output_archives(output_root)
    server = get_download_server()
    if not compact:
        st.caption("Output folder")
        st.code(str(paths.root))
        if st.button("Open output folder", type="primary", width="stretch"):
            os.startfile(str(paths.root))
            st.success("Output folder opened.")

    downloads = [
        ("Download full output ZIP", paths.full_zip, lambda: paths.full_zip.exists() and paths.report_csv.exists()),
        ("Download processed images ZIP", paths.processed_zip, lambda: any(paths.processed_images.glob("*.png"))),
        ("Download transparent images ZIP", paths.transparent_zip, lambda: any(paths.transparent_images.glob("*.png"))),
        ("Download report.csv", paths.report_csv, lambda: paths.report_csv.exists() and paths.report_csv.stat().st_size > 0),
        ("Download debug crops ZIP", paths.debug_zip, lambda: any(paths.debug_crops.glob("*.png"))),
    ]
    for label, path, is_available in downloads:
        if path.exists() and path.stat().st_size > 0 and is_available():
            href = download_url(server, path)
            st.markdown(
                f'<a class="direct-download" href="{href}" download="{path.name}">{label} ({file_size_label(path)})</a>',
                unsafe_allow_html=True,
            )
        else:
            st.caption(f"{path.name} is not available yet.")


def main() -> None:
    st.set_page_config(page_title="Sunaar Photo Tagger", page_icon=None, layout="wide")
    init_state()
    apply_theme()
    with st.sidebar:
        st.subheader("Sunaar Tagger")
        page = st.radio(
            "Navigate",
            ["Dashboard", "Upload Photos", "Settings", "Processing", "Review Required", "Report", "Download"],
            index=["Dashboard", "Upload Photos", "Settings", "Processing", "Review Required", "Report", "Download"].index(
                st.session_state.active_page
            ),
        )
        st.session_state.active_page = page
        st.divider()
        st.caption("Upload -> Settings -> Process -> Review -> Download")

    if page == "Dashboard":
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
    elif page == "Review Required":
        render_review_page()
    elif page == "Report":
        render_report_page()
    elif page == "Download":
        render_download_page()


if __name__ == "__main__":
    main()
