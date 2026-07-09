from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import streamlit as st

from .models import (
    STATUS_DUPLICATE_TAG,
    STATUS_ERROR,
    STATUS_OCR_FAILED,
    STATUS_OK,
    STATUS_REVIEW_REQUIRED,
    STATUS_TAG_NOT_FOUND,
)


FRIENDLY_STATUS = {
    STATUS_OK: "Ready",
    STATUS_REVIEW_REQUIRED: "Needs review",
    STATUS_DUPLICATE_TAG: "Duplicate saved",
    STATUS_OCR_FAILED: "Could not read tag",
    STATUS_TAG_NOT_FOUND: "Tag not found",
    STATUS_ERROR: "Could not process",
}


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #f7f5f1;
          --panel: rgba(255, 255, 255, 0.84);
          --ink: #15120f;
          --muted: #6d665c;
          --border: rgba(41, 35, 28, 0.10);
          --gold: #b78935;
          --deep: #102d28;
          --rose: #c43b74;
        }
        .stApp { background: radial-gradient(circle at 18% 0%, #fff9ed 0, #f7f5f1 34%, #f2f0ec 100%); color: var(--ink); }
        section[data-testid="stSidebar"] { background: rgba(255,255,255,0.74); border-right: 1px solid var(--border); }
        section[data-testid="stSidebar"] * { color: #28231e; }
        .block-container { padding-top: 1.3rem; max-width: 1320px; }
        h1, h2, h3 { letter-spacing: 0; }
        div[data-testid="stMetric"] {
          background: rgba(255,255,255,0.74);
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 16px;
          box-shadow: 0 14px 40px rgba(45, 34, 20, 0.06);
        }
        .stButton > button[kind="primary"] {
          background: linear-gradient(135deg, #102d28 0%, #b78935 100%);
          border: 1px solid rgba(183,137,53,0.28);
          color: #fff;
          min-height: 48px;
          border-radius: 8px;
          box-shadow: 0 14px 34px rgba(16,45,40,0.18);
        }
        .stButton > button[kind="primary"]:hover {
          border-color: rgba(183,137,53,0.62);
          filter: brightness(1.03);
        }
        .linear-card {
          background: var(--panel);
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 18px;
          box-shadow: 0 16px 48px rgba(31, 25, 18, 0.07);
          backdrop-filter: blur(18px);
        }
        .small-muted { color: var(--muted); font-size: 0.92rem; }
        .process-panel {
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 16px 18px;
          background: rgba(255,255,255,0.82);
          box-shadow: 0 14px 42px rgba(41, 35, 28, 0.06);
          margin: 8px 0 16px;
        }
        .process-panel.ready { border-color: rgba(183,137,53,0.24); }
        .process-panel strong { display: block; font-size: 1rem; margin-bottom: 4px; }
        .process-panel span { color: var(--muted); font-size: 0.94rem; line-height: 1.5; }
        .dashboard-canvas {
          position: relative;
          overflow: hidden;
          color: var(--ink);
        }
        .hero-shell {
          position: relative;
          overflow: hidden;
          min-height: 630px;
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 38px 40px;
          background:
            linear-gradient(110deg, rgba(16,45,40,0.95) 0%, rgba(16,45,40,0.78) 42%, rgba(255,255,255,0.2) 100%),
            var(--hero-image);
          background-size: cover;
          background-position: center;
          box-shadow: 0 28px 80px rgba(16,45,40,0.18);
        }
        .hero-grid { display: grid; grid-template-columns: 1fr; gap: 24px; align-items: center; }
        .hero-copy { color: white; z-index: 2; }
        .hero-copy .eyebrow { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.14em; color: rgba(255,255,255,0.72); }
        .hero-copy h1 { font-size: 3.55rem; line-height: 1.04; margin: 10px 0 14px; color: white; }
        .hero-copy p { max-width: 540px; color: rgba(255,255,255,0.78); font-size: 1.02rem; line-height: 1.6; }
        .hero-stats { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }
        .hero-pill {
          border: 1px solid rgba(255,255,255,0.24);
          background: rgba(255,255,255,0.13);
          color: white;
          border-radius: 999px;
          padding: 10px 13px;
          font-size: 0.85rem;
          backdrop-filter: blur(16px);
        }
        .stage { position: relative; min-height: 430px; perspective: 1200px; }
        .demo-label {
          position: absolute;
          top: 0;
          left: 0;
          z-index: 4;
          border: 1px solid rgba(255,255,255,0.26);
          border-radius: 999px;
          padding: 8px 11px;
          background: rgba(255,255,255,0.14);
          color: rgba(255,255,255,0.86);
          font-size: 0.72rem;
          font-weight: 700;
          backdrop-filter: blur(18px);
        }
        .photo-card {
          position: absolute;
          width: min(68%, 390px);
          left: 4%;
          top: 42px;
          border-radius: 8px;
          overflow: hidden;
          border: 1px solid rgba(255,255,255,0.35);
          box-shadow: 0 34px 90px rgba(0,0,0,0.32);
          transform-style: preserve-3d;
          animation: floatMain 7s ease-in-out infinite;
          background: rgba(255,255,255,0.15);
        }
        .photo-card.secondary { width: 31%; left: 61%; top: 16px; animation: floatSmall 6.4s ease-in-out infinite; }
        .photo-card.tertiary { width: 29%; left: 63%; top: 245px; animation: floatSmall 7.4s ease-in-out infinite reverse; }
        .photo-card img { display: block; width: 100%; height: 100%; object-fit: cover; }
        .glass-panel {
          position: absolute;
          left: 0;
          right: 0;
          bottom: 18px;
          border-radius: 8px;
          border: 1px solid rgba(255,255,255,0.28);
          background: rgba(255,255,255,0.16);
          color: white;
          padding: 12px;
          backdrop-filter: blur(22px);
          box-shadow: 0 20px 70px rgba(0,0,0,0.18);
        }
        .workflow-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; }
        .workflow-step {
          min-height: 60px;
          border: 1px solid rgba(255,255,255,0.2);
          border-radius: 8px;
          padding: 8px;
          background: rgba(255,255,255,0.1);
          font-size: 0.72rem;
          color: rgba(255,255,255,0.78);
        }
        .workflow-step strong { display: block; color: #fff; font-size: 0.82rem; margin-bottom: 2px; }
        .shine {
          position: absolute;
          inset: -40%;
          background: linear-gradient(105deg, transparent 32%, rgba(255,255,255,0.22) 48%, transparent 62%);
          animation: shine 5.8s ease-in-out infinite;
          pointer-events: none;
        }
        .scroll-cue {
          position: absolute;
          left: 48px;
          bottom: 32px;
          color: rgba(255,255,255,0.78);
          font-size: 0.86rem;
          display: inline-flex;
          align-items: center;
          gap: 10px;
        }
        .scroll-cue .line {
          width: 42px;
          height: 1px;
          background: rgba(255,255,255,0.45);
          animation: cuePulse 1.8s ease-in-out infinite;
        }
        .story-wrap {
          margin-top: 34px;
          display: grid;
          grid-template-columns: 1fr;
          gap: 34px;
          align-items: start;
        }
        .story-copy {
          display: flex;
          flex-direction: column;
          gap: 22px;
        }
        .story-step {
          min-height: 440px;
          display: flex;
          align-items: center;
          opacity: 0.48;
          transform: translateY(24px);
          animation: storyReveal both linear;
          animation-timeline: view();
          animation-range: entry 8% cover 44%;
        }
        .story-step-inner {
          border: 1px solid var(--border);
          border-radius: 8px;
          background: rgba(255,255,255,0.72);
          padding: 26px;
          box-shadow: 0 22px 70px rgba(41, 35, 28, 0.08);
        }
        .story-step .kicker {
          color: var(--gold);
          font-size: 0.76rem;
          text-transform: uppercase;
          letter-spacing: 0.14em;
          font-weight: 700;
          margin-bottom: 12px;
        }
        .story-step h2 {
          margin: 0 0 12px;
          font-size: 2.35rem;
          line-height: 1.08;
          color: var(--ink);
        }
        .story-step p {
          color: var(--muted);
          line-height: 1.65;
          margin: 0;
          font-size: 1rem;
        }
        .story-sticky {
          position: relative;
          top: auto;
          min-height: 620px;
          border: 1px solid var(--border);
          border-radius: 8px;
          background:
            linear-gradient(145deg, rgba(255,255,255,0.9), rgba(246,241,231,0.78)),
            var(--sticky-image);
          background-size: cover;
          background-position: center;
          overflow: hidden;
          box-shadow: 0 34px 100px rgba(40, 31, 19, 0.16);
        }
        .story-sticky::before {
          content: "";
          position: absolute;
          inset: 0;
          background: linear-gradient(180deg, rgba(16,45,40,0.08), rgba(16,45,40,0.72));
        }
        .device {
          position: absolute;
          inset: 40px 40px 40px 40px;
          min-height: auto;
          border-radius: 8px;
          border: 1px solid rgba(255,255,255,0.42);
          background: rgba(255,255,255,0.18);
          backdrop-filter: blur(24px);
          box-shadow: 0 26px 80px rgba(0,0,0,0.2);
          overflow: hidden;
          animation: deviceFloat 8s ease-in-out infinite;
        }
        .device-top {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 14px 16px;
          color: rgba(255,255,255,0.86);
          font-size: 0.82rem;
          border-bottom: 1px solid rgba(255,255,255,0.2);
          background: rgba(16,45,40,0.32);
        }
        .device-body {
          padding: 18px;
          color: white;
        }
        .mock-preview {
          height: 150px;
          border-radius: 8px;
          overflow: hidden;
          position: relative;
          border: 1px solid rgba(255,255,255,0.25);
          background: rgba(255,255,255,0.12);
        }
        .mock-preview img {
          width: 100%;
          height: 100%;
          object-fit: cover;
          filter: saturate(1.02) contrast(1.02);
        }
        .mock-tag {
          position: absolute;
          right: 18px;
          top: 18px;
          background: rgba(255,255,255,0.92);
          color: #15120f;
          border-radius: 8px;
          padding: 10px 14px;
          font-size: 1.42rem;
          font-weight: 800;
          letter-spacing: 0;
          box-shadow: 0 14px 34px rgba(0,0,0,0.16);
          animation: tagScan 3.6s ease-in-out infinite;
        }
        .scan-line {
          position: absolute;
          left: 0;
          right: 0;
          top: 0;
          height: 2px;
          background: rgba(255,255,255,0.92);
          box-shadow: 0 0 22px rgba(255,255,255,0.82);
          animation: scan 3.6s ease-in-out infinite;
        }
        .pipeline {
          display: grid;
          gap: 8px;
          margin-top: 14px;
        }
        .pipeline-row {
          display: grid;
          grid-template-columns: 34px 1fr auto;
          align-items: center;
          gap: 10px;
          padding: 9px;
          border-radius: 8px;
          background: rgba(255,255,255,0.13);
          border: 1px solid rgba(255,255,255,0.16);
          color: rgba(255,255,255,0.84);
          animation: rowGlow 6s ease-in-out infinite;
        }
        .pipeline-row:nth-child(2) { animation-delay: 0.5s; }
        .pipeline-row:nth-child(3) { animation-delay: 1s; }
        .pipeline-row:nth-child(4) { animation-delay: 1.5s; }
        .pipeline-row:nth-child(5) { animation-delay: 2s; }
        .pipeline-icon {
          width: 32px;
          height: 32px;
          display: grid;
          place-items: center;
          border-radius: 8px;
          background: rgba(255,255,255,0.18);
          color: #fff;
          font-weight: 800;
        }
        .pipeline-row strong { color: #fff; display: block; font-size: 0.92rem; }
        .pipeline-row span { font-size: 0.78rem; color: rgba(255,255,255,0.7); }
        .pipeline-row em { font-style: normal; font-size: 0.76rem; color: #fff; }
        .dashboard-final {
          margin-top: 34px;
          border-radius: 8px;
          border: 1px solid var(--border);
          background:
            linear-gradient(120deg, rgba(16,45,40,0.94), rgba(183,137,53,0.82)),
            var(--final-image);
          background-size: cover;
          background-position: center;
          color: white;
          padding: 44px;
          box-shadow: 0 28px 90px rgba(16,45,40,0.14);
        }
        .dashboard-final h2 { color: white; font-size: 2.7rem; margin: 0 0 12px; line-height: 1.08; }
        .dashboard-final p { max-width: 720px; color: rgba(255,255,255,0.78); line-height: 1.6; margin: 0; }
        .dashboard-actions {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 12px;
          margin-top: 24px;
        }
        .dashboard-action {
          border: 1px solid rgba(255,255,255,0.22);
          background: rgba(255,255,255,0.12);
          border-radius: 8px;
          padding: 16px;
          backdrop-filter: blur(16px);
        }
        .dashboard-action strong { display: block; margin-bottom: 5px; }
        .dashboard-action span { color: rgba(255,255,255,0.72); font-size: 0.9rem; }
        .badge {
          display: inline-flex;
          align-items: center;
          border-radius: 999px;
          padding: 4px 9px;
          font-size: 0.78rem;
          border: 1px solid var(--border);
          background: #fff;
        }
        .badge.ok { color: #12613a; background: #edf8f1; border-color: #d3ecd9; }
        .badge.review { color: #835400; background: #fff8e7; border-color: #f4dfaa; }
        .badge.error { color: #9a1d23; background: #fff0f0; border-color: #f3caca; }
        .badge.duplicate { color: #7148a7; background: #f5f0ff; border-color: #decdfd; }
        .stepper { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin: 8px 0 18px; }
        .stepper .step {
          border: 1px solid var(--border);
          background: rgba(255,255,255,0.72);
          border-radius: 8px;
          padding: 12px;
          font-size: 0.9rem;
          color: var(--muted);
        }
        .stepper .step.active { color: var(--ink); border-color: rgba(183,137,53,0.45); box-shadow: inset 0 0 0 1px rgba(183,137,53,0.12); }
        @keyframes floatMain {
          0%, 100% { transform: translate3d(0, 0, 0) rotateX(0deg) rotateY(-4deg); }
          50% { transform: translate3d(0, -16px, 26px) rotateX(2deg) rotateY(3deg); }
        }
        @keyframes floatSmall {
          0%, 100% { transform: translate3d(0, 0, 40px) rotateZ(0deg); }
          50% { transform: translate3d(0, 12px, 80px) rotateZ(2deg); }
        }
        @keyframes shine {
          0%, 55% { transform: translateX(-85%); opacity: 0; }
          65% { opacity: 1; }
          100% { transform: translateX(85%); opacity: 0; }
        }
        @keyframes cuePulse {
          0%, 100% { transform: scaleX(0.4); transform-origin: left; opacity: 0.5; }
          50% { transform: scaleX(1); opacity: 1; }
        }
        @keyframes storyReveal {
          from { opacity: 0.28; transform: translateY(44px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes deviceFloat {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-12px); }
        }
        @keyframes scan {
          0%, 100% { transform: translateY(14px); opacity: 0; }
          18%, 80% { opacity: 1; }
          50% { transform: translateY(168px); }
        }
        @keyframes tagScan {
          0%, 100% { box-shadow: 0 14px 34px rgba(0,0,0,0.16); }
          50% { box-shadow: 0 0 0 4px rgba(183,137,53,0.22), 0 20px 40px rgba(0,0,0,0.2); }
        }
        @keyframes rowGlow {
          0%, 100% { background: rgba(255,255,255,0.12); transform: translateX(0); }
          50% { background: rgba(255,255,255,0.22); transform: translateX(4px); }
        }
        @media (min-width: 1250px) {
          .hero-grid { grid-template-columns: minmax(280px, 0.95fr) minmax(320px, 1.05fr); }
          .story-wrap { grid-template-columns: minmax(340px, 0.88fr) minmax(420px, 1.12fr); }
          .story-sticky { position: sticky; top: 28px; }
        }
        @media (max-width: 900px) {
          .hero-copy h1 { font-size: 2.7rem; }
          .hero-grid { grid-template-columns: 1fr; }
          .hero-shell { padding: 24px; min-height: auto; }
          .stage { min-height: 390px; }
          .story-sticky { min-height: 620px; }
          .story-step { min-height: auto; }
          .dashboard-actions { grid-template-columns: 1fr; }
          .dashboard-final h2 { font-size: 2.1rem; }
          .workflow-row, .stepper { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def brand_asset_paths(project_root: Path) -> list[Path]:
    brand_dir = project_root / "assets" / "brand"
    dashboard_dir = brand_dir / "dashboard"
    preferred_dir = dashboard_dir if dashboard_dir.exists() else brand_dir
    return sorted([path for path in preferred_dir.glob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])


def render_stepper(active: str) -> None:
    steps = ["Upload", "Settings", "Process", "Review", "Download"]
    html = '<div class="stepper">'
    for step in steps:
        cls = "step active" if step == active else "step"
        html += f'<div class="{cls}"><strong>{step}</strong><br><span>{steps.index(step) + 1} of 5</span></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_dashboard(project_root: Path, summary=None) -> None:
    assets = brand_asset_paths(project_root)
    style = "--hero-image: linear-gradient(120deg, #102d28, #b78935);"
    sticky_style = "--sticky-image: linear-gradient(120deg, #102d28, #f6efe0);"
    final_style = "--final-image: linear-gradient(120deg, #102d28, #b78935);"
    if assets:
        style = f'--hero-image: url("{_data_uri(assets[0])}");'
        sticky_style = f'--sticky-image: url("{_data_uri(assets[min(1, len(assets) - 1)])}");'
        final_style = f'--final-image: url("{_data_uri(assets[min(2, len(assets) - 1)])}");'
    processed = getattr(summary, "processed", 0) if summary else 0
    ok = getattr(summary, "ok", 0) if summary else 0
    review = getattr(summary, "review_required", 0) if summary else 0
    elapsed = getattr(summary, "elapsed_seconds", 0.0) if summary else 0.0
    st.markdown(
        f"""
        <div class="dashboard-canvas">
          <div class="hero-shell" style='{style}'>
            <div class="hero-grid">
              <div class="hero-copy">
                <div class="eyebrow">Sunaar photo tagging studio</div>
                <h1>Photo tagging ka poora system, smooth aur safe.</h1>
                <p>Upload photos, app tag number read karega, image ko number se rename karega, aur unclear photos ko review me rakhega. Neeche scroll karo, poora flow Apple-style guide me samjho.</p>
                <div class="hero-stats">
                  <div class="hero-pill">Processed: {processed}</div>
                  <div class="hero-pill">Ready: {ok}</div>
                  <div class="hero-pill">Needs review: {review}</div>
                  <div class="hero-pill">Time: {elapsed:.1f}s</div>
                </div>
              </div>
              <div class="stage">
                <div class="demo-label">Demo workflow preview</div>
                <div class="photo-card"><div style="height:270px;background:linear-gradient(135deg,rgba(255,255,255,.2),rgba(183,137,53,.45));display:grid;place-items:center;color:white;font-weight:800;font-size:2rem;">121134</div></div>
                <div class="photo-card secondary"><div style="height:160px;background:rgba(255,255,255,.22);display:grid;place-items:center;color:white;font-weight:700;">OCR</div></div>
                <div class="photo-card tertiary"><div style="height:145px;background:rgba(255,255,255,.18);display:grid;place-items:center;color:white;font-weight:700;">ZIP</div></div>
                <div class="glass-panel">
                  <div class="workflow-row">
                    <div class="workflow-step"><strong>Upload</strong>Photos or ZIP</div>
                    <div class="workflow-step"><strong>Enhance</strong>Photo quality</div>
                    <div class="workflow-step"><strong>Read</strong>Tag number</div>
                    <div class="workflow-step"><strong>Rename</strong>121134.png</div>
                    <div class="workflow-step"><strong>Download</strong>Final ZIP</div>
                  </div>
                </div>
                <div class="shine"></div>
              </div>
            </div>
            <div class="scroll-cue"><span class="line"></span><span>Scroll karke workflow dekho</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if assets:
        st.write("")
        cols = st.columns(min(3, len(assets)))
        for index, path in enumerate(assets[:3]):
            with cols[index % len(cols)]:
                st.image(str(path), width="stretch")

    left, right = st.columns([0.92, 1.08], gap="large")
    steps = [
        ("Step 01", "Photos upload karo.", "Single photo, multiple photos, ya poora folder ZIP bana ke upload karo. App originals ko touch nahi karta, sirf output folder me final files banata hai."),
        ("Step 02", "Settings simple rakho.", "Enhancement on rakho, debug crops on rakho, aur confidence threshold default par chhodo. Duplicate tags automatically suffix ke saath save honge."),
        ("Step 03", "Start once. Wait calmly.", "Start processing ek baar dabao. First run me OCR model load ho sakta hai. Uske baad app enhance, crop, rotate aur OCR run karega."),
        ("Step 04", "Unclear tag review me jayega.", "Agar tag blurry, glare wala, ya low-confidence ho, app galat filename nahi banayega. Review screen me manually tag save kar sakte ho."),
        ("Step 05", "Final ZIP download karo.", "Processed images, report.csv, debug crops aur full output ZIP ready milta hai. Report me technical status bhi safe record hota hai."),
    ]
    with left:
        for kicker, title, body in steps:
            st.markdown(
                f"""
                <div class="story-step">
                  <div class="story-step-inner">
                    <div class="kicker">{kicker}</div>
                    <h2>{title}</h2>
                    <p>{body}</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with right:
        st.markdown(
            f"""
            <div class="story-sticky" style='{sticky_style}'>
              <div class="device">
                <div class="device-top"><span>Sunaar Tagger</span><span>Live workflow</span></div>
                <div class="device-body">
                  <div class="mock-preview">
                    <div class="mock-tag">121134</div>
                    <div class="scan-line"></div>
                  </div>
                  <div class="pipeline">
                    <div class="pipeline-row"><div class="pipeline-icon">1</div><div><strong>Upload ready</strong><span>JPG, PNG, WEBP, HEIC, ZIP</span></div><em>OK</em></div>
                    <div class="pipeline-row"><div class="pipeline-icon">2</div><div><strong>Enhance photo</strong><span>Clean sharp output PNG</span></div><em>ON</em></div>
                    <div class="pipeline-row"><div class="pipeline-icon">3</div><div><strong>Read tag</strong><span>OCR with rotations</span></div><em>OCR</em></div>
                    <div class="pipeline-row"><div class="pipeline-icon">4</div><div><strong>Rename safely</strong><span>121134.png or 121134_2.png</span></div><em>SAFE</em></div>
                    <div class="pipeline-row"><div class="pipeline-icon">5</div><div><strong>Review if unclear</strong><span>No risky wrong filenames</span></div><em>REVIEW</em></div>
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="dashboard-final" style='{final_style}'>
          <h2>Ab kaam seedha hai: Upload se start karo.</h2>
          <p>Dashboard sirf guide hai. Real kaam left sidebar ke Upload Photos page se start hota hai. Agar output me koi tag unclear ho, Review Required me manually correct kar dena.</p>
          <div class="dashboard-actions">
            <div class="dashboard-action"><strong>1. Upload Photos</strong><span>Photos select karo ya ZIP upload karo.</span></div>
            <div class="dashboard-action"><strong>2. Start Processing</strong><span>Processing page par ek baar start dabao.</span></div>
            <div class="dashboard-action"><strong>3. Download Output</strong><span>Final ZIP aur report.csv download karo.</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(status: str) -> str:
    friendly = FRIENDLY_STATUS.get(status, status)
    if status == STATUS_OK:
        cls = "ok"
    elif status == STATUS_DUPLICATE_TAG:
        cls = "duplicate"
    elif status in {STATUS_ERROR, STATUS_OCR_FAILED}:
        cls = "error"
    else:
        cls = "review"
    return f'<span class="badge {cls}">{friendly}</span>'


def render_status_badge(status: str) -> None:
    st.markdown(status_badge(status), unsafe_allow_html=True)


def display_report_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No report is available yet.")
        return
    display = frame.copy()
    display.insert(0, "result", display["status"].map(lambda value: FRIENDLY_STATUS.get(str(value), str(value))))
    st.dataframe(display, width="stretch", hide_index=True)


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="linear-card"><h3>{title}</h3><p class="small-muted">{body}</p></div>', unsafe_allow_html=True)


def process_panel(title: str, body: str, tone: str = "ready") -> None:
    st.markdown(
        f'<div class="process-panel {tone}"><strong>{title}</strong><span>{body}</span></div>',
        unsafe_allow_html=True,
    )
