# Sunaar Jewellery Photo Tagger

A Streamlit app that enhances jewellery photos, reads visible tag numbers, renames confident images, sends uncertain photos to review, creates `report.csv`, and exports downloadable ZIP files.

## What It Does

- Upload multiple images or a ZIP file.
- Keeps an active batch running in the background when the user opens another app page.
- Supports JPG, JPEG, PNG, WEBP, and HEIC when `pillow-heif` is available.
- Enhances each photo with OpenCV.
- Detects likely white/light tag areas and completes opposite-rotation OCR pairs before accepting a tag.
- Uses EasyOCR by default through a modular OCR wrapper.
- Removes backgrounds after OCR, then creates catalogue-style white PNGs and optional transparent PNGs.
- Aligns final product photos on a vertical `1200x1500` portrait canvas.
- Runs both local background models: BiRefNet supplies the clean final matte while `u2net` acts only as a conservative jewellery-preservation signal.
- Restores only narrow, connected, source-supported jewellery edges; it never lets broad U2Net wood or shadow residue overwrite the BiRefNet result.
- Saves confident white-background results as `processed_images/{tag_number}.png`.
- Saves transparent copies as `transparent_images/{tag_number}.png`.
- Sends unclear results to `review_required/REVIEW_{original}.png`.
- Sends unsafe background masks to `background_review/`.
- Lets the user manually correct OCR items and resolve background-review candidates.
- Generates `report.csv` with technical status values.
- Creates ZIP downloads for full output, processed images, transparent images, and debug crops.
- Lets the Windows user choose a save folder, save one artifact, or save every available artifact together.
- Creates an optional white-background JPEG ZIP with every included image strictly at or below `20,000` bytes.

## Install

```powershell
cd E:\Codex\2026-06-20\jewellery-photo-tagger
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

The first EasyOCR/rembg run may download OCR/background-removal model files.

## Run

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal.

## Folder Structure

```text
jewellery-photo-tagger/
  app.py
  requirements.txt
  requirements-lock.txt
  README.md
  assets/brand/
  src/
  tests/
  tools/
```

Each processing run creates a private runtime folder with:

```text
Jewellery_Output/
  processed_images/
  transparent_images/
  compressed_images_20kb/
  review_required/
  background_review/
  debug_crops/
  report.csv
  Jewellery_Output.zip
  processed_images.zip
  transparent_images.zip
  compressed_images_20kb.zip
  debug_crops.zip
```

The `compressed_images_20kb` export is created on demand from white catalogue PNGs. It never replaces or modifies the full-quality PNG or transparent output.

## Windows Package

Build the local-parity Windows ZIP from the canonical source tree:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_windows_package.ps1 `
  -WheelhouseSource E:\path\to\wheelhouse-py314-win-amd64
```

The complete offline package bundles the exact EasyOCR detector, English recognizer, U2Net preservation model, BiRefNet final-cut model, and Python 3.14 Windows dependency wheels used by the working local app. It also bundles the official signed Python 3.14.6 64-bit runtime. `Install or Repair Dependencies.bat` restores private Python when necessary and installs the pinned dependencies from the verified local wheel cache without contacting a package index. The launcher verifies Python, dependency, model, and app hashes before use.

## Review Required

Photos are sent to review when OCR fails, opposite rotations remain ambiguous, the tag is not found, confidence is below the threshold, or background cleanup could remove source-supported jewellery. OCR review shows the enhanced image, tag crop, raw OCR text, suggested tag, and a correction box.

Background review shows the complete enhanced original beside the white-background candidate. `Accept complete preview` promotes a visually confirmed candidate; `Keep original photo` preserves every jewellery part when the cutout is unsafe.

## Background Processing

After `Start processing now` is clicked, the batch belongs to its runtime output folder instead of the current Streamlit page run. Dashboard, Download, Settings, Report, and Review can be opened safely while processing continues. Those pages show live status and wait for completion before reading or rebuilding final archives, so navigation cannot restart or duplicate the batch.

Saving a correction moves the image into `processed_images`, updates `report.csv`, and rebuilds the output ZIP.

## report.csv

Columns:

- `original_filename`
- `detected_tag_number`
- `ocr_text_raw`
- `confidence_score`
- `final_filename`
- `output_folder`
- `status`
- `notes`
- `background_status`
- `background_mode`
- `transparent_filename`
- `background_notes`

Status values:

- `OK`
- `REVIEW_REQUIRED`
- `DUPLICATE_TAG`
- `OCR_FAILED`
- `TAG_NOT_FOUND`
- `ERROR`

## Limitations

- OCR accuracy depends on photo clarity.
- Very blurry or hidden tags may require manual review.
- Reflection and glare can reduce accuracy.
- The review folder is necessary to avoid wrong filenames.
- Difficult backgrounds can still require manual background review.
- 100% automatic accuracy cannot be guaranteed for poor-quality photos.
