# Sunaar Jewellery Photo Tagger

A Streamlit app that enhances jewellery photos, reads visible tag numbers, renames confident images, sends uncertain photos to review, creates `report.csv`, and exports downloadable ZIP files.

## What It Does

- Upload multiple images or a ZIP file.
- Supports JPG, JPEG, PNG, WEBP, and HEIC when `pillow-heif` is available.
- Enhances each photo with OpenCV.
- Detects likely white/light tag areas and tries OCR across rotations.
- Uses EasyOCR by default through a modular OCR wrapper.
- Removes backgrounds after OCR, then creates catalogue-style white PNGs and optional transparent PNGs.
- Aligns final product photos on a vertical `1200x1500` portrait canvas.
- Uses an object-safe `u2net` background model so jewellery details are not erased from the center.
- Saves confident white-background results as `processed_images/{tag_number}.png`.
- Saves transparent copies as `transparent_images/{tag_number}.png`.
- Sends unclear results to `review_required/REVIEW_{original}.png`.
- Sends unsafe background masks to `background_review/`.
- Lets the user manually correct review items.
- Generates `report.csv` with technical status values.
- Creates ZIP downloads for full output, processed images, transparent images, and debug crops.

## Install

```powershell
cd E:\Codex\2026-06-20\jewellery-photo-tagger
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
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
  review_required/
  background_review/
  debug_crops/
  report.csv
  Jewellery_Output.zip
  processed_images.zip
  transparent_images.zip
  debug_crops.zip
```

## Review Required

Photos are sent to review when OCR fails, the tag is not found, or confidence is below the threshold. The review screen shows the enhanced image, tag crop, raw OCR text, suggested tag, and a correction box.

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
