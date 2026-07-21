from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def download_required_models(model_cache: Path) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    model_cache = model_cache.resolve()
    easyocr_root = model_cache / "EasyOCR"
    easyocr_models = easyocr_root / "model"
    easyocr_user_network = easyocr_root / "user_network"
    u2net_root = model_cache / "u2net"
    easyocr_models.mkdir(parents=True, exist_ok=True)
    easyocr_user_network.mkdir(parents=True, exist_ok=True)
    u2net_root.mkdir(parents=True, exist_ok=True)

    os.environ["EASYOCR_MODULE_PATH"] = str(easyocr_root)
    os.environ["U2NET_HOME"] = str(u2net_root)

    import easyocr
    from rembg import new_session

    print("Downloading and validating EasyOCR detector and recognizer...")
    easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=str(easyocr_models),
        user_network_directory=str(easyocr_user_network),
        download_enabled=True,
        verbose=False,
    )
    print("Downloading and validating u2net background-removal model...")
    new_session("u2net")
    print("Downloading and validating BiRefNet final-cut model...")
    new_session("birefnet-general-lite")
    print("Required AI models are available.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cache", type=Path, required=True)
    args = parser.parse_args()
    download_required_models(args.model_cache)


if __name__ == "__main__":
    main()
