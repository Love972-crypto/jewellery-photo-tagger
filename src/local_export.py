from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable


class LocalExportError(RuntimeError):
    pass


def choose_output_folder(initial_folder: Path | None = None) -> Path | None:
    if os.name != "nt":
        raise LocalExportError("Native folder selection is available in the Windows app.")

    picker_script = r'''
import sys
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
selected = filedialog.askdirectory(
    parent=root,
    initialdir=sys.argv[1] if len(sys.argv) > 1 else "",
    mustexist=True,
    title="Choose where Sunaar output should be saved",
)
root.destroy()
if selected:
    print(selected)
'''
    startupinfo = None
    creationflags = 0
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    command = [sys.executable, "-c", picker_script]
    if initial_folder and initial_folder.exists():
        command.append(str(initial_folder))
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        startupinfo=startupinfo,
        creationflags=creationflags,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "Windows could not open the folder picker."
        raise LocalExportError(detail)
    selected = result.stdout.strip()
    if not selected:
        return None
    return validate_destination_folder(Path(selected))


def validate_destination_folder(folder: Path) -> Path:
    try:
        resolved = folder.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LocalExportError("The selected folder is no longer available.") from exc
    if not resolved.is_dir():
        raise LocalExportError("The selected location is not a folder.")
    try:
        with tempfile.NamedTemporaryFile(prefix=".sunaar_write_test_", dir=resolved, delete=True):
            pass
    except OSError as exc:
        raise LocalExportError("Sunaar cannot write to the selected folder. Choose another folder.") from exc
    return resolved


def save_artifact(source: Path, destination_folder: Path) -> Path:
    if not source.is_file() or source.stat().st_size == 0:
        raise LocalExportError(f"{source.name} is not available yet.")
    destination = validate_destination_folder(destination_folder)
    target = unique_destination(destination, source.name)
    try:
        shutil.copy2(source, target)
    except OSError as exc:
        raise LocalExportError(f"Could not save {source.name} to the selected folder.") from exc
    return target


def save_all_artifacts(
    sources: Iterable[Path],
    destination_folder: Path,
    folder_name: str | None = None,
) -> tuple[Path, list[Path]]:
    destination = validate_destination_folder(destination_folder)
    available = [path for path in sources if path.is_file() and path.stat().st_size > 0]
    if not available:
        raise LocalExportError("No output files are ready to save.")

    base_name = folder_name or f"Sunaar_Downloads_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    bundle = unique_destination(destination, base_name, is_directory=True)
    bundle.mkdir(parents=False)
    saved: list[Path] = []
    try:
        for source in available:
            target = unique_destination(bundle, source.name)
            shutil.copy2(source, target)
            saved.append(target)
    except OSError as exc:
        shutil.rmtree(bundle, ignore_errors=True)
        raise LocalExportError("Could not save every output file to the selected folder.") from exc
    return bundle, saved


def unique_destination(folder: Path, name: str, is_directory: bool = False) -> Path:
    candidate = folder / name
    if not candidate.exists():
        return candidate

    source = Path(name)
    stem = source.name if is_directory else source.stem
    suffix = "" if is_directory else source.suffix
    counter = 2
    while True:
        candidate = folder / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def open_folder(folder: Path) -> None:
    validated = validate_destination_folder(folder)
    if os.name != "nt" or not hasattr(os, "startfile"):
        raise LocalExportError("Opening folders is available in the Windows app.")
    os.startfile(str(validated))
