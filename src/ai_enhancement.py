from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .image_enhancement import load_image, save_png


def realesrgan_available(project_root: Path) -> bool:
    if os.environ.get("SUNAAR_ENABLE_REALESRGAN") != "1":
        return False
    executable, model_dir = _realesrgan_paths(project_root)
    return executable.exists() and model_dir.exists()


def upscale_safely(image: np.ndarray, scale: int = 2) -> tuple[np.ndarray, str]:
    """Deterministic HD upscale for jewellery photos. No AI tiling, no hallucinated texture."""
    if image is None or image.size == 0:
        return image, "Safe HD skipped because image was empty."

    height, width = image.shape[:2]
    upscaled = cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_LANCZOS4)

    lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.25, tileGridSize=(8, 8))
    l_channel = cv2.addWeighted(l_channel, 0.76, clahe.apply(l_channel), 0.24, 0)
    upscaled = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(upscaled, (0, 0), 0.65)
    upscaled = cv2.addWeighted(upscaled, 1.16, blur, -0.16, 0)
    return upscaled, f"Safe HD x{scale} applied without AI tiling."


def upscale_with_realesrgan(
    image: np.ndarray,
    project_root: Path,
    scale: int = 2,
    model_name: str = "realesrgan-x4plus",
    timeout_seconds: int = 240,
) -> tuple[np.ndarray, str]:
    executable, model_dir = _realesrgan_paths(project_root)
    if not executable.exists():
        return image, "Real-ESRGAN executable is not installed; kept fast enhanced image."
    if not model_dir.exists():
        return image, "Real-ESRGAN models are not installed; kept fast enhanced image."

    with tempfile.TemporaryDirectory(prefix="sunaar_realesrgan_") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "input.png"
        output_path = temp_path / "output.png"
        save_png(image, input_path)

        command = [
            str(executable),
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "-m",
            str(model_dir),
            "-n",
            model_name,
            "-s",
            str(scale),
            "-t",
            "128",
            "-f",
            "png",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(executable.parent),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
        except subprocess.TimeoutExpired:
            return image, "Real-ESRGAN timed out; kept fast enhanced image."
        except Exception as exc:
            return image, f"Real-ESRGAN could not run; kept fast enhanced image. {exc}"

        if completed.returncode != 0 or not output_path.exists():
            stderr = (completed.stderr or completed.stdout or "").strip().splitlines()
            detail = stderr[-1] if stderr else "no output"
            return image, f"Real-ESRGAN failed; kept fast enhanced image. {detail}"

        try:
            upscaled = load_image(output_path)
        except Exception as exc:
            return image, f"Real-ESRGAN output could not be read; kept fast enhanced image. {exc}"

    if _looks_tiled_or_corrupt(upscaled):
        safe, safe_note = upscale_safely(image, scale=scale)
        return safe, f"Real-ESRGAN output was rejected for tile artifacts. {safe_note}"

    return upscaled, f"AI HD enhanced with Real-ESRGAN {model_name} x{scale}."


def _looks_tiled_or_corrupt(image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape[:2]
    if height < 512 or width < 512:
        return False

    edges = cv2.Canny(gray, 80, 160)
    suspicious = 0
    checked = 0
    for step in (64, 128, 256):
        for y in range(step, height - step, step):
            band = float(np.mean(edges[max(0, y - 1) : min(height, y + 2), :]))
            around = float(np.mean(edges[max(0, y - 12) : min(height, y + 12), :])) + 1e-6
            checked += 1
            suspicious += band > around * 1.75 and band > 18
        for x in range(step, width - step, step):
            band = float(np.mean(edges[:, max(0, x - 1) : min(width, x + 2)]))
            around = float(np.mean(edges[:, max(0, x - 12) : min(width, x + 12)])) + 1e-6
            checked += 1
            suspicious += band > around * 1.75 and band > 18
    return checked > 0 and suspicious / checked > 0.22


def _realesrgan_paths(project_root: Path) -> tuple[Path, Path]:
    tools = project_root / "tools"
    executable = tools / "realesrgan-ncnn-vulkan.exe"
    if not executable.exists():
        executable = tools / "realesrgan-ncnn-vulkan" / "realesrgan-ncnn-vulkan.exe"
    return executable, tools / "models"
