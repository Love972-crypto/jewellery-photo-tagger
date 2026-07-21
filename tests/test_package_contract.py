import hashlib
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]


def test_model_manifest_locks_exact_working_models():
    manifest = json.loads((PROJECT_ROOT / "packaging" / "model-manifest.json").read_text(encoding="utf-8"))
    models = {model["relative_path"]: model for model in manifest["models"]}

    assert set(models) == {
        "EasyOCR/model/craft_mlt_25k.pth",
        "EasyOCR/model/english_g2.pth",
        "u2net/u2net.onnx",
        "u2net/birefnet-general-lite.onnx",
    }
    assert sum(model["size"] for model in models.values()) == 498_299_056
    assert all(re.fullmatch(r"[A-F0-9]{64}", model["sha256"]) for model in models.values())


def test_dependency_lock_contains_quality_critical_versions():
    lock_lines = (PROJECT_ROOT / "requirements-lock.txt").read_text(encoding="utf-8").splitlines()
    locked = {line.lower() for line in lock_lines if line.strip()}

    assert "streamlit==1.59.1" in locked
    assert "easyocr==1.7.2" in locked
    assert "rembg==2.0.76" in locked
    assert "onnxruntime==1.27.0" in locked
    assert "torch==2.13.0" in locked
    assert "opencv-python==5.0.0.93" in locked
    assert "pillow==12.3.0" in locked


def test_windows_launcher_uses_packaged_models_and_locked_dependencies():
    launcher = (PROJECT_ROOT / "packaging" / "windows" / "launcher" / "Start-SunaarTagger.ps1").read_text(encoding="utf-8")

    assert 'requirements-lock.txt' in launcher
    assert 'EASYOCR_MODULE_PATH' in launcher
    assert 'U2NET_HOME' in launcher
    assert 'Test-ModelCache' in launcher
    assert 'Test-PackageFiles' in launcher
    assert '--server.headless true' in launcher
    assert '--browser.gatherUsageStats false' in launcher
    assert 'Ensure-Models' in launcher
    assert 'src\\model_bootstrap.py' in launcher
    assert 'Install-BundledPython' in launcher
    assert 'Get-VerifiedPortablePythonArchive' in launcher
    assert 'python-portable.zip' in launcher
    assert 'python-portable-archive.json' in launcher
    assert 'ExtractToDirectory' in launcher
    assert 'Restoring private Python 3.14.6 runtime with tkinter' in launcher
    assert 'Get-AuthenticodeSignature' not in launcher
    assert 'InstallAllUsers=0' not in launcher
    assert 'import platform, sys, tkinter' in launcher
    assert 'Removing incomplete Python environment' in launcher
    assert 'The repaired Python environment is still missing tkinter' in launcher
    assert 'Join-Path $PythonRuntimeDir "python"' in launcher
    assert 'Programs\\Python\\Python314' not in launcher
    assert 'Exe = "py"' not in launcher
    assert 'Exe = "python"' not in launcher
    assert '-m pip check' in launcher
    assert 'torch, tkinter' in launcher
    assert 'Private Python 3.14.6 runtime with tkinter verified' in launcher
    assert 'Create-DesktopShortcut.ps1' in launcher
    assert 'Desktop shortcut could not be created automatically' in launcher
    assert 'Get-VerifiedWheelhouse' in launcher
    assert 'wheelhouse-manifest.json' in launcher
    assert '--no-index --find-links $WheelhouseDir' in launcher
    assert 'Installing exact Python dependencies from verified offline cache' in launcher


def test_model_bootstrap_downloads_both_background_models():
    bootstrap = (PROJECT_ROOT / "src" / "model_bootstrap.py").read_text(encoding="utf-8")

    assert 'new_session("u2net")' in bootstrap
    assert 'new_session("birefnet-general-lite")' in bootstrap


def test_windows_exe_launcher_runs_first_time_setup_automatically():
    windows_dir = PROJECT_ROOT / "packaging" / "windows"
    launcher = (windows_dir / "launcher" / "SunaarTaggerLauncher.cs").read_text(encoding="utf-8")

    assert (windows_dir / "launcher" / "SunaarTagger.ico").is_file()
    assert '.sunaar_requirements.sha256' in launcher
    assert 'Start-SunaarTagger.ps1' in launcher
    assert '"-InstallOnly"' in launcher
    assert 'RunPowerShell(launcher, string.Empty, false)' in launcher


def test_standard_windows_launcher_can_install_python_without_install_only_mode():
    launcher = (PROJECT_ROOT / "packaging" / "windows" / "launcher" / "Start-SunaarTagger.ps1").read_text(encoding="utf-8")
    missing_python_branch = launcher.split('} elseif ($null -eq $script:PythonCommand) {', 1)[1].split('} else {', 1)[0]

    assert 'Install-BundledPython' in missing_python_branch
    assert "Run 'Install or Repair Dependencies.bat' first" not in missing_python_branch


def test_windows_builder_creates_verified_portable_python_repair_archive():
    builder = (PROJECT_ROOT / "tools" / "build_windows_package.ps1").read_text(encoding="utf-8")

    assert "python-portable.zip" in builder
    assert "python-portable-archive.json" in builder
    assert "CreateFromDirectory" in builder
    assert "total_uncompressed_bytes" in builder
    assert "Get-FileHash" in builder


def test_windows_builder_can_bundle_verified_offline_dependencies():
    builder = (PROJECT_ROOT / "tools" / "build_windows_package.ps1").read_text(encoding="utf-8")

    assert "WheelhouseSource" in builder
    assert "wheelhouse-manifest.json" in builder
    assert 'python_version = "3.14"' in builder
    assert 'platform = "win_amd64"' in builder
    assert "requirements_sha256" in builder


def test_bundled_python_runtime_contract_is_exactly_locked():
    runtime_dir = PROJECT_ROOT / "packaging" / "windows" / "runtime"
    manifest = json.loads((runtime_dir / "python-runtime-manifest.json").read_text(encoding="utf-8"))

    assert manifest["python_version"] == "3.14.6"
    assert manifest["required_series"] == "3.14"
    assert manifest["architecture"] == "64bit"
    assert manifest["size"] == 30_774_112
    assert manifest["sha256"] == "14B3E9A710A3FCF0BD9B55AB6B60412BD91227563F813FC49040CABC0209E0BD"
    assert manifest["signer_contains"] == "Python Software Foundation"

    # Runtime binaries are release inputs rather than Git source. When they are
    # present locally, keep validating the exact signed payload used by builds.
    installer = runtime_dir / manifest["installer"]
    if not installer.exists():
        return

    assert installer.is_file()
    assert installer.stat().st_size == manifest["size"]
    assert hashlib.sha256(installer.read_bytes()).hexdigest().upper() == manifest["sha256"]

    portable = runtime_dir / "python"
    assert (portable / "python.exe").is_file()
    assert (portable / "python314.dll").is_file()
    assert (portable / "DLLs" / "_tkinter.pyd").is_file()
    assert (portable / "DLLs" / "tcl86t.dll").is_file()
    assert (portable / "DLLs" / "tk86t.dll").is_file()
