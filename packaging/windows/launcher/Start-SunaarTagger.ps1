param(
    [switch]$InstallOnly,
    [switch]$ForceBundledPythonInstall,
    [string]$PythonInstallTarget = ""
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$AppDir = Join-Path $Root "app"
$ModelCacheDir = Join-Path $Root "model_cache"
$ModelManifest = Join-Path $ModelCacheDir "model-manifest.json"
$PackageManifest = Join-Path $Root "package-manifest.json"
$PythonRuntimeDir = Join-Path $Root "runtime"
$PythonRuntimeManifest = Join-Path $PythonRuntimeDir "python-runtime-manifest.json"
$PythonPortableArchive = Join-Path $PythonRuntimeDir "python-portable.zip"
$PythonPortableArchiveManifest = Join-Path $PythonRuntimeDir "python-portable-archive.json"
$WheelhouseDir = Join-Path $Root "wheelhouse"
$WheelhouseManifest = Join-Path $WheelhouseDir "wheelhouse-manifest.json"
$VenvDir = Join-Path $AppDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $AppDir "requirements-lock.txt"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-PythonCommand {
    param([Parameter(Mandatory = $true)]$Candidate)

    try {
        $check = & $Candidate.Exe @($Candidate.Args) -c "import platform, sys, tkinter; print(f'{sys.version_info.major}.{sys.version_info.minor}|{platform.architecture()[0]}')" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $check) {
            return $false
        }
        return "$check".Trim() -eq "3.14|64bit"
    } catch {
        return $false
    }
}

function Find-PythonCommand {
    $target = if ($PythonInstallTarget) { $PythonInstallTarget } else { Join-Path $PythonRuntimeDir "python" }
    $candidate = [pscustomobject]@{ Exe = (Join-Path $target "python.exe"); Args = [string[]]@() }
    if (Test-PythonCommand -Candidate $candidate) {
        return $candidate
    }
    return $null
}

function Get-VerifiedPortablePythonArchive {
    if (-not (Test-Path -LiteralPath $PythonPortableArchiveManifest -PathType Leaf)) {
        throw "Private Python repair manifest is missing: $PythonPortableArchiveManifest"
    }

    try {
        $manifest = Get-Content -LiteralPath $PythonPortableArchiveManifest -Raw | ConvertFrom-Json
    } catch {
        throw "Private Python repair manifest is invalid: $($_.Exception.Message)"
    }

    $archiveName = [System.IO.Path]::GetFileName([string]$manifest.archive)
    if (-not $archiveName -or $archiveName -ne [string]$manifest.archive) {
        throw "Private Python repair manifest contains an invalid archive name."
    }

    $archivePath = Join-Path $PythonRuntimeDir $archiveName
    if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
        throw "Private Python repair archive is missing: $archiveName"
    }

    $item = Get-Item -LiteralPath $archivePath
    if ($item.Length -ne [long]$manifest.size) {
        throw "Private Python repair archive size verification failed."
    }

    $hash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash
    if (-not $hash.Equals([string]$manifest.sha256, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Private Python repair archive checksum verification failed."
    }

    return [pscustomobject]@{
        Path = $archivePath
        Manifest = $manifest
    }
}

function Install-BundledPython {
    $verified = Get-VerifiedPortablePythonArchive
    $target = $PythonInstallTarget
    if (-not $target) {
        $target = Join-Path $PythonRuntimeDir "python"
    }
    $target = [System.IO.Path]::GetFullPath($target)

    $resolvedRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd("\")
    if (-not $target.StartsWith($resolvedRoot + "\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to repair an unsafe Python runtime path: $target"
    }

    $temporary = "$target.restore.$PID"
    if (Test-Path -LiteralPath $temporary) {
        Remove-Item -LiteralPath $temporary -Recurse -Force
    }

    Write-Step "Restoring private Python 3.14.6 runtime with tkinter"
    try {
        New-Item -ItemType Directory -Force -Path $temporary | Out-Null
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [System.IO.Compression.ZipFile]::ExtractToDirectory($verified.Path, $temporary)

        $temporaryCandidate = [pscustomobject]@{
            Exe = (Join-Path $temporary "python.exe")
            Args = [string[]]@()
        }
        if (-not (Test-PythonCommand -Candidate $temporaryCandidate)) {
            throw "Restored private Python failed the Python 3.14, 64-bit, or tkinter verification."
        }

        if (Test-Path -LiteralPath $target) {
            Write-Step "Removing incomplete private Python runtime"
            Remove-Item -LiteralPath $target -Recurse -Force
        }
        Move-Item -LiteralPath $temporary -Destination $target
    } finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Recurse -Force
        }
    }

    $candidate = [pscustomobject]@{ Exe = (Join-Path $target "python.exe"); Args = [string[]]@() }
    if (-not (Test-PythonCommand -Candidate $candidate)) {
        throw "Private Python restore completed, but the final runtime could not be verified at $target."
    }

    Write-Host "Private Python 3.14.6 64-bit with tkinter restored and verified." -ForegroundColor Green
    return $candidate
}

function Invoke-BasePython {
    param([Parameter(Mandatory = $true)][string[]]$ExtraArgs)
    & $script:PythonCommand.Exe @($script:PythonCommand.Args + $ExtraArgs)
}

function Test-ModelCache {
    param([switch]$Quiet)
    if (-not (Test-Path -LiteralPath $ModelManifest)) {
        throw "Model manifest is missing: $ModelManifest"
    }
    $manifest = Get-Content -LiteralPath $ModelManifest -Raw | ConvertFrom-Json
    foreach ($model in $manifest.models) {
        $relative = ([string]$model.relative_path).Replace("/", "\")
        $path = Join-Path $ModelCacheDir $relative
        if (-not (Test-Path -LiteralPath $path)) {
            if ($Quiet) { return $false }
            throw "Required model is missing: $($model.name)"
        }
        $item = Get-Item -LiteralPath $path
        if ($item.Length -ne [long]$model.size) {
            if ($Quiet) { return $false }
            throw "Model size check failed: $($model.name)"
        }
        $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        if ($hash -ne [string]$model.sha256) {
            if ($Quiet) { return $false }
            throw "Model checksum failed: $($model.name)"
        }
    }
    return $true
}

function Remove-InvalidModelFiles {
    $manifest = Get-Content -LiteralPath $ModelManifest -Raw | ConvertFrom-Json
    foreach ($model in $manifest.models) {
        $relative = ([string]$model.relative_path).Replace("/", "\")
        $path = Join-Path $ModelCacheDir $relative
        if (Test-Path -LiteralPath $path) {
            $item = Get-Item -LiteralPath $path
            $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
            if ($item.Length -ne [long]$model.size -or $hash -ne [string]$model.sha256) {
                Remove-Item -LiteralPath $path -Force
            }
        }
    }
}

function Ensure-Models {
    if (Test-ModelCache -Quiet) {
        Write-Step "Exact OCR and background models already available"
        return
    }

    Write-Step "Downloading exact OCR and background models"
    Write-Host "Internet is required only for this first model setup."
    Remove-InvalidModelFiles
    $bootstrap = Join-Path $AppDir "src\model_bootstrap.py"
    $env:PYTHONIOENCODING = "utf-8"
    & $VenvPython $bootstrap --model-cache $ModelCacheDir
    if ($LASTEXITCODE -ne 0) {
        throw "Model download failed. Keep the internet connected and run Install or Repair again."
    }
    if (-not (Test-ModelCache)) {
        throw "Downloaded model verification failed."
    }
}

function Test-PackageFiles {
    if (-not (Test-Path -LiteralPath $PackageManifest)) {
        throw "Package manifest is missing: $PackageManifest"
    }
    $manifest = Get-Content -LiteralPath $PackageManifest -Raw | ConvertFrom-Json
    foreach ($file in $manifest.files) {
        $relative = ([string]$file.path).Replace("/", "\")
        $path = Join-Path $Root $relative
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Package file is missing: $($file.path)"
        }
        $item = Get-Item -LiteralPath $path
        if ($item.Length -ne [long]$file.size) {
            throw "Package file size check failed: $($file.path)"
        }
        $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        if ($hash -ne [string]$file.sha256) {
            throw "Package checksum failed: $($file.path)"
        }
    }
}

function Get-VerifiedWheelhouse {
    $directoryExists = Test-Path -LiteralPath $WheelhouseDir -PathType Container
    $manifestExists = Test-Path -LiteralPath $WheelhouseManifest -PathType Leaf
    if (-not $directoryExists -and -not $manifestExists) {
        return $null
    }
    if (-not $directoryExists -or -not $manifestExists) {
        throw "The bundled offline dependency cache is incomplete. Re-extract the ZIP and run setup again."
    }

    try {
        $manifest = Get-Content -LiteralPath $WheelhouseManifest -Raw | ConvertFrom-Json
    } catch {
        throw "The bundled offline dependency manifest is invalid: $($_.Exception.Message)"
    }

    $requirementsHash = (Get-FileHash -LiteralPath $Requirements -Algorithm SHA256).Hash
    if (-not $requirementsHash.Equals([string]$manifest.requirements_sha256, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "The bundled offline dependencies do not match this app version. Re-extract the ZIP."
    }
    if ([string]$manifest.python_version -ne "3.14" -or [string]$manifest.platform -ne "win_amd64") {
        throw "The bundled offline dependencies target an unsupported Python or Windows platform."
    }

    $verifiedCount = 0
    foreach ($file in $manifest.files) {
        $name = [System.IO.Path]::GetFileName([string]$file.name)
        if (-not $name -or $name -ne [string]$file.name -or -not $name.EndsWith(".whl", [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "The bundled offline dependency manifest contains an unsafe filename."
        }
        $path = Join-Path $WheelhouseDir $name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "A bundled offline dependency is missing: $name"
        }
        $item = Get-Item -LiteralPath $path
        if ($item.Length -ne [long]$file.size) {
            throw "Bundled offline dependency size verification failed: $name"
        }
        $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        if (-not $hash.Equals([string]$file.sha256, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Bundled offline dependency checksum verification failed: $name"
        }
        $verifiedCount++
    }
    if ($verifiedCount -lt 1 -or $verifiedCount -ne [int]$manifest.file_count) {
        throw "The bundled offline dependency cache file count is invalid."
    }

    return $manifest
}

function Ensure-Venv {
    $venvHealthy = $false
    if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
        & $VenvPython -c "import platform, sys, tkinter; assert sys.version_info[:2] == (3, 14) and platform.architecture()[0] == '64bit'" *> $null
        $venvHealthy = $LASTEXITCODE -eq 0
    }
    if ($venvHealthy) {
        return
    }

    if (Test-Path -LiteralPath $VenvDir) {
        $resolvedApp = [System.IO.Path]::GetFullPath($AppDir).TrimEnd("\")
        $resolvedVenv = [System.IO.Path]::GetFullPath($VenvDir).TrimEnd("\")
        if (-not $resolvedVenv.StartsWith($resolvedApp + "\", [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to repair an unsafe environment path: $resolvedVenv"
        }
        Write-Step "Removing incomplete Python environment"
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    }

    Write-Step "Creating complete local Python environment"
    Invoke-BasePython -ExtraArgs @("-m", "venv", $VenvDir)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the local Python environment."
    }
    & $VenvPython -c "import tkinter" *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "The repaired Python environment is still missing tkinter. Run Install or Repair again."
    }
}

function Ensure-Dependencies {
    $hash = (Get-FileHash -LiteralPath $Requirements -Algorithm SHA256).Hash
    $marker = Join-Path $VenvDir ".sunaar_requirements.sha256"
    $installedHash = ""
    if (Test-Path -LiteralPath $marker) {
        $installedHash = (Get-Content -LiteralPath $marker -Raw).Trim()
    }
    if ($installedHash -eq $hash) {
        & $VenvPython -m pip check *> $null
        $pipHealthy = $LASTEXITCODE -eq 0
        & $VenvPython -c "import cv2, easyocr, onnxruntime, rembg, streamlit, torch, tkinter" *> $null
        $importsHealthy = $LASTEXITCODE -eq 0
        if ($pipHealthy -and $importsHealthy) {
            Write-Step "Exact Python dependencies already installed"
            return
        }
        Write-Step "Repairing incomplete Python dependencies"
    }

    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    $env:PIP_NO_INPUT = "1"
    $offlineCache = Get-VerifiedWheelhouse
    if ($null -ne $offlineCache) {
        Write-Step "Installing exact Python dependencies from verified offline cache"
        & $VenvPython -m pip install --no-index --find-links $WheelhouseDir -r $Requirements
        if ($LASTEXITCODE -ne 0) {
            throw "Offline dependency install failed. Re-extract the complete ZIP and run Install or Repair again."
        }
    } else {
        Write-Step "Installing exact Python dependencies"
        & $VenvPython -m pip install --upgrade pip
        & $VenvPython -m pip install -r $Requirements
        if ($LASTEXITCODE -ne 0) {
            throw "Dependency install failed. Keep the internet connected and run Install or Repair again."
        }
    }
    & $VenvPython -m pip check
    if ($LASTEXITCODE -ne 0) {
        throw "Installed dependencies are inconsistent. Run Install or Repair again while online."
    }
    & $VenvPython -c "import cv2, easyocr, onnxruntime, rembg, streamlit, torch, tkinter"
    if ($LASTEXITCODE -ne 0) {
        throw "A required app module is unavailable after installation. Run Install or Repair again."
    }
    Set-Content -LiteralPath $marker -Value $hash -Encoding ASCII
}

function Test-PortFree {
    param([int]$Port)
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($null -ne $listener) {
            $listener.Stop()
        }
    }
}

function Get-FreePort {
    foreach ($port in 8501..8510) {
        if (Test-PortFree -Port $port) {
            return $port
        }
    }
    throw "No free local port found between 8501 and 8510."
}

if (-not (Test-Path -LiteralPath $AppDir)) {
    throw "App folder is missing: $AppDir"
}
if (-not (Test-Path -LiteralPath $Requirements)) {
    throw "Dependency lock is missing: $Requirements"
}

Write-Step "Verifying exact Sunaar app files"
Test-PackageFiles

$script:PythonCommand = Find-PythonCommand
if ($ForceBundledPythonInstall) {
    if (-not $InstallOnly) {
        throw "ForceBundledPythonInstall is supported only with InstallOnly."
    }
    $script:PythonCommand = Install-BundledPython
} elseif ($null -eq $script:PythonCommand) {
    $script:PythonCommand = Install-BundledPython
} else {
    Write-Step "Private Python 3.14.6 runtime with tkinter verified"
}
Ensure-Venv
Ensure-Dependencies

$env:EASYOCR_MODULE_PATH = Join-Path $ModelCacheDir "EasyOCR"
$env:U2NET_HOME = Join-Path $ModelCacheDir "u2net"
Ensure-Models

$shortcutScript = Join-Path $PSScriptRoot "Create-DesktopShortcut.ps1"
$desktopShortcut = Join-Path ([Environment]::GetFolderPath("Desktop")) "Sunaar Jewellery Tagger.lnk"
if (-not (Test-Path -LiteralPath $desktopShortcut) -and (Test-Path -LiteralPath $shortcutScript)) {
    try {
        & $shortcutScript
    } catch {
        Write-Warning "Desktop shortcut could not be created automatically: $($_.Exception.Message)"
    }
}

if ($InstallOnly) {
    Write-Step "Ready"
    Write-Host "Dependencies, source files, and exact AI models are verified." -ForegroundColor Green
    exit 0
}

$port = Get-FreePort
$url = "http://127.0.0.1:$port/"

Write-Step "Starting Sunaar Jewellery Tagger"
Write-Host "Starting local app at $url"
Write-Host "Keep this window open while using the app."

$processInfo = New-Object System.Diagnostics.ProcessStartInfo
$processInfo.FileName = $VenvPython
$processInfo.Arguments = "-m streamlit run `"app.py`" --server.address 127.0.0.1 --server.port $port --server.headless true --browser.gatherUsageStats false"
$processInfo.WorkingDirectory = $AppDir
$processInfo.UseShellExecute = $false
$processInfo.CreateNoWindow = $false
$streamlitProcess = [System.Diagnostics.Process]::Start($processInfo)

$ready = $false
for ($attempt = 0; $attempt -lt 120; $attempt++) {
    if ($streamlitProcess.HasExited) {
        throw "The local app stopped before it became ready."
    }
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $client.Connect("127.0.0.1", $port)
        $ready = $true
        break
    } catch {
        Start-Sleep -Milliseconds 500
    } finally {
        $client.Dispose()
    }
}

if (-not $ready) {
    $streamlitProcess.Kill()
    throw "The local app did not become ready within 60 seconds."
}

Write-Host "Opening $url"
Start-Process $url
try {
    $streamlitProcess.WaitForExit()
    if ($streamlitProcess.ExitCode -ne 0) {
        throw "The local app stopped with exit code $($streamlitProcess.ExitCode)."
    }
} finally {
    if (-not $streamlitProcess.HasExited) {
        $streamlitProcess.Kill()
    }
}
