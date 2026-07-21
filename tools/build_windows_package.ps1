param(
    [string]$OutputRoot = "",
    [switch]$OnlineModels,
    [switch]$IncludeUnsignedExe,
    [string]$WheelhouseSource = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = Split-Path -Parent $RepoRoot
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $WorkspaceRoot "release"
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$PackageName = if ($OnlineModels) { "Sunaar-Jewellery-Tagger-Windows-Online-Setup" } else { "Sunaar-Jewellery-Tagger-Windows" }
$StageRoot = Join-Path $OutputRoot $PackageName
$ZipPath = Join-Path $OutputRoot "$PackageName.zip"

function Assert-SafeBuildPath {
    param([string]$Path)
    $full = [System.IO.Path]::GetFullPath($Path).TrimEnd("\")
    $workspace = [System.IO.Path]::GetFullPath($WorkspaceRoot).TrimEnd("\")
    if (-not $full.StartsWith($workspace + "\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Build path must stay inside the workspace: $full"
    }
    if ($full -eq $workspace -or $full -eq [System.IO.Path]::GetPathRoot($full).TrimEnd("\")) {
        throw "Refusing unsafe build path: $full"
    }
}

Assert-SafeBuildPath -Path $OutputRoot
Assert-SafeBuildPath -Path $StageRoot
Assert-SafeBuildPath -Path $ZipPath

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
if (Test-Path -LiteralPath $StageRoot) {
    Remove-Item -LiteralPath $StageRoot -Recurse -Force
}
if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}
New-Item -ItemType Directory -Force -Path (Join-Path $StageRoot "app") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $StageRoot "model_cache") | Out-Null

$appItems = @(
    "app.py",
    "README.md",
    "PROJECT_DOCUMENTATION.md",
    "requirements.txt",
    "requirements-lock.txt",
    "assets",
    "src"
)
foreach ($item in $appItems) {
    $source = Join-Path $RepoRoot $item
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Required app item is missing: $source"
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $StageRoot "app") -Recurse -Force
}

$sampleReadme = Join-Path $RepoRoot "sample_data\README.md"
$sampleDestination = Join-Path $StageRoot "app\sample_data"
New-Item -ItemType Directory -Force -Path $sampleDestination | Out-Null
Copy-Item -LiteralPath $sampleReadme -Destination $sampleDestination -Force

Get-ChildItem -LiteralPath (Join-Path $StageRoot "app") -Directory -Recurse -Force |
    Where-Object { $_.Name -eq "__pycache__" } |
    Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath (Join-Path $StageRoot "app") -File -Recurse |
    Where-Object { $_.Extension -in @(".pyc", ".pyo") } |
    Remove-Item -Force

$requiredStagedFiles = @(
    "app\app.py",
    "app\requirements.txt",
    "app\requirements-lock.txt",
    "app\src\processor.py",
    "app\src\compressed_export.py",
    "app\src\local_export.py"
)
foreach ($relative in $requiredStagedFiles) {
    $required = Join-Path $StageRoot $relative
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required staged app file is missing: $relative"
    }
}

$windowsTemplate = Join-Path $RepoRoot "packaging\windows"
Get-ChildItem -LiteralPath $windowsTemplate -Force | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $StageRoot -Recurse -Force
}

if ($IncludeUnsignedExe) {
    $launcherSource = Join-Path $StageRoot "launcher\SunaarTaggerLauncher.cs"
    $launcherIcon = Join-Path $StageRoot "launcher\SunaarTagger.ico"
    $launcherExe = Join-Path $StageRoot "Sunaar Jewellery Tagger.exe"
    $compilerCandidates = @(
        (Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"),
        (Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe")
    )
    $compiler = $compilerCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (-not $compiler) {
        throw "Windows C# compiler was not found, so the optional launcher EXE could not be built."
    }
    if (-not (Test-Path -LiteralPath $launcherSource -PathType Leaf) -or
        -not (Test-Path -LiteralPath $launcherIcon -PathType Leaf)) {
        throw "Launcher source or icon is missing from the Windows template."
    }

    & $compiler @(
        "/nologo",
        "/target:winexe",
        "/optimize+",
        "/reference:System.dll",
        "/reference:System.Windows.Forms.dll",
        "/win32icon:$launcherIcon",
        "/out:$launcherExe",
        $launcherSource
    )
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $launcherExe -PathType Leaf)) {
        throw "Sunaar Jewellery Tagger launcher EXE compilation failed."
    }
}

$requiredWindowsFiles = @(
    "Install or Repair Dependencies.bat",
    "Run Sunaar Jewellery Tagger.bat",
    "Create Desktop Shortcut.bat",
    "launcher\SunaarTaggerLauncher.cs",
    "launcher\SunaarTagger.ico",
    "launcher\Start-SunaarTagger.ps1",
    "runtime\python-runtime-manifest.json",
    "runtime\python-3.14.6-amd64.exe",
    "runtime\python\python.exe",
    "runtime\python\python314.dll",
    "runtime\python\DLLs\_tkinter.pyd",
    "runtime\python\DLLs\tcl86t.dll",
    "runtime\python\DLLs\tk86t.dll"
)
if ($IncludeUnsignedExe) {
    $requiredWindowsFiles += "Sunaar Jewellery Tagger.exe"
}
foreach ($relative in $requiredWindowsFiles) {
    $required = Join-Path $StageRoot $relative
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required Windows package file is missing: $relative"
    }
}

$pythonRuntimeManifestPath = Join-Path $StageRoot "runtime\python-runtime-manifest.json"
$pythonRuntimeManifest = Get-Content -LiteralPath $pythonRuntimeManifestPath -Raw | ConvertFrom-Json
$pythonInstallerName = [System.IO.Path]::GetFileName([string]$pythonRuntimeManifest.installer)
if (-not $pythonInstallerName -or $pythonInstallerName -ne [string]$pythonRuntimeManifest.installer) {
    throw "Python runtime manifest contains an invalid installer name."
}
$pythonInstallerPath = Join-Path (Join-Path $StageRoot "runtime") $pythonInstallerName
$pythonInstaller = Get-Item -LiteralPath $pythonInstallerPath
$pythonInstallerHash = (Get-FileHash -LiteralPath $pythonInstallerPath -Algorithm SHA256).Hash
if ($pythonInstaller.Length -ne [long]$pythonRuntimeManifest.size -or
    -not $pythonInstallerHash.Equals([string]$pythonRuntimeManifest.sha256, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Bundled Python installer failed size or checksum verification."
}
$pythonSignature = Get-AuthenticodeSignature -LiteralPath $pythonInstallerPath
$pythonSigner = if ($null -ne $pythonSignature.SignerCertificate) { [string]$pythonSignature.SignerCertificate.Subject } else { "" }
if ([string]$pythonSignature.Status -ne "Valid" -or
    $pythonSigner.IndexOf([string]$pythonRuntimeManifest.signer_contains, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
    throw "Bundled Python installer failed digital-signature verification."
}

$portablePythonDir = Join-Path $StageRoot "runtime\python"
$portablePython = Join-Path $portablePythonDir "python.exe"
$portableCheck = & $portablePython -c "import platform, sys, tkinter; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}|{platform.architecture()[0]}|{tkinter.TkVersion}')"
if ($LASTEXITCODE -ne 0 -or "$portableCheck".Trim() -ne "3.14.6|64bit|8.6") {
    throw "Bundled private Python runtime failed the version, architecture, or tkinter check."
}
$portableSignature = Get-AuthenticodeSignature -LiteralPath $portablePython
$portableSigner = if ($null -ne $portableSignature.SignerCertificate) { [string]$portableSignature.SignerCertificate.Subject } else { "" }
if ([string]$portableSignature.Status -ne "Valid" -or
    $portableSigner.IndexOf("Python Software Foundation", [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
    throw "Bundled private Python runtime failed digital-signature verification."
}

$portableFiles = Get-ChildItem -LiteralPath $portablePythonDir -Recurse -File | Sort-Object FullName
$portableEntries = @()
$portableBytes = [long]0
foreach ($file in $portableFiles) {
    $portableBytes += $file.Length
    $portableEntries += [ordered]@{
        path = $file.FullName.Substring($portablePythonDir.Length + 1).Replace("\", "/")
        size = $file.Length
        sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    }
}
$portableManifest = [ordered]@{
    version = 1
    python_version = "3.14.6"
    architecture = "64bit"
    tkinter_version = "8.6"
    file_count = $portableFiles.Count
    total_bytes = $portableBytes
    files = $portableEntries
}
$portableManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $StageRoot "runtime\python-portable-manifest.json") -Encoding UTF8

$portableArchivePath = Join-Path $StageRoot "runtime\python-portable.zip"
$portableArchiveManifestPath = Join-Path $StageRoot "runtime\python-portable-archive.json"
if (Test-Path -LiteralPath $portableArchivePath) {
    Remove-Item -LiteralPath $portableArchivePath -Force
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $portablePythonDir,
    $portableArchivePath,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $false
)
$portableArchive = Get-Item -LiteralPath $portableArchivePath
$portableArchiveManifest = [ordered]@{
    version = 1
    archive = $portableArchive.Name
    size = $portableArchive.Length
    sha256 = (Get-FileHash -LiteralPath $portableArchivePath -Algorithm SHA256).Hash
    python_version = "3.14.6"
    architecture = "64bit"
    tkinter_version = "8.6"
    file_count = $portableFiles.Count
    total_uncompressed_bytes = $portableBytes
}
$portableArchiveManifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $portableArchiveManifestPath -Encoding UTF8

$modelManifestSource = Join-Path $RepoRoot "packaging\model-manifest.json"
$modelManifest = Get-Content -LiteralPath $modelManifestSource -Raw | ConvertFrom-Json
foreach ($model in $modelManifest.models) {
    $source = Join-Path $env:USERPROFILE (([string]$model.source_path).Replace("/", "\"))
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Model source is missing: $source"
    }
    $sourceItem = Get-Item -LiteralPath $source
    $sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
    if ($sourceItem.Length -ne [long]$model.size -or $sourceHash -ne [string]$model.sha256) {
        throw "Model verification failed before packaging: $($model.name)"
    }
    $relative = ([string]$model.relative_path).Replace("/", "\")
    $includeModel = -not $OnlineModels
    if ($includeModel) {
        $destination = Join-Path (Join-Path $StageRoot "model_cache") $relative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }
}
Copy-Item -LiteralPath $modelManifestSource -Destination (Join-Path $StageRoot "model_cache\model-manifest.json") -Force

if ($WheelhouseSource) {
    $WheelhouseSource = [System.IO.Path]::GetFullPath($WheelhouseSource)
    if (-not (Test-Path -LiteralPath $WheelhouseSource -PathType Container)) {
        throw "Wheelhouse source folder is missing: $WheelhouseSource"
    }
    $wheelFiles = @(Get-ChildItem -LiteralPath $WheelhouseSource -File -Filter "*.whl" | Sort-Object Name)
    if ($wheelFiles.Count -lt 1) {
        throw "Wheelhouse source contains no .whl files: $WheelhouseSource"
    }
    $stagedWheelhouse = Join-Path $StageRoot "wheelhouse"
    New-Item -ItemType Directory -Force -Path $stagedWheelhouse | Out-Null
    $wheelEntries = @()
    $wheelBytes = [long]0
    foreach ($wheel in $wheelFiles) {
        $destination = Join-Path $stagedWheelhouse $wheel.Name
        Copy-Item -LiteralPath $wheel.FullName -Destination $destination -Force
        $stagedWheel = Get-Item -LiteralPath $destination
        $wheelBytes += $stagedWheel.Length
        $wheelEntries += [ordered]@{
            name = $stagedWheel.Name
            size = $stagedWheel.Length
            sha256 = (Get-FileHash -LiteralPath $stagedWheel.FullName -Algorithm SHA256).Hash
        }
    }
    $wheelhouseManifest = [ordered]@{
        version = 1
        python_version = "3.14"
        platform = "win_amd64"
        requirements_sha256 = (Get-FileHash -LiteralPath (Join-Path $StageRoot "app\requirements-lock.txt") -Algorithm SHA256).Hash
        file_count = $wheelEntries.Count
        total_bytes = $wheelBytes
        files = $wheelEntries
    }
    $wheelhouseManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $stagedWheelhouse "wheelhouse-manifest.json") -Encoding UTF8
}

$packageModes = @($(if ($OnlineModels) { "Internet-once exact model setup" } else { "Bundled exact models" }))
$packageModes += $(if ($WheelhouseSource) { "Bundled verified offline dependencies" } else { "Internet-once dependency setup" })
Set-Content -LiteralPath (Join-Path $StageRoot "PACKAGE_MODE.txt") -Value ($packageModes -join "; ") -Encoding ASCII

$manifestFiles = Get-ChildItem -LiteralPath (Join-Path $StageRoot "app") -Recurse -File | Sort-Object FullName
if ($manifestFiles.Count -lt 20) {
    throw "Staged app is incomplete: only $($manifestFiles.Count) files were found."
}
$manifestEntries = @()
foreach ($file in $manifestFiles) {
    $relative = $file.FullName.Substring($StageRoot.Length + 1).Replace("\", "/")
    $manifestEntries += [ordered]@{
        path = $relative
        size = $file.Length
        sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    }
}
$packageManifest = [ordered]@{
    version = 1
    created_utc = [DateTime]::UtcNow.ToString("o")
    files = $manifestEntries
}
$packageManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $StageRoot "package-manifest.json") -Encoding UTF8

Compress-Archive -LiteralPath $StageRoot -DestinationPath $ZipPath -CompressionLevel Optimal

Write-Host ""
Write-Host "Windows package ready:" -ForegroundColor Green
Write-Host $ZipPath
Write-Host ("Size: {0:N1} MB" -f ((Get-Item -LiteralPath $ZipPath).Length / 1MB))
