$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$ExePath = Join-Path $Root "Sunaar Jewellery Tagger.exe"
$RunScript = Join-Path $Root "Run Sunaar Jewellery Tagger.bat"
$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "Sunaar Jewellery Tagger.lnk"

if (-not (Test-Path -LiteralPath $ExePath) -and -not (Test-Path -LiteralPath $RunScript)) {
    throw "Launcher files are missing from the package."
}

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
if (Test-Path -LiteralPath $ExePath) {
    $Shortcut.TargetPath = $ExePath
    $Shortcut.Arguments = ""
    $Shortcut.IconLocation = "$ExePath,0"
} else {
    $Shortcut.TargetPath = $env:ComSpec
    $Shortcut.Arguments = "/c `"`"$RunScript`"`""
    $Shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,168"
}
$Shortcut.WorkingDirectory = $Root
$Shortcut.Description = "Sunaar Jewellery Tagger"
$Shortcut.Save()

Write-Host "Desktop shortcut created: $ShortcutPath" -ForegroundColor Green
