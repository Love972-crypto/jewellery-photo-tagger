@echo off
title Sunaar Jewellery Tagger - Install or Repair
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher\Start-SunaarTagger.ps1" -InstallOnly
if errorlevel 1 (
    echo.
    echo SETUP FAILED. Read the error above, keep the internet connected, and run this file again.
    echo Press any key to close this window.
    pause >nul
    exit /b 1
)
echo.
echo SETUP COMPLETE. Python, dependencies, app files, and AI models are verified.
echo Press any key to close this window.
pause >nul
