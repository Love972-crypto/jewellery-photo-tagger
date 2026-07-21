@echo off
title Sunaar Jewellery Tagger
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher\Start-SunaarTagger.ps1"
echo.
echo Sunaar Jewellery Tagger stopped. Press any key to close this window.
pause >nul
