@echo off
title Sunaar Jewellery Tagger - Desktop Shortcut
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher\Create-DesktopShortcut.ps1"
echo.
pause
