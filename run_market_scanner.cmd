@echo off
title AI Hloubkovy Skener Trhu & Rada AI Modelu
cd /d "%~dp0"
call .\venv\Scripts\activate.bat
python main.py --scan
echo.
pause
