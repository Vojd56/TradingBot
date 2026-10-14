@echo off
title AI Planovac Skenovani Trhu (09:30 a 16:00 CET)
cd /d "%~dp0"
call .\venv\Scripts\activate.bat
python main.py --schedule-scans
echo.
pause
