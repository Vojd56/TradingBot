@echo off
title AI Trading Bot - Intelligence Cache Status
cd /d "%~dp0"
call .\venv\Scripts\activate.bat
python main.py --research-status
echo.
pause
