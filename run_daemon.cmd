@echo off
title AI Trading Bot (24/7 Continuous Research & Live Trader)
cd /d "%~dp0"
call .\venv\Scripts\activate.bat
python main.py --daemon
echo.
pause
