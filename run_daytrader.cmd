@echo off
title AI Day Trader (Leveraged Momentum & Scalping Engine)
cd /d "%~dp0"
call .\venv\Scripts\activate.bat
python main.py --daytrade --interval 60
echo.
pause
