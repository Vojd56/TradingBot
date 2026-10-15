@echo off
title AI Quantitative Trader (8-Module Architecture, Carver Volatility Targeting & EV_net)
cd /d "%~dp0"
call .\venv\Scripts\activate.bat
python main.py --quant --interval 300
echo.
pause
