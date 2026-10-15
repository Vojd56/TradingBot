@echo off
title AI Quantitative Trader - Carver Volatility Targeting and EV_net
cd /d "%~dp0"
call .\venv\Scripts\activate.bat
python main.py --quant --interval 300
echo.
pause
