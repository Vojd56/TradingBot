@echo off
title AI Trade Memory & Self-Reflection Lessons
cd /d "%~dp0"
call .\venv\Scripts\activate.bat
python main.py --trade-memory
echo.
pause
