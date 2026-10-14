@echo off
cd /d "%~dp0"
call .\venv\Scripts\activate.bat
python main.py --run-once --force
echo.
pause
