@echo off
title Agent Runner - Window Capture
cd /d "%~dp0"

echo ========================================================
echo   Minecraft AI: Real-Time Window Capture and Recorder
echo   Tick Rate:  20 ticks/sec (50ms interval)
echo   Resolution: 640x480 (standard VGA 4:3)
echo ========================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found! Please ensure .venv exists.
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Launching Window Capture Recorder...
echo.
python record.py

echo.
echo ========================================================
echo Capture session ended. Videos saved in: out_vid\
echo ========================================================
pause
