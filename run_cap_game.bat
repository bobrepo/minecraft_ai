@echo off
title Agent Runner - cap_game Gameplay Recorder
cd /d "%~dp0"

echo ========================================================
echo   Minecraft AI: Gameplay & Input Telemetry Recorder (cap_game)
echo   Output Folder: train_videos\
echo   Tick Rate:     20 ticks/sec (50ms interval)
echo   Resolution:    640x480 (VGA 4:3)
echo ========================================================
echo.
echo Controls:
echo   [F6]   : START / PAUSE RECORDING (Audio Beep Feedback)
echo   [ESC]  : STOP & SAVE CURRENT SESSION TO train_videos\
echo   Ctrl+C : Exit Recorder
echo ========================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found! Please ensure .venv exists.
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Launching cap_game Recorder...
echo.
python cap_game\recorder.py

echo.
echo ========================================================
echo Recorded sessions saved in train_videos\.
echo Run run_train.bat to train the model on this data!
echo ========================================================
pause
