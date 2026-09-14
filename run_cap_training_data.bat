@echo off
title Agent Runner - Training Data Telemetry Recorder
cd /d "%~dp0"

echo ========================================================
echo   Minecraft AI: Gameplay & Input Telemetry Recorder
echo   Program:    run_cap_training_data
echo   Output:     train_videos\
echo   Tick Rate:  20 ticks/sec (50ms interval)
echo   Resolution: 640x480 (VGA 4:3)
echo ========================================================
echo.
echo Telemetry Captured:
echo   - Video Stream (640x480 @ 20 FPS)
echo   - Timestamped Keystrokes (WASD, Sprint, Jump, Shift, Attack)
echo   - Hardware Relative Mouse Movement (dx, dy)
echo.
echo Controls:
echo   [F6]   : START / PAUSE RECORDING (Audio Beep Confirmation)
echo   [ESC]  : STOP & FINALIZE CURRENT SESSION TO train_videos\
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

echo Launching cap_training_data Recorder...
echo.
python cap_game\cap_training_data.py

echo.
echo ========================================================
echo Recorded session pairs saved into train_videos\.
echo Run run_train.bat to pre-train the model!
echo ========================================================
pause
