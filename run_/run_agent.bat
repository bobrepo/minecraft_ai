@echo off
title Agent Runner - PvP Combat Agent
cd /d "%~dp0\.."

echo ========================================================
echo   Minecraft AI: Autonomous PvP Combat Agent
echo   Tick Rate:  20 decisions/sec (50ms interval)
echo   Resolution: 640x480
echo ========================================================
echo Controls:
echo   [RUN AI] Button : Click on Desktop Overlay to Start/Stop
echo   [F6]            : TOGGLE AI ON / OFF (Audio Beep Confirmation)
echo   [ESC]           : INSTANT EMERGENCY STOP (Pauses & releases all keys)
echo   [x] / Ctrl+C    : Stop Agent cleanly
echo ========================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found! Please ensure .venv exists.
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Launching High-Speed PvP Combat Agent...
echo.

if exist "models\pvp_model.pth" (
    echo [INFO] Found trained model weights in models\pvp_model.pth! Loading weights...
    python agent.py --model "models\pvp_model.pth"
) else (
    echo [INFO] Running in high-speed visual combat mode...
    python agent.py
)

echo.
echo ========================================================
echo PvP Agent session ended. All keys released.
echo ========================================================
pause
