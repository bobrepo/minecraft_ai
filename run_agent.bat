@echo off
title Minecraft AI - PvP Combat Agent (20 TPS)
cd /d "%~dp0"

echo ========================================================
echo   Minecraft AI: Autonomous PvP Combat Agent
echo   Tick Rate:  20 decisions/sec (50ms interval)
echo   Resolution: 640x480
echo ========================================================
echo.
echo Controls:
echo   [F6]   : TOGGLE AI ON / OFF (Emergency Killswitch)
echo   [q]    : Quit Agent (in preview HUD window)
echo   Ctrl+C : Stop Agent in this command prompt
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
