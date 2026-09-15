@echo off
title Aim Agent (60 FPS)
cd /d "%~dp0\.."

echo ========================================================
echo   Minecraft AI: Aim Agent (Stage 1)
echo   Target:     Yellow (#FFF500) Enemy Body
echo   Vision:     Crosshair-Centric Outward Scan (Sub-0.5ms)
echo   Capture:    Async DirectX/OpenGL Frame Buffer
echo   Aim Mode:   FULL 2D LOCK (Horizontal Yaw + Vertical Pitch)
echo   Rate:       60 FPS Ultra-Smooth Zero-Bounce Tracking
echo ========================================================
echo.
echo Controls:
echo   [F6]            : TOGGLE AIM ON / OFF (Audio Beep Feedback)
echo   [ESC]           : INSTANT EMERGENCY STOP
echo   [RUN AI] Button : Desktop Overlay Start/Stop
echo   Ctrl+C          : Exit Agent
echo ========================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found! Please ensure .venv exists.
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Launching Aim Agent (60 FPS)...
echo.
python aim_agent.py --fps 60

echo.
echo ========================================================
echo Aim Agent session finished.
echo ========================================================
pause
