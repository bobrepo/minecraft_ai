@echo off
title Camera Perturber (Sparring Trainer)
cd /d "%~dp0\.."

echo ========================================================
echo   Minecraft AI: Camera Perturber (Sparring Trainer)
echo   Goal:     High-Displacement Camera Shifts for RL Training
echo   Interval: Randomized 0.5s - 1.3s
echo   Strength: Default 600px (+/-360px to +/-1080px kicks)
echo   Presets:  --preset light (300px), medium (450px), strong (600px), extreme (900px), insane (1400px)
echo ========================================================
echo.
echo Controls:
echo   [F6]        : START / TOGGLE (Training Mode - Perturbations Active)
echo   [F7]        : STOP / BYPASS  (Clean Match Mode - Perturber Inactive)
echo   [Page Up]   : Increase kick distance (+100px)
echo   [Page Down] : Decrease kick distance (-100px)
echo   [ESC]       : Emergency Pause All
echo   Ctrl+C      : Stop Script
echo ========================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found! Please ensure .venv exists.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

python camera_perturber.py %*

pause
