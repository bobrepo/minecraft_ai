@echo off
title Stage 1: Pure Aim RL Trainer
cd /d "%~dp0\.."

echo ========================================================
echo   Minecraft AI: Stage 1 Pure Reinforcement Learning Aim
echo   Goal:       Pure Opponent Aiming (No movement WASD/jump)
echo   Algorithm:  Branching Dueling Double Deep Q-Network
echo   Detector:   Crosshair-Centric Cyan Highlight + Red/Purple Lock (Sub-0.5ms)
echo   Rate:       Locked 60.00 FPS (16.6ms Interval)
echo   Training:   Asynchronous Background GPU Optimization
echo ========================================================
echo.
echo Reward System:
echo   Dead-Center Lock:     +12.0 pts/tick
echo   Aim Alignment (Dense): +8.0 * (1 - dist_norm)^1.5
echo   Aim Progress:         +6.0 * (prev_dist - cur_dist)
echo   Lock Duration Streak: +1.5 * streak
echo   Sky Penalty:          -2.5 pts (Looking at empty sky)
echo ========================================================
echo Controls:
echo   [F6]            : TOGGLE RL AIM BOT ON / OFF (Audio Beep Feedback)
echo   [ESC]           : INSTANT EMERGENCY STOP
echo   [RUN AI] Button : Desktop Overlay Start/Stop
echo   Ctrl+C          : Stop Agent & Save Weights to models\pure_aim_model.pth
echo ========================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found! Please ensure .venv exists.
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Launching Stage 1 Pure Aim RL...
echo.
python pure_aim_rl.py

echo.
echo ========================================================
echo Pure Aim RL finished. Weights saved to models\pure_aim_model.pth!
echo ========================================================
pause
