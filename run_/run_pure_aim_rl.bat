@echo off
title Stage 1: Pure Aim RL Trainer
cd /d "%~dp0\.."

echo ========================================================
echo   Minecraft AI: Stage 1 Pure Reinforcement Learning Aim
echo   Target:     Yellow (#FFF500) Enemy Body
echo   Algorithm:  Branching Dueling Double Deep Q-Network (BD-DQN)
echo   Aim Mode:   FULL 2D AIM RL (Horizontal Yaw + Vertical Pitch Simultaneous Learning)
echo   Rate:       Locked 60.00 FPS (16.6ms Interval)
echo   Training:   Asynchronous Background GPU Optimization (Continual Learning)
echo   Checkpoints: Rolling 20-min saves to saves\ (Max 10 models FIFO)
echo ========================================================
echo.
echo Reward System:
echo   Dwell Lock Requirement: 10+ ticks sustained on enemy
echo   Steady Lock Reward:     +8.0 to +12.0 pts/tick (Dwell Streak)
echo   Moving / Sweeping Aim:  0.0 pts (Continuous move resets streak)
echo   Target Regression:      -1.5 to -5.0 pts (Moving away from target)
echo   Off-Target Delay:       -2.0 to -5.0 pts (Staring without locking)
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
