@echo off
title Agent Runner - RL Sparring
cd /d "%~dp0"

echo ========================================================
echo   Minecraft AI: Live Reinforcement Learning (RL) Agent
echo   Algorithm:  Branching Dueling Deep Q-Network (BDQ)
echo   Device:     NVIDIA GeForce RTX 3050 (Live GPU Learning)
echo   Tick Rate:  20 decisions/sec (50ms interval)
echo ========================================================
echo.
echo Reward System:
echo   Aim Centering:        +2.0 max (crosshair on enemy)
echo   3-Block Ideal Range:  +2.0 (optimal spacing)
echo   Circle-Strafing:      +0.5 (dodging attacks)
echo   Sweep Hit:            +10.0 (grounded attack)
echo   Critical Hit:         +25.0 (jumping / falling attack)
echo   Knockback Hit:        +40.0 (sprint attack)
echo   Miss / Whiff:         -2.0  (swinging out of reach)
echo ========================================================
echo.
echo Controls:
echo   [F6]   : TOGGLE RL BOT ON / OFF (Emergency Killswitch)
echo   [q]    : Quit & Save Weights (in HUD window)
echo   Ctrl+C : Stop Agent in terminal
echo ========================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found! Please ensure .venv exists.
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Launching Reinforcement Learning PvP Agent...
echo.
python rl_agent.py

echo.
echo ========================================================
echo RL Agent session finished. Weights saved to models\rl_pvp_model.pth!
echo ========================================================
pause
