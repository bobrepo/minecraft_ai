@echo off
title Agent Runner - Behavioral Cloning GPU Trainer
cd /d "%~dp0\.."

echo ========================================================
echo   Minecraft AI: Behavioral Cloning GPU Model Trainer
echo   Model:  BranchingQNetwork (RTX 3050 Mixed Precision)
echo   Inputs: Synchronized video & keystroke telemetry from train_videos\
echo ========================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found! Please ensure .venv exists.
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting GPU pre-training on recorded sessions in train_videos\...
echo.
python train.py

echo.
echo ========================================================
echo Pre-training finished. Weights updated in models\rl_pvp_model.pth!
echo Run run_\run_rl.bat to continue learning with RL on top of this model!
echo ========================================================
pause
