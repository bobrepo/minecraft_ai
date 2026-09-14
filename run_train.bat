@echo off
title Minecraft AI - GPU Model Trainer
cd /d "%~dp0"

echo ========================================================
echo   Minecraft AI: Self-Supervised GPU Trainer
echo   Model: MinecraftPvPCNN (RTX 3050 Mixed Precision)
echo ========================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found! Please ensure .venv exists.
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting GPU training on recorded combat sessions in out_vid\...
echo.
python train.py

echo.
echo ========================================================
echo Training finished. Weights updated in models\pvp_model.pth!
echo ========================================================
pause
