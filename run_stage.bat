@echo off
setlocal
rem Run one or more stages, forwarding all arguments to src.main.
rem   run_stage.bat train_eval --models xgboost --splits split_80_20
cd /d "%~dp0"

if "%~1"=="" (
    echo Usage: run_stage.bat ^<stage^> [more stages] [--models ...] [--splits ...] [--sample 0.05] [--force]
    echo Stages: download preprocess split train_eval aggregate visualize report
    exit /b 1
)

call "%~dp0setup_env.bat"
if errorlevel 1 exit /b 1

".venv\Scripts\python.exe" -m src.main --stage %*
exit /b %errorlevel%
