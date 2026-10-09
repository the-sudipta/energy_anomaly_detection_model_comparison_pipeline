@echo off
setlocal
rem Create .venv and install requirements once. Safe to call repeatedly.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo         Install Python 3.10 or newer from https://www.python.org/downloads/
    echo         and tick "Add python.exe to PATH" during installation.
    exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo [ERROR] Python 3.10 or newer is required. Found:
    python --version
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 exit /b 1
)

set "STAMP=.venv\.requirements_installed"
fc /b requirements.txt "%STAMP%" >nul 2>nul
if errorlevel 1 (
    echo Installing requirements ...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Installing requirements failed. See the messages above.
        exit /b 1
    )
    copy /y requirements.txt "%STAMP%" >nul
) else (
    echo Environment already up to date.
)
exit /b 0
