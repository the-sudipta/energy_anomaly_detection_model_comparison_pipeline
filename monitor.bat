@echo off
setlocal
rem Open the live progress monitor for a running (or finished) pipeline.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run run.bat once first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m monitor.server
pause
