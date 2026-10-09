@echo off
setlocal
rem Run the whole benchmark: environment, all 7 stages, then open the report.
cd /d "%~dp0"

call "%~dp0setup_env.bat"
if errorlevel 1 goto :failed

rem Open the live progress monitor in its own minimised window (it also opens the browser).
start "Benchmark live monitor" /min ".venv\Scripts\python.exe" -m monitor.server

".venv\Scripts\python.exe" -m src.main --stage all %*
if errorlevel 1 goto :failed

if exist "outputs\REPORT.html" start "" "outputs\REPORT.html"
echo.
echo Finished. Results are in the outputs folder.
pause
exit /b 0

:failed
set "CODE=%errorlevel%"
echo.
echo [ERROR] The pipeline stopped with exit code %CODE%. Check the messages above
echo         and the newest file in outputs\logs.
pause
exit /b %CODE%
