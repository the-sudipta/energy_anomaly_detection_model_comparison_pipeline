@echo off
setlocal
rem Run the whole benchmark: environment, all 7 stages, then open the report.
rem With earlier results present (and no arguments), a menu asks what to do.
cd /d "%~dp0"

call "%~dp0setup_env.bat"
if errorlevel 1 goto :failed

set "ARGS=%*"
if not "%~1"=="" goto :run
if not exist "outputs\metrics\*.json" goto :run

:menu
echo.
echo  ================================================================
echo   Energy anomaly benchmark - earlier results were found
echo  ================================================================
echo    [1] View the existing results (report + live 3D monitor)
echo    [2] Run the pipeline, reusing finished work (fast if nothing changed)
echo    [3] Retrain everything from scratch (about 45-60 minutes)
echo    [4] Test a meter reading of your own against all four models
echo    [5] Exit
echo.
choice /c 12345 /n /m "  Choose 1-5: "
if errorlevel 5 exit /b 0
if errorlevel 4 goto :predict
if errorlevel 3 (set "ARGS=--force" & goto :run)
if errorlevel 2 goto :run
if errorlevel 1 goto :view

:view
start "Benchmark live monitor" /min ".venv\Scripts\python.exe" -m monitor.server
if exist "outputs\REPORT.html" (start "" "outputs\REPORT.html") else (echo No report yet - choose option 2 first.)
goto :menu

:predict
".venv\Scripts\python.exe" -m src.predict
choice /c yn /n /m "  Test another reading? [y/n] "
if errorlevel 2 goto :menu
goto :predict

:run
rem Open the live progress monitor in its own minimised window (it also opens the browser).
start "Benchmark live monitor" /min ".venv\Scripts\python.exe" -m monitor.server

".venv\Scripts\python.exe" -m src.main --stage all %ARGS%
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
