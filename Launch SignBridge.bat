@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ========================================
echo SignBridge - first-run setup and launch
echo ========================================
echo.

set "SB_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%SB_PY%" (
    echo Creating a local Python 3.14 virtual environment...
    where py >nul 2>&1
    if errorlevel 1 goto no_python
    py -3.14 --version >nul 2>&1
    if errorlevel 1 goto no_python
    py -3.14 -m venv "%~dp0.venv"
    if errorlevel 1 goto setup_failed
)

"%SB_PY%" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,14) and sys.maxsize > 2**32 else 1)"
if errorlevel 1 goto wrong_python

echo Checking and installing the pinned dependencies if needed...
"%SB_PY%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto setup_failed

echo.
echo Checking the SignBridge environment...
"%SB_PY%" "%~dp0scripts\check_environment.py"
if errorlevel 1 goto setup_failed

if /I "%~1"=="--setup-only" (
    echo.
    echo Setup check complete. The app was not started because --setup-only was requested.
    exit /b 0
)

echo.
echo Starting SignBridge. Close the SignBridge window to exit.
"%SB_PY%" "%~dp0app.py"
if errorlevel 1 goto app_failed
exit /b 0

:no_python
echo Python 3.14 (64-bit) and the Windows Python Launcher were not found.
echo Install CPython 3.14 (64-bit), then double-click this launcher again.
goto failed

:wrong_python
echo The existing .venv is not 64-bit CPython 3.14.
echo Rename or remove the .venv folder manually, then double-click this launcher again.
goto failed

:setup_failed
echo Setup did not complete. Check the messages above and your internet connection,
echo then double-click this launcher again.
goto failed

:app_failed
echo SignBridge exited with an error. Check the local log under %%LOCALAPPDATA%%\SignBridge\logs\.
goto failed

:failed
echo.
pause
exit /b 1
