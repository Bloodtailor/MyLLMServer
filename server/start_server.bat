@echo off
REM Run from this script's own directory no matter where it was launched from
pushd "%~dp0"

echo Starting LLM Server...
echo.

REM Get the IP address to display to the user
ipconfig | findstr /C:"IPv4 Address"
echo.
echo Note the IP address above - you'll need to enter it in your Android app.
echo.

if not exist "venv\Scripts\python.exe" (
    echo ERROR: No virtual environment found in "%CD%\venv".
    echo Run: python setup_environment.py
    echo.
    popd
    pause
    exit /b 1
)

set "PYTHON=venv\Scripts\python.exe"

REM Install the pinned dependency set if the venv is empty/incomplete.
REM Note: requirements.txt pins the CPU-only llama-cpp-python wheel from PyPI.
REM For GPU offload, run setup_environment.py instead.
"%PYTHON%" -c "import flask" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Installing dependencies from requirements.txt...
    "%PYTHON%" -m pip install -r requirements.txt
    REM Must be "if errorlevel", not the percent form: inside a parenthesized
    REM block cmd.exe substitutes percent-variables at parse time, i.e. before
    REM pip has run, so the percent form would always see the stale exit code.
    if errorlevel 1 (
        echo ERROR: Dependency installation failed.
        popd
        pause
        exit /b 1
    )
)

echo.
echo Starting server on port 5000...
echo Press Ctrl+C to stop the server.
echo.

"%PYTHON%" server.py

REM In case the server exits
echo.
echo Server stopped.
popd
pause
