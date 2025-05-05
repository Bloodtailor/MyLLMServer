@echo off
echo Starting LLM Server...
echo.

REM Get the IP address to display to the user
ipconfig | findstr /C:"IPv4 Address"
echo.
echo Note the IP address above - you'll need to enter it in your Android app.
echo.

REM Activate virtual environment if it exists, otherwise use system Python
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found, using system Python...
)

REM Check if required packages are installed
echo Checking required packages...
python -c "import flask" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Installing Flask...
    pip install flask
)

python -c "import flask_cors" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Installing Flask-CORS...
    pip install flask-cors
)

python -c "import llama_cpp" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Installing llama-cpp-python...
    pip install llama-cpp-python
)

echo.
echo Starting server on port 5000...
echo Press Ctrl+C to stop the server.
echo.

REM Start the server
python server.py

REM In case the server exits
echo.
echo Server stopped.
pause