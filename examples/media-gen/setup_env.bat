@echo off
REM Setup script for Media Generation Example (Windows)
REM This script creates a virtual environment and installs dependencies

echo Setting up Media Generation Example environment...

REM Check if Python 3 is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python 3 is required but not installed.
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

echo.
echo ✅ Environment setup complete!
echo.
echo To activate the environment in the future:
echo   venv\Scripts\activate.bat
echo.
echo To run the example:
echo   python example_usage.py
echo.
echo To run tests:
echo   cd tests ^&^& python test_dummy_media_gen.py 