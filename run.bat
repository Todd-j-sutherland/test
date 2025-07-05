@echo off
REM run.bat - Simple script to run the ASX Bank Trading Analysis System on Windows

echo ASX Bank Trading Analysis System
echo ================================

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Creating...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Check if .env file exists
if not exist ".env" (
    echo Environment file not found. Creating from template...
    copy .env.example .env
    echo Please edit .env file with your settings before running the system.
    pause
    exit /b 1
)

REM Run the system
echo Starting analysis...
python main.py analyze

echo Done!
pause
