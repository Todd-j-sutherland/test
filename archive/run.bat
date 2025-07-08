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
echo.
echo Choose your option:
echo 1. Run original analysis (main.py)
echo 2. Run enhanced analysis (enhanced_main.py) [RECOMMENDED]
echo 3. Run backtesting system (backtesting_system.py)
echo 4. Run test suite (test_improvements.py)
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    python main.py analyze
) else if "%choice%"=="2" (
    python enhanced_main.py
) else if "%choice%"=="3" (
    python backtesting_system.py
) else if "%choice%"=="4" (
    python test_improvements.py
) else (
    echo Invalid choice. Running enhanced analysis by default...
    python enhanced_main.py
)

echo Done!
pause
