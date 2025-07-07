#!/bin/bash
# run.sh - Simple script to run the ASX Bank Trading Analysis System

echo "ASX Bank Trading Analysis System"
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating..."
    python -m venv venv
fi

# Activate virtual environment
if [ -f "venv/Scripts/activate" ]; then
    # Windows
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    # Linux/Mac
    source venv/bin/activate
else
    echo "Error: Could not find virtual environment activation script"
    exit 1
fi

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Environment file not found. Creating from template..."
    cp .env.example .env
    echo "Please edit .env file with your settings before running the system."
    exit 1
fi

# Run the system
echo "Starting analysis..."
echo
echo "Choose your option:"
echo "1. Run original analysis (main.py)"
echo "2. Run enhanced analysis (enhanced_main.py) [RECOMMENDED]"
echo "3. Run backtesting system (backtesting_system.py)"
echo "4. Run test suite (test_improvements.py)"
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        python main.py analyze
        ;;
    2)
        python enhanced_main.py
        ;;
    3)
        python backtesting_system.py
        ;;
    4)
        python test_improvements.py
        ;;
    *)
        echo "Invalid choice. Running enhanced analysis by default..."
        python enhanced_main.py
        ;;
esac

echo "Done!"
