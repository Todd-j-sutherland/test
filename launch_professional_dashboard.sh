#!/bin/bash

# Launch script for Professional ASX Bank Analytics Dashboard
# This script sets up the environment and runs the dashboard

echo "🚀 Starting ASX Bank Analytics Dashboard..."
echo "=====================================\n"

# Navigate to project directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Install/update required packages
echo "📚 Installing required packages..."
pip install -q streamlit plotly numpy pandas yfinance python-dotenv scikit-learn

# Check if dashboard file exists
if [ ! -f "professional_dashboard.py" ]; then
    echo "❌ Error: professional_dashboard.py not found!"
    exit 1
fi

# Launch the dashboard
echo "🎯 Launching Professional Dashboard on port 8503..."
echo "\n📊 Dashboard Features:"
echo "  • Real-time sentiment analysis"
echo "  • Technical analysis integration"
echo "  • Professional visualization"
echo "  • Multi-bank comparison"
echo "  • Historical trends analysis"
echo "\n🌐 Access URLs:"
echo "  • Local:    http://localhost:8503"
echo "  • Network:  http://$(ipconfig getifaddr en0):8503"
echo "\n⌨️  Press Ctrl+C to stop the dashboard"
echo "=====================================\n"

# Run streamlit
streamlit run professional_dashboard.py --server.port 8503

echo "\n🛑 Dashboard stopped."
