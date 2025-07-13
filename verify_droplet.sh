#!/bin/bash
# DigitalOcean Droplet Verification Script
# Run this script to verify your trading system is properly deployed

set -e

echo "🚀 DIGITALOCEAN DROPLET VERIFICATION"
echo "======================================"
echo "Timestamp: $(date)"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo ""

# Check system info
echo "📋 SYSTEM INFORMATION"
echo "---------------------"
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"')"
echo "Python: $(python3 --version)"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2 " total, " $3 " used, " $4 " available"}')"
echo "Disk: $(df -h / | tail -1 | awk '{print $2 " total, " $3 " used, " $4 " available"}')"
echo ""

# Check if we're in the right directory
echo "📁 DIRECTORY VERIFICATION"
echo "-------------------------"
echo "Current directory: $(pwd)"
if [ ! -f "daily_manager.py" ]; then
    echo "❌ Not in trading_analysis directory! Looking for it..."
    
    # Try to find the directory
    if [ -d "/root/trading_analysis" ]; then
        echo "✅ Found at /root/trading_analysis"
        cd /root/trading_analysis
    elif [ -d "~/trading_analysis" ]; then
        echo "✅ Found at ~/trading_analysis"
        cd ~/trading_analysis
    elif [ -d "./trading_analysis" ]; then
        echo "✅ Found at ./trading_analysis"
        cd ./trading_analysis
    else
        echo "❌ trading_analysis directory not found!"
        echo "Please run: git clone https://github.com/Todd-j-sutherland/trading_analysis.git"
        exit 1
    fi
else
    echo "✅ In trading_analysis directory"
fi

echo "Working directory: $(pwd)"
echo "Files present: $(ls -la | wc -l) items"
echo ""

# Check Git status
echo "🔧 GIT REPOSITORY STATUS"
echo "------------------------"
if [ -d ".git" ]; then
    echo "✅ Git repository detected"
    echo "Remote URL: $(git remote get-url origin 2>/dev/null || echo 'No remote configured')"
    echo "Current branch: $(git branch --show-current 2>/dev/null || echo 'No branch info')"
    echo "Latest commit: $(git log --oneline -1 2>/dev/null || echo 'No commit history')"
    echo "Status: $(git status --porcelain | wc -l) uncommitted changes"
else
    echo "❌ Not a Git repository"
fi
echo ""

# Check Python environment
echo "🐍 PYTHON ENVIRONMENT"
echo "---------------------"
if [ -d ".venv" ]; then
    echo "✅ Virtual environment found"
    source .venv/bin/activate
    echo "Activated venv: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment found"
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "✅ Virtual environment created and activated"
fi

echo "Python path: $(which python3)"
echo "Pip version: $(pip --version)"
echo ""

# Check requirements
echo "📦 PACKAGE REQUIREMENTS"
echo "-----------------------"
if [ -f "requirements-production.txt" ]; then
    echo "✅ Production requirements file found"
    REQ_FILE="requirements-production.txt"
elif [ -f "requirements.txt" ]; then
    echo "⚠️  Using development requirements (consider requirements-production.txt)"
    REQ_FILE="requirements.txt"
else
    echo "❌ No requirements file found!"
    exit 1
fi

echo "Requirements file: $REQ_FILE"
echo "Installing/updating packages..."
pip install -r $REQ_FILE > /tmp/pip_install.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Package installation successful"
else
    echo "❌ Package installation failed! Check /tmp/pip_install.log"
    tail -10 /tmp/pip_install.log
fi
echo ""

# Check PyTorch version and security
echo "🔥 PYTORCH & SECURITY CHECK"
echo "---------------------------"
python3 -c "
import torch
import sys
print(f'PyTorch version: {torch.__version__}')

# Check version for security fix
version_parts = torch.__version__.split('.')
major, minor = int(version_parts[0]), int(version_parts[1])
if major > 2 or (major == 2 and minor >= 6):
    print('✅ PyTorch version includes CVE-2025-32434 security fix')
else:
    print('⚠️  PyTorch version may have security vulnerability CVE-2025-32434')

# Test transformer loading
try:
    from transformers import pipeline
    sentiment = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    result = sentiment('This is a test of transformer models on the droplet.')
    print(f'✅ Transformer test successful: {result[0][\"label\"]} ({result[0][\"score\"]:.3f})')
except Exception as e:
    print(f'❌ Transformer test failed: {e}')

print('✅ PyTorch security and transformer check complete')
"
echo ""

# Check environment configuration
echo "🔧 ENVIRONMENT CONFIGURATION"
echo "----------------------------"
if [ -f ".env" ]; then
    echo "✅ .env file found"
    
    # Check SKIP_TRANSFORMERS setting
    if grep -q "SKIP_TRANSFORMERS=1" .env; then
        echo "⚠️  SKIP_TRANSFORMERS=1 detected (development setting)"
        echo "For production, consider setting SKIP_TRANSFORMERS=0"
    elif grep -q "SKIP_TRANSFORMERS=0" .env; then
        echo "✅ SKIP_TRANSFORMERS=0 (production setting)"
    else
        echo "ℹ️  SKIP_TRANSFORMERS not explicitly set"
    fi
    
    echo "Environment variables loaded: $(grep -c '^[A-Z]' .env || echo 0)"
else
    echo "❌ .env file not found!"
    echo "Creating basic .env file..."
    cp .env.example .env 2>/dev/null || echo "No .env.example found either"
fi
echo ""

# Test core system components
echo "🧪 SYSTEM COMPONENT TESTS"
echo "-------------------------"

# Test daily_manager
if [ -f "daily_manager.py" ]; then
    echo "Testing daily_manager.py..."
    timeout 10 python3 daily_manager.py status > /tmp/daily_manager_test.log 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ daily_manager.py working"
        grep -E "(Samples|Win Rate|Performance)" /tmp/daily_manager_test.log || echo "No specific metrics found"
    else
        echo "⚠️  daily_manager.py test timeout or error"
        tail -3 /tmp/daily_manager_test.log
    fi
else
    echo "❌ daily_manager.py not found"
fi

# Test enhanced_main
if [ -f "enhanced_main.py" ]; then
    echo "Testing enhanced_main.py import..."
    python3 -c "
try:
    import enhanced_main
    print('✅ enhanced_main.py imports successfully')
except Exception as e:
    print(f'❌ enhanced_main.py import failed: {e}')
"
else
    echo "❌ enhanced_main.py not found"
fi

# Test core modules
echo "Testing core modules..."
python3 -c "
import sys
import os
sys.path.append('core')
sys.path.append('src')

modules_to_test = [
    ('smart_collector', 'core/smart_collector.py'),
    ('news_trading_analyzer', 'core/news_trading_analyzer.py'),
    ('advanced_paper_trading', 'core/advanced_paper_trading.py')
]

for module_name, file_path in modules_to_test:
    if os.path.exists(file_path):
        try:
            module = __import__(module_name)
            print(f'✅ {module_name} imports successfully')
        except Exception as e:
            print(f'⚠️  {module_name} import issue: {e}')
    else:
        print(f'❌ {file_path} not found')
"
echo ""

# Check data directories
echo "📊 DATA DIRECTORY STRUCTURE"
echo "---------------------------"
REQUIRED_DIRS=("data" "data/historical" "data/cache" "data/ml_models" "logs" "reports")

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ $dir/ exists ($(ls $dir | wc -l) files)"
    else
        echo "⚠️  $dir/ missing - creating..."
        mkdir -p "$dir"
        echo "✅ $dir/ created"
    fi
done
echo ""

# Test network connectivity
echo "🌐 NETWORK CONNECTIVITY"
echo "-----------------------"
echo "Testing external connections..."

# Test Yahoo Finance (key data source)
python3 -c "
import requests
try:
    response = requests.get('https://finance.yahoo.com', timeout=5)
    if response.status_code == 200:
        print('✅ Yahoo Finance accessible')
    else:
        print(f'⚠️  Yahoo Finance returned status {response.status_code}')
except Exception as e:
    print(f'❌ Yahoo Finance connection failed: {e}')

# Test GitHub (for updates)
try:
    response = requests.get('https://github.com', timeout=5)
    if response.status_code == 200:
        print('✅ GitHub accessible')
    else:
        print(f'⚠️  GitHub returned status {response.status_code}')
except Exception as e:
    print(f'❌ GitHub connection failed: {e}')
"
echo ""

# Final system test
echo "🎯 FINAL SYSTEM TEST"
echo "--------------------"
echo "Running comprehensive system verification..."

python3 -c "
# Test the complete system pipeline
import os
import sys
sys.path.append('tools')

try:
    # Test data collection capability
    import yfinance as yf
    data = yf.download('CBA.AX', period='5d', interval='1d', progress=False)
    if not data.empty:
        print(f'✅ Data collection test: Retrieved {len(data)} days of CBA.AX data')
    else:
        print('⚠️  Data collection test: No data retrieved')
        
    # Test ML pipeline
    from src.ml_training_pipeline import MLTrainingPipeline
    pipeline = MLTrainingPipeline()
    print('✅ ML Pipeline initialized successfully')
    
    # Test sentiment analysis
    from transformers import pipeline as hf_pipeline
    sentiment_analyzer = hf_pipeline('sentiment-analysis')
    result = sentiment_analyzer('The market is performing well today.')
    print(f'✅ Sentiment analysis test: {result[0][\"label\"]} confidence')
    
    print('✅ All core components working correctly')
    
except Exception as e:
    print(f'⚠️  System test encountered issue: {e}')
    print('System may still function with reduced capabilities')
"

echo ""
echo "🎉 VERIFICATION COMPLETE"
echo "========================"
echo "Timestamp: $(date)"

# System status summary
echo ""
echo "📋 DEPLOYMENT STATUS SUMMARY:"
echo "✅ = Working correctly"
echo "⚠️  = Working with warnings" 
echo "❌ = Needs attention"
echo ""
echo "Next steps:"
echo "1. If all green ✅: Run 'python3 daily_manager.py morning' to start"
echo "2. If warnings ⚠️ : Review warnings but system should work"
echo "3. If errors ❌: Fix issues before proceeding"
echo ""
echo "Useful commands:"
echo "  python3 daily_manager.py status    # Quick status check"
echo "  python3 daily_manager.py morning   # Start trading day"
echo "  python3 enhanced_main.py           # Run main system"
echo ""
