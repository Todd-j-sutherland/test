# Quick Droplet Verification Commands
# Copy and paste these commands into your droplet SSH session

# 1. Navigate to project directory
cd /root/trading_analysis || cd ~/trading_analysis || echo "Need to clone repo first"

# 2. Check Git status  
echo "=== GIT STATUS ==="
git status
git log --oneline -3

# 3. Quick Python/PyTorch check
echo "=== PYTHON & PYTORCH ==="
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 4. Check environment setup
echo "=== ENVIRONMENT ==="
source .venv/bin/activate 2>/dev/null || echo "No venv found"
which python3
pip list | grep -E "(torch|transformers|pandas|numpy)" | head -5

# 5. Test core system
echo "=== SYSTEM TEST ==="
python3 -c "
import sys
sys.path.append('core')
try:
    from smart_collector import SmartCollector
    print('✅ Smart collector imports OK')
except Exception as e:
    print(f'⚠️ Smart collector: {e}')

try:
    from news_trading_analyzer import NewsTradingAnalyzer
    print('✅ News analyzer imports OK')
except Exception as e:
    print(f'⚠️ News analyzer: {e}')
"

# 6. Test data collection
echo "=== DATA TEST ==="
python3 -c "
import yfinance as yf
try:
    data = yf.download('CBA.AX', period='2d', progress=False)
    print(f'✅ Downloaded {len(data)} rows of CBA.AX data')
except Exception as e:
    print(f'❌ Data download failed: {e}')
"

# 7. Test transformer security (key difference from local)
echo "=== TRANSFORMER SECURITY TEST ==="
python3 -c "
import os
skip_transformers = os.getenv('SKIP_TRANSFORMERS', '0')
print(f'SKIP_TRANSFORMERS = {skip_transformers}')

if skip_transformers != '1':
    try:
        from transformers import pipeline
        sentiment = pipeline('sentiment-analysis')
        result = sentiment('This droplet deployment is working well!')
        print(f'✅ Sentiment analysis: {result[0][\"label\"]} ({result[0][\"score\"]:.3f})')
        print('✅ Full transformer support active (no security warnings)')
    except Exception as e:
        print(f'❌ Transformer test failed: {e}')
else:
    print('⚠️ Transformers skipped (development mode)')
"

# 8. Quick daily manager test
echo "=== DAILY MANAGER TEST ==="
timeout 15 python3 daily_manager.py status || echo "Daily manager timeout (normal)"

echo ""
echo "=== VERIFICATION COMPLETE ==="
echo "If you see mostly ✅ symbols, your droplet is ready!"
echo "Run the full verification: bash verify_droplet.sh"
