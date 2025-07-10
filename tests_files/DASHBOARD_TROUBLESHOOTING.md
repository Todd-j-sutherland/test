# Dashboard Troubleshooting Guide

## ✅ Working Launch Command

The following command is confirmed to work:

```bash
cd /Users/toddsutherland/Repos/trading_analysis && source .venv/bin/activate && python launch_dashboard_auto.py
```

## 🔧 Common Issues and Solutions

### 1. Streamlit Email Prompt
**Problem**: Streamlit shows email prompt and dashboard won't start
**Solution**: Use the `launch_dashboard_auto.py` script which bypasses this automatically

### 2. "Site Cannot Be Reached"
**Problem**: Browser shows "site cannot be reached" at localhost:8501
**Solution**: This happens when Streamlit is stuck at the email prompt. Use the automated launcher.

### 3. Dashboard Not Loading
**Problem**: Dashboard starts but doesn't load properly
**Solution**: Check that all dependencies are installed:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Technical Analysis Errors
**Problem**: Technical analysis fails to load data
**Solution**: Ensure yfinance is installed:
```bash
pip install yfinance
```

### 5. Virtual Environment Issues
**Problem**: Commands don't work or packages missing
**Solution**: Always activate the virtual environment first:
```bash
source .venv/bin/activate
```

## 🎯 Quick Testing

To test if everything is working:

1. **Test technical analysis:**
   ```bash
   python test_technical_analysis.py
   ```

2. **Launch dashboard:**
   ```bash
   python launch_dashboard_auto.py
   ```

3. **Open browser to:** http://localhost:8501

## 📋 System Requirements

- Python 3.8+
- Virtual environment activated
- All dependencies from requirements.txt installed
- Internet connection for market data

## 🚀 Features Available

- News sentiment analysis for ASX banks
- Technical indicators (RSI, MACD, momentum)
- Combined trading recommendations
- Interactive charts and visualizations
- Individual bank analysis pages
