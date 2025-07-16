# 🚀 Quick Start Guide

Get up and running with the Trading Analysis System in minutes!

## 📋 Prerequisites

- **Python 3.8+** installed on your system
- **Git** for cloning the repository
- **Virtual environment** support (recommended)

## ⚡ 5-Minute Setup

### 1. Clone & Setup
```bash
# Clone the repository
git clone https://github.com/your-username/trading_analysis.git
cd trading_analysis

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

### 2. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

### 3. Test Installation
```bash
# Check if everything is working
python -m app.main status
```

You should see output like:
```
📊 QUICK STATUS CHECK
Trading Analysis System Status: ✅ Operational
✅ Enhanced Sentiment Integration: Available
✅ Core modules: Loaded successfully
✅ Dashboard: Ready to launch
```

## 🎯 First Commands

### Check System Status
```bash
python -m app.main status
```
- Validates all components are working
- Shows system health
- Confirms data availability

### Morning Analysis
```bash
python -m app.main morning --dry-run
```
- Runs morning market analysis
- Collects latest news sentiment
- Generates trading insights
- `--dry-run` flag shows what would happen without making changes

### Evening Summary
```bash
python -m app.main evening --dry-run
```
- Daily performance summary
- Market closure analysis
- Sentiment trend analysis

### Launch Dashboard
```bash
python -m app.main dashboard
```
- Opens interactive web interface
- Real-time market visualization
- Professional trading dashboard
- Access at `http://localhost:8501`

## 📊 Dashboard Tour

When you launch the dashboard, you'll see:

1. **📈 Market Overview**
   - Real-time sentiment scores for major ASX banks
   - Confidence metrics and trend indicators
   - Market regime analysis

2. **🏦 Bank Analysis**
   - Detailed analysis for CBA, WBC, ANZ, NAB, MQG
   - Historical sentiment trends
   - Technical indicator integration

3. **📰 News Sentiment**
   - Real-time news analysis
   - Sentiment scoring with confidence levels
   - Source attribution and timestamps

4. **📊 Technical Charts**
   - Interactive Plotly visualizations
   - Sentiment correlation analysis
   - Historical trend analysis

## 🧪 Validate Installation

Run the test suite to ensure everything is working:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v        # Unit tests (19 tests)
python -m pytest tests/integration/ -v # Integration tests (44 tests)
```

Expected output:
```
======================== 63 passed in 2.45s ========================
```

## 🔧 Configuration

### Environment Setup (Optional)
Create a `.env` file for custom configuration:
```bash
# Copy example configuration
cp .env.example .env

# Edit with your preferences
nano .env
```

Common settings:
```env
# Logging level
LOG_LEVEL=INFO

# Data directory
DATA_DIR=data

# Dashboard port
DASHBOARD_PORT=8501
```

### Data Initialization
The system will automatically create necessary directories:
- `data/` - Market and sentiment data storage
- `logs/` - Application logging
- `reports/` - Generated analysis reports

## 💡 Common Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `status` | System health check | `python -m app.main status` |
| `morning` | Morning briefing | `python -m app.main morning` |
| `evening` | Evening summary | `python -m app.main evening` |
| `dashboard` | Web interface | `python -m app.main dashboard` |

## 🎯 Next Steps

### Explore the Dashboard
1. Launch dashboard: `python -m app.main dashboard`
2. Navigate to different sections
3. Interact with charts and visualizations
4. Review sentiment analysis results

### Customize Analysis
1. Check `app/config/settings.py` for configuration options
2. Modify sentiment analysis parameters in `app/config/ml_config.yaml`
3. Add new data sources in `app/core/data/collectors/`

### Scheduled Operations
Set up daily automation:
```bash
# Add to crontab for daily automation
# Morning analysis at 9:00 AM
0 9 * * * cd /path/to/trading_analysis && source venv/bin/activate && python -m app.main morning

# Evening summary at 6:00 PM
0 18 * * * cd /path/to/trading_analysis && source venv/bin/activate && python -m app.main evening
```

## 🆘 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**Port Already in Use (Dashboard)**
```bash
# Use different port
python -m streamlit run app/dashboard/main.py --server.port 8502
```

**Missing Data Directories**
```bash
# The system creates these automatically, but you can create manually:
mkdir -p data logs reports
```

**Permission Issues**
```bash
# Ensure proper permissions
chmod +x venv/bin/activate
```

### Getting Help

1. **Check logs**: Look in `logs/` directory for error details
2. **Run tests**: `python -m pytest tests/ -v` to identify issues
3. **Verbose mode**: Add `--verbose` flag to commands for detailed output
4. **System status**: Always start with `python -m app.main status`

## 🔗 Key Files

- **Main Entry**: `app/main.py` - CLI interface
- **Configuration**: `app/config/settings.py` - System settings
- **Dashboard**: `app/dashboard/main.py` - Web interface
- **Core Logic**: `app/core/sentiment/` - Analysis engine
- **Tests**: `tests/` - Validation suite

## ✅ Success Indicators

You'll know everything is working when:
- ✅ `python -m app.main status` shows all green checkmarks
- ✅ Dashboard loads without errors at `http://localhost:8501`
- ✅ Morning/evening commands complete successfully
- ✅ Test suite passes: `python -m pytest tests/ -v`

**🎉 You're ready to start trading with AI-powered insights!**
