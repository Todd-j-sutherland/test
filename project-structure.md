# ASX Bank Trading Analysis System - Project Structure

## Directory Structure
```
asx-bank-analyzer/
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── setup.py                   # Installation script
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore file
│
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuration settings
│
├── src/
│   ├── __init__.py
│   ├── data_feed.py          # Data fetching from free APIs
│   ├── technical_analysis.py # Technical indicators
│   ├── fundamental_analysis.py # Fundamental metrics
│   ├── news_sentiment.py     # News and sentiment analysis
│   ├── risk_calculator.py    # Risk/reward calculations
│   ├── market_predictor.py   # Market predictions
│   ├── alert_system.py       # Alert notifications
│   └── report_generator.py   # Report generation
│
├── utils/
│   ├── __init__.py
│   ├── cache_manager.py      # Data caching
│   ├── validators.py         # Data validation
│   └── helpers.py            # Helper functions
│
├── tests/
│   ├── __init__.py
│   ├── test_data_feed.py
│   ├── test_analysis.py
│   └── test_predictions.py
│
├── data/
│   ├── cache/               # Cached API responses
│   └── historical/          # Historical data storage
│
├── reports/                 # Generated reports
├── logs/                    # Application logs
└── docs/                    # Additional documentation
```

## Installation Instructions

### 1. Clone or Create the Project
```bash
mkdir asx-bank-analyzer
cd asx-bank-analyzer
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### 3. Create requirements.txt
```txt
# Core dependencies
yfinance==0.2.28
pandas==2.1.3
numpy==1.24.3
requests==2.31.0
beautifulsoup4==4.12.2
lxml==4.9.3

# Technical Analysis
ta==0.10.2
pandas-ta==0.3.14b0

# Data Processing
python-dotenv==1.0.0
schedule==1.2.0

# Visualization (for reports)
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0

# Web scraping
selenium==4.15.2
webdriver-manager==4.0.1

# Natural Language Processing (free)
textblob==0.17.1
vaderSentiment==3.3.2

# Reporting
jinja2==3.1.2
weasyprint==60.1  # Optional: for PDF reports

# Development
pytest==7.4.3
black==23.11.0
flake8==6.1.0

# Optional: Discord/Telegram alerts
discord-webhook==1.3.0
python-telegram-bot==20.6
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Create .env File
```bash
# Copy from .env.example
cp .env.example .env
```

Edit .env with your settings:
```env
# Alert Settings (optional)
DISCORD_WEBHOOK_URL=your_discord_webhook_url_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_app_password_here

# Cache Settings
CACHE_DURATION_MINUTES=15
MAX_CACHE_SIZE_MB=100

# Analysis Settings
DEFAULT_ANALYSIS_PERIOD=3mo
CONFIDENCE_THRESHOLD=70
RISK_TOLERANCE=medium

# Data Sources (all free)
USE_YAHOO_FINANCE=true
USE_WEB_SCRAPING=true
USE_FREE_NEWS_API=true
```

### 6. Create Initial Directories
```bash
mkdir -p src config utils tests data/cache data/historical reports logs docs
```

### 7. Create __init__.py Files
```bash
touch src/__init__.py config/__init__.py utils/__init__.py tests/__init__.py
```

## Running the System

### First Run - Test Analysis
```bash
python main.py analyze
```

### Analyze Specific Bank
```bash
python main.py bank CBA
python main.py bank WBC
```

### Start Automated Scheduler
```bash
python main.py schedule
```

### Run Tests
```bash
pytest tests/
```

## Quick Start Guide

1. **Basic Analysis** - Just run `python main.py analyze` to get immediate insights on all major banks

2. **Morning Routine** - Set up a cron job or Windows Task Scheduler:
   ```bash
   # Linux/Mac crontab -e
   30 8 * * 1-5 cd /path/to/asx-bank-analyzer && /path/to/venv/bin/python main.py analyze
   ```

3. **View Reports** - Check the `reports/` folder for HTML reports that open in your browser

4. **Set Up Alerts** - Add your Discord webhook URL to .env to get instant notifications

## Features Overview

### 🎯 Risk/Reward Analysis
- Automatic stop loss calculation
- Position sizing recommendations
- Risk score (0-100)
- Reward potential estimation

### 📊 Market Prediction
- Bullish/Bearish signals
- Confidence levels
- Support/resistance levels
- Trend strength indicators

### 📰 Sentiment Analysis
- Australian financial news scanning
- RBA announcement monitoring
- Market sentiment scoring
- Social media sentiment (Reddit/Twitter)

### 🔔 Smart Alerts
- Strong buy/sell signals
- Risk warnings
- Dividend announcements
- Major news events

### 📈 Technical Indicators
- RSI, MACD, Bollinger Bands
- Moving averages (SMA, EMA)
- Volume analysis
- Custom Pine Script indicators

### 💰 Fundamental Analysis
- P/E ratios
- Dividend yields
- Book value
- ROE and other metrics

## Customization

### Adding New Banks
Edit `config/settings.py`:
```python
BANK_SYMBOLS = ['CBA.AX', 'WBC.AX', 'ANZ.AX', 'NAB.AX', 'MQG.AX', 'BEN.AX']
```

### Changing Alert Thresholds
Edit `config/settings.py`:
```python
ALERT_THRESHOLDS = {
    'strong_buy': 70,
    'buy': 50,
    'sell': -50,
    'strong_sell': -70
}
```

### Adding Custom Indicators
Add to `src/technical_analysis.py`:
```python
def calculate_custom_indicator(self, data):
    # Your custom logic here
    pass
```

## Troubleshooting

### Common Issues

1. **"No module named 'yfinance'"**
   - Make sure virtual environment is activated
   - Run: `pip install -r requirements.txt`

2. **"Connection error"**
   - Check internet connection
   - Yahoo Finance may be temporarily down

3. **"No data returned"**
   - ASX market might be closed
   - Check if symbol is correct (must end with .AX)

### Debug Mode
```bash
# Run with debug logging
python main.py analyze --debug
```

## Performance Tips

1. **Use Caching** - The system caches data for 15 minutes by default
2. **Limit API Calls** - Don't run analysis more than once per 15 minutes
3. **Schedule Wisely** - Run intensive analysis outside market hours

## Next Steps

1. Start with paper trading using the signals
2. Track performance in the reports
3. Adjust thresholds based on results
4. Consider adding paid data sources as you profit

## Support

- Check `docs/` folder for detailed documentation
- Logs are in `logs/trading_system.log`
- Reports are saved in `reports/` with timestamps
