# ASX Bank Trading Analysis System

## Overview
A comprehensive free trading analysis system for Australian Securities Exchange (ASX) banking stocks. This system provides technical analysis, fundamental analysis, sentiment analysis, and market predictions using only free data sources.

## Features
- **📊 Technical Analysis**: RSI, MACD, Bollinger Bands, Moving Averages, and more
- **💰 Fundamental Analysis**: P/E ratios, dividend yields, financial metrics
- **📰 Sentiment Analysis**: News sentiment and market mood analysis
- **⚖️ Risk/Reward Analysis**: Automatic stop-loss and position sizing
- **🎯 Market Predictions**: Bullish/bearish signals with confidence levels
- **🔔 Smart Alerts**: Discord, Telegram, and email notifications
- **📈 Comprehensive Reports**: HTML reports with interactive charts
- **🤖 Automated Scheduling**: Daily analysis and monitoring

## Quick Start

### Option 1: Use the Run Scripts (Recommended)
```bash
# Windows
run.bat

# Linux/Mac
chmod +x run.sh
./run.sh
```

### Option 2: Manual Setup
1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   copy .env.example .env
   # Edit .env file with your settings
   ```

4. **Run analysis:**
   ```bash
   python main.py analyze
   ```

## Usage

### Basic Commands
```bash
# Analyze all banks
python main.py analyze

# Analyze specific bank
python main.py bank --symbol CBA

# Generate report only
python main.py report

# Start automated scheduler
python main.py schedule

# Enable debug mode
python main.py analyze --debug
```

### Supported Banks
- **CBA.AX** - Commonwealth Bank of Australia
- **WBC.AX** - Westpac Banking Corporation
- **ANZ.AX** - Australia and New Zealand Banking Group
- **NAB.AX** - National Australia Bank
- **MQG.AX** - Macquarie Group

## Configuration

### Environment Variables (.env)
```env
# Alert Settings (optional)
DISCORD_WEBHOOK_URL=your_webhook_url
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Analysis Settings
DEFAULT_ANALYSIS_PERIOD=3mo
CONFIDENCE_THRESHOLD=70
RISK_TOLERANCE=medium

# Cache Settings
CACHE_DURATION_MINUTES=15
MAX_CACHE_SIZE_MB=100
```

### Customization
Edit `config/settings.py` to customize:
- Bank symbols to analyze
- Technical indicator parameters
- Risk management settings
- Alert thresholds
- Report formatting

## Data Sources
All data sources are **100% FREE**:
- **Market Data**: Yahoo Finance API
- **News**: Web scraping from financial news sites
- **Economic Data**: RBA, ABS public data
- **Sentiment**: TextBlob and VaderSentiment (free NLP)

## Output

### Console Output
```
ASX BANK ANALYSIS COMPLETE
==================================================
CBA.AX: BUY (Confidence: 75%)
WBC.AX: HOLD (Confidence: 45%)
ANZ.AX: SELL (Confidence: 60%)
NAB.AX: BUY (Confidence: 80%)
MQG.AX: HOLD (Confidence: 35%)

Detailed report saved to: reports/analysis_2024-01-15_14-30-25.html
```

### Report Files
- **Location**: `reports/` directory
- **Format**: Interactive HTML with charts
- **Content**: Detailed analysis for each bank
- **Charts**: Price charts, indicators, risk metrics

### Alerts
- **Discord**: Rich embeds with color-coded signals
- **Telegram**: Formatted messages with key metrics
- **Email**: HTML emails with detailed analysis
- **Console**: Real-time logging

## Project Structure
```
trading_analysis/
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── run.bat / run.sh          # Quick start scripts
├── setup.py                  # Package installation
├── config/
│   └── settings.py           # Configuration settings
├── src/
│   ├── data_feed.py          # Data fetching
│   ├── technical_analysis.py # Technical indicators
│   ├── fundamental_analysis.py # Fundamental metrics
│   ├── news_sentiment.py     # News sentiment analysis
│   ├── risk_calculator.py    # Risk/reward calculations
│   ├── market_predictor.py   # Market predictions
│   ├── alert_system.py       # Alert notifications
│   └── report_generator.py   # Report generation
├── utils/
│   ├── cache_manager.py      # Data caching
│   ├── validators.py         # Data validation
│   └── helpers.py            # Helper functions
├── tests/                    # Unit tests
├── data/
│   ├── cache/               # Cached API responses
│   └── historical/          # Historical data
├── reports/                 # Generated reports
└── logs/                    # Application logs
```

## Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_data_feed.py

# Run with coverage
python -m pytest tests/ --cov=src
```

## Scheduling

### Windows Task Scheduler
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger (e.g., daily at 9:30 AM)
4. Set action: `python main.py analyze`
5. Set start in: `C:\path\to\trading_analysis`

### Linux/Mac Cron
```bash
# Edit crontab
crontab -e

# Add line for daily analysis at 9:30 AM
30 9 * * 1-5 cd /path/to/trading_analysis && python main.py analyze
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure virtual environment is activated
   - Run: `pip install -r requirements.txt`

2. **No Data Retrieved**
   - Check internet connection
   - Verify bank symbols are correct (must end with .AX)
   - Markets may be closed

3. **Permission Errors**
   - Ensure write permissions for `reports/`, `logs/`, and `data/` directories

4. **Module Not Found**
   - Check Python path
   - Ensure you're in the correct directory

### Debug Mode
```bash
python main.py analyze --debug
```

### Check Logs
```bash
# View recent logs
tail -f logs/trading_system.log

# Windows
type logs\trading_system.log
```

## Performance Tips
1. **Use caching** - Data is cached for 15 minutes by default
2. **Limit API calls** - Don't run analysis more frequently than every 15 minutes
3. **Monitor memory** - Cache size is limited to 100MB by default
4. **Schedule wisely** - Run intensive analysis outside market hours

## Disclaimer
This software is for educational and research purposes only. It is not financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions. The authors are not responsible for any financial losses.

## License
MIT License - see LICENSE file for details

## Support
- Check the `docs/` folder for detailed documentation
- Review logs in `logs/trading_system.log`
- Reports are saved in `reports/` with timestamps
- Open an issue on GitHub for bugs or feature requests

---

**Happy Trading! 🚀**
