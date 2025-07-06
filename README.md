# ASX Bank Trading Analysis System

A comprehensive free trading analysis system for Australian Stock Exchange (ASX) banking stocks. This system analyzes technical indicators, fundamental metrics, news sentiment, and generates trading signals with risk/reward calculations.

## 🚀 Features

- **Technical Analysis**: RSI, MACD, Moving Averages, Bollinger Bands, Support/Resistance levels
- **Fundamental Analysis**: P/E ratios, ROE, dividend yields, bank-specific metrics
- **News Sentiment Analysis**: Real-time news sentiment from RSS feeds and social media
- **Risk Management**: Position sizing, stop-loss calculations, risk/reward ratios
- **Market Prediction**: ML-based prediction engine with confidence scores
- **Automated Alerts**: Discord, Telegram, and email notifications
- **Comprehensive Reports**: HTML reports with charts and analysis
- **Data Caching**: Efficient caching to minimize API calls

## 📊 Supported Banks

- **CBA.AX** - Commonwealth Bank of Australia
- **WBC.AX** - Westpac Banking Corporation
- **ANZ.AX** - ANZ Banking Group
- **NAB.AX** - National Australia Bank
- **MQG.AX** - Macquarie Group

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection for data feeds

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd trading_analysis
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Environment Configuration (Optional)

Create a `.env` file in the root directory for alert notifications:

```bash
# Discord Webhook (Optional)
DISCORD_WEBHOOK_URL=your_discord_webhook_url

# Telegram Bot (Optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Email Alerts (Optional)
EMAIL_ADDRESS=your_email@example.com
EMAIL_PASSWORD=your_app_password
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
```

## 📈 Usage

### Command Line Interface

The system provides several commands:

#### 1. Analyze All Banks
```bash
python main.py analyze
```
Performs complete analysis of all ASX banks and generates a report.

#### 2. Analyze Single Bank
```bash
python main.py bank --symbol CBA.AX
```
Analyzes a specific bank with detailed output.

#### 3. Generate Report Only
```bash
python main.py report
```
Generates analysis report without running new analysis.

#### 4. Start Automated Scheduler
```bash
python main.py schedule
```
Runs automated analysis at market open (9:30 AM) and afternoon check (3:00 PM).

#### 5. Debug Mode
```bash
python main.py analyze --debug
```
Runs with detailed debug logging.

### Python API

You can also use the system programmatically:

```python
from main import ASXBankTradingSystem

# Initialize system
system = ASXBankTradingSystem()

# Analyze single bank
result = system.analyze_bank('CBA.AX')
print(f"Signal: {result['prediction']['direction']}")
print(f"Confidence: {result['prediction']['confidence']}%")

# Analyze all banks
results = system.analyze_all_banks()
for symbol, analysis in results.items():
    if 'prediction' in analysis:
        print(f"{symbol}: {analysis['prediction']['direction']}")
```

## 🏗️ System Architecture

### Core Components

1. **Data Feed (`src/data_feed.py`)**
   - Fetches real-time and historical data from Yahoo Finance
   - Handles data caching and error recovery
   - Provides company information and market data

2. **Technical Analysis (`src/technical_analysis.py`)**
   - Calculates 20+ technical indicators
   - Identifies trends and patterns
   - Generates buy/sell signals

3. **Fundamental Analysis (`src/fundamental_analysis.py`)**
   - Analyzes company financials
   - Bank-specific metrics (Tier 1 ratio, NIM, etc.)
   - Peer comparison and valuation

4. **News Sentiment (`src/news_sentiment.py`)**
   - Scrapes news from multiple sources
   - Sentiment analysis using VADER and TextBlob
   - Social media sentiment tracking

5. **Risk Calculator (`src/risk_calculator.py`)**
   - Position sizing calculations
   - Stop-loss and take-profit levels
   - Risk/reward ratios and Kelly Criterion

6. **Market Predictor (`src/market_predictor.py`)**
   - Combines all analysis inputs
   - Generates weighted predictions
   - Confidence scoring system

7. **Alert System (`src/alert_system.py`)**
   - Multi-channel notifications
   - Threshold-based alerts
   - Alert history and statistics

8. **Report Generator (`src/report_generator.py`)**
   - HTML report generation
   - Charts and visualizations
   - Export capabilities

### Data Flow

```
Data Sources → Data Feed → Analysis Engines → Prediction → Alerts/Reports
     ↓              ↓           ↓              ↓           ↓
- Yahoo Finance → Cache → Technical Analysis → Market → Discord
- RSS Feeds          → Fundamental Analysis → Predictor → Telegram  
- Web Scraping       → Sentiment Analysis            → Email
                     → Risk Calculator               → Console
```

## 📋 Analysis Output

### Signal Types
- **bullish**: Strong upward momentum expected
- **bearish**: Strong downward momentum expected  
- **neutral**: Sideways movement or unclear direction

### Confidence Levels
- **0-25%**: Low confidence - avoid trading
- **25-50%**: Moderate confidence - small positions
- **50-75%**: High confidence - standard positions
- **75-100%**: Very high confidence - larger positions

### Risk Scores
- **0-25**: Low risk - conservative entry
- **25-50**: Moderate risk - standard position
- **50-75**: High risk - reduced position
- **75-100**: Very high risk - avoid or hedge

## 📊 Reports

Reports are generated in HTML format and saved to the `reports/` directory:

- **Daily Reports**: Complete analysis of all banks
- **Charts**: Price charts with technical indicators
- **Tables**: Comparative analysis across banks
- **Alerts**: Recent alerts and notifications

## 🔧 Configuration

### Settings (`config/settings.py`)

Key configuration options:

```python
# Analysis periods
DEFAULT_ANALYSIS_PERIOD = "3mo"  # 3 months of data

# Technical indicators
TECHNICAL_INDICATORS = {
    'RSI': {'period': 14},
    'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
    'SMA': {'periods': [20, 50, 200]},
    'EMA': {'periods': [12, 26]}
}

# Risk parameters
RISK_PARAMETERS = {
    'max_risk_per_trade': 0.02,  # 2% max risk
    'stop_loss_atr_multiplier': 2.0,
    'take_profit_multiplier': 2.0
}

# Alert thresholds
ALERT_THRESHOLDS = {
    'price_breakout': 0.03,  # 3% price movement
    'volume_spike': 2.0,     # 2x average volume
    'rsi_overbought': 70,
    'rsi_oversold': 30
}
```

## 🗂️ Directory Structure

```
trading_analysis/
├── main.py                 # Main entry point
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── config/
│   ├── __init__.py
│   └── settings.py        # Configuration settings
├── src/                   # Core modules
│   ├── __init__.py
│   ├── data_feed.py       # Data fetching
│   ├── technical_analysis.py
│   ├── fundamental_analysis.py
│   ├── news_sentiment.py
│   ├── risk_calculator.py
│   ├── market_predictor.py
│   ├── alert_system.py
│   └── report_generator.py
├── utils/                 # Utility modules
│   ├── __init__.py
│   ├── cache_manager.py   # Data caching
│   ├── helpers.py         # Helper functions
│   └── validators.py      # Data validation
├── tests/                 # Unit tests
│   ├── __init__.py
│   ├── test_analysis.py
│   ├── test_data_feed.py
│   └── test_predictions.py
├── data/                  # Data storage
│   ├── cache/            # Cached API responses
│   └── historical/       # Historical data
├── reports/              # Generated reports
├── logs/                 # Log files
└── docs/                 # Documentation
```

## 🔍 Troubleshooting

### Common Issues

1. **No Data Available**
   - Check internet connection
   - Verify symbol format (should end with .AX)
   - Try clearing cache: `rm -rf data/cache/*`

2. **Import Errors**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

3. **Low Confidence Scores**
   - Normal during sideways markets
   - Increase analysis period in settings
   - Check if sufficient historical data available

4. **Alert Not Working**
   - Verify .env file configuration
   - Check webhook URLs and tokens
   - Review alert thresholds in settings

### Debug Mode

Enable debug logging for detailed information:

```bash
python main.py analyze --debug
```

This will show:
- API calls and responses
- Analysis calculations
- Error details
- Performance metrics

## 📅 Automated Scheduling

The system can run automatically:

### Market Hours Schedule
- **9:30 AM**: Morning analysis (after market open)
- **3:00 PM**: Afternoon check (before market close)

### Custom Schedule
Modify the schedule in `main.py`:

```python
# Custom schedule examples
schedule.every().day.at("08:00").do(self.run_daily_analysis)
schedule.every().hour.do(self.run_quick_check)
schedule.every().monday.at("17:00").do(self.run_weekly_report)
```

## 🔒 Security Notes

- **API Keys**: Store sensitive information in `.env` file
- **Data Privacy**: All data is processed locally
- **Rate Limits**: Built-in rate limiting to respect API limits
- **Error Handling**: Graceful degradation when services unavailable

## 📈 Performance Optimization

### Data Caching
- **Historical Data**: Cached for 1 hour
- **Company Info**: Cached for 4 hours
- **News Sentiment**: Cached for 30 minutes

### Resource Usage
- **Memory**: ~100-200MB typical usage
- **CPU**: Low usage except during analysis
- **Network**: Minimal with caching enabled

## 🚨 Disclaimer

**This software is for educational and informational purposes only. It is not financial advice and should not be used as the sole basis for investment decisions. Always conduct your own research and consider consulting with a qualified financial advisor before making investment decisions.**

**Key Points:**
- Past performance does not guarantee future results
- All investments carry risk of loss
- The system may have bugs or incorrect calculations
- Market conditions can change rapidly
- Always use proper risk management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the logs in `logs/trading_system.log`
3. Open an issue on GitHub
4. Use debug mode for detailed information

---

**Happy Trading! 📊🚀**
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
python3 main.py analyze

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
