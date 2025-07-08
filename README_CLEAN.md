# News Trading Analyzer 📰📊

## Quick Start

```bash
# Analyze single bank
python news_trading_analyzer.py --symbol CBA.AX

# Analyze all major banks
python news_trading_analyzer.py --all

# Get detailed analysis with export
python news_trading_analyzer.py --all --detailed --export
```

## What It Does

This system analyzes news sources to provide **trading sentiment scores** for Australian banks:

- **Collects news** from RSS feeds, Yahoo Finance, Google News, and Reddit
- **Analyzes sentiment** using traditional methods + ML trading features
- **Provides trading recommendations** with confidence levels
- **Exports results** in JSON format for integration

## Supported Banks

- **CBA.AX** - Commonwealth Bank of Australia
- **WBC.AX** - Westpac Banking Corporation  
- **ANZ.AX** - Australia and New Zealand Banking Group
- **NAB.AX** - National Australia Bank
- **MQG.AX** - Macquarie Group

## Sample Output

```
MARKET OVERVIEW - AUSTRALIAN BANKS
Average Sentiment: 0.036
Most Bullish: NAB.AX (0.137)
Most Bearish: CBA.AX (-0.044)

Individual Bank Analysis:
NAB.AX   | BUY          | Score:  0.137 | Confidence: 1.000
MQG.AX   | HOLD         | Score:  0.086 | Confidence: 0.960
WBC.AX   | HOLD         | Score:  0.022 | Confidence: 0.980
```

## How It Works

1. **Multi-Source Data Collection**: Gathers news from multiple sources
2. **Advanced Sentiment Analysis**: Uses TextBlob, VADER, and ML features
3. **Trading-Specific Features**: Extracts 30+ trading-relevant features
4. **Confidence-Based Weighting**: Dynamically adjusts based on data quality
5. **Trading Recommendations**: Provides actionable buy/sell/hold signals

## Installation

```bash
# Dependencies are already installed in the virtual environment
# If you need to reinstall:
pip install beautifulsoup4 feedparser textblob vaderSentiment praw yfinance pandas numpy scikit-learn optuna scipy python-dotenv
```

## Project Structure

```
├── news_trading_analyzer.py    # 🎯 Main entry point
├── src/
│   ├── news_sentiment.py       # Core sentiment analysis
│   ├── ml_trading_config.py    # ML feature engineering
│   └── ...                     # Supporting modules
├── reports/                    # 📊 Output files
└── logs/                       # 📝 System logs
```

## Features

- ✅ **Multi-source news collection**
- ✅ **Advanced ML feature engineering**  
- ✅ **Confidence-based analysis**
- ✅ **Trading strategy recommendations**
- ✅ **JSON export for integration**
- ✅ **Robust error handling**

## Command Options

```bash
python news_trading_analyzer.py [OPTIONS]

Options:
  --symbol, -s SYMBOL   Analyze specific bank (CBA.AX, WBC.AX, etc.)
  --all, -a             Analyze all major banks
  --detailed, -d        Include detailed breakdown
  --export, -e          Export results to JSON
  --log-level LEVEL     Set logging level (DEBUG, INFO, WARNING, ERROR)
```

## Output Explained

- **Sentiment Score**: -1 (very bearish) to +1 (very bullish)
- **Confidence**: 0 to 1 based on data quality and consistency
- **Trading Signal**: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
- **Strategy**: Conservative, Moderate, or Aggressive approach
- **News Count**: Number of articles analyzed

## Integration

Use the `--export` flag to generate JSON files in the `reports/` directory. These contain:

- Sentiment scores and confidence levels
- Trading recommendations with parameters
- Detailed component breakdowns
- Recent headlines and significant events
- ML feature analysis

Perfect for integration with trading platforms or further analysis tools.

---

🎯 **Main Goal**: Analyze news sources and provide trading sentiment scores to assist with investment decisions.
