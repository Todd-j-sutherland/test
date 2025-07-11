# Technical Analysis Integration

## 🎯 Overview

The trading analysis system now includes comprehensive technical analysis with momentum indicators alongside the existing news sentiment analysis. This provides a complete picture for making informed trading decisions.

## 📈 Features Added

### Technical Indicators
- **RSI (Relative Strength Index)** - Momentum oscillator (14-period)
- **MACD** - Moving Average Convergence Divergence with signal line
- **Moving Averages** - SMA (20, 50, 200) and EMA (12, 26)
- **Volume Analysis** - Volume momentum and unusual volume detection

### Momentum Analysis
- **Price Momentum** - 1-day, 5-day, and 20-day price changes
- **Momentum Score** - Comprehensive score (-100 to +100) combining all factors
- **Momentum Strength Classification** - From "Very Strong Bearish" to "Very Strong Bullish"
- **Volume Momentum** - Identifies high volume periods

### Dashboard Integration
- **Momentum Chart** - Visual representation of momentum scores
- **Technical Signals Chart** - Buy/Sell recommendations from technical analysis
- **Combined Analysis** - News sentiment vs technical signals comparison
- **Individual Bank Analysis** - Detailed technical metrics for each bank

## 🚀 Quick Start

The fastest way to get the dashboard running:

```bash
cd /Users/toddsutherland/Repos/trading_analysis && source .venv/bin/activate && python launch_dashboard_auto.py
```

This will automatically:
- Activate the virtual environment
- Skip the Streamlit email prompt
- Launch the dashboard at http://localhost:8501

## 🚀 Usage

### Running the Enhanced Dashboard

```bash
# Option 1: Use the automated launcher (recommended)
python launch_dashboard_auto.py

# Option 2: Run manually
streamlit run news_analysis_dashboard.py
# (At the email prompt, just press Enter)
```

### Testing Technical Analysis

```bash
# Test the technical analysis module
python test_technical_analysis.py
```

### Manual Technical Analysis

```python
from src.technical_analysis import TechnicalAnalyzer, get_market_data

# Initialize analyzer
analyzer = TechnicalAnalyzer()

# Get market data
data = get_market_data('CBA.AX', period='3mo')

# Perform analysis
analysis = analyzer.analyze('CBA.AX', data)

# Access results
print(f"Momentum Score: {analysis['momentum']['score']}")
print(f"Recommendation: {analysis['recommendation']}")
print(f"RSI: {analysis['indicators']['rsi']}")
```

## 📊 Dashboard Sections

### 1. Technical Analysis & Momentum Tab
- **Momentum Analysis**: Visual momentum scores for all banks
- **Technical Signals**: Buy/Sell recommendations
- **Combined Analysis**: News sentiment vs technical analysis

### 2. Individual Bank Analysis
Each bank now shows:
- Current price and momentum score
- RSI, MACD, and moving average signals
- Trend direction and strength
- Volume momentum analysis
- Combined recommendation (sentiment + technical)

### 3. Trading Recommendations
Combines multiple factors:
- News sentiment score
- Technical recommendation
- Momentum strength
- Volume confirmation

## 🎯 Momentum Indicators Explained

### Momentum Score (-100 to +100)
- **+50 to +100**: Very Strong Bullish
- **+20 to +50**: Strong Bullish  
- **+5 to +20**: Moderate Bullish
- **-5 to +5**: Neutral
- **-20 to -5**: Moderate Bearish
- **-50 to -20**: Strong Bearish
- **-100 to -50**: Very Strong Bearish

### Technical Signals
- **STRONG_BUY**: Multiple bullish indicators + strong momentum
- **BUY**: Moderate bullish signals
- **HOLD**: Mixed or neutral signals
- **SELL**: Moderate bearish signals  
- **STRONG_SELL**: Multiple bearish indicators + weak momentum

## 🔧 Configuration

Technical indicators can be configured in `src/technical_analysis.py`:

```python
self.settings = {
    'RSI': {'period': 14, 'overbought': 70, 'oversold': 30},
    'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
    'SMA': {'periods': [20, 50, 200]},
    'EMA': {'periods': [12, 26]},
    'VOLUME': {'ma_period': 20}
}
```

## 📋 Data Requirements

- **Market Data**: Fetched automatically via yfinance
- **Historical Period**: 3 months default (configurable)
- **Update Frequency**: Manual refresh or cached data
- **Symbols Supported**: All ASX-listed securities

## 🎉 Benefits

1. **Comprehensive Analysis**: Combines news sentiment with technical momentum
2. **Risk Assessment**: Multiple confirmation signals reduce false positives
3. **Momentum Detection**: Identifies trend changes and strength
4. **Visual Dashboard**: Easy-to-understand charts and metrics
5. **Expandable**: Can easily add more indicators as needed

## 🔮 Future Enhancements

- **Bollinger Bands**: Support and resistance levels
- **Stochastic Oscillator**: Additional momentum indicator
- **ATR (Average True Range)**: Volatility measurement
- **Volume Profile**: Price-volume analysis
- **Fibonacci Retracements**: Key support/resistance levels
- **Real-time Updates**: Live data feeds
- **Backtesting**: Historical performance testing

This technical analysis integration provides the foundation for sophisticated momentum-based trading strategies while maintaining the simplicity and clarity of the original news analysis system.
