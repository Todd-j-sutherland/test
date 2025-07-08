# News-Based Trading Analysis System 🏦

## 🚀 Overview

The News-Based Trading Analysis System is a comprehensive tool designed for analyzing Australian banking stocks using advanced news sentiment analysis. It integrates various data sources, performs ML-enhanced sentiment scoring, and provides trading insights based on news sentiment with full Python 3.12 compatibility for transformer models.

## ✨ Key Features

### 🔥 Core Capabilities
- **Advanced News Sentiment Analysis** - VADER sentiment with ML feature engineering
- **Real-time News Processing** - Live news feeds and sentiment scoring
- **Trading Signal Generation** - HOLD/BUY/SELL signals with confidence levels
- **Multi-Bank Analysis** - Individual or all major Australian banks
- **Python 3.12 Compatible** - Full support for modern ML libraries and transformers

### 📊 What It Analyzes
- **CBA.AX** - Commonwealth Bank of Australia
- **WBC.AX** - Westpac Banking Corporation
- **ANZ.AX** - Australia and New Zealand Banking Group
- **NAB.AX** - National Australia Bank
- **MQG.AX** - Macquarie Group

## 🎯 Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd trading_analysis

# Install dependencies
pip install -r requirements.txt
```

### Run Analysis
```bash
# Full enhanced analysis (recommended)
python enhanced_main.py

# Synchronous processing
python enhanced_main.py --sync

# Analyze specific symbols
python enhanced_main.py --symbols CBA.AX ANZ.AX
```

### Example Output
```
📊 INDIVIDUAL STOCK ANALYSIS:
--------------------------------------------------
📈 CBA.AX: $178.00 | NEUTRAL (9.1%) | Risk: LOW (36/100) | Data: VALIDATED
    🔍 VaR(95%): 2.2% | Max Drawdown: 7.0%
📈 WBC.AX: $33.63 | NEUTRAL (1.0%) | Risk: LOW (26/100) | Data: VALIDATED
    🔍 VaR(95%): 1.9% | Max Drawdown: 6.6%

🎯 PORTFOLIO RISK ANALYSIS:
--------------------------------------------------
📊 Portfolio Volatility: 13.1%
📊 Portfolio VaR (95%): 1.3%
📊 Sharpe Ratio: 1.59
📊 Diversification: EXCELLENT
📊 Concentration Risk: MEDIUM
```

## 🏗️ System Architecture

### Core Components
```
Enhanced ASX Trading System
├── 🚀 Async Processing Engine
├── 🔍 Data Validation Pipeline
├── 📊 Advanced Risk Manager
├── 🤖 Machine Learning Predictor
├── 📈 Technical Analysis
├── 💰 Fundamental Analysis
├── 📰 Sentiment Analysis
└── 📋 Report Generator
```

### Data Flow
1. **Data Collection** - Fetch market data from multiple sources
2. **Validation** - Quality checks and anomaly detection
3. **Analysis** - Technical, fundamental, and sentiment analysis
4. **Prediction** - ML-based market direction forecasting
5. **Risk Assessment** - VaR, drawdown, and portfolio risk metrics
6. **Report Generation** - Comprehensive analysis reports

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.12** (Required for full ML/transformer compatibility)
- **pyenv** (recommended for Python version management)

### Step 1: Ensure Python 3.12 is Available
```bash
# Install pyenv if not already installed (macOS)
brew install pyenv

# Install Python 3.12.7
pyenv install 3.12.7

# Set Python 3.12 for this project
cd /path/to/trading_analysis
pyenv local 3.12.7

# Verify Python version
python --version  # Should show Python 3.12.7
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment with Python 3.12
python -m venv .venv312

# Activate on macOS/Linux
source .venv312/bin/activate

# Activate on Windows
.venv312\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Install additional required packages
pip install requests yfinance beautifulsoup4 feedparser textblob vaderSentiment optuna

# Optional: Install transformer models for advanced sentiment analysis
pip install transformers torch
```

### Step 4: Verify Installation
```bash
# Test the main analyzer
python news_trading_analyzer.py --help

# Run a quick analysis
python news_trading_analyzer.py --symbol CBA.AX --detailed
```

## 🚀 Usage

### Command Line Interface
```bash
# Analyze a specific bank
python news_trading_analyzer.py --symbol CBA.AX --detailed

# Analyze all major banks
python news_trading_analyzer.py --all

# Export results to JSON
python news_trading_analyzer.py --symbol CBA.AX --export

# Set detailed logging
python news_trading_analyzer.py --symbol CBA.AX --log-level DEBUG
```

### Supported Banks
- **CBA.AX** - Commonwealth Bank of Australia
- **WBC.AX** - Westpac Banking Corporation  
- **ANZ.AX** - Australia and New Zealand Banking Group
- **NAB.AX** - National Australia Bank
- **MQG.AX** - Macquarie Group

### Output Example
```
============================================================
NEWS TRADING ANALYSIS: CBA.AX
============================================================
Sentiment Score: -0.027
Confidence: 0.990
Trading Signal: HOLD
Recommendation: HOLD
Strategy: moderate
News Articles: 18
============================================================
```

## 🔧 Environment Management

### Switching Between Python Versions
```bash
# Check current Python version
python --version

# List available versions with pyenv
pyenv versions

# Switch to Python 3.12 for this project
pyenv local 3.12.7

# Activate the virtual environment
source .venv312/bin/activate
```

### Troubleshooting
If you encounter "Module not found" errors:
```bash
# Ensure you're in the correct virtual environment
which python  # Should point to .venv312/bin/python

# Reinstall dependencies if needed
pip install -r requirements.txt

# Check if all packages are installed
pip list | grep -E "(numpy|pandas|requests|yfinance)"
```
