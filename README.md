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

### 📊 Dashboard (Recommended)
Launch the interactive dashboard with news sentiment and technical analysis:

```bash
# Quick launch (recommended)
cd /Users/toddsutherland/Repos/trading_analysis && source .venv/bin/activate && python launch_dashboard_auto.py

# Or run manually
streamlit run news_analysis_dashboard.py
```

The dashboard will open at http://localhost:8501 and includes:
- News sentiment analysis
- Technical indicators (RSI, MACD, momentum)
- Combined trading recommendations
- Individual bank analysis

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

### Quick Start (No Transformer Downloads)
```bash
# Run analyzer without downloading large ML models (recommended for quick testing)
SKIP_TRANSFORMERS=1 python news_trading_analyzer.py --symbol CBA.AX

# Or use the simple runner script
python run_analyzer.py --symbol CBA.AX

# Analyze all banks quickly
SKIP_TRANSFORMERS=1 python news_trading_analyzer.py --all
```

### Advanced Mode (With Transformers)
If you want to enable advanced transformer-based sentiment analysis:
```bash
# Install transformer libraries (will download ~268MB of models)
pip install transformers torch

# Run without SKIP_TRANSFORMERS flag
python news_trading_analyzer.py --symbol CBA.AX --detailed
```

**Note:** Transformer models provide more accurate sentiment analysis but require downloading large model files (~268MB). For most use cases, the VADER sentiment analysis (default) provides excellent results without any downloads.

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

---

## 📁 File Structure & Definitions

### 🎯 **Active Core Files** (Used by the application)

#### **Main Entry Points**
- **`news_trading_analyzer.py`** - 🎯 **PRIMARY ENTRY POINT**
  - Command-line interface for news sentiment analysis
  - Orchestrates all analysis components
  - Provides trading recommendations with confidence scores
  - Exports results to JSON format

- **`run_analyzer.py`** - 🚀 **SIMPLE RUNNER**
  - Convenience script that runs analyzer without transformer downloads
  - Sets `SKIP_TRANSFORMERS=1` environment variable automatically
  - Ideal for quick testing and daily use

#### **Core Analysis Engine (`src/`)**
- **`src/news_sentiment.py`** - 📰 **CORE SENTIMENT ANALYSIS**
  - Multi-source news collection (RSS, Yahoo Finance, Google News, Reddit)
  - Advanced sentiment analysis using VADER, TextBlob, and optional transformers
  - ML-enhanced feature engineering for trading-specific insights
  - Dynamic confidence weighting and sentiment aggregation

- **`src/ml_trading_config.py`** - 🤖 **ML FEATURE ENGINEERING**
  - Extracts 30+ trading-relevant features from news text
  - Financial entity recognition (currencies, percentages, companies)
  - Temporal analysis (urgency detection, time references)
  - Trading pattern recognition (bull/bear language, action words)
  - Model optimization and hyperparameter tuning

- **`src/news_impact_analyzer.py`** - 📊 **NEWS IMPACT CORRELATION**
  - Analyzes correlation between news sentiment and price movements
  - Event impact analysis with time decay modeling
  - Predictive metrics for trading accuracy
  - Multi-symbol comparative analysis

- **`src/sentiment_history.py`** - 📈 **HISTORICAL TRACKING**
  - Stores and manages historical sentiment data
  - Trend analysis and momentum calculations
  - Data persistence and retrieval for correlation studies

- **`src/data_feed.py`** - 📡 **MARKET DATA INTERFACE**
  - Fetches real-time and historical market data using yfinance
  - Provides price data for sentiment-price correlation analysis
  - Handles data validation and error recovery

#### **Configuration & Support**
- **`config/settings.py`** - ⚙️ **CONFIGURATION MANAGEMENT**
  - Application settings and parameters
  - News source URLs and API configurations
  - Trading strategy thresholds and weights

- **`config/__init__.py`** - 📦 **PACKAGE MARKER**
  - Makes config directory a Python package

- **`utils/cache_manager.py`** - 💾 **CACHING SYSTEM**
  - Implements intelligent caching for news and market data
  - Reduces API calls and improves performance
  - TTL-based cache expiration

- **`utils/__init__.py`** - 📦 **PACKAGE MARKER**
  - Makes utils directory a Python package

- **`src/__init__.py`** - 📦 **PACKAGE MARKER**
  - Makes src directory a Python package

#### **Configuration Files**
- **`requirements.txt`** - 📋 **PYTHON DEPENDENCIES**
  - Complete list of required Python packages
  - Version specifications for reproducible environments

- **`.python-version`** - 🐍 **PYTHON VERSION LOCK**
  - Specifies Python 3.12.7 for pyenv compatibility
  - Ensures consistent Python version across environments

- **`.gitignore`** - 🚫 **GIT IGNORE RULES**
  - Excludes logs, virtual environments, cache files, and temporary data
  - Prevents sensitive and generated files from being committed

### 🗂️ **Directory Structure**
- **`data/`** - 💾 **DATA STORAGE**
  - `data/cache/` - Cached API responses and news data
  - `data/historical/` - Historical price and sentiment data
  - `data/sentiment_history/` - Stored sentiment analysis results

- **`logs/`** - 📝 **APPLICATION LOGS**
  - `news_trading_analyzer.log` - Main application log
  - `enhanced_trading_system.log` - Legacy system logs

- **`reports/`** - 📊 **ANALYSIS OUTPUTS**
  - Generated analysis reports in JSON and HTML formats
  - Backtesting results and correlation studies

### ❌ **Unused Files** (Safe to remove or ignore)

#### **Cleanup & Development Files**
- **`cleanup_project.py`** - 🧹 **ONE-TIME CLEANUP SCRIPT**
  - Used once to organize project structure
  - Can be safely removed or archived

- **`install_transformers.sh`** - 📥 **MANUAL INSTALLATION SCRIPT**
  - Shell script for installing transformer libraries
  - Not automatically used by the application

#### **Legacy & Backup Files**
- **`backtesting_system.py`** - 📈 **LEGACY BACKTESTING**
  - Old backtesting system not integrated with current analyzer
  - Standalone system not used by main application

- **`archive/`** - 📦 **ARCHIVED FILES**
  - Contains old main files, demos, and unused components
  - Preserved for reference but not actively used

- **`backup_before_cleanup/`** - 💾 **PRE-CLEANUP BACKUP**
  - Backup of files before project reorganization
  - Can be safely removed once satisfied with current system

#### **Documentation & Alternatives**
- **`README_CLEAN.md`** - 📖 **ALTERNATIVE README**
  - Simplified version of documentation
  - Redundant with main README.md

- **`IMPROVEMENTS_README.md`** - 📋 **DEVELOPMENT NOTES**
  - Development notes and improvement ideas
  - Not part of production system

- **`PROJECT_CLEANUP_SUMMARY.md`** - 📝 **CLEANUP DOCUMENTATION**
  - Documents the cleanup process performed
  - Reference document, not operational

#### **Test Files**
- **`tests/test_*.py`** - 🧪 **UNIT TESTS**
  - Test files for development and validation
  - Not used in production runtime but valuable for development

#### **Unused Utilities**
- **`utils/helpers.py`** - 🔧 **UNUSED HELPER FUNCTIONS**
  - Helper functions not currently imported by any active component

- **`utils/validators.py`** - ✅ **UNUSED VALIDATION FUNCTIONS**
  - Data validation functions not currently used

### 🎯 **Summary**

**Active System (12 core files):**
```
news_trading_analyzer.py     # Main entry point
run_analyzer.py             # Simple runner
src/news_sentiment.py       # Core sentiment analysis
src/ml_trading_config.py    # ML feature engineering  
src/news_impact_analyzer.py # Impact correlation
src/sentiment_history.py    # Historical tracking
src/data_feed.py            # Market data
config/settings.py          # Configuration
utils/cache_manager.py      # Caching
requirements.txt            # Dependencies
.python-version             # Python version
.gitignore                  # Git ignore rules
```

**The system is lean, focused, and production-ready with minimal dependencies and clear separation of concerns.**
