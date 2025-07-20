# 🚀 Trading Analysis System

A comprehensive **AI-powered trading analysis platform** that combines sentiment analysis, technical indicators, and market intelligence to provide actionable trading insights for Australian Stock Exchange (ASX) securities.

## ✨ Features

- **🧠 Enhanced Sentiment Analysis** - Multi-layered sentiment scoring with temporal analysis
- **📊 Professional Dashboard** - Interactive Streamlit-based web interface
- **📈 Technical Analysis** - Advanced technical indicators and pattern recognition
- **🔄 Automated Data Collection** - Real-time news and market data integration
- **📱 Daily Operations** - Morning briefings and evening summaries
- **🧪 Comprehensive Testing** - Full test suite with 63+ passing tests
- **🏗️ Professional Architecture** - Industry-standard Python package structure

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/trading_analysis.git
cd trading_analysis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```bash
# Check system status
python -m app.main status

# Run morning analysis
source ../trading_venv/bin/activate && ../trading_evn python -m app.main morning

# Run evening summary
python -m app.main evening

# Launch interactive dashboard
python -m app.main dashboard

# 💻 Local Setup
source .venv312/bin/activate
export PYTHONPATH=/Users/toddsutherland/Repos/trading_analysis
cd /Users/toddsutherland/Repos/trading_analysis

# 🌐 Remote Server Setup
ssh -i ~/.ssh/id_rsa root@170.64.199.151
cd test
source ../trading_venv/bin/activate
export PYTHONPATH=/root/test

# Run dashboard on remote server (accessible via browser)
streamlit run app/dashboard/enhanced_main.py --server.port 8501 --server.address 0.0.0.0
```

## 📁 Project Structure

```
trading_analysis/
├── app/                          # 🏗️ Main application package
│   ├── main.py                   # CLI entry point
│   ├── config/                   # ⚙️ Configuration management
│   ├── core/                     # 🧠 Core business logic
│   │   ├── sentiment/            # Sentiment analysis components
│   │   ├── analysis/             # Technical analysis tools
│   │   ├── data/                 # Data management & collection
│   │   └── trading/              # Trading logic & signals
│   ├── dashboard/                # 📊 Web interface components
│   ├── services/                 # 🔄 Business services
│   └── utils/                    # 🛠️ Utility functions
├── tests/                        # 🧪 Comprehensive test suite
├── docs/                         # 📚 Documentation
├── data/                         # 📈 Data storage
└── logs/                         # 📝 Application logs
```

## 🎯 Core Commands

| Command | Description | Example |
|---------|-------------|---------|
| `status` | System health check | `python -m app.main status` |
| `morning` | Morning briefing | `python -m app.main morning` |
| `evening` | Evening summary | `python -m app.main evening` |
| `dashboard` | Launch web interface | `python -m app.main dashboard` |

## 📊 Dashboard Features

The professional dashboard provides:

- **📈 Market Overview** - Real-time sentiment across major ASX banks
- **🏦 Bank Analysis** - Detailed analysis of CBA, WBC, ANZ, NAB, MQG
- **📰 News Sentiment** - Real-time news analysis with confidence scoring
- **📊 Technical Charts** - Interactive visualizations and trend analysis
- **🎯 Position Risk** - Risk assessment and position management tools

## 🧠 Enhanced Sentiment Analysis

Our advanced sentiment engine features:

- **Multi-layered Scoring** - Combines multiple sentiment models
- **Temporal Analysis** - Time-weighted sentiment trends
- **Confidence Metrics** - Statistical confidence in sentiment scores
- **Market Context** - Volatility and regime-aware adjustments
- **Integration Ready** - Easy integration with existing trading systems

## 🧪 Testing & Quality

- **63+ Tests** - Comprehensive test coverage
- **Unit Tests** - Core functionality validation (19 tests)
- **Integration Tests** - System integration validation (44 tests)
- **Continuous Validation** - Automated testing pipeline
- **Professional Standards** - Industry-standard code quality

## 📚 Documentation

Comprehensive documentation is organized in the `docs/` directory:

### 🚀 Getting Started
- **[Quick Start Guide](docs/QUICK_START.md)** - 5-minute setup and first commands
- **[Features Overview](docs/FEATURES.md)** - Complete feature breakdown and use cases
- **[Architecture Guide](docs/ARCHITECTURE.md)** - Technical architecture and design patterns

### 📖 Project History & Development
- **[Project Restructuring](docs/RESTRUCTURE_COMPLETE.md)** - Complete architectural migration details
- **[File Organization](docs/ORGANIZATION_COMPLETE.md)** - File structure evolution and cleanup
- **[Testing Framework](docs/TESTING_COMPLETE_SUMMARY.md)** - Comprehensive testing implementation
- **[Legacy Cleanup](docs/CLEANUP_COMPLETE.md)** - Migration from legacy structure

### 🎯 Quick References
- **System Commands** - See [Quick Start Guide](docs/QUICK_START.md#-first-commands)
- **Dashboard Features** - See [Features Overview](docs/FEATURES.md#-professional-dashboard)
- **API Usage** - See [Architecture Guide](docs/ARCHITECTURE.md#-entry-points)

## 🛠️ Development

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v        # Unit tests
python -m pytest tests/integration/ -v # Integration tests
```

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run with development logging
python -m app.main status --verbose
```

### Code Quality
- **Type Hints** - Full type annotation coverage
- **Logging** - Comprehensive logging with rotation
- **Error Handling** - Graceful error recovery
- **Documentation** - Inline documentation and docstrings

## 🔧 Configuration

Configuration is managed through:
- **Environment Variables** - `.env` file support
- **YAML Configuration** - `app/config/ml_config.yaml`
- **Settings Module** - `app/config/settings.py`

## 📈 Performance

- **Optimized Data Processing** - Efficient pandas operations
- **Async Components** - Non-blocking data collection
- **Memory Management** - Optimized for large datasets
- **Caching** - Intelligent data caching strategies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Key Components

- **Enhanced Sentiment Scoring** - `app/core/sentiment/enhanced_scoring.py`
- **Professional Dashboard** - `app/dashboard/main.py`
- **Daily Operations Manager** - `app/services/daily_manager.py`
- **Technical Analysis** - `app/core/analysis/technical.py`
- **Data Collection** - `app/core/data/collectors/`

## 🎯 Recent Updates

- ✅ **Complete Project Restructuring** - Professional Python package architecture
- ✅ **Enhanced Sentiment System** - Multi-layered sentiment analysis with confidence metrics
- ✅ **Legacy Cleanup** - 100+ legacy files organized and archived
- ✅ **Comprehensive Testing** - 63+ tests ensuring system reliability
- ✅ **Professional Dashboard** - Modern web interface with interactive charts

---

**🚀 Ready to start trading with AI-powered insights!**


send
scp -i ~/.ssh/id_rsa -r trading_analysis/data root@170.64.199.151:/root/test/

scp -i ~/.ssh/id_rsa -r root@170.64.199.151:/root/test/data trading_analysis/data


# Connect to your server
ssh -i ~/.ssh/id_rsa root@170.64.199.151

# Navigate to project and activate environment
cd test
source ../trading_venv/bin/activate
export PYTHONPATH=/root/test

# Run different commands
python app/main.py status          # System health check
python app/main.py ml-scores       # ML trading analysis
python app/main.py news           # News sentiment analysis

# Start dashboard
streamlit run app/dashboard/enhanced_main.py --server.port 8504 --server.address 0.0.0.0



ssh -i ~/.ssh/id_rsa root@170.64.199.151 "cd test && source ../trading_venv/bin/activate && export PYTHONPATH=/root/test && streamlit run app/dashboard/enhanced_main.py --server.port 8504 --server.address 0.0.0.0"