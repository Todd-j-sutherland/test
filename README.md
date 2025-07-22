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

# Run morning analysis (Stage 1 - continuous monitoring)
python -m app.main morning

# Run evening summary (automatically uses Stage 2 when memory permits)
python -m app.main evening

# Run evening with enhanced Stage 2 analysis
export USE_TWO_STAGE_ANALYSIS=1 && export SKIP_TRANSFORMERS=0 && python -m app.main evening

# Launch interactive dashboard
python -m app.main dashboard

# Check two-stage system health
python -c "
import os
os.environ['USE_TWO_STAGE_ANALYSIS'] = '1'
from app.core.sentiment.two_stage_analyzer import TwoStageAnalyzer
analyzer = TwoStageAnalyzer()
print('✅ Two-stage system operational')
"

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

# 🤖 Remote Two-Stage Analysis
# Morning routine (Stage 1 continuous monitoring)
ssh -i ~/.ssh/id_rsa root@170.64.199.151 'cd /root/test && source /root/trading_venv/bin/activate && export USE_TWO_STAGE_ANALYSIS=1 && export SKIP_TRANSFORMERS=1 && python -m app.main morning'

# Evening enhanced analysis (Stage 2 when memory permits)
ssh -i ~/.ssh/id_rsa root@170.64.199.151 'cd /root/test && source /root/trading_venv/bin/activate && export USE_TWO_STAGE_ANALYSIS=1 && export SKIP_TRANSFORMERS=0 && python -m app.main evening'

# System health check
ssh -i ~/.ssh/id_rsa root@170.64.199.151 'cd /root/test && source /root/trading_venv/bin/activate && python -c "
import os
os.environ[\"USE_TWO_STAGE_ANALYSIS\"] = \"1\"
print(\"🏥 SYSTEM HEALTH CHECK\")
print(\"=\" * 50)
import subprocess
result = subprocess.run([\"ps\", \"aux\"], capture_output=True, text=True)
print(\"✅ Smart Collector:\", \"Running\" if \"news_collector\" in result.stdout else \"Not Running\")
result = subprocess.run([\"free\", \"-m\"], capture_output=True, text=True)
for line in result.stdout.split(\"\\n\"):
    if \"Mem:\" in line:
        parts = line.split()
        used, total = int(parts[2]), int(parts[1])
        print(f\"💾 Memory: {used}MB/{total}MB ({100*used/total:.1f}%)\")
"'
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
| `morning` | Morning briefing (Stage 1) | `python -m app.main morning` |
| `evening` | Evening summary (Stage 1+2) | `python -m app.main evening` |
| `dashboard` | Launch web interface | `python -m app.main dashboard` |
| `news` | News sentiment analysis | `python -m app.main news` |

### 🤖 Two-Stage Analysis Commands

| Mode | Memory Usage | Quality | Command |
|------|-------------|---------|---------|
| **Stage 1 Only** | ~100MB | 70% accuracy | `export SKIP_TRANSFORMERS=1 && python -m app.main morning` |
| **Stage 2 Enhanced** | ~800MB | 85-95% accuracy | `export USE_TWO_STAGE_ANALYSIS=1 && export SKIP_TRANSFORMERS=0 && python -m app.main evening` |
| **Memory Optimized** | Auto-detect | Adaptive | `export USE_TWO_STAGE_ANALYSIS=1 && python -m app.main evening` |

## 📊 Dashboard Features

The professional dashboard provides:

- **📈 Market Overview** - Real-time sentiment across major ASX banks
- **🏦 Bank Analysis** - Detailed analysis of CBA, WBC, ANZ, NAB, MQG
- **📰 News Sentiment** - Real-time news analysis with confidence scoring
- **📊 Technical Charts** - Interactive visualizations and trend analysis
- **🎯 Position Risk** - Risk assessment and position management tools

## 🧠 Enhanced Sentiment Analysis

Our advanced **two-stage sentiment engine** features:

### Stage 1: Continuous Monitoring (Always Running)
- **Multi-layered Scoring** - TextBlob + VADER sentiment models
- **Memory Efficient** - ~100MB usage for continuous operation
- **Real-time Collection** - 30-minute smart collector intervals
- **ML Feature Engineering** - 10 trading features extracted
- **Base Quality** - 70% accuracy for rapid analysis

### Stage 2: Enhanced Analysis (On-Demand)
- **FinBERT Integration** - Financial domain-specific sentiment
- **Advanced Models** - RoBERTa + emotion detection + news classification
- **High Quality** - 85-95% accuracy with transformer models
- **Comprehensive Analysis** - Processes ALL daily data for maximum quality
- **Memory Intelligent** - Automatically activates when resources permit

### Key Features
- **Temporal Analysis** - Time-weighted sentiment trends
- **Confidence Metrics** - Statistical confidence in sentiment scores
- **Market Context** - Volatility and regime-aware adjustments
- **Quality Escalation** - Automatic upgrade from Stage 1 to Stage 2
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

### Two-Stage Analysis Environment Variables

| Variable | Values | Purpose |
|----------|--------|---------|
| `USE_TWO_STAGE_ANALYSIS` | `0`/`1` | Enable intelligent two-stage analysis |
| `SKIP_TRANSFORMERS` | `0`/`1` | Control transformer model loading |
| `FINBERT_ONLY` | `0`/`1` | Load only FinBERT (memory optimized) |

### Memory Optimization Examples
```bash
# Maximum quality (requires ~800MB+ memory)
export USE_TWO_STAGE_ANALYSIS=1
export SKIP_TRANSFORMERS=0

# Memory constrained (uses ~100MB memory)
export USE_TWO_STAGE_ANALYSIS=1
export SKIP_TRANSFORMERS=1

# Balanced mode (FinBERT only, ~400MB memory)
export USE_TWO_STAGE_ANALYSIS=1
export FINBERT_ONLY=1
```

## 📈 Performance

- **Two-Stage Architecture** - Intelligent memory management (100MB → 800MB as needed)
- **Optimized Data Processing** - Efficient pandas operations
- **Async Components** - Non-blocking data collection
- **Memory Management** - Automatic Stage 1 ↔ Stage 2 switching
- **Caching** - Intelligent data caching strategies
- **Quality Escalation** - 70% → 95% accuracy when memory permits

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Key Components

- **Two-Stage Sentiment Analyzer** - `app/core/sentiment/two_stage_analyzer.py`
- **Enhanced Sentiment Scoring** - `app/core/sentiment/enhanced_scoring.py`
- **Professional Dashboard** - `app/dashboard/main.py`
- **Daily Operations Manager** - `app/services/daily_manager.py`
- **Technical Analysis** - `app/core/analysis/technical.py`
- **Smart Data Collection** - `app/core/data/collectors/`
- **Memory Optimization** - Intelligent transformer loading

## 🎯 Recent Updates

- ✅ **Two-Stage ML System** - Intelligent memory management with quality escalation (70% → 95%)
- ✅ **Complete Project Restructuring** - Professional Python package architecture
- ✅ **Enhanced Sentiment System** - Multi-layered sentiment analysis with confidence metrics
- ✅ **Memory Optimization** - Automatic Stage 1 ↔ Stage 2 switching based on available resources
- ✅ **Legacy Cleanup** - 100+ legacy files organized and archived
- ✅ **Comprehensive Testing** - 63+ tests ensuring system reliability
- ✅ **Professional Dashboard** - Modern web interface with interactive charts
- ✅ **Smart Collector** - Background data collection with 30-minute intervals

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


# Stage 1 only (memory optimized)
export SKIP_TRANSFORMERS=1 && python -m app.main morning

# Stage 2 enhanced (full quality)
export USE_TWO_STAGE_ANALYSIS=1 && export SKIP_TRANSFORMERS=0 && python -m app.main evening

# System health check
python -c "import os; os.environ['USE_TWO_STAGE_ANALYSIS']='1'; ..."