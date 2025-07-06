# ASX Bank Trading Analysis - Quick Start Guide

## 🚀 Quick Setup (5 minutes)

### 1. Prerequisites Check
```bash
# Check Python version (need 3.8+)
python --version

# Check pip is installed
pip --version
```

### 2. Setup Environment
```bash
# Clone or download the project
cd trading_analysis

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Test Installation
```bash
# Quick test - analyze CBA
python main.py bank --symbol CBA.AX

# Should show something like:
# Analysis for CBA.AX:
# Current Price: $178.00
# Signal: neutral
# Confidence: 8.79%
# Risk Score: 65
```

### 4. Run Full Analysis
```bash
# Analyze all banks
python main.py analyze

# Check the report
ls reports/
# Open the latest HTML file in browser
```

## 🎯 Basic Commands

| Command | Description | Example |
|---------|-------------|---------|
| `python main.py analyze` | Analyze all banks | Full system analysis |
| `python main.py bank --symbol CBA.AX` | Single bank | Detailed CBA analysis |
| `python main.py report` | Generate report | Create HTML report |
| `python main.py schedule` | Auto trading | Run every market day |

## 📊 Understanding Output

### Signal Types
- **bullish** 📈 = Buy signal
- **bearish** 📉 = Sell signal  
- **neutral** ➡️ = Hold/wait

### Confidence Levels
- **0-25%** = Don't trade
- **25-50%** = Small position
- **50-75%** = Normal position
- **75-100%** = Strong signal

### Risk Scores
- **0-25** = Low risk
- **25-50** = Medium risk
- **50-75** = High risk
- **75-100** = Very high risk

## 🔧 Optional: Setup Alerts

Create `.env` file for notifications:

```bash
# Discord (get webhook from Discord server settings)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook_url

# Telegram (create bot with @BotFather)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Email (use app password, not regular password)
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
```

## 🚨 Troubleshooting

### ❌ Common Issues

**"ModuleNotFoundError"**
```bash
# Make sure virtual environment is activated
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate    # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**"No data available"**
```bash
# Check internet connection
# Clear cache and try again
rm -rf data/cache/*
python main.py bank --symbol CBA.AX
```

**"Permission denied"**
```bash
# On macOS/Linux, make script executable
chmod +x main.py
```

### ✅ Health Check

```bash
# Test each component
python -c "from src.data_feed import ASXDataFeed; print('Data Feed: OK')"
python -c "from src.technical_analysis import TechnicalAnalyzer; print('Technical: OK')"
python -c "from src.fundamental_analysis import FundamentalAnalyzer; print('Fundamental: OK')"
```

## 📈 Your First Analysis

### Step-by-Step Walkthrough

1. **Start with one bank:**
   ```bash
   python main.py bank --symbol CBA.AX
   ```

2. **Look at the output:**
   - Current Price: Stock price right now
   - Signal: What the system recommends
   - Confidence: How sure the system is
   - Risk Score: How risky this trade is

3. **Run full analysis:**
   ```bash
   python main.py analyze
   ```

4. **Check the report:**
   - Open `reports/daily_report_YYYYMMDD_HHMMSS.html` in browser
   - Review charts and analysis
   - Compare all banks side-by-side

5. **Set up automation (optional):**
   ```bash
   python main.py schedule
   ```
   - Runs automatically during market hours
   - Sends alerts when configured

## 🎓 Learning the System

### Week 1: Basics
- Run daily analysis manually
- Learn to read the signals
- Understand confidence scores
- Practice with paper trading

### Week 2: Advanced
- Set up automated alerts
- Customize settings in `config/settings.py`
- Learn risk management principles
- Study the generated reports

### Week 3: Optimization
- Fine-tune alert thresholds
- Add custom indicators
- Optimize for your trading style
- Backtest strategies

## 📚 Resources

- **Full README.md**: Complete documentation
- **config/settings.py**: All configuration options
- **logs/trading_system.log**: Detailed system logs
- **reports/**: Generated analysis reports

## ⚠️ Important Reminders

1. **This is NOT financial advice**
2. **Always do your own research**
3. **Start with paper trading**
4. **Use proper risk management**
5. **Markets can be unpredictable**

---

🎉 **You're ready to start!** Run `python main.py analyze` and explore your first analysis.

For detailed documentation, see the main README.md file.
