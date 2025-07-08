# 📰 News Analysis Dashboard

A comprehensive dashboard for visualizing ASX bank news sentiment analysis results.

## 🌟 Features

### Dashboard Displays:
- **Each Symbol Analyzed**: Shows all banks that have been analyzed (CBA, WBC, ANZ, NAB, MQG)
- **News Articles**: Displays relevant news articles found for each bank
- **Sentiment Impact**: Shows how news articles affected the scoring
- **Confidence Legend**: Clear explanation of confidence score requirements for trading decisions

### Key Information Shown:
1. **Sentiment Score**: -1.0 (very negative) to +1.0 (very positive)
2. **Confidence Level**: HIGH (≥0.8), MEDIUM (0.6-0.8), LOW (<0.6)
3. **Trading Signal**: Strong Buy/Buy/Hold/Sell/Strong Sell
4. **News Article Count**: Number of articles analyzed
5. **Component Breakdown**: How news, Reddit, events, volume, momentum, and ML contributed
6. **Recent Headlines**: Latest news headlines that influenced the analysis
7. **Significant Events**: Detected events like earnings, dividends, management changes

## 🚀 Usage

### Option 1: Quick HTML Dashboard (Recommended)
```bash
# Run analysis and generate dashboard in one command
python run_analysis_and_dashboard.py
```

This will:
1. Analyze all banks using the news trading analyzer
2. Generate an HTML dashboard that opens in your browser
3. Display all results in a beautiful, mobile-friendly interface

### Option 2: Manual Steps
```bash
# Step 1: Analyze all banks first
python news_trading_analyzer.py --all

# Step 2: Generate HTML dashboard
python generate_dashboard.py
```

### Option 3: Interactive Streamlit Dashboard
```bash
# Install required packages (if not already installed)
pip install streamlit plotly pandas

# Run interactive dashboard
streamlit run news_analysis_dashboard.py
```

## 📊 Dashboard Sections

### 1. Market Overview
- Average sentiment across all banks
- Confidence levels
- Total news articles analyzed
- Bullish vs bearish bank count

### 2. Confidence Legend & Decision Criteria
- **HIGH CONFIDENCE (≥0.8)**: Execute trades with full position sizing
- **MEDIUM CONFIDENCE (0.6-0.8)**: Execute trades with reduced position sizing  
- **LOW CONFIDENCE (<0.6)**: Avoid trading, wait for better signals

### 3. Sentiment Score Scale
- **Very Negative (-1.0 to -0.5)**: Strong Sell
- **Negative (-0.5 to -0.2)**: Sell
- **Neutral (-0.2 to +0.2)**: Hold
- **Positive (+0.2 to +0.5)**: Buy
- **Very Positive (+0.5 to +1.0)**: Strong Buy

### 4. Individual Bank Analysis
For each bank, displays:
- Current sentiment score and confidence
- Trading recommendation (Buy/Sell/Hold)
- Breakdown of sentiment components
- Recent news headlines
- Significant events detected
- Reddit sentiment analysis (if available)

## 🎨 Dashboard Features

- **Mobile-Responsive**: Works on desktop, tablet, and mobile
- **Real-time Data**: Shows latest analysis results
- **Color-Coded**: Green for positive, red for negative, gray for neutral
- **Interactive Elements**: Expandable sections for detailed analysis
- **Print-Friendly**: Optimized for printing reports

## 📁 Files Generated

- `news_analysis_dashboard.html`: Static HTML dashboard (auto-opens in browser)
- Individual bank analysis data in `data/sentiment_history/`

## 🔄 Refreshing Data

To update the dashboard with latest analysis:
```bash
python run_analysis_and_dashboard.py
```

This will re-analyze all banks and regenerate the dashboard with fresh data.

## ⚙️ Requirements

- Python 3.7+
- Internet connection (for fetching news)
- Optional: streamlit, plotly, pandas (for interactive dashboard)

## 🎯 Use Cases

1. **Morning Trading Prep**: Check overnight sentiment changes
2. **Decision Making**: Use confidence levels to determine trade sizing
3. **News Monitoring**: See which headlines are driving sentiment
4. **Risk Management**: Identify low-confidence signals to avoid
5. **Market Overview**: Get quick snapshot of banking sector sentiment

## 📈 Interpreting Results

### High-Quality Signals (Confidence ≥ 0.8)
- Multiple news sources agree
- Significant events detected
- Consistent sentiment across sources
- **Action**: Safe to trade with standard position sizing

### Medium-Quality Signals (Confidence 0.6-0.8)
- Some sources agree
- Moderate news volume
- **Action**: Trade with reduced position sizing

### Low-Quality Signals (Confidence < 0.6)
- Limited or conflicting sources
- Low news volume
- **Action**: Avoid trading, monitor for better signals

Enjoy your comprehensive news analysis dashboard! 🚀
