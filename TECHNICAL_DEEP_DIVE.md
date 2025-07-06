# Technical Deep Dive: How the Analysis Works

This document explains the technical details of how the ASX Bank Trading Analysis System generates its predictions and signals.

## 🧠 Analysis Pipeline

### 1. Data Collection Phase

```
Yahoo Finance API → Raw OHLCV Data → Cache → Processed Data
        ↓
Company Fundamentals → Financial Ratios → Normalized Metrics
        ↓
News Sources → Raw Articles → Sentiment Scores → Weighted Sentiment
```

**Key Data Points Collected:**
- **Price Data**: Open, High, Low, Close, Volume (62 days default)
- **Financial Data**: P/E, ROE, Dividend Yield, Market Cap, etc.
- **News Data**: Headlines, sentiment scores, publication dates
- **Market Context**: Index performance, sector trends

### 2. Technical Analysis Engine

#### Indicators Calculated

**Momentum Indicators:**
- **RSI (Relative Strength Index)**: Measures overbought/oversold conditions
  ```python
  RSI = 100 - (100 / (1 + RS))
  RS = Average Gain / Average Loss (14 periods)
  ```
  - Values > 70: Potentially overbought
  - Values < 30: Potentially oversold

- **MACD (Moving Average Convergence Divergence)**: Trend following momentum
  ```python
  MACD Line = EMA12 - EMA26
  Signal Line = EMA9 of MACD Line
  Histogram = MACD Line - Signal Line
  ```

**Trend Indicators:**
- **SMA (Simple Moving Average)**: 20, 50, 200 period averages
- **EMA (Exponential Moving Average)**: 12, 26 period averages
- **Bollinger Bands**: Price volatility bands
  ```python
  Middle Band = SMA20
  Upper Band = SMA20 + (2 × Standard Deviation)
  Lower Band = SMA20 - (2 × Standard Deviation)
  ```

**Volatility Indicators:**
- **ATR (Average True Range)**: Price volatility measurement
- **Bollinger Band Width**: Volatility expansion/contraction

**Volume Indicators:**
- **Volume SMA**: Average volume trends
- **Volume Spikes**: Unusual volume detection

#### Signal Generation Logic

Each indicator generates a score from -100 to +100:

```python
def calculate_rsi_signal(rsi_value):
    if rsi_value > 70:
        return -30  # Overbought (bearish)
    elif rsi_value < 30:
        return 30   # Oversold (bullish)
    else:
        return 0    # Neutral

def calculate_macd_signal(macd, signal):
    if macd > signal:
        return 20   # Bullish crossover
    else:
        return -20  # Bearish crossover
```

**Overall Technical Score:**
```python
technical_score = (
    rsi_signal * 0.3 +
    macd_signal * 0.25 +
    trend_signal * 0.25 +
    volume_signal * 0.2
)
```

### 3. Fundamental Analysis Engine

#### Bank-Specific Metrics

**Core Banking Ratios:**
- **Return on Equity (ROE)**: Profitability measure
  ```
  ROE = Net Income / Shareholders' Equity
  ```
- **Net Interest Margin (NIM)**: Interest earning efficiency
- **Tier 1 Capital Ratio**: Financial strength
- **Cost-to-Income Ratio**: Operational efficiency

**Valuation Metrics:**
- **P/E Ratio**: Price relative to earnings
- **Price-to-Book Ratio**: Price relative to book value
- **Dividend Yield**: Income generation

#### Scoring Algorithm

```python
def score_fundamental_metric(value, benchmarks):
    if value >= benchmarks['excellent']:
        return 100
    elif value >= benchmarks['good']:
        return 75
    elif value >= benchmarks['fair']:
        return 50
    else:
        return 25

# Example benchmarks for ROE
roe_benchmarks = {
    'excellent': 0.15,  # 15%+
    'good': 0.12,       # 12-15%
    'fair': 0.10,       # 10-12%
    'poor': 0.08        # <10%
}
```

**Peer Comparison:**
- Ranks bank against ASX bank peers
- Adjusts score based on relative performance
- Accounts for sector-wide trends

### 4. Sentiment Analysis Engine

#### News Sentiment Processing

**Data Sources:**
- RSS feeds from financial news sites
- Company announcements
- Market commentary

**Sentiment Calculation:**
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

def analyze_sentiment(text):
    # VADER sentiment (good for social media/news)
    vader_score = analyzer.polarity_scores(text)['compound']
    
    # TextBlob sentiment (general purpose)
    blob_score = TextBlob(text).sentiment.polarity
    
    # Weighted combination
    final_score = (vader_score * 0.6) + (blob_score * 0.4)
    return final_score  # Range: -1 (very negative) to +1 (very positive)
```

**Aggregation Logic:**
```python
def calculate_overall_sentiment(news_items):
    scores = []
    weights = []
    
    for item in news_items:
        sentiment = analyze_sentiment(item['title'] + ' ' + item['summary'])
        
        # Weight by recency (newer = more important)
        days_old = (now - item['date']).days
        weight = max(0.1, 1.0 - (days_old / 30))
        
        scores.append(sentiment)
        weights.append(weight)
    
    # Weighted average
    weighted_sentiment = sum(s*w for s,w in zip(scores, weights)) / sum(weights)
    return weighted_sentiment
```

### 5. Risk Calculation Engine

#### Position Sizing

**Kelly Criterion Implementation:**
```python
def calculate_kelly_percentage(win_probability, avg_win, avg_loss):
    if avg_loss == 0:
        return 0
    
    win_loss_ratio = avg_win / avg_loss
    kelly = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio
    
    # Cap at 25% for safety
    return min(kelly, 0.25)
```

**ATR-Based Stop Loss:**
```python
def calculate_stop_loss(current_price, atr, multiplier=2.0):
    stop_distance = atr * multiplier
    stop_loss_price = current_price - stop_distance
    stop_loss_percentage = (stop_distance / current_price) * 100
    
    return {
        'price': stop_loss_price,
        'percentage': stop_loss_percentage,
        'distance': stop_distance
    }
```

**Risk Score Calculation:**
```python
def calculate_risk_score(volatility, beta, debt_ratio, market_conditions):
    base_risk = 50  # Neutral starting point
    
    # Volatility adjustment
    if volatility > 0.25:  # High volatility
        base_risk += 20
    elif volatility < 0.15:  # Low volatility
        base_risk -= 10
    
    # Beta adjustment
    if beta > 1.5:  # High beta (more volatile than market)
        base_risk += 15
    elif beta < 0.8:  # Low beta (less volatile)
        base_risk -= 10
    
    # Debt adjustment
    if debt_ratio > 0.6:  # High debt
        base_risk += 10
    
    return min(100, max(0, base_risk))
```

### 6. Market Prediction Engine

#### Weighted Scoring System

```python
prediction_weights = {
    'technical': 0.35,      # 35% weight
    'fundamental': 0.25,    # 25% weight
    'sentiment': 0.20,      # 20% weight
    'market_structure': 0.15, # 15% weight
    'seasonality': 0.05     # 5% weight
}

def calculate_prediction(analysis_results):
    weighted_score = 0
    
    for component, weight in prediction_weights.items():
        component_score = analysis_results[component]['score']
        weighted_score += component_score * weight
    
    # Determine direction and confidence
    if weighted_score > 30:
        direction = 'bullish'
        strength = 'strong' if weighted_score > 60 else 'moderate'
    elif weighted_score < -30:
        direction = 'bearish'
        strength = 'strong' if weighted_score < -60 else 'moderate'
    else:
        direction = 'neutral'
        strength = 'weak'
    
    confidence = abs(weighted_score)
    
    return {
        'direction': direction,
        'strength': strength,
        'confidence': confidence,
        'score': weighted_score
    }
```

#### Time Frame Analysis

**Short-term (1-5 days):**
- Heavy weight on technical indicators
- Recent news sentiment
- Options activity (if available)

**Medium-term (1-4 weeks):**
- Balanced technical and fundamental
- Earnings expectations
- Sector trends

**Long-term (1-3 months):**
- Heavy weight on fundamentals
- Economic indicators
- Structural changes

### 7. Confidence Calculation

The confidence score represents how sure the system is about its prediction:

```python
def calculate_confidence(predictions, market_conditions):
    base_confidence = abs(predictions['weighted_score'])
    
    # Adjustment factors
    adjustments = []
    
    # Data quality
    if all_data_available:
        adjustments.append(1.0)
    else:
        adjustments.append(0.8)
    
    # Market volatility
    if market_volatility < 0.20:  # Stable market
        adjustments.append(1.1)
    else:  # High volatility
        adjustments.append(0.9)
    
    # Agreement between indicators
    agreement = calculate_indicator_agreement(predictions)
    adjustments.append(agreement)
    
    # Apply adjustments
    final_confidence = base_confidence
    for adj in adjustments:
        final_confidence *= adj
    
    return min(100, final_confidence)
```

### 8. Signal Strength Categories

| Score Range | Signal | Interpretation | Action |
|-------------|--------|----------------|---------|
| 75-100 | Very Strong | High conviction trade | Full position |
| 50-74 | Strong | Good probability | Standard position |
| 25-49 | Moderate | Uncertain outcome | Small position |
| 0-24 | Weak | Unclear direction | Avoid/Wait |

### 9. Real-time Monitoring

**Alert Triggers:**
```python
def check_alert_conditions(current_analysis, previous_analysis):
    alerts = []
    
    # Signal change alert
    if current_analysis['direction'] != previous_analysis['direction']:
        alerts.append({
            'type': 'SIGNAL_CHANGE',
            'message': f"Signal changed from {previous_analysis['direction']} to {current_analysis['direction']}"
        })
    
    # Confidence threshold alert
    if current_analysis['confidence'] > 70:
        alerts.append({
            'type': 'HIGH_CONFIDENCE',
            'message': f"High confidence {current_analysis['direction']} signal ({current_analysis['confidence']:.1f}%)"
        })
    
    return alerts
```

## 🎯 Key Performance Metrics

**System Accuracy Tracking:**
- Signal accuracy over time
- Confidence calibration
- Risk-adjusted returns
- Maximum drawdown

**Quality Indicators:**
- Data freshness
- API success rates
- Analysis completion time
- Error rates

## 🔧 Customization Points

**Adjustable Parameters:**
- Indicator periods (RSI: 14, MACD: 12,26,9)
- Weight distributions
- Alert thresholds
- Risk multipliers

**Adding New Indicators:**
```python
def custom_indicator(data):
    # Your calculation here
    return indicator_value

# Add to technical analysis
def _calculate_indicators(self, data):
    indicators = {}
    # ... existing indicators
    indicators['custom'] = custom_indicator(data)
    return indicators
```

This technical foundation ensures the system provides reliable, data-driven analysis while remaining transparent and customizable for different trading styles and risk preferences.
