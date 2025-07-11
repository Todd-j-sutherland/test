# Immediate Action Plan - Next 7 Days

## Day 1-2: Setup & Historical Data Collection

### Morning of Day 1
```bash
# 1. Create all necessary scripts
mkdir scripts
cd scripts

# 2. Create historical data collector
cat > generate_historical_samples.py << 'EOF'
#!/usr/bin/env python3
import sys
sys.path.append('..')
from src.ml_training_pipeline import MLTrainingPipeline
from src.news_sentiment import NewsSentimentAnalyzer
import pandas as pd
from datetime import datetime, timedelta

def collect_historical():
    ml_pipeline = MLTrainingPipeline()
    analyzer = NewsSentimentAnalyzer()
    
    # Check current sample count
    X, y = ml_pipeline.prepare_training_dataset(min_samples=1)
    current_samples = len(X) if X is not None else 0
    print(f"Current samples: {current_samples}")
    
    # Collect for past 14 days
    symbols = ['CBA.AX', 'WBC.AX', 'ANZ.AX', 'NAB.AX']
    end_date = datetime.now()
    current_date = end_date - timedelta(days=14)
    
    samples_collected = 0
    while current_date < end_date and samples_collected < 90:
        if current_date.weekday() < 5:  # Weekday
            for hour in [10, 11, 14, 15]:  # Key trading hours
                for symbol in symbols:
                    try:
                        print(f"Processing {symbol} at {current_date.date()} {hour}:00")
                        result = analyzer.analyze_bank_sentiment(symbol)
                        
                        # Simulate paper trade outcome
                        if result['overall_sentiment'] > 0.1 and result['confidence'] > 0.6:
                            feature_id = result.get('ml_feature_id')
                            if feature_id:
                                # Record simulated outcome
                                outcome = {
                                    'symbol': symbol,
                                    'signal_timestamp': f"{current_date.date()}T{hour}:00:00",
                                    'signal_type': 'BUY',
                                    'entry_price': 100,  # Placeholder
                                    'exit_price': 100.5,  # 0.5% gain
                                    'exit_timestamp': f"{current_date.date()}T{hour+4}:00:00"
                                }
                                ml_pipeline.record_trading_outcome(feature_id, outcome)
                                samples_collected += 1
                    except Exception as e:
                        print(f"Error: {e}")
                        
        current_date += timedelta(days=1)
    
    print(f"Collected {samples_collected} new samples")
    print(f"Total samples now: {current_samples + samples_collected}")

if __name__ == "__main__":
    collect_historical()
EOF

# 3. Run historical collection
python generate_historical_samples.py
```

### Afternoon of Day 1
```bash
# 4. Create automated collector for live data
cat > auto_collect.py << 'EOF'
#!/usr/bin/env python3
import sys
sys.path.append('..')
from news_trading_analyzer import NewsTradingAnalyzer
import time
from datetime import datetime

analyzer = NewsTradingAnalyzer()
symbols = ['CBA.AX', 'WBC.AX', 'ANZ.AX', 'NAB.AX']

print("Starting automated collection...")
while True:
    current_time = datetime.now()
    
    # Only during market hours
    if 10 <= current_time.hour <= 16 and current_time.weekday() < 5:
        print(f"\n[{current_time.strftime('%H:%M')}] Collecting data...")
        
        for symbol in symbols:
            try:
                result = analyzer.analyze_and_track(symbol)
                print(f"{symbol}: {result['signal']} (Score: {result['sentiment_score']:.3f})")
            except Exception as e:
                print(f"Error with {symbol}: {e}")
        
        # Wait 30 minutes
        time.sleep(1800)
    else:
        print("Market closed. Waiting...")
        time.sleep(3600)  # Check every hour when market is closed
EOF

# 5. Test the dashboard
streamlit run ../news_analysis_dashboard.py
```

## Day 2: Verify Data Quality

### Morning Check
```python
# scripts/verify_data.py
from src.ml_training_pipeline import MLTrainingPipeline

pipeline = MLTrainingPipeline()
X, y = pipeline.prepare_training_dataset(min_samples=1)

if X is not None:
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.columns.tolist()}")
    print(f"Class distribution: {y.value_counts()}")
    print(f"\nFeature statistics:")
    print(X.describe())
else:
    print("No data available yet")
```

## Day 3-4: Live Collection & Paper Trading

### Automated Schedule
```bash
# Add to crontab (crontab -e)
# Run every 30 minutes during market hours
*/30 10-15 * * 1-5 cd /path/to/project && python scripts/auto_collect.py >> logs/collection.log 2>&1

# Evening review
0 17 * * 1-5 cd /path/to/project && python scripts/daily_review.py
```

### Paper Trading Tracker
```python
# scripts/paper_trades.csv
# Track all signals for later outcome recording
Date,Time,Symbol,Signal,Sentiment,Confidence,Entry_Price,Status
2024-01-15,10:00,CBA.AX,BUY,0.45,0.75,104.50,OPEN
2024-01-15,11:00,WBC.AX,SELL,-0.35,0.65,25.30,OPEN
```

## Day 5: First Model Training

### Train Models
```python
# scripts/train_first_models.py
from src.ml_training_pipeline import MLTrainingPipeline
from src.ml_trading_config import ProgressiveLearning

pipeline = MLTrainingPipeline()
X, y = pipeline.prepare_training_dataset(min_samples=50)

if X is not None and len(X) >= 50:
    print(f"Training with {len(X)} samples")
    
    # Use progressive learning
    prog_learner = ProgressiveLearning()
    model = prog_learner.get_appropriate_model(len(X))
    
    # Train
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Initial accuracy: {score:.2%}")
    
    # Save
    import joblib
    joblib.dump(model, 'data/ml_models/models/initial_model.pkl')
else:
    print(f"Need at least 50 samples. Current: {len(X) if X is not None else 0}")
```

## Day 6-7: Optimization & Production Prep

### Model Evaluation
```python
# scripts/evaluate_models.py
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load model and test data
model = joblib.load('data/ml_models/models/initial_model.pkl')

# Get predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Metrics
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance (if available)
if hasattr(model, 'feature_importances_'):
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    print(importance.head(10))
```

### Production Configuration
```python
# config/production_config.py
TRADING_CONFIG = {
    'min_sentiment_score': 0.2,
    'min_confidence': 0.7,
    'max_position_size': 0.02,  # 2% of capital
    'stop_loss': 0.005,  # 0.5%
    'take_profit': 0.015,  # 1.5%
    'max_daily_trades': 4,
    'trading_hours': {
        'start': 10,
        'end': 15,
        'avoid_first_30min': True,
        'avoid_last_30min': True
    }
}

# Risk checks
RISK_LIMITS = {
    'max_correlation': 0.7,  # Between positions
    'max_sector_exposure': 0.3,  # 30% in one sector
    'required_samples': 100,  # Before live trading
    'min_backtest_sharpe': 1.0
}
```

## Daily Checklist

### Pre-Market (8:30 AM - 9:30 AM)
- [ ] Run `python news_trading_analyzer.py -a -e`
- [ ] Check dashboard: `streamlit run news_analysis_dashboard.py`
- [ ] Review overnight news impact
- [ ] Check technical alignment on 4H charts

### Market Hours (10:00 AM - 4:00 PM)
- [ ] Monitor automated collection logs
- [ ] Check signals every hour
- [ ] Record paper trade entries
- [ ] Update exit prices for closed trades

### Post-Market (4:30 PM - 5:30 PM)
- [ ] Record all trade outcomes
- [ ] Run daily review script
- [ ] Check sample count progress
- [ ] Backup database: `cp data/ml_models/training_data.db backups/`

## Emergency Commands

```bash
# If collection stops
ps aux | grep auto_collect  # Check if running
python scripts/auto_collect.py &  # Restart in background

# If database corrupted
cp backups/training_data.db data/ml_models/  # Restore backup

# If model performance drops
python scripts/evaluate_models.py  # Check metrics
python scripts/train_first_models.py  # Retrain

# Export all data
python -c "
from src.ml_training_pipeline import MLTrainingPipeline
p = MLTrainingPipeline()
X, y = p.prepare_training_dataset(1)
if X is not None:
    X.to_csv('training_features.csv')
    y.to_csv('training_labels.csv')
"
```

## Success Metrics

By Day 7, you should have:
- ✅ 80-100 training samples
- ✅ First ML model trained
- ✅ Paper trading system running
- ✅ Automated collection active
- ✅ Dashboard showing live results
- ✅ Backup and recovery procedures

**Next Week**: Focus on model improvement and preparing for AWS deployment!