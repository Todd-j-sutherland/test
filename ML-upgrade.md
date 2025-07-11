# ML Training Optimization Guide - From 10 to 100 Samples

## Your Current Situation
- **Samples**: 10 (need 100 minimum)
- **Timeframes**: 1-hour primary, 4-hour and 15-minute for confirmation
- **Goal**: Build reliable ML models for ASX bank trading

## Fast Track to 100 Samples (7-10 Days)

### Method 1: Historical Data Generation (Fastest - 3 Days)

```python
# scripts/generate_historical_samples.py

import pandas as pd
from datetime import datetime, timedelta
from src.news_sentiment import NewsSentimentAnalyzer
from src.data_feed import ASXDataFeed
from src.ml_training_pipeline import MLTrainingPipeline

def generate_historical_samples():
    analyzer = NewsSentimentAnalyzer()
    data_feed = ASXDataFeed()
    ml_pipeline = MLTrainingPipeline()
    
    # Banks to analyze
    symbols = ['CBA.AX', 'WBC.AX', 'ANZ.AX', 'NAB.AX']
    
    # Go back 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # For each day
    current_date = start_date
    while current_date < end_date:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
            
        # For each trading hour (10 AM - 4 PM AEST)
        for hour in range(10, 16):
            for symbol in symbols:
                try:
                    # Simulate analysis at this time
                    print(f"Analyzing {symbol} at {current_date} {hour}:00")
                    
                    # Get sentiment (this will use cached/historical news)
                    sentiment_result = analyzer.analyze_bank_sentiment(symbol)
                    
                    # Store the feature
                    feature_id = ml_pipeline.collect_training_data(sentiment_result, symbol)
                    
                    # Get price at entry and 4 hours later
                    entry_price = data_feed.get_historical_price(symbol, current_date, hour)
                    exit_price = data_feed.get_historical_price(symbol, current_date, hour + 4)
                    
                    # Record outcome
                    outcome_data = {
                        'symbol': symbol,
                        'signal_timestamp': f"{current_date}T{hour}:00:00",
                        'signal_type': 'BUY' if sentiment_result['overall_sentiment'] > 0 else 'SELL',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'exit_timestamp': f"{current_date}T{hour+4}:00:00",
                        'max_drawdown': 0.5  # Simplified
                    }
                    
                    ml_pipeline.record_trading_outcome(feature_id, outcome_data)
                    
                except Exception as e:
                    print(f"Error processing {symbol} at {current_date}: {e}")
        
        current_date += timedelta(days=1)
    
    print("Historical sample generation complete!")

# Run it
if __name__ == "__main__":
    generate_historical_samples()
```

### Method 2: High-Frequency Collection (5-7 Days)

```python
# scripts/rapid_collector.py

import schedule
import time
from datetime import datetime

class RapidDataCollector:
    def __init__(self):
        self.symbols = ['CBA.AX', 'WBC.AX', 'ANZ.AX', 'NAB.AX']
        self.analyzer = NewsTradingAnalyzer()
        self.sample_count = 10  # Starting count
        
    def collect_samples(self):
        """Collect samples every 30 minutes during market hours"""
        current_hour = datetime.now().hour
        
        # Only during market hours (10 AM - 4 PM AEST)
        if 10 <= current_hour <= 16:
            for symbol in self.symbols:
                result = self.analyzer.analyze_and_track(symbol)
                if result.get('signal') != 'HOLD':
                    self.sample_count += 1
                    print(f"Sample {self.sample_count}: {symbol} - {result['signal']}")
            
            # Paper trade simulation
            self.simulate_trade_outcomes()
    
    def simulate_trade_outcomes(self):
        """Simulate trade outcomes after 1-4 hours"""
        # Check trades that are 1-4 hours old
        # Record their outcomes based on actual price movements
        pass
    
    def run(self):
        # Schedule every 30 minutes
        schedule.every(30).minutes.do(self.collect_samples)
        
        print("Starting rapid data collection...")
        while self.sample_count < 100:
            schedule.run_pending()
            time.sleep(60)
            
        print(f"Target reached! {self.sample_count} samples collected.")

collector = RapidDataCollector()
collector.run()
```

### Method 3: Multi-Timeframe Collection (7 Days)

```python
# Collect on multiple timeframes simultaneously
def collect_multiframe_samples():
    timeframes = {
        '15min': {'samples_per_hour': 4, 'hold_period': '1H'},
        '1H': {'samples_per_hour': 1, 'hold_period': '4H'},
        '4H': {'samples_per_hour': 0.25, 'hold_period': '1D'}
    }
    
    for tf_name, tf_config in timeframes.items():
        # Analyze on each timeframe
        # Each provides different training samples
        pass
```

## Optimizing Your ML Pipeline

### 1. Feature Engineering for 1-Hour Trading

```python
# Add to ml_trading_config.py

class HourlyTradingFeatures:
    def __init__(self):
        self.timeframes = ['15M', '1H', '4H']
    
    def extract_features(self, symbol, timestamp):
        features = {}
        
        # 1. Multi-timeframe alignment
        features['tf_alignment_score'] = self.calculate_timeframe_alignment()
        
        # 2. Hour-specific patterns
        hour = timestamp.hour
        features['is_opening_hour'] = int(hour == 10)
        features['is_closing_hour'] = int(hour == 15)
        features['is_lunch_hour'] = int(12 <= hour <= 13)
        
        # 3. Momentum features for 1H
        features['momentum_15m'] = self.get_momentum(symbol, '15M', periods=4)
        features['momentum_1h'] = self.get_momentum(symbol, '1H', periods=3)
        features['momentum_4h'] = self.get_momentum(symbol, '4H', periods=2)
        
        # 4. Volume patterns
        features['volume_vs_avg_1h'] = self.get_volume_ratio(symbol, '1H')
        features['volume_acceleration'] = self.get_volume_acceleration(symbol)
        
        # 5. News decay (important for 1H trading)
        features['news_age_hours'] = self.get_news_age_hours()
        features['news_decay_factor'] = np.exp(-features['news_age_hours'] / 6)
        
        # 6. Technical structure
        features['distance_from_vwap'] = self.get_vwap_distance(symbol)
        features['support_resistance_distance'] = self.get_sr_distance(symbol)
        
        return features
    
    def calculate_timeframe_alignment(self):
        """Check if 15M, 1H, and 4H are aligned"""
        signals = {
            '15M': self.get_signal('15M'),
            '1H': self.get_signal('1H'),
            '4H': self.get_signal('4H')
        }
        
        # Full alignment = 1.0, partial = 0.5, no alignment = 0
        if all(s == signals['1H'] for s in signals.values()):
            return 1.0
        elif signals['1H'] == signals['4H']:
            return 0.7
        elif signals['15M'] == signals['1H']:
            return 0.5
        else:
            return 0.0
```

### 2. Label Engineering for Better Training

```python
# Improve your labeling strategy

def create_advanced_labels(price_data, entry_time, symbol):
    """Create more nuanced labels than just profitable/unprofitable"""
    
    entry_price = price_data.loc[entry_time, 'Close']
    
    # Check multiple exit points
    labels = {}
    
    # 1 hour exit
    exit_1h = price_data.loc[entry_time + pd.Timedelta(hours=1), 'Close']
    ret_1h = (exit_1h - entry_price) / entry_price
    
    # 2 hour exit  
    exit_2h = price_data.loc[entry_time + pd.Timedelta(hours=2), 'Close']
    ret_2h = (exit_2h - entry_price) / entry_price
    
    # 4 hour exit
    exit_4h = price_data.loc[entry_time + pd.Timedelta(hours=4), 'Close']
    ret_4h = (exit_4h - entry_price) / entry_price
    
    # Maximum adverse excursion (MAE)
    period_low = price_data.loc[entry_time:entry_time + pd.Timedelta(hours=4), 'Low'].min()
    mae = (period_low - entry_price) / entry_price
    
    # Maximum favorable excursion (MFE)
    period_high = price_data.loc[entry_time:entry_time + pd.Timedelta(hours=4), 'High'].max()
    mfe = (period_high - entry_price) / entry_price
    
    # Create multi-class labels
    labels['binary'] = int(ret_4h > 0.002)  # 0.2% after costs
    
    # Multi-class (more informative)
    if ret_4h > 0.01:  # 1%+
        labels['multiclass'] = 'strong_win'
    elif ret_4h > 0.002:
        labels['multiclass'] = 'small_win'
    elif ret_4h > -0.002:
        labels['multiclass'] = 'breakeven'
    elif ret_4h > -0.01:
        labels['multiclass'] = 'small_loss'
    else:
        labels['multiclass'] = 'large_loss'
    
    # Regression targets
    labels['return_1h'] = ret_1h
    labels['return_2h'] = ret_2h
    labels['return_4h'] = ret_4h
    labels['max_drawdown'] = mae
    labels['max_profit'] = mfe
    
    # Risk-adjusted label
    if mae < -0.005:  # Stop would have been hit
        labels['risk_adjusted'] = 'stopped_out'
    elif mfe > 0.01 and ret_4h > 0.005:
        labels['risk_adjusted'] = 'good_trade'
    else:
        labels['risk_adjusted'] = 'poor_trade'
    
    return labels

### 3. Model Selection for Small Data (10-100 samples)

```python
# Best models for limited data

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

def get_small_data_models():
    """Models that work well with 10-100 samples"""
    
    models = {
        # 1. Logistic Regression - great baseline, doesn't overfit easily
        'logistic': LogisticRegression(
            penalty='l2',
            C=0.1,  # Strong regularization
            class_weight='balanced',
            max_iter=1000
        ),
        
        # 2. Random Forest - limit complexity
        'rf_conservative': RandomForestClassifier(
            n_estimators=50,  # Fewer trees
            max_depth=3,      # Shallow trees
            min_samples_leaf=5,  # Require more samples per leaf
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        ),
        
        # 3. SVM - good for small data
        'svm': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True
        ),
        
        # 4. XGBoost - configured for small data
        'xgb_small': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.5,
            min_child_weight=5,  # Prevent overfitting
            reg_alpha=1.0,       # L1 regularization
            reg_lambda=1.0,      # L2 regularization
            scale_pos_weight=2,
            eval_metric='logloss'
        )
    }
    
    return models

# Cross-validation for small datasets
from sklearn.model_selection import TimeSeriesSplit, LeaveOneOut

def validate_small_dataset(X, y, model):
    """Proper validation for small datasets"""
    
    # For 10-30 samples: Leave-One-Out
    if len(X) < 30:
        cv = LeaveOneOut()
        print("Using Leave-One-Out CV due to small sample size")
    # For 30-100 samples: Time Series Split with 3 folds
    else:
        cv = TimeSeriesSplit(n_splits=3)
        print("Using TimeSeriesSplit with 3 folds")
    
    scores = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

### 4. Data Augmentation Techniques

```python
# Augment your limited dataset

def augment_trading_data(X, y, n_synthetic=50):
    """Generate synthetic samples to boost training data"""
    
    from imblearn.over_sampling import SMOTE, ADASYN
    import numpy as np
    
    # 1. SMOTE for generating synthetic samples
    smote = SMOTE(
        sampling_strategy='auto',
        k_neighbors=min(5, len(X)-1),  # Adjust for small data
        random_state=42
    )
    
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # 2. Add noise-based augmentation
    X_augmented = []
    y_augmented = []
    
    for i in range(len(X)):
        # Original sample
        X_augmented.append(X.iloc[i])
        y_augmented.append(y.iloc[i])
        
        # Create variations with small noise
        for j in range(2):  # 2 variations per sample
            noise = np.random.normal(0, 0.05, X.shape[1])
            X_noisy = X.iloc[i] + noise
            
            # Slightly perturb continuous features only
            for col in X.columns:
                if 'sentiment' in col or 'score' in col:
                    X_noisy[col] = np.clip(X_noisy[col], -1, 1)
            
            X_augmented.append(X_noisy)
            y_augmented.append(y.iloc[i])  # Keep same label
    
    # 3. Time-shift augmentation
    # Shift sentiment features by small amounts to simulate different timing
    
    return pd.DataFrame(X_augmented), pd.Series(y_augmented)
```

### 5. Ensemble Strategy for Reliability

```python
# Ensemble to reduce overfitting with small data

class SmallDataEnsemble:
    def __init__(self):
        self.base_models = get_small_data_models()
        self.ensemble_predictions = {}
        
    def fit(self, X, y):
        """Train all base models"""
        for name, model in self.base_models.items():
            print(f"Training {name}...")
            
            # Use different random subsets for each model
            n_samples = int(0.8 * len(X))
            indices = np.random.choice(len(X), n_samples, replace=False)
            
            X_subset = X.iloc[indices]
            y_subset = y.iloc[indices]
            
            model.fit(X_subset, y_subset)
            
            # Store out-of-bag predictions for stacking
            oob_indices = [i for i in range(len(X)) if i not in indices]
            if oob_indices:
                self.ensemble_predictions[name] = {
                    'indices': oob_indices,
                    'predictions': model.predict_proba(X.iloc[oob_indices])[:, 1]
                }
    
    def predict_proba(self, X):
        """Weighted ensemble prediction"""
        predictions = []
        weights = {
            'logistic': 0.3,      # Stable baseline
            'rf_conservative': 0.3,
            'svm': 0.2,
            'xgb_small': 0.2
        }
        
        for name, model in self.base_models.items():
            pred = model.predict_proba(X)[:, 1]
            weighted_pred = pred * weights[name]
            predictions.append(weighted_pred)
        
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0)
        
        # Return as probability array
        return np.column_stack([1 - ensemble_pred, ensemble_pred])
```

### 6. Progressive Learning Strategy

```python
# Start simple, add complexity as data grows

class ProgressiveLearning:
    def __init__(self):
        self.sample_thresholds = {
            10: 'simple_rules',
            30: 'basic_ml',
            60: 'ensemble',
            100: 'advanced_ml',
            200: 'deep_learning'
        }
    
    def get_appropriate_model(self, n_samples):
        """Select model based on data availability"""
        
        if n_samples < 10:
            print("⚠️  Too few samples. Using rule-based approach.")
            return self.get_rule_based_system()
            
        elif n_samples < 30:
            print("📊 Using simple ML with heavy regularization")
            return LogisticRegression(C=0.01, class_weight='balanced')
            
        elif n_samples < 60:
            print("🌲 Using Random Forest with constraints")
            return RandomForestClassifier(
                n_estimators=30,
                max_depth=3,
                min_samples_leaf=5
            )
            
        elif n_samples < 100:
            print("🚀 Using ensemble approach")
            return SmallDataEnsemble()
            
        else:
            print("💪 Sufficient data for advanced models")
            return self.get_advanced_model()
    
    def get_rule_based_system(self):
        """Simple rules when ML isn't viable"""
        
        class RuleBasedTrader:
            def fit(self, X, y):
                # Calculate simple thresholds from data
                self.sentiment_threshold = X['sentiment_score'].quantile(0.7)
                self.confidence_threshold = X['confidence'].quantile(0.6)
                
            def predict_proba(self, X):
                # Simple rules
                signals = (
                    (X['sentiment_score'] > self.sentiment_threshold) &
                    (X['confidence'] > self.confidence_threshold) &
                    (X['momentum_1h'] > 0)
                ).astype(float)
                
                # Return as probability array
                return np.column_stack([1 - signals, signals])
        
        return RuleBasedTrader()
```

## Practical Daily Workflow

### Morning Routine (7:00 AM - 9:30 AM)
```bash
# 1. Check overnight news impact
python news_trading_analyzer.py -a -e

# 2. Review dashboard
streamlit run news_analysis_dashboard.py

# 3. Update ML models if new data available
python scripts/check_and_retrain.py
```

### Market Hours (10:00 AM - 4:00 PM)
```bash
# Every hour on the hour
0 10-16 * * 1-5 cd /path/to/project && python news_trading_analyzer.py -a

# 15 minutes after the hour (for 1H bar completion)
15 10-16 * * 1-5 cd /path/to/project && python scripts/record_outcomes.py

# Check 4H alignment at 10 AM and 2 PM
0 10,14 * * 1-5 cd /path/to/project && python news_trading_analyzer.py -a --technical
```

### Evening Review (5:00 PM - 6:00 PM)
```python
# scripts/daily_review.py

def daily_performance_review():
    # 1. Calculate hit rate
    todays_trades = get_todays_trades()
    hit_rate = calculate_hit_rate(todays_trades)
    
    # 2. Update ML training data
    for trade in todays_trades:
        if trade['completed']:
            record_outcome(trade)
    
    # 3. Generate report
    report = {
        'date': datetime.now().date(),
        'trades': len(todays_trades),
        'hit_rate': hit_rate,
        'samples_collected': count_new_samples(),
        'total_samples': get_total_samples()
    }
    
    print(f"\n📊 Daily Review - {report['date']}")
    print(f"Trades: {report['trades']}")
    print(f"Hit Rate: {report['hit_rate']:.1%}")
    print(f"New Samples: {report['samples_collected']}")
    print(f"Total Samples: {report['total_samples']}/100")
    
    # 4. Retrain if significant new data
    if report['samples_collected'] >= 10:
        print("\n🔄 Retraining models with new data...")
        retrain_models()
```

## Monitoring & Debugging

### Key Metrics to Track
```python
# scripts/ml_monitor.py

class MLMonitor:
    def __init__(self):
        self.metrics = {
            'prediction_accuracy': [],
            'confidence_calibration': [],
            'feature_importance': {},
            'drift_detection': []
        }
    
    def log_prediction(self, features, prediction, actual_outcome):
        # Track accuracy
        self.metrics['prediction_accuracy'].append({
            'timestamp': datetime.now(),
            'predicted': prediction['class'],
            'confidence': prediction['probability'],
            'actual': actual_outcome,
            'correct': prediction['class'] == actual_outcome
        })
        
        # Check confidence calibration
        self.check_calibration()
        
        # Monitor feature drift
        self.check_feature_drift(features)
    
    def generate_ml_health_report(self):
        recent_predictions = self.metrics['prediction_accuracy'][-50:]
        
        report = {
            'accuracy': sum(p['correct'] for p in recent_predictions) / len(recent_predictions),
            'avg_confidence': np.mean([p['confidence'] for p in recent_predictions]),
            'calibration_error': self.calculate_calibration_error(),
            'top_features': self.get_top_features(),
            'drift_alert': self.check_for_drift()
        }
        
        return report
```

## Quick Troubleshooting

### Common Issues & Solutions

1. **Low Confidence Scores**
   ```python
   # Add more data sources
   analyzer.add_news_source('Reuters', 'https://reuters.com/rss/...')
   analyzer.enable_twitter_sentiment = True
   ```

2. **Poor Model Performance**
   ```python
   # Check feature quality
   feature_correlations = X.corr()
   low_correlation_features = feature_correlations[abs(feature_correlations[y]) < 0.05]
   print(f"Remove these features: {low_correlation_features.index.tolist()}")
   ```

3. **Slow Data Collection**
   ```python
   # Parallel processing
   from concurrent.futures import ThreadPoolExecutor
   
   with ThreadPoolExecutor(max_workers=4) as executor:
       results = executor.map(analyzer.analyze_bank_sentiment, symbols)
   ```

Remember: **Quality > Quantity** for your first 100 samples. Focus on high-confidence signals during major market hours (10 AM - 12 PM, 2 PM - 3:30 PM) for best results.