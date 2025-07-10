Machine Learning Enhancement - Deep Dive Implementation
Overview of Current ML State
Your project already has ML components in:

ml_trading_config.py - Feature engineering and model optimization
news_sentiment.py - Basic ML model initialization
news_impact_analyzer.py - Some ML feature caching logic

However, the ML models aren't being properly trained or updated with real trading outcomes. Let's fix this.
1. Create a New ML Training Pipeline
First, create a new file src/ml_training_pipeline.py:
python#!/usr/bin/env python3
"""
Machine Learning Training Pipeline for Trading Sentiment Analysis
Handles data collection, labeling, training, and model updating
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import json
import os
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
import sqlite3

logger = logging.getLogger(__name__)

class MLTrainingPipeline:
    """Manages the complete ML training lifecycle"""
    
    def __init__(self, data_dir: str = "data/ml_models"):
        self.data_dir = data_dir
        self.models_dir = os.path.join(data_dir, "models")
        self.training_data_dir = os.path.join(data_dir, "training_data")
        self.ensure_directories()
        
        # Initialize database for training data
        self.db_path = os.path.join(self.data_dir, "training_data.db")
        self.init_database()
        
        # Model versioning
        self.model_version = self.get_latest_model_version()
        
    def ensure_directories(self):
        """Create necessary directories"""
        for dir_path in [self.data_dir, self.models_dir, self.training_data_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def init_database(self):
        """Initialize SQLite database for training data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for storing features and outcomes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                sentiment_score REAL,
                confidence REAL,
                news_count INTEGER,
                reddit_sentiment REAL,
                event_score REAL,
                technical_score REAL,
                ml_features TEXT,  -- JSON string of additional features
                feature_version TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_id INTEGER,
                symbol TEXT NOT NULL,
                signal_timestamp DATETIME NOT NULL,
                signal_type TEXT,  -- BUY, SELL, HOLD
                entry_price REAL,
                exit_price REAL,
                exit_timestamp DATETIME,
                return_pct REAL,
                max_drawdown REAL,
                outcome_label INTEGER,  -- 1 for profitable, 0 for loss
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (feature_id) REFERENCES sentiment_features (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT,
                model_type TEXT,
                training_date DATETIME,
                validation_score REAL,
                test_score REAL,
                precision_score REAL,
                recall_score REAL,
                parameters TEXT,  -- JSON string
                feature_importance TEXT,  -- JSON string
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def collect_training_data(self, sentiment_data: Dict, symbol: str) -> int:
        """
        Store sentiment analysis results for future training
        
        Args:
            sentiment_data: Output from analyze_bank_sentiment
            symbol: Stock symbol
            
        Returns:
            feature_id for linking with outcomes
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract ML features if available
        ml_features = {}
        if 'ml_trading_details' in sentiment_data.get('sentiment_components', {}):
            ml_features = sentiment_data['sentiment_components']['ml_trading_details']
        
        cursor.execute('''
            INSERT INTO sentiment_features 
            (symbol, timestamp, sentiment_score, confidence, news_count, 
             reddit_sentiment, event_score, technical_score, ml_features, feature_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            sentiment_data['timestamp'],
            sentiment_data['overall_sentiment'],
            sentiment_data['confidence'],
            sentiment_data['news_count'],
            sentiment_data.get('reddit_sentiment', {}).get('average_sentiment', 0),
            sentiment_data.get('sentiment_components', {}).get('events', 0),
            0,  # Technical score - to be added
            json.dumps(ml_features),
            "1.0"
        ))
        
        feature_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return feature_id
    
    def record_trading_outcome(self, feature_id: int, outcome_data: Dict):
        """
        Record the actual outcome of a trading signal
        
        Args:
            feature_id: ID from collect_training_data
            outcome_data: Dict containing trade results
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate return percentage
        return_pct = ((outcome_data['exit_price'] - outcome_data['entry_price']) / 
                      outcome_data['entry_price']) * 100
        
        # Label: 1 if profitable (including fees), 0 if loss
        # Assuming 0.1% fee per trade (0.2% round trip)
        net_return = return_pct - 0.2
        outcome_label = 1 if net_return > 0 else 0
        
        cursor.execute('''
            INSERT INTO trading_outcomes
            (feature_id, symbol, signal_timestamp, signal_type, entry_price,
             exit_price, exit_timestamp, return_pct, max_drawdown, outcome_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feature_id,
            outcome_data['symbol'],
            outcome_data['signal_timestamp'],
            outcome_data['signal_type'],
            outcome_data['entry_price'],
            outcome_data['exit_price'],
            outcome_data['exit_timestamp'],
            return_pct,
            outcome_data.get('max_drawdown', 0),
            outcome_label
        ))
        
        conn.commit()
        conn.close()
    
    def prepare_training_dataset(self, min_samples: int = 100) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare dataset for training from collected data
        
        Returns:
            X: Feature matrix
            y: Labels (profitable or not)
        """
        conn = sqlite3.connect(self.db_path)
        
        # Join features with outcomes
        query = '''
            SELECT 
                sf.*,
                to.outcome_label,
                to.return_pct,
                to.signal_type
            FROM sentiment_features sf
            INNER JOIN trading_outcomes to ON sf.id = to.feature_id
            WHERE to.outcome_label IS NOT NULL
            ORDER BY sf.timestamp
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) < min_samples:
            logger.warning(f"Insufficient training data: {len(df)} samples (minimum: {min_samples})")
            return None, None
        
        # Prepare features
        feature_columns = [
            'sentiment_score', 'confidence', 'news_count', 
            'reddit_sentiment', 'event_score'
        ]
        
        X = df[feature_columns].copy()
        
        # Add engineered features
        X['sentiment_confidence_interaction'] = X['sentiment_score'] * X['confidence']
        X['news_volume_category'] = pd.cut(X['news_count'], bins=[0, 5, 10, 20, 100], labels=[0, 1, 2, 3])
        X['news_volume_category'] = X['news_volume_category'].astype(int)
        
        # Add time-based features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        X['hour'] = df['timestamp'].dt.hour
        X['day_of_week'] = df['timestamp'].dt.dayofweek
        X['is_market_hours'] = ((X['hour'] >= 10) & (X['hour'] <= 16)).astype(int)
        
        # Parse ML features from JSON
        if 'ml_features' in df.columns:
            ml_features_expanded = pd.json_normalize(df['ml_features'].apply(json.loads))
            X = pd.concat([X, ml_features_expanded], axis=1)
        
        # Labels
        y = df['outcome_label']
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, any]:
        """
        Train multiple models and select best performer
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        import xgboost as xgb
        
        # Time series split for financial data
        tscv = TimeSeriesSplit(n_splits=5)
        
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                class_weight='balanced',
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                scale_pos_weight=len(y[y==0])/len(y[y==1]),  # Handle imbalance
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        }
        
        # Scale features for neural network and logistic regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        best_model = None
        best_score = -1
        model_scores = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Use scaled data for certain models
            if name in ['neural_network', 'logistic_regression']:
                X_train = X_scaled
            else:
                X_train = X
            
            # Cross-validation scores
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_cv_train, y_cv_train)
                
                # Use probability for better threshold tuning
                y_pred_proba = model.predict_proba(X_cv_val)[:, 1]
                
                # Find optimal threshold for trading (minimize false positives)
                thresholds = np.arange(0.3, 0.8, 0.05)
                best_threshold = 0.5
                best_f1 = 0
                
                for threshold in thresholds:
                    y_pred = (y_pred_proba > threshold).astype(int)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_cv_val, y_pred, average='binary'
                    )
                    
                    # Prioritize precision for trading (avoid false signals)
                    weighted_score = 0.7 * precision + 0.3 * f1
                    
                    if weighted_score > best_f1:
                        best_f1 = weighted_score
                        best_threshold = threshold
                
                cv_scores.append(best_f1)
            
            avg_score = np.mean(cv_scores)
            model_scores[name] = {
                'avg_cv_score': avg_score,
                'best_threshold': best_threshold,
                'model': model
            }
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = name
        
        logger.info(f"Best model: {best_model} with score: {best_score:.4f}")
        
        # Train best model on full dataset
        final_model = models[best_model]
        if best_model in ['neural_network', 'logistic_regression']:
            final_model.fit(X_scaled, y)
            # Save scaler
            joblib.dump(scaler, os.path.join(self.models_dir, 'feature_scaler.pkl'))
        else:
            final_model.fit(X, y)
        
        # Save model and metadata
        self.save_model(final_model, best_model, model_scores[best_model], X.columns.tolist())
        
        return {
            'best_model': best_model,
            'model_scores': model_scores,
            'feature_columns': X.columns.tolist()
        }
    
    def save_model(self, model, model_type: str, performance: Dict, feature_columns: List[str]):
        """Save trained model with metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version = f"v_{timestamp}"
        
        # Save model
        model_path = os.path.join(self.models_dir, f"{model_type}_{version}.pkl")
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'version': version,
            'model_type': model_type,
            'training_date': timestamp,
            'performance': performance,
            'feature_columns': feature_columns,
            'model_path': model_path
        }
        
        metadata_path = os.path.join(self.models_dir, f"metadata_{version}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update current model symlink
        current_model_path = os.path.join(self.models_dir, 'current_model.pkl')
        current_metadata_path = os.path.join(self.models_dir, 'current_metadata.json')
        
        if os.path.exists(current_model_path):
            os.remove(current_model_path)
        if os.path.exists(current_metadata_path):
            os.remove(current_metadata_path)
        
        os.symlink(model_path, current_model_path)
        os.symlink(metadata_path, current_metadata_path)
        
        logger.info(f"Model saved: {model_path}")
    
    def get_latest_model_version(self) -> Optional[str]:
        """Get the latest model version"""
        try:
            metadata_path = os.path.join(self.models_dir, 'current_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return metadata['version']
        except Exception as e:
            logger.error(f"Error loading model version: {e}")
        
        return None
    
    def update_model_online(self, new_samples: List[Tuple[Dict, int]]):
        """
        Online learning - update model with new samples
        
        Args:
            new_samples: List of (features_dict, outcome_label) tuples
        """
        if len(new_samples) < 10:
            logger.info("Insufficient samples for online update")
            return
        
        try:
            # Load current model
            model_path = os.path.join(self.models_dir, 'current_model.pkl')
            if not os.path.exists(model_path):
                logger.warning("No current model found for online update")
                return
            
            model = joblib.load(model_path)
            
            # Check if model supports partial_fit
            if hasattr(model, 'partial_fit'):
                # Prepare new data
                X_new = pd.DataFrame([s[0] for s in new_samples])
                y_new = pd.Series([s[1] for s in new_samples])
                
                # Load scaler if needed
                scaler_path = os.path.join(self.models_dir, 'feature_scaler.pkl')
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    X_new = scaler.transform(X_new)
                
                # Update model
                model.partial_fit(X_new, y_new)
                
                # Save updated model
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                updated_path = os.path.join(self.models_dir, f"updated_{timestamp}.pkl")
                joblib.dump(model, updated_path)
                
                logger.info(f"Model updated online with {len(new_samples)} samples")
            else:
                logger.info("Current model doesn't support online learning")
                
        except Exception as e:
            logger.error(f"Error in online model update: {e}")
2. Integrate ML Pipeline with News Sentiment Analyzer
Update news_sentiment.py to use the ML pipeline:
python# Add to imports in news_sentiment.py
from src.ml_training_pipeline import MLTrainingPipeline

# Update __init__ method
def __init__(self):
    # ... existing code ...
    
    # Initialize ML training pipeline
    self.ml_pipeline = MLTrainingPipeline()
    
    # Load latest trained model
    self.ml_model = self._load_ml_model()
    self.ml_feature_columns = []

def _load_ml_model(self):
    """Load the latest trained ML model"""
    try:
        model_path = os.path.join("data/ml_models/models", "current_model.pkl")
        metadata_path = os.path.join("data/ml_models/models", "current_metadata.json")
        
        if os.path.exists(model_path) and os.path.exists(metadata_path):
            model = joblib.load(model_path)
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.ml_feature_columns = metadata['feature_columns']
            self.ml_threshold = metadata['performance'].get('best_threshold', 0.5)
            
            logger.info(f"Loaded ML model version: {metadata['version']}")
            return model
        else:
            logger.warning("No trained ML model found")
            return None
            
    except Exception as e:
        logger.error(f"Error loading ML model: {e}")
        return None

# Update analyze_bank_sentiment method
def analyze_bank_sentiment(self, symbol: str) -> Dict:
    # ... existing sentiment analysis code ...
    
    # Store data for ML training
    if result and self.ml_pipeline:
        feature_id = self.ml_pipeline.collect_training_data(result, symbol)
        result['ml_feature_id'] = feature_id
    
    # Add ML prediction if model is available
    if self.ml_model:
        ml_prediction = self._get_ml_prediction(result)
        result['ml_prediction'] = ml_prediction
    
    return result

def _get_ml_prediction(self, sentiment_data: Dict) -> Dict:
    """Get ML model prediction for the sentiment data"""
    try:
        # Prepare features
        features = {
            'sentiment_score': sentiment_data['overall_sentiment'],
            'confidence': sentiment_data['confidence'],
            'news_count': sentiment_data['news_count'],
            'reddit_sentiment': sentiment_data.get('reddit_sentiment', {}).get('average_sentiment', 0),
            'event_score': sentiment_data.get('sentiment_components', {}).get('events', 0),
            'sentiment_confidence_interaction': sentiment_data['overall_sentiment'] * sentiment_data['confidence']
        }
        
        # Add time features
        timestamp = datetime.fromisoformat(sentiment_data['timestamp'])
        features['hour'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()
        features['is_market_hours'] = 1 if 10 <= timestamp.hour <= 16 else 0
        
        # Convert to DataFrame with correct column order
        X = pd.DataFrame([features])[self.ml_feature_columns]
        
        # Get prediction probability
        prob = self.ml_model.predict_proba(X)[0, 1]
        
        # Apply threshold
        prediction = 'PROFITABLE' if prob > self.ml_threshold else 'UNPROFITABLE'
        
        return {
            'prediction': prediction,
            'probability': float(prob),
            'confidence': abs(prob - 0.5) * 2,  # Convert to 0-1 confidence scale
            'threshold': self.ml_threshold
        }
        
    except Exception as e:
        logger.error(f"Error getting ML prediction: {e}")
        return {
            'prediction': 'UNKNOWN',
            'probability': 0.5,
            'confidence': 0,
            'error': str(e)
        }
3. Create a Trading Outcome Tracker
Create src/trading_outcome_tracker.py:
python#!/usr/bin/env python3
"""
Trading Outcome Tracker
Records actual trading results for ML training
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TradingOutcomeTracker:
    """Tracks trading signals and their outcomes"""
    
    def __init__(self, ml_pipeline):
        self.ml_pipeline = ml_pipeline
        self.active_trades = {}
        self.load_active_trades()
    
    def record_signal(self, symbol: str, signal_data: Dict):
        """Record a trading signal for tracking"""
        trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.active_trades[trade_id] = {
            'symbol': symbol,
            'feature_id': signal_data.get('ml_feature_id'),
            'signal_type': signal_data['trading_recommendation']['action'],
            'signal_timestamp': signal_data['timestamp'],
            'entry_price': None,  # To be filled when trade executes
            'sentiment_score': signal_data['overall_sentiment'],
            'confidence': signal_data['confidence'],
            'ml_prediction': signal_data.get('ml_prediction', {})
        }
        
        self.save_active_trades()
        return trade_id
    
    def update_trade_execution(self, trade_id: str, execution_data: Dict):
        """Update trade with execution details"""
        if trade_id in self.active_trades:
            self.active_trades[trade_id].update({
                'entry_price': execution_data['price'],
                'entry_timestamp': execution_data['timestamp'],
                'position_size': execution_data.get('size', 0)
            })
            self.save_active_trades()
    
    def close_trade(self, trade_id: str, exit_data: Dict):
        """Close a trade and record outcome"""
        if trade_id not in self.active_trades:
            logger.warning(f"Trade {trade_id} not found")
            return
        
        trade = self.active_trades[trade_id]
        
        # Calculate outcome
        outcome_data = {
            'symbol': trade['symbol'],
            'signal_timestamp': trade['signal_timestamp'],
            'signal_type': trade['signal_type'],
            'entry_price': trade['entry_price'],
            'exit_price': exit_data['price'],
            'exit_timestamp': exit_data['timestamp'],
            'max_drawdown': exit_data.get('max_drawdown', 0)
        }
        
        # Record to ML pipeline
        if trade['feature_id']:
            self.ml_pipeline.record_trading_outcome(
                trade['feature_id'], 
                outcome_data
            )
        
        # Remove from active trades
        del self.active_trades[trade_id]
        self.save_active_trades()
        
        logger.info(f"Trade {trade_id} closed and recorded")
    
    def check_stale_trades(self, days: int = 30):
        """Check for trades that should be closed"""
        cutoff_date = datetime.now() - timedelta(days=days)
        stale_trades = []
        
        for trade_id, trade in self.active_trades.items():
            trade_date = datetime.fromisoformat(trade['signal_timestamp'])
            if trade_date < cutoff_date:
                stale_trades.append(trade_id)
        
        return stale_trades
    
    def save_active_trades(self):
        """Save active trades to file"""
        with open('data/active_trades.json', 'w') as f:
            json.dump(self.active_trades, f, indent=2)
    
    def load_active_trades(self):
        """Load active trades from file"""
        try:
            with open('data/active_trades.json', 'r') as f:
                self.active_trades = json.load(f)
        except FileNotFoundError:
            self.active_trades = {}
4. Create ML Model Retraining Script
Create scripts/retrain_ml_models.py:
python#!/usr/bin/env python3
"""
Script to retrain ML models with collected data
Run this periodically (e.g., weekly via cron)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml_training_pipeline import MLTrainingPipeline
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Retrain ML models')
    parser.add_argument('--min-samples', type=int, default=500,
                      help='Minimum samples required for training')
    parser.add_argument('--evaluate-only', action='store_true',
                      help='Only evaluate current model, don\'t retrain')
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MLTrainingPipeline()
    
    # Prepare dataset
    logger.info("Preparing training dataset...")
    X, y = pipeline.prepare_training_dataset(min_samples=args.min_samples)
    
    if X is None:
        logger.error("Insufficient data for training")
        return
    
    logger.info(f"Dataset prepared: {len(X)} samples")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    
    if args.evaluate_only:
        # Just evaluate current model
        logger.info("Evaluation mode - skipping training")
        # Add evaluation code here
        return
    
    # Train models
    logger.info("Training models...")
    results = pipeline.train_models(X, y)
    
    logger.info("Training completed!")
    logger.info(f"Best model: {results['best_model']}")
    logger.info("Model scores:")
    for model, scores in results['model_scores'].items():
        logger.info(f"  {model}: {scores['avg_cv_score']:.4f}")

if __name__ == "__main__":
    main()
5. Update News Trading Analyzer
Update news_trading_analyzer.py to integrate the outcome tracker:
python# Add to imports
from src.trading_outcome_tracker import TradingOutcomeTracker

# Update __init__
def __init__(self):
    # ... existing code ...
    
    # Initialize outcome tracker
    if hasattr(self.sentiment_analyzer, 'ml_pipeline'):
        self.outcome_tracker = TradingOutcomeTracker(
            self.sentiment_analyzer.ml_pipeline
        )
    else:
        self.outcome_tracker = None

# Add method to analyze with tracking
def analyze_and_track(self, symbol: str) -> Dict:
    """Analyze sentiment and track for ML training"""
    result = self.analyze_single_bank(symbol, detailed=True)
    
    # Record signal if it's actionable
    if self.outcome_tracker and result.get('signal') not in ['HOLD', None]:
        trade_id = self.outcome_tracker.record_signal(symbol, result)
        result['trade_id'] = trade_id
        logger.info(f"Recorded trade signal: {trade_id}")
    
    return result

# Add method to close trades (to be called by your execution system)
def close_trade(self, trade_id: str, exit_price: float):
    """Close a trade and record outcome"""
    if self.outcome_tracker:
        exit_data = {
            'price': exit_price,
            'timestamp': datetime.now().isoformat()
        }
        self.outcome_tracker.close_trade(trade_id, exit_data)
6. Create Backtesting Integration
Create src/ml_backtester.py:
python#!/usr/bin/env python3
"""
ML Model Backtester
Tests ML predictions against historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class MLBacktester:
    """Backtest ML model predictions"""
    
    def __init__(self, ml_pipeline, data_feed):
        self.ml_pipeline = ml_pipeline
        self.data_feed = data_feed
    
    def backtest_predictions(self, symbol: str, start_date: str, end_date: str) -> Dict:
        """
        Backtest ML predictions against actual price movements
        """
        # Get historical sentiment data
        conn = sqlite3.connect(self.ml_pipeline.db_path)
        query = '''
            SELECT * FROM sentiment_features 
            WHERE symbol = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        '''
        
        sentiment_df = pd.read_sql_query(query, conn, params=[symbol, start_date, end_date])
        conn.close()
        
        if sentiment_df.empty:
            return {'error': 'No historical data found'}
        
        # Get price data
        price_data = self.data_feed.get_historical_data(
            symbol, 
            period='1y'  # Adjust as needed
        )
        
        # Simulate trades based on ML predictionsRetryTContinueEditpython       # Simulate trades based on ML predictions
       trades = []
       capital = 10000  # Starting capital
       position = 0
       
       for idx, row in sentiment_df.iterrows():
           # Prepare features for ML prediction
           features = {
               'sentiment_score': row['sentiment_score'],
               'confidence': row['confidence'],
               'news_count': row['news_count'],
               'reddit_sentiment': row['reddit_sentiment'],
               'event_score': row['event_score']
           }
           
           # Get ML prediction
           if self.ml_pipeline.get_latest_model_version():
               prediction = self._get_ml_prediction(features, row['timestamp'])
               
               # Find matching price data
               trade_date = pd.to_datetime(row['timestamp']).date()
               price_row = price_data[price_data.index.date == trade_date]
               
               if not price_row.empty:
                   current_price = price_row['Close'].iloc[0]
                   
                   # Execute trades based on prediction
                   if prediction['prediction'] == 'PROFITABLE' and position == 0:
                       # Buy signal
                       position = capital / current_price
                       trades.append({
                           'date': trade_date,
                           'type': 'BUY',
                           'price': current_price,
                           'quantity': position,
                           'ml_confidence': prediction['confidence']
                       })
                   
                   elif prediction['prediction'] == 'UNPROFITABLE' and position > 0:
                       # Sell signal
                       capital = position * current_price
                       trades.append({
                           'date': trade_date,
                           'type': 'SELL',
                           'price': current_price,
                           'quantity': position,
                           'ml_confidence': prediction['confidence'],
                           'return': (current_price - trades[-1]['price']) / trades[-1]['price']
                       })
                       position = 0
       
       # Calculate performance metrics
       metrics = self._calculate_backtest_metrics(trades, capital)
       
       return {
           'trades': trades,
           'metrics': metrics,
           'final_capital': capital,
           'total_return': (capital - 10000) / 10000
       }
   
   def _calculate_backtest_metrics(self, trades: List[Dict], final_capital: float) -> Dict:
       """Calculate backtesting performance metrics"""
       if not trades:
           return {
               'total_trades': 0,
               'win_rate': 0,
               'sharpe_ratio': 0,
               'max_drawdown': 0
           }
       
       # Extract returns
       returns = [t.get('return', 0) for t in trades if 'return' in t]
       
       # Win rate
       winning_trades = sum(1 for r in returns if r > 0)
       win_rate = winning_trades / len(returns) if returns else 0
       
       # Sharpe ratio (simplified)
       if returns:
           avg_return = np.mean(returns)
           std_return = np.std(returns)
           sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
       else:
           sharpe_ratio = 0
       
       # Max drawdown
       cumulative_returns = np.cumprod(1 + np.array(returns))
       running_max = np.maximum.accumulate(cumulative_returns)
       drawdown = (cumulative_returns - running_max) / running_max
       max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
       
       return {
           'total_trades': len(trades),
           'win_rate': win_rate,
           'sharpe_ratio': sharpe_ratio,
           'max_drawdown': max_drawdown,
           'avg_return_per_trade': np.mean(returns) if returns else 0,
           'best_trade': max(returns) if returns else 0,
           'worst_trade': min(returns) if returns else 0
       }
7. Update Dashboard for ML Insights
Update news_analysis_dashboard.py to display ML predictions:
python# Add to imports
import sqlite3
import joblib

# Add ML section to display_bank_analysis method
def display_bank_analysis(self, symbol: str, data: List[Dict]):
    # ... existing code ...
    
    # Add ML Prediction Section
    st.markdown("#### 🤖 Machine Learning Prediction")
    
    latest = self.get_latest_analysis(data)
    ml_prediction = latest.get('ml_prediction', {})
    
    if ml_prediction and ml_prediction.get('prediction') != 'UNKNOWN':
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            prediction = ml_prediction['prediction']
            if prediction == 'PROFITABLE':
                st.metric("ML Signal", "📈 PROFITABLE", delta="BUY")
            else:
                st.metric("ML Signal", "📉 UNPROFITABLE", delta="AVOID")
        
        with col2:
            probability = ml_prediction.get('probability', 0)
            st.metric("Probability", f"{probability:.1%}")
        
        with col3:
            ml_confidence = ml_prediction.get('confidence', 0)
            st.metric("ML Confidence", f"{ml_confidence:.1%}")
        
        with col4:
            threshold = ml_prediction.get('threshold', 0.5)
            st.metric("Decision Threshold", f"{threshold:.2f}")
        
        # Show ML model performance if available
        if st.button("📊 Show ML Model Performance"):
            self.display_ml_performance(symbol)
    else:
        st.info("ML predictions not available. Train models using: `python scripts/retrain_ml_models.py`")

def display_ml_performance(self, symbol: str):
    """Display ML model performance metrics"""
    try:
        # Load ML pipeline to access database
        from src.ml_training_pipeline import MLTrainingPipeline
        ml_pipeline = MLTrainingPipeline()
        
        conn = sqlite3.connect(ml_pipeline.db_path)
        
        # Get model performance
        query = '''
            SELECT * FROM model_performance 
            ORDER BY training_date DESC 
            LIMIT 5
        '''
        
        performance_df = pd.read_sql_query(query, conn)
        
        if not performance_df.empty:
            st.markdown("##### Recent Model Performance")
            
            # Display metrics
            for idx, row in performance_df.iterrows():
                with st.expander(f"Model: {row['model_type']} - {row['model_version']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Validation Score", f"{row['validation_score']:.3f}")
                    with col2:
                        st.metric("Precision", f"{row['precision_score']:.3f}")
                    with col3:
                        st.metric("Recall", f"{row['recall_score']:.3f}")
                    
                    # Show feature importance if available
                    if row['feature_importance']:
                        importance = json.loads(row['feature_importance'])
                        st.markdown("**Top Features:**")
                        for feature, score in sorted(importance.items(), 
                                                   key=lambda x: x[1], 
                                                   reverse=True)[:5]:
                            st.write(f"- {feature}: {score:.3f}")
        
        # Get recent predictions accuracy
        query = '''
            SELECT 
                COUNT(*) as total_predictions,
                SUM(CASE WHEN outcome_label = 1 THEN 1 ELSE 0 END) as profitable_trades,
                AVG(return_pct) as avg_return
            FROM trading_outcomes
            WHERE symbol = ?
            AND exit_timestamp > datetime('now', '-30 days')
        '''
        
        accuracy_df = pd.read_sql_query(query, conn, params=[symbol])
        
        if not accuracy_df.empty and accuracy_df['total_predictions'].iloc[0] > 0:
            st.markdown("##### Recent Trading Performance (30 days)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total = accuracy_df['total_predictions'].iloc[0]
                st.metric("Total Trades", total)
            
            with col2:
                profitable = accuracy_df['profitable_trades'].iloc[0]
                win_rate = profitable / total if total > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1%}")
            
            with col3:
                avg_return = accuracy_df['avg_return'].iloc[0]
                st.metric("Avg Return", f"{avg_return:.2f}%")
        
        conn.close()
        
    except Exception as e:
        st.error(f"Error loading ML performance: {e}")
8. Create Automated Training Schedule
Create scripts/schedule_ml_training.py:
python#!/usr/bin/env python3
"""
Automated ML training scheduler
Set up as cron job for regular model updates
"""

import schedule
import time
import subprocess
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training():
    """Run ML model training"""
    try:
        logger.info(f"Starting ML training at {datetime.now()}")
        
        # Run training script
        result = subprocess.run(
            ['python', 'scripts/retrain_ml_models.py', '--min-samples', '500'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("Training completed successfully")
            logger.info(result.stdout)
        else:
            logger.error(f"Training failed: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Error running training: {e}")

def check_data_quality():
    """Check if we have enough quality data for training"""
    try:
        from src.ml_training_pipeline import MLTrainingPipeline
        pipeline = MLTrainingPipeline()
        
        X, y = pipeline.prepare_training_dataset(min_samples=100)
        
        if X is not None:
            logger.info(f"Data check passed: {len(X)} samples available")
            
            # Check class balance
            class_balance = y.value_counts().to_dict()
            logger.info(f"Class balance: {class_balance}")
            
            # Warn if imbalanced
            if min(class_balance.values()) / max(class_balance.values()) < 0.3:
                logger.warning("Warning: Class imbalance detected")
        else:
            logger.warning("Insufficient data for training")
            
    except Exception as e:
        logger.error(f"Error checking data quality: {e}")

# Schedule tasks
schedule.every().sunday.at("02:00").do(run_training)
schedule.every().day.at("06:00").do(check_data_quality)

if __name__ == "__main__":
    logger.info("ML training scheduler started")
    
    # Run initial check
    check_data_quality()
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute
9. Add ML Configuration
Create config/ml_config.yaml:
yaml# ML Model Configuration
model_settings:
  # Minimum samples required for training
  min_training_samples: 500
  
  # Model types to train
  models:
    - random_forest
    - xgboost
    - gradient_boosting
    - neural_network
    - logistic_regression
  
  # Feature engineering settings
  features:
    # Base features from sentiment analysis
    base_features:
      - sentiment_score
      - confidence
      - news_count
      - reddit_sentiment
      - event_score
    
    # Engineered features
    engineered_features:
      - sentiment_confidence_interaction
      - news_volume_category
      - hour
      - day_of_week
      - is_market_hours
    
    # Technical features (if available)
    technical_features:
      - rsi
      - macd_signal
      - momentum_score
      - volume_ratio
  
  # Training settings
  training:
    test_size: 0.2
    cv_splits: 5  # Time series splits
    random_state: 42
    
    # Class weight strategies
    class_weight: balanced
    
    # Optimization metric
    optimization_metric: precision  # Focus on avoiding false positives
    
  # Prediction thresholds
  thresholds:
    default: 0.5
    conservative: 0.65
    aggressive: 0.35
    
  # Online learning settings
  online_learning:
    enabled: true
    min_batch_size: 10
    update_frequency: daily
    
# Model performance thresholds
performance_thresholds:
  min_precision: 0.6
  min_recall: 0.4
  min_f1_score: 0.5
  min_auc_roc: 0.65
  
# Data quality requirements
data_quality:
  min_samples_per_class: 50
  max_class_imbalance_ratio: 3.0
  feature_missing_threshold: 0.1  # Max 10% missing values
10. Testing the ML Pipeline
Create tests/test_ml_pipeline.py:
python#!/usr/bin/env python3
"""
Tests for ML training pipeline
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

from src.ml_training_pipeline import MLTrainingPipeline

class TestMLPipeline(unittest.TestCase):
    
    def setUp(self):
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.pipeline = MLTrainingPipeline(data_dir=self.test_dir)
    
    def tearDown(self):
        # Clean up
        shutil.rmtree(self.test_dir)
    
    def test_collect_training_data(self):
        """Test collecting sentiment data for training"""
        sentiment_data = {
            'symbol': 'CBA.AX',
            'timestamp': datetime.now().isoformat(),
            'overall_sentiment': 0.5,
            'confidence': 0.8,
            'news_count': 10,
            'reddit_sentiment': {'average_sentiment': 0.3},
            'sentiment_components': {'events': 0.2}
        }
        
        feature_id = self.pipeline.collect_training_data(sentiment_data, 'CBA.AX')
        self.assertIsNotNone(feature_id)
        self.assertGreater(feature_id, 0)
    
    def test_record_trading_outcome(self):
        """Test recording trading outcomes"""
        # First create a feature entry
        sentiment_data = {
            'symbol': 'CBA.AX',
            'timestamp': datetime.now().isoformat(),
            'overall_sentiment': 0.5,
            'confidence': 0.8,
            'news_count': 10
        }
        
        feature_id = self.pipeline.collect_training_data(sentiment_data, 'CBA.AX')
        
        # Record outcome
        outcome_data = {
            'symbol': 'CBA.AX',
            'signal_timestamp': datetime.now().isoformat(),
            'signal_type': 'BUY',
            'entry_price': 100.0,
            'exit_price': 105.0,
            'exit_timestamp': (datetime.now() + timedelta(days=5)).isoformat()
        }
        
        self.pipeline.record_trading_outcome(feature_id, outcome_data)
        
        # Verify data was stored
        X, y = self.pipeline.prepare_training_dataset(min_samples=1)
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertEqual(len(X), 1)
        self.assertEqual(y.iloc[0], 1)  # Should be profitable
    
    def test_model_training(self):
        """Test model training with synthetic data"""
        # Create synthetic training data
        n_samples = 200
        
        # Generate features
        np.random.seed(42)
        X = pd.DataFrame({
            'sentiment_score': np.random.uniform(-1, 1, n_samples),
            'confidence': np.random.uniform(0.3, 1, n_samples),
            'news_count': np.random.randint(1, 20, n_samples),
            'reddit_sentiment': np.random.uniform(-1, 1, n_samples),
            'event_score': np.random.uniform(-0.5, 0.5, n_samples),
            'sentiment_confidence_interaction': np.random.uniform(-1, 1, n_samples),
            'news_volume_category': np.random.randint(0, 4, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'is_market_hours': np.random.randint(0, 2, n_samples)
        })
        
        # Generate labels (with some correlation to features)
        y = pd.Series((X['sentiment_score'] + X['confidence'] + 
                      np.random.normal(0, 0.5, n_samples)) > 0.5).astype(int)
        
        # Train models
        results = self.pipeline.train_models(X, y)
        
        self.assertIn('best_model', results)
        self.assertIn('model_scores', results)
        self.assertGreater(len(results['model_scores']), 0)

if __name__ == '__main__':
    unittest.main()
Implementation Steps

Start with the ML Pipeline:
bash# Create the ML pipeline file
touch src/ml_training_pipeline.py
# Add the code above

Update News Sentiment Analyzer:

Add ML pipeline initialization
Add feature collection after each analysis
Add ML prediction to results


Create the Outcome Tracker:
bashtouch src/trading_outcome_tracker.py

Set up the Database:
bash# The pipeline will auto-create the SQLite database
python -c "from src.ml_training_pipeline import MLTrainingPipeline; MLTrainingPipeline()"

Collect Initial Data:

Run the system normally for a few weeks to collect data
Or backfill with historical data if available


Train Initial Models:
bashpython scripts/retrain_ml_models.py --min-samples 100

Set up Automated Training (cron job):
bash# Add to crontab
0 2 * * 0 /usr/bin/python /path/to/scripts/retrain_ml_models.py


Key Benefits

Self-Improving System: Models get better with more data
Objective Performance Tracking: Know exactly how well predictions work
Adaptive Thresholds: Automatically finds optimal decision boundaries
Feature Importance: Understand what really drives profitable trades
Multiple Model Ensemble: Reduces overfitting risk
Online Learning: Can adapt to market regime changes

Monitoring and Maintenance

Monitor Data Quality:
python# Add to your monitoring script
def check_ml_health():
    pipeline = MLTrainingPipeline()
    X, y = pipeline.prepare_training_dataset()
    
    if X is not None:
        print(f"Training samples: {len(X)}")
        print(f"Class balance: {y.value_counts().to_dict()}")
        print(f"Features: {X.columns.tolist()}")

Track Model Performance:

Set up alerts if model performance degrades
Compare new model performance before deployment
A/B test new models against current ones


Regular Maintenance:

Clean old training data (>1 year)
Retrain models monthly or when performance drops
Update feature engineering based on insights



This ML enhancement will transform your system from rule-based to data-driven, continuously learning and improving from actual trading outcomes.