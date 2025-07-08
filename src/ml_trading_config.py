# ml_trading_optimizer.py
"""
ML Model Optimization for Trading Sentiment Analysis
Fine-tunes models specifically for financial text and trading signals
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, precision_recall_curve
from typing import Dict, List, Tuple, Optional  # Add Optional to imports
import optuna
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class TradingModelOptimizer:
    """Optimize ML models for trading-specific sentiment analysis"""
    
    def __init__(self, ml_analyzer):
        self.ml_analyzer = ml_analyzer
        self.best_params = {}
        self.optimization_history = []
        
    def trading_score(self, y_true, y_pred_proba):
        """
        Custom scoring function that considers trading profitability
        """
        # Convert probabilities to trading decisions
        # Assuming y_pred_proba is for the positive class
        threshold = 0.6  # Conservative threshold
        predictions = (y_pred_proba > threshold).astype(int)
        
        # Calculate trading metrics
        true_positives = np.sum((predictions == 1) & (y_true == 1))
        false_positives = np.sum((predictions == 1) & (y_true == 0))
        false_negatives = np.sum((predictions == 0) & (y_true == 1))
        
        # Trading-specific scoring
        # Reward true positives (profitable trades)
        # Heavily penalize false positives (losing trades)
        profit = true_positives * 1.0  # Average profit per correct trade
        loss = false_positives * 1.5   # Higher penalty for losses
        missed = false_negatives * 0.3  # Opportunity cost
        
        score = profit - loss - missed
        
        # Normalize by number of trades
        total_trades = predictions.sum() + 1  # Avoid division by zero
        
        return score / total_trades
    
    def optimize_random_forest(self, X_train, y_train, X_val, y_val):
        """Optimize Random Forest for trading"""
        
        param_distributions = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
            'bootstrap': [True, False],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
        
        # Custom scorer
        trading_scorer = make_scorer(self.trading_score, needs_proba=True)
        
        # Random search
        rf_random = RandomizedSearchCV(
            estimator=self.ml_analyzer.models['random_forest'],
            param_distributions=param_distributions,
            n_iter=100,
            cv=5,
            scoring=trading_scorer,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        logger.info("Optimizing Random Forest...")
        rf_random.fit(X_train, y_train)
        
        self.best_params['random_forest'] = rf_random.best_params_
        
        # Evaluate on validation set
        val_score = rf_random.score(X_val, y_val)
        
        logger.info(f"Best Random Forest params: {rf_random.best_params_}")
        logger.info(f"Validation score: {val_score:.3f}")
        
        return rf_random.best_estimator_
    
    def optimize_with_optuna(self, X_train, y_train, X_val, y_val):
        """Use Optuna for advanced hyperparameter optimization"""
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }
            
            # Train model with suggested parameters
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate using trading score
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            score = self.trading_score(y_val, y_pred_proba)
            
            return score
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        
        logger.info(f"Best Optuna params: {study.best_params}")
        logger.info(f"Best score: {study.best_value:.3f}")
        
        self.best_params['gradient_boosting_optuna'] = study.best_params
        
        return study
    
    def find_optimal_threshold(self, model, X_val, y_val):
        """Find optimal probability threshold for trading decisions"""
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate precision-recall for different thresholds
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)
        
        # Calculate F1 scores
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        
        # Find threshold that maximizes F1
        best_threshold_idx = np.argmax(f1_scores[:-1])  # Exclude last element
        best_threshold = thresholds[best_threshold_idx]
        
        # Also calculate trading-specific optimal threshold
        trading_scores = []
        threshold_range = np.arange(0.3, 0.8, 0.05)
        
        for threshold in threshold_range:
            predictions = (y_pred_proba > threshold).astype(int)
            score = self.trading_score(y_val, y_pred_proba)
            trading_scores.append(score)
        
        best_trading_threshold = threshold_range[np.argmax(trading_scores)]
        
        return {
            'f1_optimal': best_threshold,
            'trading_optimal': best_trading_threshold,
            'precision_at_optimal': precisions[best_threshold_idx],
            'recall_at_optimal': recalls[best_threshold_idx]
        }


class FeatureEngineer:
    """Advanced feature engineering for financial text"""
    
    def __init__(self):
        self.feature_stats = {}
        
    def create_trading_features(self, texts: List[str], 
                              market_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Create advanced features for trading sentiment analysis"""
        
        features_list = []
        
        for text in texts:
            features = {}
            
            # Time-based features
            features.update(self._extract_temporal_features(text))
            
            # Financial entity features
            features.update(self._extract_financial_entities(text))
            
            # Market context features (if available)
            if market_data is not None:
                features.update(self._extract_market_context_features(market_data))
            
            # Linguistic complexity features
            features.update(self._extract_complexity_features(text))
            
            # Domain-specific patterns
            features.update(self._extract_trading_patterns(text))
            
            features_list.append(features)
        
        # Convert to numpy array
        feature_names = list(features_list[0].keys())
        feature_matrix = np.array([[f[name] for name in feature_names] 
                                  for f in features_list])
        
        return feature_matrix, feature_names
    
    def _extract_temporal_features(self, text: str) -> Dict[str, float]:
        """Extract time-related features"""
        
        features = {}
        
        # Time references
        time_words = ['today', 'yesterday', 'tomorrow', 'week', 'month', 'quarter', 'year']
        for word in time_words:
            features[f'contains_{word}'] = 1 if word in text.lower() else 0
        
        # Urgency indicators
        urgency_words = ['immediate', 'urgent', 'breaking', 'just', 'now']
        features['urgency_score'] = sum(1 for word in urgency_words if word in text.lower())
        
        return features
    
    def _extract_financial_entities(self, text: str) -> Dict[str, float]:
        """Extract financial entity features"""
        
        import re
        
        features = {}
        
        # Currency mentions
        currency_pattern = r'\$[\d,]+\.?\d*[MBK]?'
        currency_matches = re.findall(currency_pattern, text)
        features['currency_count'] = len(currency_matches)
        
        # Percentage mentions
        percent_pattern = r'\d+\.?\d*\s*%'
        percent_matches = re.findall(percent_pattern, text)
        features['percentage_count'] = len(percent_matches)
        
        # Company mentions (simplified)
        company_pattern = r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:Inc|Corp|Ltd|Limited|Bank|Group)\b'
        company_matches = re.findall(company_pattern, text)
        features['company_mentions'] = len(company_matches)
        
        # Financial metrics
        metrics = ['revenue', 'profit', 'earnings', 'margin', 'growth', 'loss', 'debt']
        for metric in metrics:
            features[f'metric_{metric}'] = 1 if metric in text.lower() else 0
        
        return features
    
    def _extract_market_context_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract features from market context"""
        
        features = {}
        
        if len(market_data) > 0:
            # Recent volatility
            returns = market_data['Close'].pct_change().dropna()
            features['market_volatility'] = returns.std() if len(returns) > 0 else 0
            
            # Trend
            if len(market_data) >= 5:
                recent_trend = (market_data['Close'].iloc[-1] - market_data['Close'].iloc[-5]) / market_data['Close'].iloc[-5]
                features['market_trend'] = recent_trend
            else:
                features['market_trend'] = 0
            
            # Volume spike
            if 'Volume' in market_data:
                avg_volume = market_data['Volume'].mean()
                recent_volume = market_data['Volume'].iloc[-1] if len(market_data) > 0 else avg_volume
                features['volume_spike'] = recent_volume / (avg_volume + 1e-8)
            else:
                features['volume_spike'] = 1.0
        
        return features
    
    def _extract_complexity_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic complexity features"""
        
        features = {}
        
        # Basic metrics
        words = text.split()
        sentences = text.split('.')
        
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['avg_sentence_length'] = len(words) / (len(sentences) + 1)
        
        # Readability (simplified Flesch score)
        syllable_count = sum(self._count_syllables(word) for word in words)
        if len(words) > 0 and len(sentences) > 0:
            features['readability'] = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllable_count / len(words))
        else:
            features['readability'] = 0
        
        return features
    
    def _extract_trading_patterns(self, text: str) -> Dict[str, float]:
        """Extract trading-specific patterns"""
        
        features = {}
        
        # Bull/bear indicators
        bullish_words = ['bull', 'bullish', 'rally', 'surge', 'soar', 'jump', 'gain']
        bearish_words = ['bear', 'bearish', 'plunge', 'crash', 'fall', 'drop', 'decline']
        
        features['bullish_score'] = sum(1 for word in bullish_words if word in text.lower())
        features['bearish_score'] = sum(1 for word in bearish_words if word in text.lower())
        
        # Action words
        action_words = ['buy', 'sell', 'hold', 'upgrade', 'downgrade']
        for action in action_words:
            features[f'action_{action}'] = 1 if action in text.lower() else 0
        
        # Confidence indicators
        confidence_high = ['definitely', 'certainly', 'surely', 'confident']
        confidence_low = ['maybe', 'perhaps', 'possibly', 'uncertain']
        
        features['confidence_high'] = sum(1 for word in confidence_high if word in text.lower())
        features['confidence_low'] = sum(1 for word in confidence_low if word in text.lower())
        
        return features
    
    def _count_syllables(self, word: str) -> int:
        """Simple syllable counter"""
        vowels = 'aeiouAEIOU'
        count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e'):
            count -= 1
        
        # Ensure at least one syllable
        return max(1, count)


# Configuration presets for different trading styles
TRADING_CONFIGS = {
    'conservative': {
        'threshold': 0.7,
        'min_confidence': 0.8,
        'position_size_multiplier': 0.5,
        'stop_loss_multiplier': 1.5,
        'take_profit_multiplier': 2.0
    },
    'moderate': {
        'threshold': 0.6,
        'min_confidence': 0.65,
        'position_size_multiplier': 0.75,
        'stop_loss_multiplier': 2.0,
        'take_profit_multiplier': 3.0
    },
    'aggressive': {
        'threshold': 0.5,
        'min_confidence': 0.5,
        'position_size_multiplier': 1.0,
        'stop_loss_multiplier': 2.5,
        'take_profit_multiplier': 4.0
    }
}

# Save configuration
def save_trading_config(config_name: str, config: Dict, filepath: str = "configs/trading_config.json"):
    """Save trading configuration"""
    
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    configs = {}
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            configs = json.load(f)
    
    configs[config_name] = {
        'config': config,
        'created': datetime.now().isoformat(),
        'performance': {}
    }
    
    with open(filepath, 'w') as f:
        json.dump(configs, f, indent=2)
    
    logger.info(f"Saved trading config: {config_name}")