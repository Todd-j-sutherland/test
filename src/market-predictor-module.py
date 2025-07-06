# src/market_predictor.py
"""
Market prediction module that combines all analysis for bullish/bearish predictions
Uses machine learning concepts without external ML libraries
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import json

# Import settings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Settings

logger = logging.getLogger(__name__)

class MarketPredictor:
    """Predicts market direction using multiple indicators"""
    
    def __init__(self):
        self.settings = Settings()
        self.prediction_weights = {
            'technical': 0.35,
            'fundamental': 0.25,
            'sentiment': 0.20,
            'market_structure': 0.10,
            'seasonality': 0.10
        }
    
    def predict(self, symbol: str, technical: Dict, fundamental: Dict, sentiment: Dict) -> Dict:
        """Generate comprehensive market prediction"""
        
        try:
            # Calculate individual predictions
            tech_prediction = self._technical_prediction(technical)
            fund_prediction = self._fundamental_prediction(fundamental)
            sent_prediction = self._sentiment_prediction(sentiment)
            market_prediction = self._market_structure_prediction(technical)
            seasonal_prediction = self._seasonal_prediction(symbol)
            
            # Combine predictions
            predictions = {
                'technical': tech_prediction,
                'fundamental': fund_prediction,
                'sentiment': sent_prediction,
                'market_structure': market_prediction,
                'seasonality': seasonal_prediction
            }
            
            # Calculate weighted prediction
            weighted_score = sum(
                predictions[key]['score'] * self.prediction_weights[key]
                for key in predictions
            )
            
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
            
            # Calculate time horizons
            short_term = self._predict_timeframe(predictions, 'short')  # 1-5 days
            medium_term = self._predict_timeframe(predictions, 'medium')  # 1-4 weeks
            long_term = self._predict_timeframe(predictions, 'long')  # 1-3 months
            
            # Generate price targets
            price_targets = self._calculate_price_targets(
                technical, 
                weighted_score,
                direction
            )
            
            # Identify key factors
            key_factors = self._identify_key_factors(predictions)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'direction': direction,
                'strength': strength,
                'confidence': abs(weighted_score),
                'score': weighted_score,
                'timeframes': {
                    'short_term': short_term,
                    'medium_term': medium_term,
                    'long_term': long_term
                },
                'price_targets': price_targets,
                'predictions': predictions,
                'key_factors': key_factors,
                'recommendation': self._generate_recommendation(
                    direction, strength, weighted_score
                )
            }
            
        except Exception as e:
            logger.error(f"Error in prediction for {symbol}: {str(e)}")
            return self._default_prediction()
    
    def _technical_prediction(self, technical: Dict) -> Dict:
        """Generate prediction based on technical analysis"""
        
        score = 0
        factors = []
        
        # Trend analysis
        trend = technical.get('trend', {})
        if trend.get('primary') == 'bullish':
            score += 30
            factors.append('Bullish trend confirmed')
        elif trend.get('primary') == 'bearish':
            score -= 30
            factors.append('Bearish trend confirmed')
        
        # Signal strength
        overall_signal = technical.get('overall_signal', 0)
        score += overall_signal * 0.5  # Scale down to reasonable range
        
        # Support/Resistance proximity
        sr = technical.get('support_resistance', {})
        key_levels = technical.get('key_levels', {})
        
        if key_levels:
            current_price = key_levels.get('current_price', 0)
            distance_from_high = key_levels.get('distance_from_high', 0)
            
            if distance_from_high > 20:
                score += 10
                factors.append('Trading well off highs')
            elif distance_from_high < 5:
                score -= 10
                factors.append('Near resistance levels')
        
        # Volume analysis
        volume = technical.get('indicators', {}).get('volume', {})
        if volume.get('ratio', 1) > 1.5:
            score += 5
            factors.append('Above average volume')
        
        return {
            'score': max(-100, min(100, score)),
            'factors': factors,
            'confidence': min(abs(score), 100)
        }
    
    def _fundamental_prediction(self, fundamental: Dict) -> Dict:
        """Generate prediction based on fundamental analysis"""
        
        score = 0
        factors = []
        
        # Valuation metrics
        valuation = fundamental.get('valuation', {})
        if valuation.get('rating') == 'undervalued':
            score += 20
            factors.append('Undervalued stock')
        elif valuation.get('rating') == 'overvalued':
            score -= 20
            factors.append('Overvalued stock')
        
        # Financial health
        metrics = fundamental.get('metrics', {})
        roe = metrics.get('roe', 0)
        if roe > 0.15:
            score += 10
            factors.append('Strong ROE')
        
        return {
            'score': max(-100, min(100, score)),
            'factors': factors,
            'confidence': min(abs(score), 100)
        }
    
    def _sentiment_prediction(self, sentiment: Dict) -> Dict:
        """Generate prediction based on sentiment analysis"""
        
        score = 0
        factors = []
        
        overall_sentiment = sentiment.get('overall_sentiment', 0)
        score = overall_sentiment * 50  # Scale to -50 to 50
        
        if overall_sentiment > 0.3:
            factors.append('Positive market sentiment')
        elif overall_sentiment < -0.3:
            factors.append('Negative market sentiment')
        
        return {
            'score': max(-100, min(100, score)),
            'factors': factors,
            'confidence': min(abs(score), 100)
        }
    
    def _market_structure_prediction(self, technical: Dict) -> Dict:
        """Generate prediction based on market structure"""
        
        score = 0
        factors = []
        
        # Simple market structure analysis
        trend = technical.get('trend', {})
        if trend.get('structure') == 'bullish':
            score += 15
            factors.append('Bullish market structure')
        elif trend.get('structure') == 'bearish':
            score -= 15
            factors.append('Bearish market structure')
        
        return {
            'score': max(-100, min(100, score)),
            'factors': factors,
            'confidence': min(abs(score), 100)
        }
    
    def _seasonal_prediction(self, symbol: str) -> Dict:
        """Generate prediction based on seasonal patterns"""
        
        score = 0
        factors = []
        
        # Australian banks seasonal patterns
        month = datetime.now().month
        
        # Dividend season (February/August)
        if month in [2, 8]:
            score += 5
            factors.append('Dividend season')
        
        # End of financial year (June)
        if month == 6:
            score -= 5
            factors.append('End of financial year')
        
        return {
            'score': max(-100, min(100, score)),
            'factors': factors,
            'confidence': min(abs(score), 100)
        }
    
    def _predict_timeframe(self, predictions: Dict, timeframe: str) -> Dict:
        """Predict for specific timeframe"""
        
        weights = {
            'short': {'technical': 0.6, 'sentiment': 0.3, 'market_structure': 0.1},
            'medium': {'technical': 0.4, 'fundamental': 0.4, 'sentiment': 0.2},
            'long': {'fundamental': 0.6, 'technical': 0.2, 'sentiment': 0.2}
        }
        
        weight_set = weights.get(timeframe, weights['medium'])
        
        score = sum(
            predictions[key]['score'] * weight
            for key, weight in weight_set.items()
            if key in predictions
        )
        
        return {
            'prediction': 'bullish' if score > 20 else 'bearish' if score < -20 else 'neutral',
            'score': score,
            'confidence': min(abs(score), 100)
        }
    
    def _calculate_price_targets(self, technical: Dict, score: float, direction: str) -> Dict:
        """Calculate price targets based on prediction"""
        
        key_levels = technical.get('key_levels', {})
        current_price = key_levels.get('current_price', 0)
        
        if current_price == 0:
            return {'upside': 0, 'downside': 0}
        
        # Calculate potential moves based on score
        percentage_move = abs(score) * 0.001  # 0.1% per point
        
        if direction == 'bullish':
            upside = current_price * (1 + percentage_move)
            downside = current_price * 0.95  # 5% downside
        else:
            upside = current_price * 1.05  # 5% upside
            downside = current_price * (1 - percentage_move)
        
        return {
            'upside': upside,
            'downside': downside,
            'current': current_price
        }
    
    def _identify_key_factors(self, predictions: Dict) -> List[str]:
        """Identify key factors driving the prediction"""
        
        factors = []
        
        for pred_type, pred_data in predictions.items():
            if pred_data.get('score', 0) != 0:
                factors.extend(pred_data.get('factors', []))
        
        return factors[:5]  # Top 5 factors
    
    def _generate_recommendation(self, direction: str, strength: str, score: float) -> Dict:
        """Generate trading recommendation"""
        
        if direction == 'bullish':
            if strength == 'strong':
                action = 'STRONG BUY'
            else:
                action = 'BUY'
        elif direction == 'bearish':
            if strength == 'strong':
                action = 'STRONG SELL'
            else:
                action = 'SELL'
        else:
            action = 'HOLD'
        
        return {
            'action': action,
            'confidence': min(abs(score), 100),
            'reasoning': f"{direction} {strength} prediction based on analysis"
        }
    
    def _default_prediction(self) -> Dict:
        """Return default prediction when analysis fails"""
        
        return {
            'direction': 'neutral',
            'strength': 'weak',
            'confidence': 0,
            'score': 0,
            'timeframes': {
                'short_term': {'prediction': 'neutral', 'score': 0},
                'medium_term': {'prediction': 'neutral', 'score': 0},
                'long_term': {'prediction': 'neutral', 'score': 0}
            },
            'price_targets': {'upside': 0, 'downside': 0},
            'key_factors': ['Analysis temporarily unavailable'],
            'recommendation': {'action': 'HOLD', 'confidence': 0}
        }
