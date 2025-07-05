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
            