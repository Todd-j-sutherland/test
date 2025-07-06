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
            distance_from_low = key_levels.get('distance_from_low', 0)
            
            # Closeness to support/resistance levels
            if distance_from_low < 0.05 * current_price:  # Within 5% of a low
                score += 10
                factors.append('Approaching support level')
            if distance_from_high < 0.05 * current_price:  # Within 5% of a high
                score -= 10
                factors.append('Approaching resistance level')
        
        return {
            'score': score,
            'factors': factors,
            'signal': 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral',
            'strength': min(abs(score) / 50, 1.0)  # Normalize strength to 0-1
        }
        
        # except Exception as e:
        #     logger.warning(f"Error in technical prediction: {str(e)}")
        #     return {'score': 0, 'factors': [], 'signal': 'neutral', 'strength': 0}
    
    def _fundamental_prediction(self, fundamental: Dict) -> Dict:
        """Generate prediction based on fundamental analysis"""
        
        score = 0
        factors = []
        
        try:
            # Valuation analysis
            valuation = fundamental.get('valuation', {})
            if valuation.get('rating') == 'undervalued':
                score += 20
                factors.append('Stock appears undervalued')
            elif valuation.get('rating') == 'overvalued':
                score -= 20
                factors.append('Stock appears overvalued')
            
            # Fundamental score
            fund_score = fundamental.get('score', 50)
            if fund_score > 70:
                score += 25
                factors.append('Strong fundamental metrics')
            elif fund_score < 30:
                score -= 25
                factors.append('Weak fundamental metrics')
            
            # Growth analysis
            growth = fundamental.get('growth', {})
            earnings_growth = growth.get('earnings_growth', 0)
            if earnings_growth > 10:
                score += 15
                factors.append('Strong earnings growth')
            elif earnings_growth < -5:
                score -= 15
                factors.append('Declining earnings')
            
            return {
                'score': score,
                'factors': factors,
                'signal': 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral',
                'strength': min(abs(score) / 50, 1.0)
            }
            
        except Exception as e:
            logger.warning(f"Error in fundamental prediction: {str(e)}")
            return {'score': 0, 'factors': [], 'signal': 'neutral', 'strength': 0}
    
    def _sentiment_prediction(self, sentiment: Dict) -> Dict:
        """Generate prediction based on sentiment analysis"""
        
        score = 0
        factors = []
        
        try:
            # Overall sentiment
            overall_sentiment = sentiment.get('overall_sentiment', 0)
            score += overall_sentiment * 40  # Scale sentiment to reasonable range
            
            if overall_sentiment > 0.3:
                factors.append('Positive news sentiment')
            elif overall_sentiment < -0.3:
                factors.append('Negative news sentiment')
            
            # News count consideration
            news_count = sentiment.get('news_count', 0)
            if news_count > 10:
                factors.append('High news volume')
            
            return {
                'score': score,
                'factors': factors,
                'signal': 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral',
                'strength': min(abs(score) / 30, 1.0)
            }
            
        except Exception as e:
            logger.warning(f"Error in sentiment prediction: {str(e)}")
            return {'score': 0, 'factors': [], 'signal': 'neutral', 'strength': 0}
    
    def _market_structure_prediction(self, technical: Dict) -> Dict:
        """Generate prediction based on market structure"""
        
        score = 0
        factors = []
        
        try:
            # Support/Resistance levels
            sr = technical.get('support_resistance', {})
            key_levels = technical.get('key_levels', {})
            
            if key_levels:
                distance_from_high = key_levels.get('distance_from_high', 0)
                distance_from_low = key_levels.get('distance_from_low', 0)
                
                if distance_from_low > 50:  # Near highs
                    score += 10
                    factors.append('Price near recent highs')
                elif distance_from_high > 50:  # Near lows
                    score -= 10
                    factors.append('Price near recent lows')
            
            # Volume profile
            volume = technical.get('indicators', {}).get('volume', {})
            vol_ratio = volume.get('ratio', 1)
            if vol_ratio > 1.5:
                score += 5
                factors.append('Above average volume')
            
            return {
                'score': score,
                'factors': factors,
                'signal': 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral',
                'strength': min(abs(score) / 20, 1.0)
            }
            
        except Exception as e:
            logger.warning(f"Error in market structure prediction: {str(e)}")
            return {'score': 0, 'factors': [], 'signal': 'neutral', 'strength': 0}
    
    def _seasonal_prediction(self, symbol: str) -> Dict:
        """Generate prediction based on seasonal patterns"""
        
        score = 0
        factors = []
        
        try:
            # Simple seasonal analysis
            current_month = datetime.now().month
            
            # Banks typically perform better in certain months
            if current_month in [2, 3, 4]:  # Reporting season
                score += 5
                factors.append('Earnings season approaching')
            elif current_month in [11, 12]:  # End of year
                score += 3
                factors.append('End of year positioning')
            
            return {
                'score': score,
                'factors': factors,
                'signal': 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral',
                'strength': min(abs(score) / 10, 1.0)
            }
            
        except Exception as e:
            logger.warning(f"Error in seasonal prediction: {str(e)}")
            return {'score': 0, 'factors': [], 'signal': 'neutral', 'strength': 0}
    
    def _predict_timeframe(self, predictions: Dict, timeframe: str) -> Dict:
        """Generate prediction for specific timeframe"""
        
        try:
            # Weight predictions differently for different timeframes
            if timeframe == 'short':
                # Short term: technical and sentiment more important
                weights = {'technical': 0.5, 'sentiment': 0.3, 'market_structure': 0.2}
            elif timeframe == 'medium':
                # Medium term: balanced approach
                weights = {'technical': 0.3, 'fundamental': 0.3, 'sentiment': 0.2, 'market_structure': 0.2}
            else:  # long term
                # Long term: fundamental more important
                weights = {'fundamental': 0.5, 'technical': 0.2, 'sentiment': 0.1, 'seasonality': 0.2}
            
            # Calculate weighted score
            weighted_score = 0
            for key, weight in weights.items():
                if key in predictions:
                    weighted_score += predictions[key]['score'] * weight
            
            # Determine direction
            if weighted_score > 15:
                direction = 'bullish'
                strength = 'strong' if weighted_score > 30 else 'moderate'
            elif weighted_score < -15:
                direction = 'bearish'
                strength = 'strong' if weighted_score < -30 else 'moderate'
            else:
                direction = 'neutral'
                strength = 'weak'
            
            return {
                'direction': direction,
                'strength': strength,
                'score': weighted_score,
                'confidence': min(abs(weighted_score) / 30, 1.0)
            }
            
        except Exception as e:
            logger.warning(f"Error in timeframe prediction: {str(e)}")
            return {'direction': 'neutral', 'strength': 'weak', 'score': 0, 'confidence': 0}
    
    def _calculate_price_targets(self, technical: Dict, score: float, direction: str) -> Dict:
        """Calculate price targets based on analysis"""
        
        try:
            key_levels = technical.get('key_levels', {})
            current_price = key_levels.get('current_price', 0)
            
            if current_price <= 0:
                return {'upside_target': 0, 'downside_target': 0}
            
            # Calculate targets based on ATR or percentage
            atr = technical.get('indicators', {}).get('atr', current_price * 0.02)
            
            if direction == 'bullish':
                upside_target = current_price + (atr * 2)
                downside_target = current_price - (atr * 1)
            elif direction == 'bearish':
                upside_target = current_price + (atr * 1)
                downside_target = current_price - (atr * 2)
            else:
                upside_target = current_price + (atr * 1.5)
                downside_target = current_price - (atr * 1.5)
            
            return {
                'current_price': current_price,
                'upside_target': round(upside_target, 2),
                'downside_target': round(downside_target, 2),
                'risk_reward_ratio': round((upside_target - current_price) / (current_price - downside_target), 2) if downside_target < current_price else 0
            }
            
        except Exception as e:
            logger.warning(f"Error calculating price targets: {str(e)}")
            return {'upside_target': 0, 'downside_target': 0, 'risk_reward_ratio': 0}
    
    def _identify_key_factors(self, predictions: Dict) -> List[str]:
        """Identify key factors driving the prediction"""
        
        key_factors = []
        
        for category, pred in predictions.items():
            factors = pred.get('factors', [])
            for factor in factors:
                if factor not in key_factors:
                    key_factors.append(factor)
        
        return key_factors[:5]  # Return top 5 factors
    
    def _generate_recommendation(self, direction: str, strength: str, score: float) -> str:
        """Generate trading recommendation"""
        
        if direction == 'bullish':
            if strength == 'strong':
                return 'STRONG BUY'
            elif strength == 'moderate':
                return 'BUY'
            else:
                return 'HOLD'
        elif direction == 'bearish':
            if strength == 'strong':
                return 'STRONG SELL'
            elif strength == 'moderate':
                return 'SELL'
            else:
                return 'HOLD'
        else:
            return 'HOLD'
    
    def _default_prediction(self) -> Dict:
        """Return default prediction when analysis fails"""
        
        return {
            'symbol': '',
            'timestamp': datetime.now().isoformat(),
            'direction': 'neutral',
            'strength': 'weak',
            'confidence': 0,
            'score': 0,
            'timeframes': {
                'short_term': {'direction': 'neutral', 'strength': 'weak', 'score': 0, 'confidence': 0},
                'medium_term': {'direction': 'neutral', 'strength': 'weak', 'score': 0, 'confidence': 0},
                'long_term': {'direction': 'neutral', 'strength': 'weak', 'score': 0, 'confidence': 0}
            },
            'price_targets': {'upside_target': 0, 'downside_target': 0, 'risk_reward_ratio': 0},
            'predictions': {},
            'key_factors': ['Analysis temporarily unavailable'],
            'recommendation': 'HOLD'
        }
