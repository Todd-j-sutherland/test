"""
Historical sentiment tracking module
Stores and analyzes sentiment trends over time
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SentimentHistoryManager:
    """Manages historical sentiment data and trend analysis"""
    
    def __init__(self, data_dir: str = "data/sentiment_history"):
        self.data_dir = data_dir
        self.ensure_data_dir()
    
    def ensure_data_dir(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def store_sentiment(self, symbol: str, sentiment_data: Dict):
        """Store sentiment data with timestamp"""
        try:
            # Add timestamp if not present
            if 'timestamp' not in sentiment_data:
                sentiment_data['timestamp'] = datetime.now().isoformat()
            
            # Load existing data
            history = self.load_sentiment_history(symbol)
            
            # Add new data point
            history.append(sentiment_data)
            
            # Keep only last 30 days of data
            cutoff_date = datetime.now() - timedelta(days=30)
            history = [
                entry for entry in history 
                if datetime.fromisoformat(entry['timestamp']) > cutoff_date
            ]
            
            # Save updated history
            filename = os.path.join(self.data_dir, f"{symbol}_history.json")
            with open(filename, 'w') as f:
                json.dump(history, f, indent=2, default=str)
            
            logger.info(f"Stored sentiment data for {symbol}")
            
        except Exception as e:
            logger.error(f"Error storing sentiment for {symbol}: {e}")
    
    def load_sentiment_history(self, symbol: str) -> List[Dict]:
        """Load historical sentiment data for a symbol"""
        try:
            filename = os.path.join(self.data_dir, f"{symbol}_history.json")
            
            if not os.path.exists(filename):
                return []
            
            with open(filename, 'r') as f:
                history = json.load(f)
            
            # Sort by timestamp
            history.sort(key=lambda x: x['timestamp'])
            
            return history
            
        except Exception as e:
            logger.error(f"Error loading sentiment history for {symbol}: {e}")
            return []
    
    def get_sentiment_trend(self, symbol: str, days: int = 7) -> Dict:
        """Analyze sentiment trend over specified days"""
        try:
            history = self.load_sentiment_history(symbol)
            
            if not history:
                return self._empty_trend_data()
            
            # Filter to specified timeframe
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_history = [
                entry for entry in history 
                if datetime.fromisoformat(entry['timestamp']) > cutoff_date
            ]
            
            if not recent_history:
                return self._empty_trend_data()
            
            # Extract sentiment values
            sentiments = [entry['overall_sentiment'] for entry in recent_history]
            timestamps = [entry['timestamp'] for entry in recent_history]
            news_counts = [entry.get('news_count', 0) for entry in recent_history]
            
            # Calculate trend metrics
            first_sentiment = sentiments[0]
            last_sentiment = sentiments[-1]
            trend_direction = last_sentiment - first_sentiment
            
            # Calculate volatility (standard deviation)
            avg_sentiment = sum(sentiments) / len(sentiments)
            volatility = (sum((s - avg_sentiment) ** 2 for s in sentiments) / len(sentiments)) ** 0.5
            
            # Calculate momentum (rate of change)
            momentum = 0
            if len(sentiments) > 1:
                momentum = (sentiments[-1] - sentiments[0]) / len(sentiments)
            
            # Identify significant changes
            significant_changes = []
            for i in range(1, len(sentiments)):
                change = sentiments[i] - sentiments[i-1]
                if abs(change) > 0.2:  # Significant change threshold
                    significant_changes.append({
                        'timestamp': timestamps[i],
                        'change': change,
                        'from_sentiment': sentiments[i-1],
                        'to_sentiment': sentiments[i]
                    })
            
            return {
                'symbol': symbol,
                'analysis_period_days': days,
                'data_points': len(recent_history),
                'current_sentiment': last_sentiment,
                'average_sentiment': avg_sentiment,
                'trend_direction': trend_direction,
                'trend_description': self._describe_trend(trend_direction),
                'volatility': volatility,
                'momentum': momentum,
                'significant_changes': significant_changes,
                'sentiment_range': {
                    'min': min(sentiments),
                    'max': max(sentiments),
                    'range': max(sentiments) - min(sentiments)
                },
                'news_activity': {
                    'total_news': sum(news_counts),
                    'average_per_day': sum(news_counts) / len(news_counts),
                    'max_daily_news': max(news_counts),
                    'min_daily_news': min(news_counts)
                },
                'daily_data': [
                    {
                        'date': timestamps[i][:10],
                        'sentiment': sentiments[i],
                        'news_count': news_counts[i],
                        'change_from_previous': sentiments[i] - sentiments[i-1] if i > 0 else 0
                    }
                    for i in range(len(recent_history))
                ]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment trend for {symbol}: {e}")
            return self._empty_trend_data()
    
    def _empty_trend_data(self) -> Dict:
        """Return empty trend data structure"""
        return {
            'symbol': '',
            'analysis_period_days': 0,
            'data_points': 0,
            'current_sentiment': 0,
            'average_sentiment': 0,
            'trend_direction': 0,
            'trend_description': 'no_data',
            'volatility': 0,
            'momentum': 0,
            'significant_changes': [],
            'sentiment_range': {'min': 0, 'max': 0, 'range': 0},
            'news_activity': {'total_news': 0, 'average_per_day': 0, 'max_daily_news': 0, 'min_daily_news': 0},
            'daily_data': []
        }
    
    def _describe_trend(self, trend_direction: float) -> str:
        """Convert trend direction to description"""
        if trend_direction > 0.1:
            return 'improving'
        elif trend_direction < -0.1:
            return 'deteriorating'
        else:
            return 'stable'
    
    def get_comparative_analysis(self, symbols: List[str], days: int = 7) -> Dict:
        """Compare sentiment trends across multiple symbols"""
        try:
            comparative_data = {}
            
            for symbol in symbols:
                trend_data = self.get_sentiment_trend(symbol, days)
                comparative_data[symbol] = {
                    'current_sentiment': trend_data['current_sentiment'],
                    'average_sentiment': trend_data['average_sentiment'],
                    'trend_direction': trend_data['trend_direction'],
                    'trend_description': trend_data['trend_description'],
                    'volatility': trend_data['volatility'],
                    'momentum': trend_data['momentum'],
                    'data_points': trend_data['data_points']
                }
            
            # Calculate market-wide metrics
            all_current_sentiments = [data['current_sentiment'] for data in comparative_data.values() if data['data_points'] > 0]
            all_trends = [data['trend_direction'] for data in comparative_data.values() if data['data_points'] > 0]
            
            market_sentiment = sum(all_current_sentiments) / len(all_current_sentiments) if all_current_sentiments else 0
            market_trend = sum(all_trends) / len(all_trends) if all_trends else 0
            
            # Identify leaders and laggards
            sorted_by_sentiment = sorted(
                [(symbol, data) for symbol, data in comparative_data.items() if data['data_points'] > 0],
                key=lambda x: x[1]['current_sentiment'],
                reverse=True
            )
            
            sorted_by_trend = sorted(
                [(symbol, data) for symbol, data in comparative_data.items() if data['data_points'] > 0],
                key=lambda x: x[1]['trend_direction'],
                reverse=True
            )
            
            return {
                'analysis_period_days': days,
                'symbols_analyzed': len(symbols),
                'market_sentiment': market_sentiment,
                'market_trend': market_trend,
                'market_trend_description': self._describe_trend(market_trend),
                'individual_analysis': comparative_data,
                'rankings': {
                    'by_current_sentiment': [(symbol, data['current_sentiment']) for symbol, data in sorted_by_sentiment],
                    'by_trend_direction': [(symbol, data['trend_direction']) for symbol, data in sorted_by_trend]
                },
                'leaders': {
                    'most_positive': sorted_by_sentiment[0] if sorted_by_sentiment else None,
                    'most_improving': sorted_by_trend[0] if sorted_by_trend else None
                },
                'laggards': {
                    'most_negative': sorted_by_sentiment[-1] if sorted_by_sentiment else None,
                    'most_declining': sorted_by_trend[-1] if sorted_by_trend else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error in comparative analysis: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Remove old sentiment data files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            for filename in os.listdir(self.data_dir):
                if filename.endswith('_history.json'):
                    filepath = os.path.join(self.data_dir, filename)
                    
                    # Load and filter data
                    with open(filepath, 'r') as f:
                        history = json.load(f)
                    
                    # Keep only recent data
                    filtered_history = [
                        entry for entry in history 
                        if datetime.fromisoformat(entry['timestamp']) > cutoff_date
                    ]
                    
                    # Save filtered data or remove empty files
                    if filtered_history:
                        with open(filepath, 'w') as f:
                            json.dump(filtered_history, f, indent=2, default=str)
                    else:
                        os.remove(filepath)
                        logger.info(f"Removed empty history file: {filename}")
            
            logger.info(f"Cleaned up old sentiment data (kept {days_to_keep} days)")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
