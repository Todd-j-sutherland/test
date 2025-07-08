# src/news_sentiment.py
"""
News and sentiment analysis module using free sources
Scrapes Australian financial news and analyzes sentiment
"""

import requests
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime, timedelta
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
from typing import Dict, List, Optional
import time
import praw
import os
import numpy as np  # Add numpy import at the top

# Transformers for advanced sentiment analysis
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    # Try importing torch or tensorflow
    backend_available = False
    try:
        import torch
        backend_available = True
        BACKEND_TYPE = "torch"
    except ImportError:
        try:
            import tensorflow as tf
            backend_available = True
            BACKEND_TYPE = "tensorflow"
        except ImportError:
            pass
    
    TRANSFORMERS_AVAILABLE = backend_available
    if not backend_available:
        logging.warning("Neither PyTorch nor TensorFlow available. Transformers will not work.")
        logging.warning("For Python 3.13: Consider using Python 3.11 or 3.12 for full transformer support.")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    BACKEND_TYPE = "none"
    logging.warning("Transformers not available. Install with: pip install transformers torch")

import yfinance as yf
import pandas as pd

# Import settings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Settings
from utils.cache_manager import CacheManager
from src.sentiment_history import SentimentHistoryManager
from src.news_impact_analyzer import NewsImpactAnalyzer

# Import ML trading components for enhanced analysis
try:
    from src.ml_trading_config import FeatureEngineer, TradingModelOptimizer
    ML_TRADING_AVAILABLE = True
except ImportError:
    ML_TRADING_AVAILABLE = False
    logging.warning("ML trading components not available")

logger = logging.getLogger(__name__)

class NewsSentimentAnalyzer:
    """Analyzes news sentiment from free Australian sources"""
    
    def __init__(self):
        self.settings = Settings()
        self.cache = CacheManager()
        self.vader = SentimentIntensityAnalyzer()
        self.history_manager = SentimentHistoryManager()
        self.impact_analyzer = NewsImpactAnalyzer()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Initialize Transformers models for advanced sentiment analysis
        self.transformer_models = {}
        if TRANSFORMERS_AVAILABLE:
            try:
                # Initialize multiple models for different aspects
                self._initialize_transformer_models()
            except Exception as e:
                logger.warning(f"Failed to initialize Transformers models: {e}")
                self.transformer_models = {}
        
        # Initialize Reddit client (read-only)
        self.reddit = None
        try:
            self.reddit = praw.Reddit(
                client_id="XWbagNe33Nz0xFF1nUngbA",  # Optional - uses public API
                client_secret="INNUY2kX3PgO58NElPr2_necAlKdQw",  # Optional
                user_agent="ASXBankAnalysis/1.0 by /u/tradingbot",
                check_for_async=False
            )
        except Exception as e:
            logger.warning(f"Reddit client initialization failed: {e}")
            logger.info("Reddit integration will use fallback mode")
        
        # Bank name variations for searching
        self.bank_keywords = {
            'CBA.AX': ['Commonwealth Bank', 'CommBank', 'CBA', 'commonwealth'],
            'WBC.AX': ['Westpac', 'WBC', 'westpac'],
            'ANZ.AX': ['ANZ', 'Australia and New Zealand Banking', 'anz'],
            'NAB.AX': ['National Australia Bank', 'NAB', 'nab'],
            'MQG.AX': ['Macquarie', 'MQG', 'Macquarie Group', 'macquarie']
        }
        
        # Reddit subreddits for financial discussion
        self.financial_subreddits = [
            'AusFinance',
            'ASX_Bets', 
            'fiaustralia',
            'AusEcon',
            'AusProperty',
            'SecurityAnalysis',
            'investing',
            'stocks'
        ]
        
        # Initialize ML Trading components for enhanced analysis
        self.feature_engineer = None
        self.ml_models = {}
        self.trading_features_cache = {}
        
        if ML_TRADING_AVAILABLE:
            try:
                self.feature_engineer = FeatureEngineer()
                logger.info("✅ ML Trading feature engineer initialized")
                
                # Initialize basic models for sentiment scoring
                self._initialize_ml_trading_models()
                
            except Exception as e:
                logger.warning(f"Failed to initialize ML trading components: {e}")
                self.feature_engineer = None
    
    def _initialize_transformer_models(self):
        """Initialize various transformer models for sentiment analysis"""
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers backend not available. Skipping transformer model initialization.")
            logger.info("For Python 3.13 users: Consider using Python 3.11 or 3.12 for full transformer support.")
            return
        
        try:
            logger.info(f"Initializing transformer models with {BACKEND_TYPE} backend...")
            
            # Start with a simple model to test if transformers work
            logger.info("Testing transformer functionality with basic model...")
            logger.warning("⚠️  This will download ~268MB model files. Set SKIP_TRANSFORMERS=1 to disable.")
            
            # Check if user wants to skip transformer downloads
            if os.environ.get('SKIP_TRANSFORMERS', '0') == '1':
                logger.info("🚫 Skipping transformer downloads (SKIP_TRANSFORMERS=1)")
                return
            
            test_model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
            
            # Test the model
            test_result = test_model("This is a test sentence.")
            logger.info("✅ Basic transformer model working!")
            
            # Store the working basic model
            self.transformer_models['basic'] = test_model
            
            # Try to load more advanced models
            try:
                # 1. General Financial Sentiment - FinBERT
                logger.info("Loading FinBERT for financial sentiment analysis...")
                self.transformer_models['financial'] = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    return_all_scores=True
                )
                logger.info("✅ FinBERT loaded successfully!")
                
            except Exception as e:
                logger.warning(f"Could not load FinBERT: {e}")
            
            try:
                # 2. General Sentiment - RoBERTa optimized for social media
                logger.info("Loading RoBERTa for general sentiment analysis...")
                self.transformer_models['general'] = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                logger.info("✅ RoBERTa loaded successfully!")
                
            except Exception as e:
                logger.warning(f"Could not load RoBERTa: {e}")
            
            try:
                # 3. Emotion Detection - for nuanced analysis
                logger.info("Loading emotion detection model...")
                self.transformer_models['emotion'] = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    tokenizer="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True
                )
                logger.info("✅ Emotion detection model loaded successfully!")
                
            except Exception as e:
                logger.warning(f"Could not load emotion detection model: {e}")
            
            try:
                # 4. News Classification - for identifying financial news types
                logger.info("Loading news classification model...")
                self.transformer_models['news_type'] = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
                logger.info("✅ News classification model loaded successfully!")
                
            except Exception as e:
                logger.warning(f"Could not load news classification model: {e}")
            
            logger.info(f"Transformer model initialization complete! Loaded {len(self.transformer_models)} models.")
            
        except Exception as e:
            logger.error(f"Error initializing transformer models: {e}")
            logger.warning("Falling back to traditional sentiment analysis methods only.")
            logger.info("For Python 3.13 users: Consider using Python 3.11 or 3.12 for full transformer support.")
            self.transformer_models = {}
    
    def _initialize_ml_trading_models(self):
        """Initialize ML models optimized for trading sentiment analysis"""
        
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            import numpy as np
            
            logger.info("Initializing ML trading models...")
            
            # Initialize basic models that can be trained on news data
            self.ml_models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'logistic_regression': LogisticRegression(
                    random_state=42,
                    class_weight='balanced',
                    max_iter=1000
                )
            }
            
            logger.info(f"✅ Initialized {len(self.ml_models)} ML trading models")
            
        except Exception as e:
            logger.error(f"Error initializing ML trading models: {e}")
            self.ml_models = {}
    
    def analyze_bank_sentiment(self, symbol: str) -> Dict:
        """Analyze sentiment for a specific bank"""
        
        cache_key = f"sentiment_{symbol}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            # Collect news from multiple sources
            all_news = []
            
            # RSS feeds
            rss_news = self._fetch_rss_news(symbol)
            all_news.extend(rss_news)
            
            # Yahoo Finance news
            yahoo_news = self._fetch_yahoo_news(symbol)
            all_news.extend(yahoo_news)
            
            # Web scraping
            scraped_news = self._scrape_news_sites(symbol)
            all_news.extend(scraped_news)
            
            # Reddit sentiment
            reddit_sentiment = self._get_reddit_sentiment(symbol)
            
            # Analyze sentiment
            sentiment_analysis = self._analyze_news_sentiment(all_news)
            
            # Check for specific events
            event_analysis = self._check_significant_events(all_news, symbol)
            
            # Get market context for enhanced calculation
            market_context = self._get_market_context()
            
            # Use enhanced sentiment calculation
            overall_sentiment = self._calculate_overall_sentiment_improved(
                sentiment_analysis,
                reddit_sentiment,
                event_analysis,
                market_context,
                all_news  # Pass all_news to the improved calculation
            )
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'news_count': len(all_news),
                'sentiment_scores': sentiment_analysis,
                'reddit_sentiment': reddit_sentiment,
                'significant_events': event_analysis,
                'overall_sentiment': overall_sentiment['score'],
                'sentiment_components': overall_sentiment['components'],
                'confidence': overall_sentiment['confidence'],
                'recent_headlines': [news['title'] for news in all_news[:5]]
            }
            logger.info(f"Sentiment analysis result for {symbol}: {result}")
            
            # Cache for 30 minutes
            self.cache.set(cache_key, result, expiry_minutes=30)
            
            # Store in historical data
            self.history_manager.store_sentiment(symbol, result)
            
            # Add trend analysis
            trend_data = self.history_manager.get_sentiment_trend(symbol, days=7)
            result['trend_analysis'] = trend_data
            
            # Add impact correlation analysis (if sufficient data)
            if trend_data['data_points'] >= 5:
                impact_analysis = self.impact_analyzer.analyze_sentiment_price_correlation(symbol, days=14)
                result['impact_analysis'] = impact_analysis
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
            return self._default_sentiment()
    
    def _calculate_overall_sentiment_improved(self, news_sentiment: Dict, 
                                            reddit_sentiment: Dict, 
                                            events: Dict,
                                            market_context: Dict,
                                            all_news: List[Dict] = None) -> Dict:
        """Enhanced sentiment calculation with ML trading features and dynamic weighting"""
        
        # Base weights - now includes ML trading features
        weights = {
            'news': 0.35,           # Reduced to make room for ML features
            'reddit': 0.15,
            'events': 0.2,          # Reduced slightly
            'volume': 0.1,
            'momentum': 0.1,
            'ml_trading': 0.1       # New ML trading component
        }
        
        # Adjust weights based on data quality
        news_count = news_sentiment.get('news_count', 0) if isinstance(news_sentiment, dict) else 0
        reddit_posts = reddit_sentiment.get('posts_analyzed', 0)
        
        # Check if transformers are being used and their confidence
        transformer_confidence = 0
        if isinstance(news_sentiment, dict) and 'method_breakdown' in news_sentiment:
            method_breakdown = news_sentiment['method_breakdown']
            transformer_confidence = method_breakdown.get('transformer', {}).get('confidence', 0)
        
        # Calculate ML trading features and score
        ml_trading_result = {'ml_score': 0, 'confidence': 0}
        if all_news and self.feature_engineer:
            try:
                # Fetch market data for ML features (optional)
                market_data = self._get_market_data_for_ml()
                ml_trading_result = self._calculate_ml_trading_score(all_news, market_data)
                logger.info(f"ML Trading Score: {ml_trading_result['ml_score']:.3f}, Confidence: {ml_trading_result['confidence']:.3f}")
            except Exception as e:
                logger.warning(f"ML trading score calculation failed: {e}")
        
        ml_score = ml_trading_result['ml_score']
        ml_confidence = ml_trading_result['confidence']
        
        # Boost news weight if transformers are working well
        if transformer_confidence > 0.8:
            weights['news'] += 0.05  # High confidence transformer boosts news reliability
        elif transformer_confidence > 0.6:
            weights['news'] += 0.03  # Medium confidence
        
        # Boost ML trading weight if ML confidence is high
        if ml_confidence > 0.8:
            weights['ml_trading'] += 0.05
            weights['news'] -= 0.02  # Slight reduction to maintain balance
            weights['events'] -= 0.03
        elif ml_confidence > 0.6:
            weights['ml_trading'] += 0.03
            weights['news'] -= 0.01
            weights['events'] -= 0.02
        
        # Dynamic weight adjustment based on data availability
        if news_count < 5:
            weights['news'] *= 0.5
            weights['reddit'] += weights['news'] * 0.2  # Transfer some weight to reddit
            weights['events'] += weights['news'] * 0.15
            weights['ml_trading'] += weights['news'] * 0.15  # Transfer some to ML trading
        
        if reddit_posts < 3:
            weights['reddit'] *= 0.3
            weights['news'] += weights['reddit'] * 0.4  # Transfer weight back to news
            weights['events'] += weights['reddit'] * 0.15
            weights['ml_trading'] += weights['reddit'] * 0.15
        
        # Reduce ML trading weight if ML features are unavailable
        if not self.feature_engineer or ml_confidence < 0.3:
            transferred_weight = weights['ml_trading']
            weights['ml_trading'] = 0
            weights['news'] += transferred_weight * 0.6
            weights['events'] += transferred_weight * 0.4
        
        # Calculate component scores
        news_score = news_sentiment.get('average_sentiment', 0) if isinstance(news_sentiment, dict) else 0
        reddit_score = reddit_sentiment.get('average_sentiment', 0)
        
        # Enhanced event scoring with decay
        events_score = self._calculate_event_impact_score(events)
        
        # Add volume-weighted sentiment
        volume_sentiment = self._calculate_volume_weighted_sentiment(news_sentiment)
        
        # Add momentum factor
        momentum_score = self._calculate_sentiment_momentum(news_sentiment)
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate weighted score
        overall = (
            news_score * normalized_weights['news'] +
            reddit_score * normalized_weights['reddit'] +
            events_score * normalized_weights['events'] +
            volume_sentiment * normalized_weights['volume'] +
            momentum_score * normalized_weights['momentum'] +
            ml_score * normalized_weights['ml_trading']
        )
        
        # Apply market context modifier
        market_modifier = market_context.get('volatility_factor', 1.0)
        overall *= market_modifier
        
        # Calculate confidence factor based on data quality including transformer and ML confidence
        confidence = self._calculate_confidence_factor(
            news_count, 
            reddit_posts, 
            len(events.get('events_detected', [])),
            transformer_confidence,
            ml_confidence  # Add ML confidence to overall confidence calculation
        )
        
        # Apply confidence adjustment (less aggressive than full multiplication)
        confidence_adjusted = overall * (0.7 + 0.3 * confidence)
        
        return {
            'score': max(-1, min(1, confidence_adjusted)),
            'components': {
                'news': news_score * normalized_weights['news'],
                'reddit': reddit_score * normalized_weights['reddit'],
                'events': events_score * normalized_weights['events'],
                'volume': volume_sentiment * normalized_weights['volume'],
                'momentum': momentum_score * normalized_weights['momentum'],
                'ml_trading': ml_score * normalized_weights['ml_trading']
            },
            'weights': normalized_weights,
            'confidence': confidence,
            'market_modifier': market_modifier,
            'ml_trading_details': ml_trading_result.get('feature_analysis', {}),
            'transformer_confidence': transformer_confidence,
            'ml_confidence': ml_confidence
        }
    
    def _calculate_event_impact_score(self, events: Dict) -> float:
        """Calculate event impact with time decay and severity weighting"""
        
        event_weights = {
            'dividend_announcement': {'base': 0.3, 'decay': 0.9},
            'earnings_report': {'base': 0.4, 'decay': 0.8},
            'scandal_investigation': {'base': -0.6, 'decay': 0.7},
            'regulatory_news': {'base': -0.4, 'decay': 0.85},
            'merger_acquisition': {'base': 0.35, 'decay': 0.95},
            'rating_change': {'base': 0.3, 'decay': 0.9},
            'management_change': {'base': -0.2, 'decay': 0.8},
            'capital_raising': {'base': -0.2, 'decay': 0.85},
            'branch_closure': {'base': -0.1, 'decay': 0.9},
            'product_launch': {'base': 0.1, 'decay': 0.95},
            'partnership_deal': {'base': 0.1, 'decay': 0.95},
            'legal_action': {'base': -0.3, 'decay': 0.8}
        }
        
        total_score = 0
        events_detected = events.get('events_detected', [])
        
        for event in events_detected:
            event_type = event['type']
            if event_type in event_weights:
                # Calculate time decay
                try:
                    event_date = datetime.fromisoformat(event['date'].replace('Z', '+00:00'))
                    days_old = (datetime.now() - event_date).days
                    decay_factor = event_weights[event_type]['decay'] ** days_old
                except:
                    decay_factor = 1.0  # No decay if date parsing fails
                
                # Apply sentiment impact from context
                base_score = event_weights[event_type]['base']
                context_modifier = event.get('sentiment_impact', 1.0)
                
                # Apply relevance modifier
                relevance = event.get('relevance', 'medium')
                relevance_modifier = {'high': 1.2, 'medium': 1.0, 'low': 0.7}.get(relevance, 1.0)
                
                score = base_score * decay_factor * context_modifier * relevance_modifier
                total_score += score
        
        # Apply diminishing returns for multiple events
        if len(events_detected) > 3:
            total_score *= (1 - 0.1 * (len(events_detected) - 3))
        
        return max(-1, min(1, total_score))
    
    def _calculate_volume_weighted_sentiment(self, news_sentiment: Dict) -> float:
        """Calculate sentiment weighted by news volume patterns"""
        
        if not isinstance(news_sentiment, dict):
            return 0
        
        news_count = news_sentiment.get('news_count', 0)
        avg_sentiment = news_sentiment.get('average_sentiment', 0)
        
        # Sentiment distribution
        distribution = news_sentiment.get('sentiment_distribution', {})
        
        # Calculate concentration (how unanimous the sentiment is)
        total_items = sum(distribution.values()) if distribution else 1
        if total_items == 0:
            return avg_sentiment
        
        # Calculate entropy (diversity of opinions)
        entropy = 0
        for count in distribution.values():
            if count > 0:
                p = count / total_items
                entropy -= p * (p if p > 0 else 1)  # Avoid log(0)
        
        # High volume with low entropy (unanimous sentiment) = stronger signal
        volume_factor = min(news_count / 10, 1.0)  # Normalize to 0-1
        unanimity_factor = 1 - entropy  # Higher when sentiment is unanimous
        
        # Weight the sentiment by volume and unanimity
        weighted_sentiment = avg_sentiment * volume_factor * (0.7 + 0.3 * unanimity_factor)
        
        return weighted_sentiment
    
    def _calculate_sentiment_momentum(self, news_sentiment: Dict) -> float:
        """Calculate momentum based on sentiment trends"""
        
        try:
            # Get recent sentiment history
            history = self.history_manager.load_sentiment_history(news_sentiment.get('symbol', ''))
            
            if len(history) < 3:
                return 0
            
            # Get last 7 days of sentiment
            recent = history[-7:] if len(history) >= 7 else history
            
            # Calculate moving averages
            if len(recent) >= 3:
                ma3 = sum(h['overall_sentiment'] for h in recent[-3:]) / 3
                ma7 = sum(h['overall_sentiment'] for h in recent) / len(recent)
                
                # Momentum is difference between short and long MA
                momentum = ma3 - ma7
                
                # Scale momentum to reasonable range
                return max(-0.5, min(0.5, momentum * 2))
            
            return 0
            
        except Exception as e:
            logger.warning(f"Error calculating sentiment momentum: {e}")
            return 0
    
    def _calculate_confidence_factor(self, news_count: int, reddit_posts: int, event_count: int, 
                                   transformer_confidence: float = 0, ml_confidence: float = 0) -> float:
        """Calculate confidence in the sentiment analysis including transformer and ML performance"""
        
        # Base confidence
        confidence = 0.5
        
        # News contribution (up to 0.25 to make room for transformer boost)
        if news_count >= 10:
            confidence += 0.25
        elif news_count >= 5:
            confidence += 0.18
        elif news_count >= 2:
            confidence += 0.1
        
        # Reddit contribution (up to 0.15)
        if reddit_posts >= 10:
            confidence += 0.15
        elif reddit_posts >= 5:
            confidence += 0.1
        elif reddit_posts >= 2:
            confidence += 0.05
        
        # Event contribution (up to 0.05)
        if event_count >= 3:
            confidence += 0.05
        elif event_count >= 1:
            confidence += 0.03
        
        # Transformer contribution (up to 0.12 - reduced to make room for ML)
        if transformer_confidence > 0:
            # High confidence transformers significantly boost overall confidence
            if transformer_confidence > 0.9:
                confidence += 0.12
            elif transformer_confidence > 0.8:
                confidence += 0.09
            elif transformer_confidence > 0.7:
                confidence += 0.06
            elif transformer_confidence > 0.6:
                confidence += 0.04
            elif transformer_confidence > 0.5:
                confidence += 0.02
        
        # ML Trading contribution (up to 0.08 - new addition)
        if ml_confidence > 0:
            # ML trading features boost confidence
            if ml_confidence > 0.9:
                confidence += 0.08
            elif ml_confidence > 0.8:
                confidence += 0.06
            elif ml_confidence > 0.7:
                confidence += 0.04
            elif ml_confidence > 0.6:
                confidence += 0.03
            elif ml_confidence > 0.5:
                confidence += 0.01
        
        return min(1.0, confidence)
    
    def _get_market_context(self) -> Dict:
        """Get market context for sentiment adjustment"""
        
        try:
            # This would ideally fetch real market data
            # For now, return default context
            return {
                'volatility_factor': 1.0,  # Would be calculated from VIX or similar
                'market_trend': 'neutral',
                'sector_momentum': 0
            }
        except Exception as e:
            logger.warning(f"Error getting market context: {e}")
            return {'volatility_factor': 1.0}
    
    def _fetch_rss_news(self, symbol: str) -> List[Dict]:
        """Fetch news from RSS feeds"""
        news_items = []
        keywords = self.bank_keywords.get(symbol, [symbol.replace('.AX', '')])
        
        for feed_name, feed_url in self.settings.NEWS_SOURCES['rss_feeds'].items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:20]:  # Check last 20 entries
                    # Check if any keyword is in title or summary
                    title = entry.get('title', '').lower()
                    summary = entry.get('summary', '').lower()
                    
                    if any(keyword.lower() in title or keyword.lower() in summary 
                          for keyword in keywords):
                        
                        # Parse date
                        published = entry.get('published_parsed', None)
                        if published:
                            pub_date = datetime.fromtimestamp(time.mktime(published))
                        else:
                            pub_date = datetime.now()
                        
                        # Only include recent news (last 7 days)
                        if pub_date > datetime.now() - timedelta(days=7):
                            news_items.append({
                                'title': entry.get('title', ''),
                                'summary': entry.get('summary', ''),
                                'source': feed_name,
                                'url': entry.get('link', ''),
                                'published': pub_date.isoformat(),
                                'relevance': 'high' if symbol.replace('.AX', '') in title else 'medium'
                            })
            
            except Exception as e:
                logger.warning(f"Error fetching RSS feed {feed_name}: {str(e)}")
                continue
        
        return news_items
    
    def _fetch_yahoo_news(self, symbol: str) -> List[Dict]:
        """Fetch news from Yahoo Finance"""
        news_items = []
        
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            for item in news:
                # Parse timestamp
                pub_timestamp = item.get('providerPublishTime')
                if pub_timestamp:
                    pub_date = datetime.fromtimestamp(pub_timestamp)
                else:
                    pub_date = datetime.now()
                
                # Only include recent news (last 7 days)
                if pub_date > datetime.now() - timedelta(days=7):
                    news_items.append({
                        'title': item.get('title', ''),
                        'summary': item.get('summary', ''),
                        'source': 'Yahoo Finance',
                        'url': item.get('link', ''),
                        'published': pub_date.isoformat(),
                        'relevance': 'high'  # Yahoo Finance news is usually relevant
                    })
        
        except Exception as e:
            logger.warning(f"Error fetching Yahoo Finance news for {symbol}: {str(e)}")
        
        return news_items
    
    def _scrape_news_sites(self, symbol: str) -> List[Dict]:
        """Scrape news from financial websites"""
        news_items = []
        keywords = self.bank_keywords.get(symbol, [symbol.replace('.AX', '')])
        
        # Example: Scrape Google News (be careful with rate limiting)
        try:
            query = '+'.join(keywords) + '+Australia+bank'
            url = f"https://news.google.com/search?q={query}&hl=en-AU&gl=AU&ceid=AU:en"
            
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find news articles (selectors may change)
                articles = soup.find_all('article')[:10]
                
                for article in articles:
                    title_elem = article.find('a', class_='JtKRv')
                    if title_elem:
                        title = title_elem.get_text()
                        
                        # Basic relevance check
                        if any(keyword.lower() in title.lower() for keyword in keywords):
                            news_items.append({
                                'title': title,
                                'summary': '',
                                'source': 'Google News',
                                'url': '',
                                'published': datetime.now().isoformat(),
                                'relevance': 'medium'
                            })
            
            # Add delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.warning(f"Error scraping Google News: {str(e)}")
        
        return news_items
    
    def _get_reddit_sentiment(self, symbol: str) -> Dict:
        """Get sentiment from Reddit financial subreddits"""
        try:
            keywords = self.bank_keywords.get(symbol, [symbol.replace('.AX', '')])
            
            reddit_data = {
                'posts_analyzed': 0,
                'average_sentiment': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'top_posts': [],
                'sentiment_distribution': {},
                'subreddit_breakdown': {}
            }
            
            if not self.reddit:
                logger.warning("Reddit client not available, using fallback")
                return reddit_data
            
            # Search across financial subreddits
            all_posts = []
            
            for subreddit_name in self.financial_subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search for posts mentioning the bank
                    for keyword in keywords:
                        try:
                            # Search recent posts (last week)
                            posts = subreddit.search(
                                keyword, 
                                sort='new', 
                                time_filter='week',
                                limit=20
                            )
                            
                            for post in posts:
                                # Check if post is recent (last 7 days)
                                post_date = datetime.fromtimestamp(post.created_utc)
                                if post_date > datetime.now() - timedelta(days=7):
                                    all_posts.append({
                                        'title': post.title,
                                        'selftext': post.selftext,
                                        'score': post.score,
                                        'num_comments': post.num_comments,
                                        'created': post_date,
                                        'subreddit': subreddit_name,
                                        'upvote_ratio': post.upvote_ratio,
                                        'url': post.url
                                    })
                        except Exception as e:
                            logger.warning(f"Error searching {subreddit_name} for {keyword}: {e}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"Error accessing subreddit {subreddit_name}: {e}")
                    continue
            
            # Analyze sentiment of collected posts
            if all_posts:
                sentiments = []
                subreddit_sentiments = {}
                
                for post in all_posts:
                    # Combine title and text for analysis
                    text = f"{post['title']} {post['selftext']}"
                    
                    # Analyze sentiment
                    blob = TextBlob(text)
                    vader_scores = self.vader.polarity_scores(text)
                    
                    # Weighted sentiment (TextBlob + VADER + Reddit metrics)
                    textblob_sentiment = blob.sentiment.polarity
                    vader_sentiment = vader_scores['compound']
                    
                    # Weight by Reddit engagement (upvotes, comments)
                    engagement_weight = min(1.0, (post['score'] + post['num_comments']) / 100)
                    upvote_influence = (post['upvote_ratio'] - 0.5) * 0.5  # Convert 0-1 to -0.25 to +0.25
                    
                    combined_sentiment = (textblob_sentiment + vader_sentiment) / 2
                    weighted_sentiment = combined_sentiment * engagement_weight + upvote_influence
                    
                    # Clamp to -1 to 1
                    weighted_sentiment = max(-1, min(1, weighted_sentiment))
                    
                    sentiments.append(weighted_sentiment)
                    
                    # Track by subreddit
                    if post['subreddit'] not in subreddit_sentiments:
                        subreddit_sentiments[post['subreddit']] = []
                    subreddit_sentiments[post['subreddit']].append(weighted_sentiment)
                    
                    # Categorize
                    if weighted_sentiment > 0.1:
                        reddit_data['bullish_count'] += 1
                    elif weighted_sentiment < -0.1:
                        reddit_data['bearish_count'] += 1
                    else:
                        reddit_data['neutral_count'] += 1
                
                # Calculate averages
                reddit_data['posts_analyzed'] = len(all_posts)
                reddit_data['average_sentiment'] = sum(sentiments) / len(sentiments)
                
                # Top posts by engagement
                reddit_data['top_posts'] = sorted(
                    all_posts, 
                    key=lambda x: x['score'] + x['num_comments'], 
                    reverse=True
                )[:5]
                
                # Sentiment distribution
                reddit_data['sentiment_distribution'] = {
                    'very_positive': sum(1 for s in sentiments if s > 0.5),
                    'positive': sum(1 for s in sentiments if 0.1 < s <= 0.5),
                    'neutral': sum(1 for s in sentiments if -0.1 <= s <= 0.1),
                    'negative': sum(1 for s in sentiments if -0.5 <= s < -0.1),
                    'very_negative': sum(1 for s in sentiments if s < -0.5)
                }
                
                # Subreddit breakdown
                for subreddit, sents in subreddit_sentiments.items():
                    reddit_data['subreddit_breakdown'][subreddit] = {
                        'posts': len(sents),
                        'average_sentiment': sum(sents) / len(sents),
                        'bullish': sum(1 for s in sents if s > 0.1),
                        'bearish': sum(1 for s in sents if s < -0.1)
                    }
            
            return reddit_data
            
        except Exception as e:
            logger.warning(f"Error fetching Reddit sentiment: {str(e)}")
            return {
                'posts_analyzed': 0,
                'average_sentiment': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'error': str(e)
            }
    
    def _analyze_news_sentiment(self, news_items: List[Dict]) -> Dict:
        """Analyze sentiment of news articles using multiple methods including transformers"""
        
        if not news_items:
            return {
                'average_sentiment': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'sentiment_distribution': {},
                'news_count': 0,
                'method_breakdown': {}
            }
        
        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        method_results = {
            'traditional': [],
            'transformer': [],
            'composite': []
        }
        
        for news in news_items:
            # Combine title and summary for analysis
            text = f"{news['title']} {news.get('summary', '')}"
            
            # Method 1: Traditional approach (TextBlob + VADER)
            traditional_sentiment = self._analyze_traditional_sentiment(text)
            method_results['traditional'].append(traditional_sentiment)
            
            # Method 2: Transformer approach (if available)
            transformer_sentiment = 0
            transformer_confidence = 0
            transformer_details = {}
            
            if self.transformer_models:
                transformer_sentiment = self._analyze_with_transformers(text)
                if 'composite' in transformer_sentiment:
                    transformer_sentiment = transformer_sentiment['composite']['score']
                    transformer_confidence = transformer_sentiment['composite']['confidence']
                    transformer_details = transformer_sentiment
                method_results['transformer'].append(transformer_sentiment)
            else:
                method_results['transformer'].append(0)
            
            # Method 3: Composite approach (weighted combination)
            composite_sentiment = self._calculate_composite_sentiment(
                traditional_sentiment, 
                transformer_sentiment, 
                transformer_confidence,
                news['relevance']
            )
            method_results['composite'].append(composite_sentiment)
            
            # Store the composite sentiment for this news item
            sentiments.append(composite_sentiment)
            
            # Store additional analysis details
            news['sentiment_analysis'] = {
                'traditional': traditional_sentiment,
                'transformer': transformer_sentiment,
                'transformer_confidence': transformer_confidence,
                'transformer_details': transformer_details,
                'composite': composite_sentiment
            }
            
            # Categorize based on composite sentiment
            if composite_sentiment > 0.1:
                positive_count += 1
            elif composite_sentiment < -0.1:
                negative_count += 1
            else:
                neutral_count += 1
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        # Calculate method performance comparison
        method_breakdown = self._analyze_method_performance(method_results)
        
        return {
            'average_sentiment': avg_sentiment,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'sentiment_distribution': {
                'very_positive': sum(1 for s in sentiments if s > 0.5),
                'positive': sum(1 for s in sentiments if 0.1 < s <= 0.5),
                'neutral': sum(1 for s in sentiments if -0.1 <= s <= 0.1),
                'negative': sum(1 for s in sentiments if -0.5 <= s < -0.1),
                'very_negative': sum(1 for s in sentiments if s < -0.5)
            },
            'strongest_sentiment': max(sentiments, key=abs) if sentiments else 0,
            'news_count': len(news_items),
            'method_breakdown': method_breakdown,
            'transformer_available': bool(self.transformer_models)
        }
    
    def _analyze_traditional_sentiment(self, text: str) -> float:
        """Analyze sentiment using traditional methods (TextBlob + VADER)"""
        # TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        compound = vader_scores['compound']  # -1 to 1
        
        # Average the two methods
        return (polarity + compound) / 2

    def _analyze_with_transformers(self, text: str) -> Dict:
        """Analyze sentiment using transformer models"""
        if not self.transformer_models:
            return {
                'composite': {'score': 0, 'confidence': 0},
                'models': {},
                'error': 'No transformer models available'
            }
        
        try:
            model_results = {}
            scores = []
            confidences = []
            
            # Truncate text if too long for transformer models (max 512 tokens typically)
            max_length = 500  # Conservative limit
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            # Analyze with each available model
            for model_name, model in self.transformer_models.items():
                try:
                    result = model(text)
                    
                    # Convert model output to standardized format
                    if isinstance(result, list) and len(result) > 0:
                        # Handle different model output formats
                        if model_name == 'financial':  # FinBERT
                            # FinBERT outputs: positive, negative, neutral
                            sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
                            max_result = max(result, key=lambda x: x['score'])
                            score = sentiment_map.get(max_result['label'].lower(), 0)
                            confidence = max_result['score']
                        elif model_name == 'general':  # RoBERTa
                            # RoBERTa outputs: LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
                            sentiment_map = {'LABEL_0': -1, 'LABEL_1': 0, 'LABEL_2': 1}
                            max_result = max(result, key=lambda x: x['score'])
                            score = sentiment_map.get(max_result['label'], 0)
                            confidence = max_result['score']
                        else:  # Basic model
                            # Basic model outputs: NEGATIVE, POSITIVE
                            sentiment_map = {'NEGATIVE': -1, 'POSITIVE': 1}
                            max_result = max(result, key=lambda x: x['score'])
                            score = sentiment_map.get(max_result['label'], 0)
                            confidence = max_result['score']
                        
                        model_results[model_name] = {
                            'score': score,
                            'confidence': confidence,
                            'label': max_result['label'],
                            'raw_result': result
                        }
                        
                        scores.append(score)
                        confidences.append(confidence)
                        
                except Exception as e:
                    logger.warning(f"Error with transformer model {model_name}: {e}")
                    model_results[model_name] = {'error': str(e)}
            
            # Calculate composite score from all models
            if scores:
                # Weight by confidence and prioritize financial models
                weighted_scores = []
                total_weight = 0
                
                for model_name, result in model_results.items():
                    if 'score' in result:
                        # Financial models get higher weight
                        model_weight = 2.0 if model_name == 'financial' else 1.0
                        confidence_weight = result['confidence']
                        total_weight += model_weight * confidence_weight
                        weighted_scores.append(result['score'] * model_weight * confidence_weight)
                
                composite_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0
                composite_confidence = sum(confidences) / len(confidences)
            else:
                composite_score = 0
                composite_confidence = 0
            
            return {
                'composite': {
                    'score': composite_score,
                    'confidence': composite_confidence
                },
                'models': model_results,
                'text_length': len(text),
                'models_used': len([r for r in model_results.values() if 'score' in r])
            }
            
        except Exception as e:
            logger.error(f"Error in transformer sentiment analysis: {e}")
            return {
                'composite': {'score': 0, 'confidence': 0},
                'models': {},
                'error': str(e)
            }
