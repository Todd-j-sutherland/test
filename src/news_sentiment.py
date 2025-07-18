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

# Import ML pipeline at the top
from src.ml_training_pipeline import MLTrainingPipeline

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

# Import enhanced keyword system
from src.bank_keywords import BankNewsFilter

logger = logging.getLogger(__name__)

# Enhanced Sentiment Integration
try:
    from src.enhanced_sentiment_integration import SentimentIntegrationManager, enhance_existing_sentiment
    ENHANCED_SENTIMENT_AVAILABLE = True
    logger.info("Enhanced sentiment integration loaded successfully")
except ImportError:
    ENHANCED_SENTIMENT_AVAILABLE = False
    logger.warning("Enhanced sentiment integration not available")

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
        
        # Initialize ML training pipeline
        self.ml_pipeline = MLTrainingPipeline()
        
        # Load latest trained model
        self.ml_model = self._load_ml_model()
        # Note: ml_feature_columns and ml_threshold are set in _load_ml_model()
        if not hasattr(self, 'ml_feature_columns'):
            self.ml_feature_columns = []
        if not hasattr(self, 'ml_threshold'):
            self.ml_threshold = 0.5
        
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
        
        # Initialize enhanced keyword filtering system
        self.news_filter = BankNewsFilter()
        
        # Bank name variations for searching (enhanced)
        self.bank_keywords = self.news_filter.bank_keywords
        
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
        
        # Initialize enhanced sentiment integration
        self.enhanced_integration = None
        if ENHANCED_SENTIMENT_AVAILABLE:
            try:
                self.enhanced_integration = SentimentIntegrationManager()
                logger.info("✅ Enhanced sentiment integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced sentiment integration: {e}")
                self.enhanced_integration = None

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
    
    def _load_ml_model(self):
        """Load the latest trained ML model"""
        try:
            import joblib
            import json
            
            model_path = os.path.join("data/ml_models/models", "current_model.pkl")
            metadata_path = os.path.join("data/ml_models/models", "current_metadata.json")
            
            if os.path.exists(model_path) and os.path.exists(metadata_path):
                model = joblib.load(model_path)
                
                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.ml_feature_columns = metadata.get('feature_columns', [])
                # Set threshold from metadata if available
                performance = metadata.get('performance', {})
                self.ml_threshold = performance.get('best_threshold', 0.5)
                
                logger.info(f"Loaded ML model: {metadata.get('model_type', 'unknown')} with {len(self.ml_feature_columns)} features")
                return model
            else:
                logger.warning("No trained ML model found")
                return None
                
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            return None
    
    def _get_ml_prediction(self, sentiment_data: Dict) -> Dict:
        """Get ML prediction from sentiment data"""
        try:
            if not self.ml_model or not hasattr(self, 'ml_feature_columns'):
                return {'prediction': 'HOLD', 'confidence': 0.0, 'error': 'No model available'}
            
            import pandas as pd
            import joblib
            
            # Load feature scaler
            scaler_path = os.path.join("data/ml_models/models", "feature_scaler.pkl")
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            else:
                scaler = None
            
            # Extract features from sentiment data
            features = {
                'sentiment_score': float(sentiment_data.get('overall_sentiment', 0)) if not isinstance(sentiment_data.get('overall_sentiment', 0), dict) else 0,
                'confidence': float(sentiment_data.get('confidence', 0.5)) if not isinstance(sentiment_data.get('confidence', 0.5), dict) else 0.5,
                'news_count': int(sentiment_data.get('news_count', 0)) if not isinstance(sentiment_data.get('news_count', 0), dict) else 0,
                'reddit_sentiment': float(sentiment_data.get('reddit_sentiment', 0)) if not isinstance(sentiment_data.get('reddit_sentiment', 0), dict) else 0,
                'event_score': float(sentiment_data.get('event_score', 0)) if not isinstance(sentiment_data.get('event_score', 0), dict) else 0,
                'sentiment_confidence_interaction': 0,  # Calculate after
                'news_volume_category': 0,  # Calculate after
                'hour': datetime.now().hour,
                'day_of_week': datetime.now().weekday(),
                'is_market_hours': 1 if 10 <= datetime.now().hour <= 16 else 0
            }
            
            # Calculate derived features
            features['sentiment_confidence_interaction'] = features['sentiment_score'] * features['confidence']
            features['news_volume_category'] = 1 if features['news_count'] > 5 else 0
            
            # Create feature vector in correct order
            feature_vector = []
            for col in self.ml_feature_columns:
                feature_vector.append(features.get(col, 0))
            
            # Convert to DataFrame for prediction
            X = pd.DataFrame([feature_vector], columns=self.ml_feature_columns)
            
            # Scale features if scaler available
            if scaler:
                X = scaler.transform(X)
            
            # Get prediction and probability
            prediction = self.ml_model.predict(X)[0]
            try:
                proba = self.ml_model.predict_proba(X)[0]
                confidence = max(proba)
            except:
                confidence = 0.6  # Default confidence
            
            # Convert prediction to signal
            signal_map = {0: 'SELL', 1: 'BUY'}
            signal = signal_map.get(prediction, 'HOLD')
            
            return {
                'prediction': signal,
                'confidence': confidence,
                'ml_score': confidence if signal == 'BUY' else -confidence if signal == 'SELL' else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting ML prediction: {e}")
            return {'prediction': 'HOLD', 'confidence': 0.0, 'error': str(e)}
    
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
            
            # Original web scraping (Google News)
            scraped_news = self._scrape_news_sites(symbol)
            all_news.extend(scraped_news)
            
            # Additional Australian news sources
            abc_news = self._scrape_abc_news(symbol)
            all_news.extend(abc_news)
            
            news_com_au = self._scrape_news_com_au(symbol)
            all_news.extend(news_com_au)
            
            motley_fool = self._scrape_motley_fool_au(symbol)
            all_news.extend(motley_fool)
            
            market_online = self._scrape_market_online(symbol)
            all_news.extend(market_online)
            
            investing_au = self._scrape_investing_au(symbol)
            all_news.extend(investing_au)
            
            # Industry and regulatory news (applies to all banks)
            aba_news = self._scrape_aba_news(symbol)
            all_news.extend(aba_news)
            
            # Official ASX announcements
            asx_announcements = self._scrape_asx_announcements(symbol)
            all_news.extend(asx_announcements)
            
            logger.info(f"Collected {len(all_news)} news articles from {7} sources for {symbol}")
            
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
            
            # Add trend analysis
            trend_data = self.history_manager.get_sentiment_trend(symbol, days=7)
            result['trend_analysis'] = trend_data
            
            # Add impact correlation analysis (if sufficient data)
            if trend_data['data_points'] >= 5:
                impact_analysis = self.impact_analyzer.analyze_sentiment_price_correlation(symbol, days=14)
                result['impact_analysis'] = impact_analysis
            
            # Store data for ML training
            if result and self.ml_pipeline:
                feature_id = self.ml_pipeline.collect_training_data(result, symbol)
                result['ml_feature_id'] = feature_id
            
            # Add ML prediction if model is available
            if self.ml_model:
                ml_prediction = self._get_ml_prediction(result)
                result['ml_prediction'] = ml_prediction
            
            # Apply Enhanced Sentiment Analysis if available
            if self.enhanced_integration:
                try:
                    enhanced_metrics = self.enhanced_integration.convert_legacy_to_enhanced(result)
                    enhanced_signals = self.enhanced_integration.generate_enhanced_trading_signals(result)
                    
                    # Add enhanced data to result
                    result['enhanced_sentiment'] = {
                        'normalized_score': enhanced_metrics.normalized_score,
                        'strength_category': enhanced_metrics.strength_category.name,
                        'confidence': enhanced_metrics.confidence,
                        'z_score': enhanced_metrics.z_score,
                        'percentile_rank': enhanced_metrics.percentile_rank,
                        'volatility_adjusted_score': enhanced_metrics.volatility_adjusted_score,
                        'market_adjusted_score': enhanced_metrics.market_adjusted_score,
                        'trading_signals': enhanced_signals,
                        'enhancement_summary': enhanced_signals['enhanced_analysis']['improvement_over_legacy']['enhancement_summary']
                    }
                    
                    logger.info(f"Enhanced sentiment analysis applied for {symbol}: Score {enhanced_metrics.normalized_score:.1f}/100, Strength: {enhanced_metrics.strength_category.name}")
                    
                except Exception as e:
                    logger.warning(f"Enhanced sentiment analysis failed for {symbol}: {e}")
                    result['enhanced_sentiment'] = None
            
            # Store in historical data (after all enhancements are added)
            self.history_manager.store_sentiment(symbol, result)
            
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
    
    def _get_market_data_for_ml(self) -> Dict:
        """Get market data for ML feature engineering"""
        try:
            # For now, return basic market context
            # This could be enhanced to fetch real market data
            return {
                'market_volatility': 0.5,  # Placeholder volatility measure
                'market_trend': 'neutral',
                'trading_volume': 1.0,     # Normalized volume
                'market_hours': self._is_market_hours(),
                'day_of_week': datetime.now().weekday()
            }
        except Exception as e:
            logger.warning(f"Error getting market data for ML: {e}")
            return {}
    
    def _calculate_ml_trading_score(self, news_items: List[Dict], market_data: Dict) -> Dict:
        """Calculate ML trading score based on news and market data"""
        try:
            if not news_items:
                return {'ml_score': 0, 'confidence': 0}
            
            # Extract features from news
            total_sentiment = 0
            sentiment_variance = 0
            news_count = len(news_items)
            
            # Calculate sentiment statistics
            sentiments = []
            for item in news_items:
                if 'sentiment_analysis' in item:
                    sentiment = item['sentiment_analysis'].get('composite', 0)
                    sentiments.append(sentiment)
                    total_sentiment += sentiment
            
            if sentiments:
                avg_sentiment = total_sentiment / len(sentiments)
                sentiment_variance = sum((s - avg_sentiment) ** 2 for s in sentiments) / len(sentiments)
            else:
                avg_sentiment = 0
            
            # Calculate ML score based on features
            ml_score = avg_sentiment
            
            # Adjust for market conditions
            market_factor = market_data.get('market_volatility', 0.5)
            if market_data.get('market_trend') == 'bullish':
                ml_score *= 1.1
            elif market_data.get('market_trend') == 'bearish':
                ml_score *= 0.9
            
            # Calculate confidence based on news volume and sentiment consistency
            base_confidence = min(0.9, news_count / 10.0) # More news = higher confidence
            variance_penalty = min(0.3, sentiment_variance)  # High variance = lower confidence
            confidence = max(0.1, base_confidence - variance_penalty)
            
            return {
                'ml_score': float(ml_score),
                'confidence': float(confidence),
                'features': {
                    'avg_sentiment': avg_sentiment,
                    'sentiment_variance': sentiment_variance,
                    'news_count': news_count,
                    'market_factor': market_factor
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating ML trading score: {e}")
            return {'ml_score': 0, 'confidence': 0}
    
    def _is_market_hours(self) -> bool:
        """Check if it's currently market hours (AEST)"""
        try:
            now = datetime.now()
            # ASX market hours: 10:00 AM - 4:00 PM AEST (Monday-Friday)
            if now.weekday() >= 5:  # Weekend
                return False
            return 10 <= now.hour < 16
        except:
            return False
    
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
        
        # Scrape additional Australian news sources
        try:
            abc_news = self._scrape_abc_news(symbol)
            news_items.extend(abc_news)
        except Exception as e:
            logger.warning(f"Error scraping ABC News: {str(e)}")
        
        try:
            news_com_au_news = self._scrape_news_com_au(symbol)
            news_items.extend(news_com_au_news)
        except Exception as e:
            logger.warning(f"Error scraping News.com.au: {str(e)}")
        
        try:
            motley_fool_au_news = self._scrape_motley_fool_au(symbol)
            news_items.extend(motley_fool_au_news)
        except Exception as e:
            logger.warning(f"Error scraping Motley Fool AU: {str(e)}")
        
        try:
            market_online_news = self._scrape_market_online(symbol)
            news_items.extend(market_online_news)
        except Exception as e:
            logger.warning(f"Error scraping The Market Online: {str(e)}")
        
        try:
            investing_au_news = self._scrape_investing_au(symbol)
            news_items.extend(investing_au_news)
        except Exception as e:
            logger.warning(f"Error scraping Investing.com AU: {str(e)}")
        
        try:
            aba_news = self._scrape_aba_news(symbol)
            news_items.extend(aba_news)
        except Exception as e:
            logger.warning(f"Error scraping ABA news: {str(e)}")
        
        try:
            asx_announcements = self._scrape_asx_announcements(symbol)
            news_items.extend(asx_announcements)
        except Exception as e:
            logger.warning(f"Error scraping ASX announcements: {str(e)}")
        
        return news_items
    
    def _scrape_abc_news(self, symbol: str) -> List[Dict]:
        """Scrape ABC News business section"""
        news_items = []
        keywords = self.bank_keywords.get(symbol, [symbol.replace('.AX', '')])
        
        try:
            # ABC News Business section
            url = "https://www.abc.net.au/news/business/"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article links
                articles = soup.find_all('a', class_='_2EXSs _3a_TV _3DJPT')[:15]
                
                for article in articles:
                    title = article.get_text(strip=True)
                    article_url = article.get('href', '')
                    
                    # Make URL absolute if relative
                    if article_url.startswith('/'):
                        article_url = f"https://www.abc.net.au{article_url}"
                    
                    # Check relevance using enhanced filtering
                    filter_result = self.news_filter.is_relevant_banking_news(title, bank_symbol=symbol)
                    
                    if filter_result['is_relevant'] and filter_result['relevance_score'] >= 0.3:
                        news_items.append({
                            'title': title,
                            'summary': '',
                            'source': 'ABC News',
                            'url': article_url,
                            'published': datetime.now().isoformat(),
                            'relevance': self._calculate_relevance(title, symbol=symbol),
                            'relevance_score': filter_result['relevance_score'],
                            'matched_keywords': filter_result['matched_keywords'],
                            'categories': filter_result['categories'],
                            'urgency_score': filter_result['urgency_score']
                        })
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            logger.warning(f"Error scraping ABC News: {str(e)}")
        
        return news_items
    
    def _scrape_news_com_au(self, symbol: str) -> List[Dict]:
        """Scrape News.com.au finance section"""
        news_items = []
        keywords = self.bank_keywords.get(symbol, [symbol.replace('.AX', '')])
        
        try:
            url = "https://www.news.com.au/finance"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article headlines
                articles = soup.find_all('h3', class_='story-headline')[:15]
                
                for article in articles:
                    title_link = article.find('a')
                    if title_link:
                        title = title_link.get_text(strip=True)
                        article_url = title_link.get('href', '')
                        
                        # Make URL absolute if relative
                        if article_url.startswith('/'):
                            article_url = f"https://www.news.com.au{article_url}"
                        
                        # Check relevance using enhanced filtering
                        filter_result = self.news_filter.is_relevant_banking_news(title, bank_symbol=symbol)
                        
                        if filter_result['is_relevant'] and filter_result['relevance_score'] >= 0.3:
                            news_items.append({
                                'title': title,
                                'summary': '',
                                'source': 'News.com.au',
                                'url': article_url,
                                'published': datetime.now().isoformat(),
                                'relevance': self._calculate_relevance(title, symbol=symbol),
                                'relevance_score': filter_result['relevance_score'],
                                'matched_keywords': filter_result['matched_keywords'],
                                'categories': filter_result['categories'],
                                'urgency_score': filter_result['urgency_score']
                            })
            
            time.sleep(1)
            
        except Exception as e:
            logger.warning(f"Error scraping News.com.au: {str(e)}")
        
        return news_items
    
    def _scrape_motley_fool_au(self, symbol: str) -> List[Dict]:
        """Scrape Motley Fool Australia"""
        news_items = []
        keywords = self.bank_keywords.get(symbol, [symbol.replace('.AX', '')])
        
        try:
            url = "https://www.fool.com.au/"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article links
                articles = soup.find_all('a', class_='text-link')[:20]
                
                for article in articles:
                    title = article.get_text(strip=True)
                    article_url = article.get('href', '')
                    
                    # Make URL absolute if relative
                    if article_url.startswith('/'):
                        article_url = f"https://www.fool.com.au{article_url}"
                    
                    # Check relevance using enhanced filtering
                    filter_result = self.news_filter.is_relevant_banking_news(title, bank_symbol=symbol)
                    
                    if filter_result['is_relevant'] and filter_result['relevance_score'] >= 0.3:
                        news_items.append({
                            'title': title,
                            'summary': '',
                            'source': 'Motley Fool Australia',
                            'url': article_url,
                            'published': datetime.now().isoformat(),
                            'relevance': self._calculate_relevance(title, symbol=symbol),
                            'relevance_score': filter_result['relevance_score'],
                            'matched_keywords': filter_result['matched_keywords'],
                            'categories': filter_result['categories'],
                            'urgency_score': filter_result['urgency_score']
                        })
            
            time.sleep(1)
            
        except Exception as e:
            logger.warning(f"Error scraping Motley Fool AU: {str(e)}")
        
        return news_items
    
    def _scrape_market_online(self, symbol: str) -> List[Dict]:
        """Scrape The Market Online for ASX news"""
        news_items = []
        keywords = self.bank_keywords.get(symbol, [symbol.replace('.AX', '')])
        
        try:
            url = "https://themarketonline.com.au/"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find news articles
                articles = soup.find_all('h2', class_='entry-title')[:15]
                
                for article in articles:
                    title_link = article.find('a')
                    if title_link:
                        title = title_link.get_text(strip=True)
                        article_url = title_link.get('href', '')
                        
                        # Check relevance
                        if any(keyword.lower() in title.lower() for keyword in keywords + ['bank', 'ASX', 'finance']):
                            news_items.append({
                                'title': title,
                                'summary': '',
                                'source': 'The Market Online',
                                'url': article_url,
                                'published': datetime.now().isoformat(),
                                'relevance': self._calculate_relevance(title, keywords)
                            })
            
            time.sleep(1)
            
        except Exception as e:
            logger.warning(f"Error scraping The Market Online: {str(e)}")
        
        return news_items
    
    def _scrape_investing_au(self, symbol: str) -> List[Dict]:
        """Scrape Investing.com Australia"""
        news_items = []
        keywords = self.bank_keywords.get(symbol, [symbol.replace('.AX', '')])
        
        try:
            # Try to get ASX-specific news
            url = f"https://au.investing.com/search/?q={symbol}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find news articles
                articles = soup.find_all('a', class_='title')[:10]
                
                for article in articles:
                    title = article.get_text(strip=True)
                    article_url = article.get('href', '')
                    
                    # Make URL absolute if relative
                    if article_url.startswith('/'):
                        article_url = f"https://au.investing.com{article_url}"
                    
                    # Check relevance
                    if any(keyword.lower() in title.lower() for keyword in keywords + ['bank', 'ASX']):
                        news_items.append({
                            'title': title,
                            'summary': '',
                            'source': 'Investing.com Australia',
                            'url': article_url,
                            'published': datetime.now().isoformat(),
                            'relevance': self._calculate_relevance(title, keywords)
                        })
            
            time.sleep(1)
            
        except Exception as e:
            logger.warning(f"Error scraping Investing.com AU: {str(e)}")
        
        return news_items
    
    def _scrape_aba_news(self, symbol: str) -> List[Dict]:
        """Scrape Australian Banking Association news"""
        news_items = []
        
        try:
            url = "https://ausbanking.org.au/news/"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find news articles
                articles = soup.find_all('h3', class_='entry-title')[:10]
                
                for article in articles:
                    title_link = article.find('a')
                    if title_link:
                        title = title_link.get_text(strip=True)
                        article_url = title_link.get('href', '')
                        
                        # ABA news is always relevant to banking sector
                        news_items.append({
                            'title': title,
                            'summary': '',
                            'source': 'Australian Banking Association',
                            'url': article_url,
                            'published': datetime.now().isoformat(),
                            'relevance': 'high'  # Industry association news is always relevant
                        })
            
            time.sleep(1)
            
        except Exception as e:
            logger.warning(f"Error scraping ABA news: {str(e)}")
        
        return news_items

    def _scrape_asx_announcements(self, symbol: str) -> List[Dict]:
        """Scrape ASX announcements for specific symbol"""
        news_items = []
        
        try:
            # Get company announcements from ASX
            company_code = symbol.replace('.AX', '')
            url = f"https://www.asx.com.au/markets/trade-our-cash-market/todays-announcements"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find announcement rows
                rows = soup.find_all('tr')[:20]
                
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        code_cell = cells[0]
                        title_cell = cells[1]
                        
                        if code_cell and company_code.upper() in code_cell.get_text().upper():
                            title = title_cell.get_text(strip=True)
                            
                            # Enhance ASX announcements with filtering analysis
                            filter_result = self.news_filter.is_relevant_banking_news(title, bank_symbol=symbol)
                            
                            news_items.append({
                                'title': f"ASX: {title}",
                                'summary': '',
                                'source': 'ASX Announcements',
                                'url': url,
                                'published': datetime.now().isoformat(),
                                'relevance': 'high',  # Official ASX announcements are highly relevant
                                'relevance_score': max(filter_result['relevance_score'], 0.8),  # Minimum 0.8 for ASX
                                'matched_keywords': filter_result['matched_keywords'],
                                'categories': filter_result['categories'] + ['official_announcement'],
                                'urgency_score': filter_result['urgency_score']
                            })
            
            time.sleep(1)
            
        except Exception as e:
            logger.warning(f"Error scraping ASX announcements: {str(e)}")
        
        return news_items

    def _calculate_relevance(self, title: str, keywords: List[str] = None, content: str = "", symbol: str = None) -> str:
        """Calculate relevance score for news article using enhanced filtering"""
        try:
            # Use enhanced filtering system
            filter_result = self.news_filter.is_relevant_banking_news(title, content, symbol)
            
            # Convert relevance score to category
            relevance_score = filter_result['relevance_score']
            
            if relevance_score >= 0.7:
                return 'high'
            elif relevance_score >= 0.4:
                return 'medium'
            elif relevance_score >= 0.2:
                return 'low'
            else:
                return 'very_low'
                
        except Exception as e:
            # Fallback to simple keyword matching
            if keywords:
                title_lower = title.lower()
                keyword_count = sum(1 for keyword in keywords if keyword.lower() in title_lower)
                
                if keyword_count >= 2:
                    return 'high'
                elif keyword_count == 1:
                    return 'medium'
                else:
                    return 'low'
            return 'low'
    
    def _get_reddit_sentiment(self, symbol: str) -> Dict:
        """Get sentiment from Reddit financial subreddits"""
        try:
            keywords = self.bank_keywords.get(symbol, [symbol.replace('.AX', '')])
            
            if not self.reddit:
                return {
                    'posts_analyzed': 0,
                    'average_sentiment': 0,
                    'bullish_count': 0,
                    'bearish_count': 0,
                    'neutral_count': 0,
                    'top_posts': [],
                    'sentiment_distribution': {},
                    'subreddit_breakdown': {}
                }
            
            subreddits = ['ASX_Bets', 'AusFinance', 'fiaustralia', 'ASX', 'investing']
            all_posts = []
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search for posts mentioning the bank
                    for keyword in keywords:
                        search_results = subreddit.search(keyword, time_filter='week', limit=10)
                        
                        for post in search_results:
                            # Basic sentiment analysis
                            text = f"{post.title} {post.selftext}"
                            blob = TextBlob(text)
                            
                            sentiment_score = blob.sentiment.polarity
                            
                            # Categorize sentiment
                            if sentiment_score > 0.1:
                                sentiment_category = 'bullish'
                            elif sentiment_score < -0.1:
                                sentiment_category = 'bearish'
                            else:
                                sentiment_category = 'neutral'
                            
                            all_posts.append({
                                'title': post.title,
                                'score': post.score,
                                'sentiment': sentiment_score,
                                'category': sentiment_category,
                                'subreddit': subreddit_name,
                                'url': f"https://reddit.com{post.permalink}"
                            })
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Error accessing subreddit {subreddit_name}: {str(e)}")
                    continue
            
            if not all_posts:
                return {
                    'posts_analyzed': 0,
                    'average_sentiment': 0,
                    'bullish_count': 0,
                    'bearish_count': 0,
                    'neutral_count': 0,
                    'top_posts': [],
                    'sentiment_distribution': {},
                    'subreddit_breakdown': {}
                }
            
            # Calculate aggregate sentiment
            avg_sentiment = sum(post['sentiment'] for post in all_posts) / len(all_posts)
            
            # Count categories
            bullish_count = len([p for p in all_posts if p['category'] == 'bullish'])
            bearish_count = len([p for p in all_posts if p['category'] == 'bearish'])
            neutral_count = len([p for p in all_posts if p['category'] == 'neutral'])
            
            # Get top posts by score
            top_posts = sorted(all_posts, key=lambda x: x['score'], reverse=True)[:5]
            
            # Sentiment distribution
            sentiment_distribution = {
                'bullish': bullish_count / len(all_posts) * 100,
                'bearish': bearish_count / len(all_posts) * 100,
                'neutral': neutral_count / len(all_posts) * 100
            }
            
            # Subreddit breakdown
            subreddit_breakdown = {}
            for post in all_posts:
                sub = post['subreddit']
                if sub not in subreddit_breakdown:
                    subreddit_breakdown[sub] = {'count': 0, 'avg_sentiment': 0}
                subreddit_breakdown[sub]['count'] += 1
            
            for sub in subreddit_breakdown:
                sub_posts = [p for p in all_posts if p['subreddit'] == sub]
                subreddit_breakdown[sub]['avg_sentiment'] = sum(p['sentiment'] for p in sub_posts) / len(sub_posts)
            
            return {
                'posts_analyzed': len(all_posts),
                'average_sentiment': avg_sentiment,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'neutral_count': neutral_count,
                'top_posts': top_posts,
                'sentiment_distribution': sentiment_distribution,
                'subreddit_breakdown': subreddit_breakdown
            }
            
        except Exception as e:
            logger.warning(f"Error getting Reddit sentiment for {symbol}: {str(e)}")
            return {
                'posts_analyzed': 0,
                'average_sentiment': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'top_posts': [],
                'sentiment_distribution': {},
                'subreddit_breakdown': {}
            }

    def _default_sentiment(self) -> Dict:
        """Return default sentiment structure when analysis fails"""
        return {
            'symbol': 'UNKNOWN',
            'timestamp': datetime.now().isoformat(),
            'news_count': 0,
            'sentiment_scores': {
                'average_sentiment': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'sentiment_distribution': {
                    'very_positive': 0,
                    'positive': 0,
                    'neutral': 0,
                    'negative': 0,
                    'very_negative': 0
                },
                'strongest_sentiment': 0.0
            },
            'reddit_sentiment': {
                'posts_analyzed': 0,
                'average_sentiment': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'top_posts': [],
                'sentiment_distribution': {},
                'subreddit_breakdown': {}
            },
            'significant_events': {},
            'overall_sentiment': 0.0,
            'sentiment_components': {},
            'confidence': 0.0,
            'recent_headlines': [],
            'trend_analysis': {},
            'impact_analysis': {}
        }
    
    def _analyze_news_sentiment(self, news_articles: List[Dict]) -> Dict:
        """Analyze sentiment from news articles using multiple methods"""
        
        if not news_articles:
            return {
                'average_sentiment': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'sentiment_distribution': {
                    'very_positive': 0,
                    'positive': 0,
                    'neutral': 0,
                    'negative': 0,
                    'very_negative': 0
                },
                'strongest_sentiment': 0.0
            }
        
        sentiments = []
        sentiment_categories = {
            'very_positive': 0,
            'positive': 0,
            'neutral': 0,
            'negative': 0,
            'very_negative': 0
        }
        
        for article in news_articles:
            title = article.get('title', '')
            summary = article.get('summary', '')
            text = f"{title} {summary}".strip()
            
            if not text:
                continue
            
            # Multiple sentiment analysis methods
            sentiment_scores = []
            
            # TextBlob sentiment
            try:
                blob = TextBlob(text)
                textblob_score = blob.sentiment.polarity
                sentiment_scores.append(textblob_score)
            except Exception:
                pass
            
            # VADER sentiment
            try:
                vader_scores = self.vader.polarity_scores(text)
                # Convert compound score to -1 to 1 range
                vader_score = vader_scores['compound']
                sentiment_scores.append(vader_score)
            except Exception:
                pass
            
            # Transformer sentiment (if available)
            if self.transformer_models.get('general_sentiment'):
                try:
                    transformer_result = self.transformer_models['general_sentiment'](text)
                    if transformer_result:
                        # Convert to -1 to 1 scale
                        if transformer_result[0]['label'] == 'POSITIVE':
                            transformer_score = transformer_result[0]['score']
                        elif transformer_result[0]['label'] == 'NEGATIVE':
                            transformer_score = -transformer_result[0]['score']
                        else:
                            transformer_score = 0.0
                        sentiment_scores.append(transformer_score)
                except Exception:
                    pass
            
            # Average the sentiment scores
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                sentiments.append(avg_sentiment)
                
                # Categorize sentiment
                if avg_sentiment > 0.5:
                    sentiment_categories['very_positive'] += 1
                elif avg_sentiment > 0.1:
                    sentiment_categories['positive'] += 1
                elif avg_sentiment < -0.5:
                    sentiment_categories['very_negative'] += 1
                elif avg_sentiment < -0.1:
                    sentiment_categories['negative'] += 1
                else:
                    sentiment_categories['neutral'] += 1
        
        if not sentiments:
            return {
                'average_sentiment': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'sentiment_distribution': sentiment_categories,
                'strongest_sentiment': 0.0
            }
        
        # Calculate overall metrics
        average_sentiment = sum(sentiments) / len(sentiments)
        positive_count = len([s for s in sentiments if s > 0.1])
        negative_count = len([s for s in sentiments if s < -0.1])
        neutral_count = len(sentiments) - positive_count - negative_count
        strongest_sentiment = max(sentiments, key=abs) if sentiments else 0.0
        
        return {
            'average_sentiment': average_sentiment,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'sentiment_distribution': sentiment_categories,
            'strongest_sentiment': strongest_sentiment
        }
    
    def _check_significant_events(self, news_articles: List[Dict], symbol: str) -> Dict:
        """Check for significant events in news articles"""
        
        events = {
            'dividend_announcement': False,
            'earnings_report': False,
            'management_change': False,
            'regulatory_news': False,
            'merger_acquisition': False,
            'scandal_investigation': False,
            'rating_change': False,
            'capital_raising': False,
            'branch_closure': False,
            'product_launch': False,
            'partnership_deal': False,
            'legal_action': False,
            'events_detected': []
        }
        
        # Keywords for different event types
        event_keywords = {
            'dividend_announcement': ['dividend', 'payout', 'distribution', 'yield'],
            'earnings_report': ['earnings', 'profit', 'revenue', 'quarterly', 'annual', 'results'],
            'management_change': ['ceo', 'chairman', 'executive', 'director', 'appointment', 'resignation'],
            'regulatory_news': ['apra', 'asic', 'rba', 'regulation', 'compliance', 'audit'],
            'merger_acquisition': ['merger', 'acquisition', 'takeover', 'bid', 'combine'],
            'scandal_investigation': ['investigation', 'scandal', 'misconduct', 'fraud', 'inquiry'],
            'rating_change': ['rating', 'upgrade', 'downgrade', 'outlook', 'moody', 'fitch', 's&p'],
            'capital_raising': ['capital raising', 'share issue', 'equity', 'funding', 'placement'],
            'branch_closure': ['branch closure', 'closing branches', 'shut down', 'consolidation'],
            'product_launch': ['launch', 'new product', 'service', 'offering', 'introduce'],
            'partnership_deal': ['partnership', 'deal', 'alliance', 'agreement', 'collaborate'],
            'legal_action': ['lawsuit', 'legal action', 'court', 'litigation', 'settlement']
        }
        
        for article in news_articles:
            title = article.get('title', '').lower()
            summary = article.get('summary', '').lower()
            text = f"{title} {summary}"
            
            for event_type, keywords in event_keywords.items():
                if any(keyword in text for keyword in keywords):
                    events[event_type] = True
                    
                    # Determine sentiment impact
                    sentiment_impact = 0.0
                    if event_type in ['dividend_announcement', 'earnings_report', 'product_launch', 'partnership_deal']:
                        sentiment_impact = 0.1  # Generally positive
                    elif event_type in ['rating_change']:
                        if any(word in text for word in ['upgrade', 'positive', 'improve']):
                            sentiment_impact = 0.15
                        elif any(word in text for word in ['downgrade', 'negative', 'lower']):
                            sentiment_impact = -0.15
                    elif event_type in ['scandal_investigation', 'legal_action']:
                        sentiment_impact = -0.2  # Generally negative
                    elif event_type in ['management_change']:
                        if any(word in text for word in ['new', 'appointed', 'joins']):
                            sentiment_impact = 0.05
                        else:
                            sentiment_impact = -0.05
                    
                    events['events_detected'].append({
                        'type': event_type,
                        'headline': article.get('title', ''),
                        'date': article.get('published', datetime.now().isoformat()),
                        'source': article.get('source', 'Unknown'),
                        'relevance': article.get('relevance', 'medium'),
                        'sentiment_impact': sentiment_impact
                    })
        
        return events

    def analyze_news_relevance(self, articles: List[Dict], symbol: str = None) -> List[Dict]:
        """
        Analyze and prioritize news articles using enhanced filtering
        
        Args:
            articles: List of news articles
            symbol: Bank symbol for enhanced filtering
            
        Returns:
            List of articles with enhanced relevance analysis, sorted by priority
        """
        enhanced_articles = []
        
        for article in articles:
            title = article.get('title', '')
            summary = article.get('summary', '')
            
            # Skip if already has enhanced data
            if 'relevance_score' in article:
                enhanced_articles.append(article)
                continue
            
            # Get enhanced filtering results
            filter_result = self.news_filter.is_relevant_banking_news(
                title, summary, symbol
            )
            
            # Add enhanced metadata
            enhanced_article = article.copy()
            enhanced_article.update({
                'enhanced_relevance_score': filter_result['relevance_score'],
                'matched_keywords': filter_result['matched_keywords'],
                'categories': filter_result['categories'],
                'urgency_score': filter_result['urgency_score'],
                'sentiment_indicators': filter_result['sentiment_indicators'],
                'priority_score': self._calculate_priority_score(filter_result)
            })
            
            # Only include relevant articles
            if filter_result['is_relevant']:
                enhanced_articles.append(enhanced_article)
        
        # Sort by priority score (highest first)
        enhanced_articles.sort(key=lambda x: x.get('priority_score', x.get('enhanced_relevance_score', 0)), reverse=True)
        
        return enhanced_articles
    
    def _calculate_priority_score(self, filter_result: dict) -> float:
        """
        Calculate priority score for news article based on multiple factors
        
        Args:
            filter_result: Result from enhanced filtering
            
        Returns:
            Priority score (0-1, higher is more important)
        """
        base_score = filter_result['relevance_score']
        urgency_bonus = filter_result['urgency_score'] * 0.3
        
        # Category bonuses
        category_bonus = 0
        categories = filter_result['categories']
        
        # High priority categories
        if any('regulatory' in cat for cat in categories):
            category_bonus += 0.2
        if any('financial_results' in cat for cat in categories):
            category_bonus += 0.2
        if any('risk' in cat for cat in categories):
            category_bonus += 0.15
        if any('leadership' in cat for cat in categories):
            category_bonus += 0.1
        if any('specific_bank' in cat for cat in categories):
            category_bonus += 0.1
        if any('official_announcement' in cat for cat in categories):
            category_bonus += 0.25  # ASX announcements get high priority
        
        # Risk sentiment bonus
        risk_indicators = len(filter_result['sentiment_indicators']['risk'])
        risk_bonus = min(risk_indicators * 0.05, 0.15)
        
        total_score = min(base_score + urgency_bonus + category_bonus + risk_bonus, 1.0)
        return total_score
    
    def get_filtered_news_summary(self, symbol: str) -> Dict:
        """
        Get a summary of news filtering performance and statistics
        
        Args:
            symbol: Bank symbol
            
        Returns:
            Dict with filtering statistics and insights
        """
        try:
            # Collect all news sources
            all_news = []
            
            # Get news from all sources
            rss_news = self._fetch_rss_news(symbol)
            yahoo_news = self._fetch_yahoo_news(symbol)
            abc_news = self._scrape_abc_news(symbol)
            news_com_au = self._scrape_news_com_au(symbol)
            motley_fool = self._scrape_motley_fool_au(symbol)
            
            all_raw_news = rss_news + yahoo_news + abc_news + news_com_au + motley_fool
            
            # Analyze with enhanced filtering
            enhanced_news = self.analyze_news_relevance(all_raw_news, symbol)
            
            # Calculate statistics
            total_articles = len(all_raw_news)
            relevant_articles = len(enhanced_news)
            
            # Category breakdown
            category_counts = {}
            urgency_distribution = {'low': 0, 'medium': 0, 'high': 0}
            
            for article in enhanced_news:
                # Count categories
                for category in article.get('categories', []):
                    category_counts[category] = category_counts.get(category, 0) + 1
                
                # Urgency distribution
                urgency = article.get('urgency_score', 0)
                if urgency >= 0.7:
                    urgency_distribution['high'] += 1
                elif urgency >= 0.3:
                    urgency_distribution['medium'] += 1
                else:
                    urgency_distribution['low'] += 1
            
            # Top categories
            top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Top keywords
            all_keywords = []
            for article in enhanced_news:
                all_keywords.extend(article.get('matched_keywords', []))
            
            keyword_counts = {}
            for keyword in all_keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
            
            top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'filtering_summary': {
                    'total_articles_found': total_articles,
                    'relevant_articles': relevant_articles,
                    'filtering_efficiency': relevant_articles / max(total_articles, 1),
                    'avg_relevance_score': sum(a.get('enhanced_relevance_score', 0) for a in enhanced_news) / max(relevant_articles, 1),
                    'avg_priority_score': sum(a.get('priority_score', 0) for a in enhanced_news) / max(relevant_articles, 1)
                },
                'category_breakdown': {
                    'top_categories': top_categories,
                    'total_categories': len(category_counts)
                },
                'urgency_analysis': urgency_distribution,
                'keyword_analysis': {
                    'top_keywords': top_keywords,
                    'unique_keywords': len(keyword_counts)
                },
                'high_priority_articles': [
                    {
                        'title': a['title'],
                        'source': a['source'],
                        'priority_score': a.get('priority_score', 0),
                        'categories': a.get('categories', [])
                    }
                    for a in enhanced_news[:5]  # Top 5 by priority
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating filtered news summary: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()

            }
    
    # ...existing code...
