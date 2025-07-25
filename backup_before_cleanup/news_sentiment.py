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

# Transformers for advanced sentiment analysis
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Falling back to TextBlob and VADER.")

# Import settings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Settings
from utils.cache_manager import CacheManager
from src.sentiment_history import SentimentHistoryManager
from src.news_impact_analyzer import NewsImpactAnalyzer

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
        
        # Initialize transformers-based sentiment analysis
        self.transformer_classifier = None
        if TRANSFORMERS_AVAILABLE:
            try:
                # For now, let's use a simpler model that doesn't require PyTorch/TensorFlow
                # We'll use VADER and TextBlob as primary, but enhance the logic
                logger.info("Transformers library available - using enhanced sentiment analysis")
                self.use_enhanced_sentiment = True
            except Exception as e:
                logger.warning(f"Enhanced sentiment analysis setup failed: {e}")
                self.use_enhanced_sentiment = False
        else:
            self.use_enhanced_sentiment = False
        
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
                market_context
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
                                            market_context: Dict) -> Dict:
        """Enhanced sentiment calculation with dynamic weighting"""
        
        # Base weights
        weights = {
            'news': 0.4,
            'reddit': 0.15,
            'events': 0.25,
            'volume': 0.1,
            'momentum': 0.1
        }
        
        # Adjust weights based on data quality
        news_count = news_sentiment.get('news_count', 0) if isinstance(news_sentiment, dict) else 0
        reddit_posts = reddit_sentiment.get('posts_analyzed', 0)
        
        # Dynamic weight adjustment based on data availability
        if news_count < 5:
            weights['news'] *= 0.5
            weights['reddit'] += weights['news'] * 0.3  # Transfer some weight to reddit
            weights['events'] += weights['news'] * 0.2
        
        if reddit_posts < 3:
            weights['reddit'] *= 0.3
            weights['news'] += weights['reddit'] * 0.5  # Transfer weight back to news
            weights['events'] += weights['reddit'] * 0.2
        
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
            momentum_score * normalized_weights['momentum']
        )
        
        # Apply market context modifier
        market_modifier = market_context.get('volatility_factor', 1.0)
        overall *= market_modifier
        
        # Calculate confidence factor based on data quality
        confidence = self._calculate_confidence_factor(news_count, reddit_posts, len(events.get('events_detected', [])))
        
        # Apply confidence adjustment (less aggressive than full multiplication)
        confidence_adjusted = overall * (0.7 + 0.3 * confidence)
        
        return {
            'score': max(-1, min(1, confidence_adjusted)),
            'components': {
                'news': news_score * normalized_weights['news'],
                'reddit': reddit_score * normalized_weights['reddit'],
                'events': events_score * normalized_weights['events'],
                'volume': volume_sentiment * normalized_weights['volume'],
                'momentum': momentum_score * normalized_weights['momentum']
            },
            'weights': normalized_weights,
            'confidence': confidence,
            'market_modifier': market_modifier
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
    
    def _calculate_confidence_factor(self, news_count: int, reddit_posts: int, event_count: int) -> float:
        """Calculate confidence in the sentiment analysis"""
        
        # Base confidence
        confidence = 0.5
        
        # News contribution (up to 0.3)
        if news_count >= 10:
            confidence += 0.3
        elif news_count >= 5:
            confidence += 0.2
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
        """Analyze sentiment of news articles using enhanced multi-method approach"""
        
        if not news_items:
            return {
                'average_sentiment': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'sentiment_distribution': {},
                'news_count': 0,
                'analysis_method': 'none'
            }
        
        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        analysis_method = "enhanced" if self.use_enhanced_sentiment else "basic"
        
        for news in news_items:
            # Combine title and summary for analysis
            text = f"{news['title']} {news.get('summary', '')}"
            
            # Use enhanced sentiment analysis if available
            if self.use_enhanced_sentiment:
                sentiment_score = self._enhanced_sentiment_analysis(text)
            else:
                sentiment_score = self._fallback_sentiment_analysis(text)
            
            # Weight by relevance
            relevance_weight = 1.0 if news['relevance'] == 'high' else 0.7
            weighted_sentiment = sentiment_score * relevance_weight
            
            sentiments.append(weighted_sentiment)
            
            # Categorize
            if weighted_sentiment > 0.1:
                positive_count += 1
            elif weighted_sentiment < -0.1:
                negative_count += 1
            else:
                neutral_count += 1
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
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
            'analysis_method': analysis_method
        }
    
    def _enhanced_sentiment_analysis(self, text: str) -> float:
        """Enhanced sentiment analysis using multiple methods and financial keywords"""
        
        # Basic sentiment from TextBlob and VADER
        basic_sentiment = self._fallback_sentiment_analysis(text)
        
        # Financial sentiment keywords analysis
        financial_sentiment = self._analyze_financial_keywords(text)
        
        # Combine the methods with weights
        # 60% basic sentiment, 40% financial keyword sentiment
        enhanced_sentiment = (basic_sentiment * 0.6) + (financial_sentiment * 0.4)
        
        return enhanced_sentiment
    
    def _analyze_financial_keywords(self, text: str) -> float:
        """Analyze financial-specific keywords for sentiment"""
        
        text_lower = text.lower()
        
        # Positive financial keywords
        positive_keywords = {
            'strong': 0.3, 'growth': 0.4, 'profit': 0.5, 'revenue': 0.3, 'beat': 0.6,
            'exceed': 0.5, 'outperform': 0.6, 'record': 0.4, 'high': 0.2, 'surge': 0.7,
            'boost': 0.4, 'gain': 0.4, 'rise': 0.3, 'increase': 0.3, 'upgrade': 0.6,
            'positive': 0.4, 'bullish': 0.7, 'dividend': 0.3, 'buyback': 0.4,
            'partnership': 0.3, 'acquisition': 0.2, 'expansion': 0.4, 'recovery': 0.5,
            'milestone': 0.4, 'breakthrough': 0.6, 'success': 0.5, 'achievement': 0.4,
            'resilient': 0.4, 'robust': 0.5, 'stable': 0.2, 'momentum': 0.3
        }
        
        # Negative financial keywords
        negative_keywords = {
            'loss': -0.5, 'decline': -0.4, 'fall': -0.3, 'drop': -0.4, 'weak': -0.4,
            'miss': -0.6, 'disappoint': -0.6, 'concern': -0.3, 'risk': -0.2, 'worry': -0.4,
            'pressure': -0.3, 'challenge': -0.3, 'struggle': -0.5, 'difficulty': -0.4,
            'downgrade': -0.6, 'negative': -0.4, 'bearish': -0.7, 'cut': -0.4,
            'reduce': -0.3, 'slash': -0.5, 'suspend': -0.6, 'crisis': -0.8,
            'scandal': -0.8, 'fraud': -0.9, 'investigation': -0.5, 'lawsuit': -0.6,
            'penalty': -0.6, 'fine': -0.5, 'breach': -0.6, 'violation': -0.5,
            'volatile': -0.3, 'uncertain': -0.3, 'unstable': -0.4
        }
        
        # Calculate sentiment based on keyword presence and frequency
        sentiment_score = 0.0
        total_weight = 0.0
        
        # Check positive keywords
        for keyword, weight in positive_keywords.items():
            count = text_lower.count(keyword)
            if count > 0:
                # Use logarithmic scaling to prevent over-weighting of repeated keywords
                impact = weight * (1 + 0.3 * (count - 1))  # Additional 30% for each repeat
                sentiment_score += impact
                total_weight += abs(weight)
        
        # Check negative keywords
        for keyword, weight in negative_keywords.items():
            count = text_lower.count(keyword)
            if count > 0:
                # Use logarithmic scaling
                impact = weight * (1 + 0.3 * (count - 1))
                sentiment_score += impact
                total_weight += abs(weight)
        
        # Normalize the score
        if total_weight > 0:
            normalized_score = sentiment_score / total_weight
        else:
            normalized_score = 0.0
        
        # Apply context modifiers
        normalized_score = self._apply_context_modifiers(text_lower, normalized_score)
        
        # Clamp to [-1, 1] range
        return max(-1, min(1, normalized_score))
    
    def _apply_context_modifiers(self, text_lower: str, base_score: float) -> float:
        """Apply context-based modifiers to sentiment score"""
        
        modified_score = base_score
        
        # Negation handling
        negation_words = ['not', 'no', 'never', 'neither', 'none', 'nothing', 'nowhere', 
                         'hardly', 'scarcely', 'barely', 'little', 'few', 'seldom', 'rarely']
        
        # Check for negations near sentiment words
        for negation in negation_words:
            if negation in text_lower:
                # Reduce the magnitude of sentiment if negation is present
                modified_score *= 0.7
        
        # Uncertainty modifiers
        uncertainty_words = ['might', 'may', 'could', 'possibly', 'perhaps', 'maybe', 
                           'uncertain', 'unclear', 'potential', 'likely', 'probable']
        
        uncertainty_count = sum(1 for word in uncertainty_words if word in text_lower)
        if uncertainty_count > 0:
            # Reduce confidence in sentiment
            modified_score *= (1 - 0.1 * uncertainty_count)
        
        # Emphasis modifiers
        emphasis_words = ['very', 'extremely', 'highly', 'significantly', 'substantially', 
                         'dramatically', 'remarkably', 'exceptionally', 'particularly']
        
        emphasis_count = sum(1 for word in emphasis_words if word in text_lower)
        if emphasis_count > 0:
            # Amplify sentiment
            modified_score *= (1 + 0.2 * emphasis_count)
        
        return modified_score
    
    def _fallback_sentiment_analysis(self, text: str) -> float:
        """Fallback sentiment analysis using TextBlob and VADER"""
        
        # TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        compound = vader_scores['compound']  # -1 to 1
        
        # Average the two methods
        combined_sentiment = (polarity + compound) / 2
        
        return combined_sentiment
    
    def _check_significant_events(self, news_items: List[Dict], symbol: str) -> Dict:
        """Check for significant events in the news with enhanced pattern matching"""
        
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
        
        # Enhanced keywords with regex patterns for better matching
        event_patterns = {
            'dividend_announcement': {
                'keywords': ['dividend', 'distribution', 'payout', 'interim dividend', 'final dividend', 'special dividend'],
                'regex': [
                    r'dividend.*\$[\d.]+',
                    r'(interim|final|special)\s+dividend',
                    r'dividend.*(?:increased|raised|maintained|cut|suspended)'
                ]
            },
            'earnings_report': {
                'keywords': ['earnings', 'profit', 'results', 'quarterly', 'half-year', 'full-year', 'net income'],
                'regex': [
                    r'(quarterly|half-year|full-year)\s+results',
                    r'profit.*\$[\d.]+(?:million|billion)',
                    r'earnings.*(?:beat|miss|exceed|below)',
                    r'net\s+income.*\$[\d.]+'
                ]
            },
            'management_change': {
                'keywords': ['CEO', 'CFO', 'director', 'chairman', 'appoint', 'resign', 'retirement', 'succession'],
                'regex': [
                    r'(CEO|CFO|chairman|director).*(?:appoint|resign|retire|step down)',
                    r'new\s+(CEO|CFO|chairman|director)',
                    r'(appoint|announce).*(?:CEO|CFO|chairman|director)',
                    r'management\s+change'
                ]
            },
            'regulatory_news': {
                'keywords': ['APRA', 'ASIC', 'regulator', 'compliance', 'investigation', 'audit', 'prudential'],
                'regex': [
                    r'(APRA|ASIC).*(?:investigation|audit|review|action)',
                    r'regulatory.*(?:action|review|investigation|compliance)',
                    r'prudential.*(?:requirement|review|standard)',
                    r'compliance.*(?:breach|issue|review)'
                ]
            },
            'merger_acquisition': {
                'keywords': ['merger', 'acquisition', 'takeover', 'buyout', 'combine', 'acquire'],
                'regex': [
                    r'(merger|acquisition|takeover).*\$[\d.]+(?:million|billion)',
                    r'acquire.*(?:stake|interest|business)',
                    r'(merge|combine)\s+with',
                    r'takeover.*(?:bid|offer)'
                ]
            },
            'scandal_investigation': {
                'keywords': ['scandal', 'probe', 'misconduct', 'penalty', 'fine', 'breach', 'violation'],
                'regex': [
                    r'(fine|penalty).*\$[\d.]+(?:million|billion)',
                    r'(scandal|misconduct|breach).*(?:investigation|probe)',
                    r'(AUSTRAC|ASIC|APRA).*(?:fine|penalty|action)',
                    r'compliance.*(?:breach|failure|issue)'
                ]
            },
            'rating_change': {
                'keywords': ['upgrade', 'downgrade', 'rating', 'outlook', 'Moody\'s', 'S&P', 'Fitch'],
                'regex': [
                    r'(Moody\'s|S&P|Fitch).*(?:upgrade|downgrade|rating)',
                    r'credit\s+rating.*(?:upgrade|downgrade|affirm)',
                    r'outlook.*(?:positive|negative|stable)',
                    r'rating.*(?:AA|A|BBB|BB|B)'
                ]
            },
            'capital_raising': {
                'keywords': ['capital raising', 'share issue', 'equity raising', 'rights issue', 'placement'],
                'regex': [
                    r'(capital|equity)\s+raising.*\$[\d.]+(?:million|billion)',
                    r'(rights|share)\s+issue',
                    r'placement.*\$[\d.]+(?:million|billion)',
                    r'raise.*capital'
                ]
            },
            'branch_closure': {
                'keywords': ['branch closure', 'branch closing', 'close branches', 'branch network'],
                'regex': [
                    r'clos(?:e|ing).*\d+.*branches?',
                    r'branch.*(?:closure|closing|reduction)',
                    r'reduce.*branch.*network'
                ]
            },
            'product_launch': {
                'keywords': ['launch', 'new product', 'introduce', 'unveil', 'digital platform'],
                'regex': [
                    r'launch.*(?:new|digital).*(?:product|service|platform)',
                    r'introduce.*(?:banking|financial).*(?:product|service)',
                    r'unveil.*(?:new|digital).*(?:platform|service)'
                ]
            },
            'partnership_deal': {
                'keywords': ['partnership', 'joint venture', 'alliance', 'collaboration', 'agreement'],
                'regex': [
                    r'(partnership|alliance).*with',
                    r'joint\s+venture',
                    r'(sign|announce).*(?:partnership|agreement|deal)',
                    r'collaboration.*with'
                ]
            },
            'legal_action': {
                'keywords': ['lawsuit', 'legal action', 'court case', 'litigation', 'class action'],
                'regex': [
                    r'(lawsuit|litigation).*(?:filed|against)',
                    r'class\s+action',
                    r'legal\s+action.*(?:taken|filed)',
                    r'court.*(?:case|action|ruling)'
                ]
            }
        }
        
        for news in news_items:
            text = f"{news['title']} {news.get('summary', '')}".lower()
            
            for event_type, patterns in event_patterns.items():
                # Check keywords
                keyword_match = any(keyword.lower() in text for keyword in patterns['keywords'])
                
                # Check regex patterns
                regex_match = False
                if 'regex' in patterns:
                    for pattern in patterns['regex']:
                        if re.search(pattern, text, re.IGNORECASE):
                            regex_match = True
                            break
                
                if keyword_match or regex_match:
                    events[event_type] = True
                    
                    # Extract relevant details
                    event_details = {
                        'type': event_type,
                        'headline': news['title'],
                        'date': news.get('published', ''),
                        'source': news.get('source', ''),
                        'relevance': news.get('relevance', 'medium'),
                        'sentiment_impact': self._calculate_event_sentiment_impact(event_type, text)
                    }
                    
                    # Try to extract specific values (amounts, percentages, etc.)
                    extracted_values = self._extract_event_values(text, event_type)
                    if extracted_values:
                        event_details['extracted_values'] = extracted_values
                    
                    events['events_detected'].append(event_details)
        
        return events
    
    def _calculate_event_sentiment_impact(self, event_type: str, text: str) -> float:
        """Calculate the expected sentiment impact of an event"""
        
        # Base sentiment impact by event type
        base_impacts = {
            'dividend_announcement': 0.3,
            'earnings_report': 0.0,  # Depends on results
            'management_change': -0.1,
            'regulatory_news': -0.4,
            'merger_acquisition': 0.2,
            'scandal_investigation': -0.6,
            'rating_change': 0.0,  # Depends on direction
            'capital_raising': -0.2,
            'branch_closure': -0.1,
            'product_launch': 0.1,
            'partnership_deal': 0.1,
            'legal_action': -0.3
        }
        
        base_impact = base_impacts.get(event_type, 0)
        
        # Adjust based on context
        if event_type == 'earnings_report':
            if any(word in text for word in ['beat', 'exceed', 'strong', 'record']):
                base_impact = 0.4
            elif any(word in text for word in ['miss', 'below', 'weak', 'disappoint']):
                base_impact = -0.4
        
        elif event_type == 'rating_change':
            if any(word in text for word in ['upgrade', 'positive', 'improve']):
                base_impact = 0.3
            elif any(word in text for word in ['downgrade', 'negative', 'lower']):
                base_impact = -0.3
        
        elif event_type == 'dividend_announcement':
            if any(word in text for word in ['increase', 'raise', 'higher']):
                base_impact = 0.4
            elif any(word in text for word in ['cut', 'reduce', 'suspend']):
                base_impact = -0.4
        
        return base_impact
    
    def _extract_event_values(self, text: str, event_type: str) -> Dict:
        """Extract specific values from event text (amounts, percentages, etc.)"""
        
        extracted = {}
        
        # Extract dollar amounts
        dollar_pattern = r'\$[\d,]+(?:\.\d{1,2})?(?:\s*(?:million|billion|m|b|mn|bn))?'
        dollar_matches = re.findall(dollar_pattern, text, re.IGNORECASE)
        if dollar_matches:
            extracted['amounts'] = dollar_matches
        
        # Extract percentages
        percentage_pattern = r'\d+(?:\.\d+)?%'
        percentage_matches = re.findall(percentage_pattern, text)
        if percentage_matches:
            extracted['percentages'] = percentage_matches
        
        # Extract numbers (for branch closures, etc.)
        if event_type == 'branch_closure':
            number_pattern = r'\d+(?:\s*(?:branches?|locations?))'
            number_matches = re.findall(number_pattern, text, re.IGNORECASE)
            if number_matches:
                extracted['branch_numbers'] = number_matches
        
        # Extract dates
        date_pattern = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}'
        date_matches = re.findall(date_pattern, text, re.IGNORECASE)
        if date_matches:
            extracted['dates'] = date_matches
        
        return extracted
    
    def _calculate_overall_sentiment(self, news_sentiment: Dict, 
                                   reddit_sentiment: Dict, events: Dict) -> float:
        """Calculate overall sentiment score (-1 to 1) - DEPRECATED: Use _calculate_overall_sentiment_improved"""
        
        # This method is kept for backward compatibility
        # It now calls the improved version and returns just the score
        market_context = self._get_market_context()
        result = self._calculate_overall_sentiment_improved(
            news_sentiment, reddit_sentiment, events, market_context
        )
        return result['score']
    
    def get_market_sentiment(self) -> Dict:
        """Get overall market sentiment"""
        
        try:
            # Analyze general market news
            market_news = self._fetch_market_news()
            
            # RBA sentiment
            rba_sentiment = self._analyze_rba_sentiment()
            
            # Economic indicators sentiment
            economic_sentiment = self._analyze_economic_sentiment()
            
            # Calculate overall market sentiment
            sentiments = []
            
            if market_news:
                sentiments.append(market_news['sentiment'])
            if rba_sentiment:
                sentiments.append(rba_sentiment['sentiment'])
            if economic_sentiment:
                sentiments.append(economic_sentiment['sentiment'])
            
            overall_market_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
            return {
                'overall_sentiment': overall_market_sentiment,
                'market_news': market_news,
                'rba_sentiment': rba_sentiment,
                'economic_sentiment': economic_sentiment,
                'sentiment_description': self._describe_sentiment(overall_market_sentiment)
            }
            
        except Exception as e:
            logger.error(f"Error getting market sentiment: {str(e)}")
            return {
                'overall_sentiment': 0,
                'sentiment_description': 'neutral'
            }
    
    def _fetch_market_news(self) -> Dict:
        """Fetch general market news"""
        
        try:
            # Use ASX market news RSS or scraping
            market_keywords = ['ASX', 'Australian market', 'All Ordinaries', 'ASX 200']
            
            # This would fetch and analyze market news
            # For now, return placeholder
            return {
                'sentiment': 0,
                'headlines': []
            }
            
        except Exception as e:
            logger.warning(f"Error fetching market news: {str(e)}")
            return None
    
    def _analyze_rba_sentiment(self) -> Dict:
        """Analyze RBA (Reserve Bank of Australia) sentiment"""
        
        try:
            # Check RBA RSS feed for recent announcements
            rba_feed = feedparser.parse(self.settings.NEWS_SOURCES['rss_feeds']['rba'])
            
            if rba_feed.entries:
                latest_entry = rba_feed.entries[0]
                title = latest_entry.get('title', '').lower()
                summary = latest_entry.get('summary', '').lower()
                
                # Simple keyword analysis for RBA stance
                hawkish_keywords = ['raise', 'increase', 'inflation', 'tighten', 'hike']
                dovish_keywords = ['cut', 'reduce', 'stimulus', 'accommodate', 'pause']
                
                hawkish_count = sum(1 for k in hawkish_keywords if k in title or k in summary)
                dovish_count = sum(1 for k in dovish_keywords if k in title or k in summary)
                
                if hawkish_count > dovish_count:
                    sentiment = -0.3  # Negative for banks (higher rates = lower margins)
                elif dovish_count > hawkish_count:
                    sentiment = 0.3   # Positive for banks
                else:
                    sentiment = 0
                
                return {
                    'sentiment': sentiment,
                    'stance': 'hawkish' if hawkish_count > dovish_count else 'dovish' if dovish_count > hawkish_count else 'neutral',
                    'latest_announcement': latest_entry.get('title', '')
                }
            
        except Exception as e:
            logger.warning(f"Error analyzing RBA sentiment: {str(e)}")
        
        return None
    
    def _analyze_economic_sentiment(self) -> Dict:
        """Analyze economic indicators sentiment"""
        
        # This would analyze recent economic data releases
        # For now, return neutral
        return {
            'sentiment': 0,
            'indicators': {}
        }
    
    def _describe_sentiment(self, score: float) -> str:
        """Convert sentiment score to description"""
        
        if score > 0.5:
            return 'very_positive'
        elif score > 0.2:
            return 'positive'
        elif score > -0.2:
            return 'neutral'
        elif score > -0.5:
            return 'negative'
        else:
            return 'very_negative'
    
    def _default_sentiment(self) -> Dict:
        """Return default sentiment when analysis fails"""
        
        return {
            'symbol': '',
            'timestamp': datetime.now().isoformat(),
            'news_count': 0,
            'sentiment_scores': {
                'average_sentiment': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            },
            'overall_sentiment': 0,
            'recent_headlines': [],
            'error': 'Sentiment analysis temporarily unavailable'
        }
    
    def get_overnight_news(self) -> List[Dict]:
        """Get overnight news for morning analysis"""
        
        overnight_news = []
        
        # Calculate overnight period (last 12 hours)
        overnight_start = datetime.now() - timedelta(hours=12)
        
        # Fetch news from all sources
        for symbol in self.settings.BANK_SYMBOLS:
            news = self._fetch_rss_news(symbol)
            
            # Filter for overnight news
            for item in news:
                try:
                    pub_date = datetime.fromisoformat(item['published'].replace('Z', '+00:00'))
                    if pub_date > overnight_start:
                        overnight_news.append(item)
                except:
                    continue
        
        # Sort by date
        overnight_news.sort(key=lambda x: x['published'], reverse=True)
        
        return overnight_news[:10]  # Top 10 overnight news items