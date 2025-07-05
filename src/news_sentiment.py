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

# Import settings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Settings
from utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class NewsSentimentAnalyzer:
    """Analyzes news sentiment from free Australian sources"""
    
    def __init__(self):
        self.settings = Settings()
        self.cache = CacheManager()
        self.vader = SentimentIntensityAnalyzer()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Bank name variations for searching
        self.bank_keywords = {
            'CBA.AX': ['Commonwealth Bank', 'CommBank', 'CBA'],
            'WBC.AX': ['Westpac', 'WBC'],
            'ANZ.AX': ['ANZ', 'Australia and New Zealand Banking'],
            'NAB.AX': ['National Australia Bank', 'NAB'],
            'MQG.AX': ['Macquarie', 'MQG', 'Macquarie Group']
        }
    
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
            
            # Web scraping
            scraped_news = self._scrape_news_sites(symbol)
            all_news.extend(scraped_news)
            
            # Reddit sentiment
            reddit_sentiment = self._get_reddit_sentiment(symbol)
            
            # Analyze sentiment
            sentiment_analysis = self._analyze_news_sentiment(all_news)
            
            # Check for specific events
            event_analysis = self._check_significant_events(all_news, symbol)
            
            # Combine all sentiment sources
            overall_sentiment = self._calculate_overall_sentiment(
                sentiment_analysis,
                reddit_sentiment,
                event_analysis
            )
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'news_count': len(all_news),
                'sentiment_scores': sentiment_analysis,
                'reddit_sentiment': reddit_sentiment,
                'significant_events': event_analysis,
                'overall_sentiment': overall_sentiment,
                'recent_headlines': [news['title'] for news in all_news[:5]]
            }
            
            # Cache for 30 minutes
            self.cache.set(cache_key, result, expiry_minutes=30)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
            return self._default_sentiment()
    
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
        """Get sentiment from Reddit (using pushshift.io or Reddit API)"""
        try:
            # For simplicity, we'll use a basic approach
            # In production, you'd want to use PRAW or the Reddit API
            
            keywords = self.bank_keywords.get(symbol, [symbol.replace('.AX', '')])
            
            # Simulated Reddit sentiment (replace with actual Reddit API calls)
            # This is just an example structure
            reddit_data = {
                'posts_analyzed': 0,
                'average_sentiment': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0
            }
            
            # You would actually fetch and analyze Reddit posts here
            # For now, return neutral sentiment
            return reddit_data
            
        except Exception as e:
            logger.warning(f"Error fetching Reddit sentiment: {str(e)}")
            return {
                'posts_analyzed': 0,
                'average_sentiment': 0
            }
    
    def _analyze_news_sentiment(self, news_items: List[Dict]) -> Dict:
        """Analyze sentiment of news articles"""
        
        if not news_items:
            return {
                'average_sentiment': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'sentiment_distribution': {}
            }
        
        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for news in news_items:
            # Combine title and summary for analysis
            text = f"{news['title']} {news.get('summary', '')}"
            
            # Use both TextBlob and VADER for better accuracy
            
            # TextBlob sentiment
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            
            # VADER sentiment
            vader_scores = self.vader.polarity_scores(text)
            compound = vader_scores['compound']  # -1 to 1
            
            # Average the two methods
            combined_sentiment = (polarity + compound) / 2
            
            # Weight by relevance
            relevance_weight = 1.0 if news['relevance'] == 'high' else 0.7
            weighted_sentiment = combined_sentiment * relevance_weight
            
            sentiments.append(weighted_sentiment)
            
            # Categorize
            if combined_sentiment > 0.1:
                positive_count += 1
            elif combined_sentiment < -0.1:
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
            'strongest_sentiment': max(sentiments, key=abs) if sentiments else 0
        }
    
    def _check_significant_events(self, news_items: List[Dict], symbol: str) -> Dict:
        """Check for significant events in the news"""
        
        events = {
            'dividend_announcement': False,
            'earnings_report': False,
            'management_change': False,
            'regulatory_news': False,
            'merger_acquisition': False,
            'scandal_investigation': False,
            'rating_change': False,
            'events_detected': []
        }
        
        # Keywords for different event types
        event_keywords = {
            'dividend_announcement': ['dividend', 'distribution', 'payout'],
            'earnings_report': ['earnings', 'profit', 'results', 'quarterly', 'half-year'],
            'management_change': ['CEO', 'CFO', 'director', 'appoint', 'resign', 'retirement'],
            'regulatory_news': ['APRA', 'ASIC', 'regulator', 'compliance', 'investigation'],
            'merger_acquisition': ['merger', 'acquisition', 'takeover', 'buyout'],
            'scandal_investigation': ['scandal', 'probe', 'misconduct', 'penalty', 'fine'],
            'rating_change': ['upgrade', 'downgrade', 'rating', 'outlook']
        }
        
        for news in news_items:
            text = f"{news['title']} {news.get('summary', '')}".lower()
            
            for event_type, keywords in event_keywords.items():
                if any(keyword in text for keyword in keywords):
                    events[event_type] = True
                    events['events_detected'].append({
                        'type': event_type,
                        'headline': news['title'],
                        'date': news.get('published', '')
                    })
        
        return events
    
    def _calculate_overall_sentiment(self, news_sentiment: Dict, 
                                   reddit_sentiment: Dict, events: Dict) -> float:
        """Calculate overall sentiment score (-1 to 1)"""
        
        # Weight different sources
        news_weight = 0.6
        reddit_weight = 0.2
        events_weight = 0.2
        
        # News sentiment score
        news_score = news_sentiment['average_sentiment']
        
        # Reddit sentiment score (if available)
        reddit_score = reddit_sentiment.get('average_sentiment', 0)
        
        # Events impact
        events_score = 0
        if events['dividend_announcement'] or events['earnings_report']:
            events_score += 0.2
        if events['rating_change']:
            events_score += 0.1
        if events['scandal_investigation'] or events['regulatory_news']:
            events_score -= 0.3
        if events['merger_acquisition']:
            events_score += 0.1
        
        # Calculate weighted overall sentiment
        overall = (news_score * news_weight + 
                  reddit_score * reddit_weight + 
                  events_score * events_weight)
        
        # Clamp to -1 to 1
        return max(-1, min(1, overall))
    
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