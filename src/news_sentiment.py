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