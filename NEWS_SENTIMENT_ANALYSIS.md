# News Sentiment Analysis - Demonstration & Enhancement Plan

## How News Sentiment Currently Works

### Data Sources
1. **RSS Feeds:**
   - RBA (Reserve Bank of Australia) announcements
   - ASX company announcements 
   - ABC Business news

2. **Web Scraping:**
   - Google News (filtered for bank-specific keywords)
   - Uses Beautiful Soup to extract headlines

3. **Sentiment Analysis:**
   - **TextBlob**: Provides polarity scores (-1 to 1)
   - **VADER**: Specialized for social media text, compound scores
   - **Combined**: Averages both methods for better accuracy

### Current Data Structure Returned

```json
{
  "symbol": "CBA.AX",
  "timestamp": "2025-07-06T14:00:11.441238",
  "news_count": 8,
  "sentiment_scores": {
    "average_sentiment": 0.104,
    "positive_count": 5,
    "negative_count": 0,
    "neutral_count": 3,
    "sentiment_distribution": {
      "very_positive": 0,
      "positive": 5,
      "neutral": 3,
      "negative": 0,
      "very_negative": 0
    },
    "strongest_sentiment": 0.238
  },
  "reddit_sentiment": {
    "posts_analyzed": 0,
    "average_sentiment": 0,
    "bullish_count": 0,
    "bearish_count": 0,
    "neutral_count": 0
  },
  "significant_events": {
    "dividend_announcement": false,
    "earnings_report": false,
    "management_change": false,
    "regulatory_news": false,
    "merger_acquisition": false,
    "scandal_investigation": false,
    "rating_change": false,
    "events_detected": []
  },
  "overall_sentiment": 0.062,
  "recent_headlines": [
    "Inspiring Aussies share their stories of doubt in new CommBank campaign",
    "CommBank harnesses near real-time, AI-powered intelligence to outsmart the scammers",
    "Gooooal! CommBank and Football Australia sign landmark deal to lift Australia's biggest game to new heights",
    "CBA completes sale of shareholding in Bank of Hangzhou",
    "Helping Australia's small businesses unlock value and reduce costs with CommBank Yello for Business"
  ]
}
```

## Analysis of Current Implementation

### ✅ What's Working Well:
1. **Multiple sentiment engines** (TextBlob + VADER) for better accuracy
2. **Event detection** - automatically identifies significant events
3. **Relevance weighting** - gives higher weight to more relevant news
4. **Caching** - avoids excessive API calls (30-minute cache)
5. **Real data extraction** - successfully fetching current news

### ⚠️ Current Limitations:

1. **Limited RSS Sources**: Only 3 RSS feeds, some may not be very active
2. **Reddit Integration**: Currently returns placeholder data (not implemented)
3. **Web Scraping Fragility**: Google News scraping may break if HTML changes
4. **No Financial News APIs**: Missing dedicated financial news sources
5. **Limited Event Keywords**: Basic keyword matching for event detection
6. **No Historical Sentiment**: No trend analysis over time

## 🚀 Suggested Enhancements

### 1. **Enhanced News Sources**
```python
# Add more comprehensive RSS feeds
ENHANCED_RSS_FEEDS = {
    'afr_rss': 'https://www.afr.com/rss/companies',
    'smh_business': 'https://www.smh.com.au/rss/business.xml',
    'financial_review': 'https://www.financialstandard.com.au/rss.xml',
    'market_index': 'https://www.marketindex.com.au/rss',
    'investing_com_au': 'https://au.investing.com/rss/news.rss'
}
```

### 2. **Reddit/Social Media Enhancement**
```python
def _get_reddit_sentiment_enhanced(self, symbol: str) -> Dict:
    """Enhanced Reddit sentiment using PRAW or web scraping"""
    try:
        import praw  # Reddit API
        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent='ASX_Analysis_Bot'
        )
        
        # Search relevant subreddits
        subreddits = ['ASX_Bets', 'AusFinance', 'SecurityAnalysis']
        posts = []
        
        for sub_name in subreddits:
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.search(symbol.replace('.AX', ''), limit=20):
                posts.append(post)
        
        # Analyze sentiment of posts and comments
        # ... sentiment analysis code
        
    except Exception as e:
        logger.warning(f"Reddit analysis failed: {e}")
        return self._fallback_reddit_sentiment()
```

### 3. **Financial News API Integration**
```python
def _fetch_financial_apis(self, symbol: str) -> List[Dict]:
    """Integrate with financial news APIs"""
    news_items = []
    
    # Alpha Vantage News (free tier available)
    try:
        alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY')
        if alpha_vantage_key:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': alpha_vantage_key,
                'limit': 50
            }
            # Process API response
    except:
        pass
    
    # Yahoo Finance News (free)
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        news = ticker.news
        for item in news:
            news_items.append({
                'title': item.get('title'),
                'summary': item.get('summary', ''),
                'source': 'Yahoo Finance',
                'published': item.get('providerPublishTime'),
                'url': item.get('link')
            })
    except:
        pass
    
    return news_items
```

### 4. **Enhanced Event Detection**
```python
def _enhanced_event_detection(self, news_items: List[Dict], symbol: str) -> Dict:
    """Enhanced event detection with NLP"""
    
    # More sophisticated keyword patterns
    event_patterns = {
        'earnings_report': [
            r'quarterly results?', r'half[- ]year results?', 
            r'profit (?:up|down|fell|rose)', r'earnings? (?:beat|miss|exceed)',
            r'revenue (?:up|down|increased|decreased)'
        ],
        'dividend_announcement': [
            r'dividend (?:declared|announced|increased|cut)',
            r'(?:interim|final) dividend', r'distribution (?:increased|maintained)'
        ],
        'regulatory_action': [
            r'APRA (?:investigation|penalty|fine)',
            r'ASIC (?:action|enforcement)', r'regulatory (?:breach|compliance)'
        ],
        'management_change': [
            r'(?:CEO|CFO|director) (?:appointed|resigned|retiring)',
            r'leadership (?:change|transition)', r'board (?:appointment|departure)'
        ],
        'credit_rating': [
            r'(?:Moody\'s|S&P|Fitch) (?:upgrade|downgrade)',
            r'credit rating (?:raised|lowered|affirmed)',
            r'outlook (?:positive|negative|stable)'
        ]
    }
    
    # Use regex for better pattern matching
    import re
    events_detected = {}
    
    for news in news_items:
        text = f"{news['title']} {news.get('summary', '')}".lower()
        
        for event_type, patterns in event_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    if event_type not in events_detected:
                        events_detected[event_type] = []
                    events_detected[event_type].append({
                        'headline': news['title'],
                        'date': news.get('published'),
                        'confidence': 'high' if symbol.replace('.AX', '').lower() in text else 'medium'
                    })
    
    return events_detected
```

### 5. **Sentiment Trend Analysis**
```python
def _analyze_sentiment_trends(self, symbol: str) -> Dict:
    """Analyze sentiment trends over time"""
    
    trends = {
        '1_day': self._get_daily_sentiment(symbol),
        '1_week': self._get_weekly_sentiment(symbol),
        '1_month': self._get_monthly_sentiment(symbol)
    }
    
    # Calculate momentum and direction
    recent_avg = sum(trends['1_day']) / len(trends['1_day']) if trends['1_day'] else 0
    weekly_avg = sum(trends['1_week']) / len(trends['1_week']) if trends['1_week'] else 0
    
    sentiment_momentum = 'improving' if recent_avg > weekly_avg else 'declining'
    
    return {
        'trends': trends,
        'momentum': sentiment_momentum,
        'volatility': self._calculate_sentiment_volatility(trends['1_week'])
    }
```

### 6. **News Quality Scoring**
```python
def _score_news_quality(self, news_item: Dict) -> float:
    """Score news quality and credibility"""
    
    quality_score = 1.0
    
    # Source credibility
    trusted_sources = ['AFR', 'Reuters', 'Bloomberg', 'ABC', 'SMH']
    if any(source.lower() in news_item['source'].lower() for source in trusted_sources):
        quality_score += 0.3
    
    # Recency bonus
    try:
        pub_date = datetime.fromisoformat(news_item['published'])
        hours_old = (datetime.now() - pub_date).total_seconds() / 3600
        if hours_old < 24:
            quality_score += 0.2
        elif hours_old < 72:
            quality_score += 0.1
    except:
        pass
    
    # Content depth (longer articles often more substantive)
    summary_length = len(news_item.get('summary', ''))
    if summary_length > 200:
        quality_score += 0.1
    
    return min(quality_score, 2.0)  # Cap at 2x weight
```

## Implementation Priority

### High Priority (Immediate)
1. ✅ **Add Yahoo Finance news integration** (free, reliable)
2. ✅ **Enhance event detection with regex patterns**
3. ✅ **Add more RSS feeds for better coverage**

### Medium Priority
4. **Reddit integration using web scraping or PRAW**
5. **Sentiment trend analysis over time**
6. **News quality scoring system**

### Low Priority (Future Enhancement)
7. **Social media monitoring (Twitter/X)**
8. **Analyst report sentiment**
9. **Machine learning sentiment models**

Would you like me to implement any of these enhancements?
