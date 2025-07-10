# config/settings.py
"""
Configuration settings for ASX Bank Trading Analysis System
All settings use FREE data sources and APIs
"""

import os
from datetime import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Central configuration for the trading system"""
    
    # ASX Bank Symbols
    BANK_SYMBOLS = [
        'CBA.AX',  # Commonwealth Bank
        'WBC.AX',  # Westpac
        'ANZ.AX',  # ANZ
        'NAB.AX',  # National Australia Bank
        'MQG.AX',  # Macquarie Group
        'SUN.AX',  # Suncorp Group
        'QBE.AX'   # QBE Insurance Group
    ]
    
    # Additional financial stocks (optional)
    EXTENDED_SYMBOLS = [
        'BEN.AX',  # Bendigo Bank
        'BOQ.AX',  # Bank of Queensland
        'SUN.AX',  # Suncorp
        'QBE.AX'   # QBE Insurance
    ]
    
    # Market indices
    MARKET_INDICES = {
        'ASX200': '^AXJO',
        'ALL_ORDS': '^AORD',
        'FINANCIALS': '^AXFJ'
    }
    
    # Trading Hours (Sydney time)
    MARKET_OPEN = time(10, 0)    # 10:00 AM
    MARKET_CLOSE = time(16, 0)   # 4:00 PM
    PRE_MARKET = time(7, 0)      # 7:00 AM
    POST_MARKET = time(17, 0)    # 5:00 PM
    
    # Analysis Parameters
    DEFAULT_ANALYSIS_PERIOD = os.getenv('DEFAULT_ANALYSIS_PERIOD', '3mo')
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 70))
    RISK_TOLERANCE = os.getenv('RISK_TOLERANCE', 'medium')  # low, medium, high
    
    # Technical Analysis Settings
    TECHNICAL_INDICATORS = {
        'RSI': {'period': 14, 'overbought': 70, 'oversold': 30},
        'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
        'BB': {'period': 20, 'std': 2},
        'SMA': {'periods': [20, 50, 200]},
        'EMA': {'periods': [12, 26]},
        'ATR': {'period': 14},
        'VOLUME': {'ma_period': 20}
    }
    
    # Risk Management
    RISK_PARAMETERS = {
        'max_position_size': 0.25,  # 25% of capital max per position
        'stop_loss_atr_multiplier': 2.0,  # 2x ATR for stop loss
        'take_profit_ratio': 2.0,  # 2:1 risk/reward minimum
        'max_daily_loss': 0.06,  # 6% max daily loss
        'max_open_positions': 3
    }
    
    # Alert Thresholds
    ALERT_THRESHOLDS = {
        'strong_buy': 70,
        'buy': 50,
        'sell': -50,
        'strong_sell': -70,
        'high_risk': 80,
        'unusual_volume': 2.0,  # 2x average volume
        'price_breakout': 0.02  # 2% move
    }
    
    # Sentiment Analysis
    SENTIMENT_WEIGHTS = {
        'news': 0.3,
        'technical': 0.4,
        'fundamental': 0.3
    }
    
    # Data Sources (all free)
    DATA_SOURCES = {
        'quotes': 'yfinance',
        'news': ['google_news', 'reddit', 'asx_announcements'],
        'economic': ['rba', 'abs'],
        'sentiment': 'textblob'  # Free NLP
    }
    
    # Cache Settings
    CACHE_DURATION_MINUTES = int(os.getenv('CACHE_DURATION_MINUTES', 15))
    MAX_CACHE_SIZE_MB = int(os.getenv('MAX_CACHE_SIZE_MB', 100))
    
    # API Keys (optional - for enhanced features)
    DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS', '')
    EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
    
    # Free News Sources (Expanded with additional Australian sources)
    NEWS_SOURCES = {
        'urls': {
            'afr': 'https://www.afr.com/companies/financial-services',
            'abc': 'https://www.abc.net.au/news/business/',
            'smh': 'https://www.smh.com.au/business/banking-and-finance',
            'reuters': 'https://www.reuters.com/world/asia-pacific/',
            'bloomberg_au': 'https://www.bloomberg.com/australia',
            # New Australian news sources
            'news_com_au': 'https://www.news.com.au/finance',
            'investing_au': 'https://au.investing.com/',
            'market_online': 'https://themarketonline.com.au/',
            'motley_fool_au': 'https://www.fool.com.au/',
            'investor_daily': 'https://www.investordaily.com.au/',
            'aba_news': 'https://ausbanking.org.au/news/',
            'asx_announcements': 'https://www.asx.com.au/markets/trade-our-cash-market/todays-announcements'
        },
        'rss_feeds': {
            # Tier 1: Essential Financial & Government Sources
            'rba': 'https://www.rba.gov.au/rss/rss-cb.xml',
            'abc_news': 'https://www.abc.net.au/news/feed/2942460/rss.xml',
            'business_news': 'https://www.businessnews.com.au/rssfeed/latest.rss',
            'smh_main': 'https://www.smh.com.au/rss/feed.xml',
            'the_age': 'https://www.theage.com.au/rss/feed.xml',
            'brisbane_times': 'https://www.brisbanetimes.com.au/rss/feed.xml',
            
            # Tier 2: Major Australian News Sources
            'news_com_au': 'https://www.news.com.au/content-feeds/latest-news-national/',
            'nine_news': 'https://www.9news.com.au/rss',
            'sbs_news': 'https://www.sbs.com.au/news/feed',
            'the_west_australian': 'https://thewest.com.au/rss',
            'perth_now': 'https://www.perthnow.com.au/feed',
            'wa_today': 'https://www.watoday.com.au/rss/feed.xml',
            
            # Tier 3: Financial & Business Focused
            'kalkine_media': 'https://kalkinemedia.com/au/feed',
            'ibt_australia': 'https://www.ibtimes.com.au/rss',
            'crikey': 'https://www.crikey.com.au/feed/',
            'tech_business_news': 'https://www.techbusinessnews.com.au/feed/',
            
            # Tier 4: Regional Coverage (Important for Mining/Resources)
            'canberra_times': 'https://www.canberratimes.com.au/rss.xml',
            'daily_telegraph': 'https://www.dailytelegraph.com.au/feed',
            'herald_sun': 'https://www.heraldsun.com.au/news/breaking-news/rss',
            'gold_coast_bulletin': 'https://www.goldcoastbulletin.com.au/feed',
            'townsville_bulletin': 'https://www.townsvillebulletin.com.au/news/rss',
            'nt_news': 'https://www.ntnews.com.au/news/rss',
            'the_mercury': 'https://www.themercury.com.au/feed',
            
            # Existing feeds (deduplicated)
            'afr_companies': 'https://www.afr.com/rss/companies',
            'market_index': 'https://www.marketindex.com.au/rss/asx-news',
            'investing_au': 'https://au.investing.com/rss/news_301.rss',
            'financial_review': 'https://www.financialstandard.com.au/rss.xml',
            'motley_fool_au': 'https://www.fool.com.au/feed/',
            'investor_daily': 'https://www.investordaily.com.au/feed/',
            'aba_news': 'https://ausbanking.org.au/feed/'
        },
        # Google Alerts integration for targeted monitoring
        'google_alerts': {
            'cba_alerts': 'Commonwealth Bank OR CBA.AX OR "CommBank"',
            'westpac_alerts': 'Westpac OR WBC.AX OR "Westpac Banking"',
            'anz_alerts': 'ANZ Bank OR ANZ.AX OR "Australia and New Zealand Banking"',
            'nab_alerts': 'NAB OR NAB.AX OR "National Australia Bank"',
            'mqg_alerts': 'Macquarie Group OR MQG.AX OR Macquarie Bank',
            'banking_sector': 'Australian banking OR RBA interest rates OR APRA regulations',
            'regulatory': 'ASIC banking OR APRA requirements OR "Reserve Bank Australia"'
        }
    }
    
    # Economic Calendar Events
    IMPORTANT_EVENTS = [
        'RBA Interest Rate Decision',  # First Tuesday of month
        'RBA Meeting Minutes',
        'Employment Data',
        'GDP',
        'Inflation (CPI)',
        'Retail Sales',
        'Housing Data'
    ]
    
    # Bank-specific settings
    BANK_PROFILES = {
        'CBA.AX': {
            'name': 'Commonwealth Bank',
            'sector_weight': 0.3,
            'dividend_months': [2, 8],
            'financial_year_end': 6
        },
        'WBC.AX': {
            'name': 'Westpac',
            'sector_weight': 0.2,
            'dividend_months': [5, 11],
            'financial_year_end': 9
        },
        'ANZ.AX': {
            'name': 'ANZ',
            'sector_weight': 0.2,
            'dividend_months': [5, 11],
            'financial_year_end': 9
        },
        'NAB.AX': {
            'name': 'National Australia Bank',
            'sector_weight': 0.2,
            'dividend_months': [5, 11],
            'financial_year_end': 9
        },
        'MQG.AX': {
            'name': 'Macquarie Group',
            'sector_weight': 0.1,
            'dividend_months': [6, 12],
            'financial_year_end': 3
        },
        'SUN.AX': {
            'name': 'Suncorp Group',
            'sector_weight': 0.05,
            'dividend_months': [3, 9],
            'financial_year_end': 6
        },
        'QBE.AX': {
            'name': 'QBE Insurance Group',
            'sector_weight': 0.05,
            'dividend_months': [3, 9],
            'financial_year_end': 12
        }
    }
    
    # Report Settings
    REPORT_SETTINGS = {
        'format': 'html',  # html, pdf, markdown
        'include_charts': True,
        'chart_library': 'plotly',  # plotly is free and interactive
        'timezone': 'Australia/Sydney'
    }
    
    # Scoring Weights for Final Recommendation
    SCORING_WEIGHTS = {
        'technical': 0.3,
        'fundamental': 0.25,
        'sentiment': 0.15,
        'risk_reward': 0.2,
        'market_condition': 0.1
    }
    
    # Position Sizing Rules
    POSITION_SIZING = {
        'method': 'risk_based',  # fixed, risk_based, kelly
        'risk_per_trade': 0.02,  # 2% risk per trade
        'account_size': 10000,   # Default account size for calculations
        'use_atr_sizing': True
    }
    
    # Backtesting Parameters (for validation)
    BACKTEST_SETTINGS = {
        'period': '1y',
        'initial_capital': 10000,
        'commission': 9.50,  # SelfWealth rate
        'slippage': 0.001    # 0.1% slippage
    }
    
    @classmethod
    def get_bank_name(cls, symbol):
        """Get bank name from symbol"""
        return cls.BANK_PROFILES.get(symbol, {}).get('name', symbol)
    
    @classmethod
    def is_dividend_month(cls, symbol):
        """Check if current month is dividend month"""
        from datetime import datetime
        current_month = datetime.now().month
        div_months = cls.BANK_PROFILES.get(symbol, {}).get('dividend_months', [])
        return current_month in div_months
    
    @classmethod
    def get_risk_level(cls):
        """Get risk level settings"""
        risk_levels = {
            'low': {'stop_loss': 0.02, 'position_size': 0.15},
            'medium': {'stop_loss': 0.03, 'position_size': 0.25},
            'high': {'stop_loss': 0.05, 'position_size': 0.35}
        }
        return risk_levels.get(cls.RISK_TOLERANCE, risk_levels['medium'])