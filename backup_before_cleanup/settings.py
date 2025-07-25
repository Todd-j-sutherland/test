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
        'MQG.AX'   # Macquarie Group
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
    
    # Free News Sources
    NEWS_SOURCES = {
        'urls': {
            'afr': 'https://www.afr.com/companies/financial-services',
            'abc': 'https://www.abc.net.au/news/business/',
            'smh': 'https://www.smh.com.au/business/banking-and-finance',
            'reuters': 'https://www.reuters.com/world/asia-pacific/',
            'bloomberg_au': 'https://www.bloomberg.com/australia'
        },
        'rss_feeds': {
            'rba': 'https://www.rba.gov.au/rss/rss-cb.xml',
            'asx': 'https://www.asx.com.au/asx/statistics/announcements.do',
            'abc_business': 'https://www.abc.net.au/news/feed/51892/rss.xml',
            'afr_companies': 'https://www.afr.com/rss/companies',
            'smh_business': 'https://www.smh.com.au/rss/business.xml',
            'market_index': 'https://www.marketindex.com.au/rss/asx-news',
            'investing_au': 'https://au.investing.com/rss/news_301.rss',
            'afr_rss': 'https://www.afr.com/rss/companies',
            'smh_business': 'https://www.smh.com.au/rss/business.xml',
            'financial_review': 'https://www.financialstandard.com.au/rss.xml',
            'market_index': 'https://www.marketindex.com.au/rss',
            'investing_com_au': 'https://au.investing.com/rss/news.rss'
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