BANK_SYMBOLS = ['CBA.AX', 'WBC.AX', 'ANZ.AX', 'NAB.AX', 'MQG.AX', 'BEN.AX']

ALERT_THRESHOLDS = {
    'strong_buy': 70,
    'buy': 50,
    'sell': -50,
    'strong_sell': -70
}

CACHE_SETTINGS = {
    'duration_minutes': 15,
    'max_size_mb': 100
}

ANALYSIS_SETTINGS = {
    'default_analysis_period': '3mo',
    'confidence_threshold': 70,
    'risk_tolerance': 'medium'
}

DATA_SOURCES = {
    'use_yahoo_finance': True,
    'use_web_scraping': True,
    'use_free_news_api': True
}