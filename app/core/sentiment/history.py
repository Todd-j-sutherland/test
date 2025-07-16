"""Sentiment history management"""

class SentimentHistoryManager:
    """Manages sentiment history data"""
    
    def __init__(self):
        self.history = []
    
    def add_sentiment_record(self, record):
        """Add a sentiment record to history"""
        self.history.append(record)
