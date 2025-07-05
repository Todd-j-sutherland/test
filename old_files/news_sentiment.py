from textblob import TextBlob
import requests

class NewsSentimentAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"

    def fetch_news(self, query, from_date, to_date):
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'sortBy': 'relevancy',
            'apiKey': self.api_key
        }
        response = requests.get(self.base_url, params=params)
        return response.json()

    def analyze_sentiment(self, articles):
        sentiments = []
        for article in articles:
            analysis = TextBlob(article['title'])
            sentiments.append(analysis.sentiment.polarity)
        return sentiments

    def get_average_sentiment(self, query, from_date, to_date):
        news_data = self.fetch_news(query, from_date, to_date)
        if news_data['status'] == 'ok':
            sentiments = self.analyze_sentiment(news_data['articles'])
            return sum(sentiments) / len(sentiments) if sentiments else 0
        return None

# Example usage:
# analyzer = NewsSentimentAnalyzer(api_key='your_api_key_here')
# average_sentiment = analyzer.get_average_sentiment('CBA', '2023-01-01', '2023-01-31')
# print(average_sentiment)