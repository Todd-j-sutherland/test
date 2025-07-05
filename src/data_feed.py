import requests
import pandas as pd
import os

class DataFeed:
    def __init__(self):
        self.api_url = "https://api.example.com/data"  # Replace with actual API URL

    def fetch_data(self, bank_symbol):
        response = requests.get(f"{self.api_url}?symbol={bank_symbol}")
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error fetching data for {bank_symbol}: {response.status_code}")

    def fetch_historical_data(self, bank_symbol, start_date, end_date):
        response = requests.get(f"{self.api_url}/historical?symbol={bank_symbol}&start={start_date}&end={end_date}")
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            raise Exception(f"Error fetching historical data for {bank_symbol}: {response.status_code}")

    def save_to_cache(self, bank_symbol, data):
        cache_dir = os.path.join(os.path.dirname(__file__), '../data/cache')
        os.makedirs(cache_dir, exist_ok=True)
        file_path = os.path.join(cache_dir, f"{bank_symbol}.json")
        with open(file_path, 'w') as f:
            f.write(data)

    def load_from_cache(self, bank_symbol):
        cache_dir = os.path.join(os.path.dirname(__file__), '../data/cache')
        file_path = os.path.join(cache_dir, f"{bank_symbol}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return f.read()
        else:
            raise FileNotFoundError(f"No cached data found for {bank_symbol}")