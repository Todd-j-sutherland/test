#!/usr/bin/env python3
"""Debug script to isolate the analyze_bank issue"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_feed import ASXDataFeed
from src.technical_analysis import TechnicalAnalyzer
from src.fundamental_analysis import FundamentalAnalyzer
from src.news_sentiment import NewsSentimentAnalyzer
from src.risk_calculator import RiskRewardCalculator
from src.market_predictor import MarketPredictor
from config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_analyze_bank():
    """Debug analyze_bank method step by step"""
    
    settings = Settings()
    symbol = "CBA.AX"
    
    print(f"\n=== Debugging analysis for {symbol} ===")
    
    # Initialize components
    data_feed = ASXDataFeed()
    technical = TechnicalAnalyzer()
    fundamental = FundamentalAnalyzer()
    sentiment = NewsSentimentAnalyzer()
    risk_calc = RiskRewardCalculator()
    predictor = MarketPredictor()
    
    try:
        # Step 1: Get market data
        print("Step 1: Getting market data...")
        market_data = data_feed.get_historical_data(symbol, period=settings.DEFAULT_ANALYSIS_PERIOD)
        if market_data is None or market_data.empty:
            print(f"ERROR: No market data for {symbol}")
            return
        print(f"Market data shape: {market_data.shape}")
        print(f"Market data columns: {market_data.columns.tolist()}")
        
        # Step 2: Technical analysis
        print("\nStep 2: Technical analysis...")
        technical_signals = technical.analyze(symbol, market_data)
        print(f"Technical signals type: {type(technical_signals)}")
        if isinstance(technical_signals, dict):
            print(f"Technical signals keys: {list(technical_signals.keys())}")
        else:
            print(f"ERROR: Technical signals is not a dict: {technical_signals}")
            return
        
        # Step 3: Fundamental analysis
        print("\nStep 3: Fundamental analysis...")
        fundamental_metrics = fundamental.analyze(symbol)
        print(f"Fundamental metrics type: {type(fundamental_metrics)}")
        if isinstance(fundamental_metrics, dict):
            print(f"Fundamental metrics keys: {list(fundamental_metrics.keys())}")
        else:
            print(f"ERROR: Fundamental metrics is not a dict: {fundamental_metrics}")
            return
        
        # Step 4: News sentiment
        print("\nStep 4: News sentiment...")
        sentiment_score = sentiment.analyze_bank_sentiment(symbol)
        print(f"Sentiment score type: {type(sentiment_score)}")
        if isinstance(sentiment_score, dict):
            print(f"Sentiment score keys: {list(sentiment_score.keys())}")
        else:
            print(f"ERROR: Sentiment score is not a dict: {sentiment_score}")
            return
        
        # Step 5: Risk/reward calculation
        print("\nStep 5: Risk/reward calculation...")
        current_price = float(market_data['Close'].iloc[-1])
        print(f"Current price: {current_price}")
        
        # Check what types we're passing
        print(f"technical_signals type: {type(technical_signals)}")
        print(f"fundamental_metrics type: {type(fundamental_metrics)}")
        
        risk_reward = risk_calc.calculate(symbol, current_price, technical_signals, fundamental_metrics)
        print(f"Risk reward type: {type(risk_reward)}")
        if isinstance(risk_reward, dict):
            print(f"Risk reward keys: {list(risk_reward.keys())}")
        else:
            print(f"ERROR: Risk reward is not a dict: {risk_reward}")
            return
        
        # Step 6: Market prediction
        print("\nStep 6: Market prediction...")
        prediction = predictor.predict(
            symbol,
            technical_signals, 
            fundamental_metrics, 
            sentiment_score
        )
        print(f"Prediction type: {type(prediction)}")
        if isinstance(prediction, dict):
            print(f"Prediction keys: {list(prediction.keys())}")
        else:
            print(f"ERROR: Prediction is not a dict: {prediction}")
            return
        
        print("\n=== All steps completed successfully! ===")
        
    except Exception as e:
        print(f"ERROR in debug_analyze_bank: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_analyze_bank()
