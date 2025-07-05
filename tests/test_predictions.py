import pytest
from src.market_predictor import MarketPredictor

def test_market_prediction():
    predictor = MarketPredictor()
    test_data = [100, 102, 101, 105, 107]  # Sample price data
    prediction = predictor.predict(test_data)
    
    assert isinstance(prediction, dict)
    assert 'signal' in prediction
    assert 'confidence' in prediction
    assert prediction['signal'] in ['bullish', 'bearish', 'neutral']
    assert 0 <= prediction['confidence'] <= 100

def test_prediction_with_insufficient_data():
    predictor = MarketPredictor()
    test_data = []  # Insufficient data
    prediction = predictor.predict(test_data)
    
    assert prediction is None  # Expecting None for insufficient data

def test_prediction_edge_cases():
    predictor = MarketPredictor()
    test_data = [100]  # Edge case with single data point
    prediction = predictor.predict(test_data)
    
    assert prediction is not None  # Should handle single data point gracefully
    assert 'signal' in prediction
    assert 'confidence' in prediction

# Additional tests can be added as needed for comprehensive coverage.