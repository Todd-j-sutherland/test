#!/usr/bin/env python3
"""
Debug ML Prediction Display
Test what's actually being returned by the ML pipeline in the dashboard context
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.ml.enhanced_pipeline import EnhancedMLPipeline

def test_prediction_conversion():
    """Test ML prediction to signal conversion"""
    print("🔍 Testing ML Prediction to Signal Conversion")
    print("=" * 50)
    
    # Initialize ML pipeline
    pipeline = EnhancedMLPipeline("data/ml_models")
    
    # Mock sentiment data (similar to what dashboard would have)
    sentiment_data = {
        'overall_sentiment': 0.033,
        'confidence': 0.59,
        'news_count': 8,
        'symbol': 'CBA.AX'
    }
    
    # Mock market data
    market_data = {
        'price': 100.0,
        'change_percent': 0.0,
        'volume': 1000000,
        'volatility': 0.15
    }
    
    # Get prediction
    prediction_result = pipeline.predict(sentiment_data, market_data, [])
    
    print(f"📊 Raw Prediction Result: {prediction_result}")
    
    if 'error' not in prediction_result:
        # Convert numeric prediction to signal string (same logic as dashboard)
        ensemble_pred = prediction_result.get('ensemble_prediction', 0)
        signal_mapping = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        ml_signal = signal_mapping.get(ensemble_pred, 'HOLD')
        
        print(f"🎯 Ensemble Prediction (numeric): {ensemble_pred}")
        print(f"📈 Converted Signal: {ml_signal}")
        print(f"📊 Confidence: {prediction_result.get('ensemble_confidence', 0)}")
        print(f"🔢 Feature Count: {prediction_result.get('feature_count', 0)}")
        
        # Test what would appear in the dataframe
        ml_row = {
            'Bank': 'CBA.AX',
            'ML Prediction': ml_signal,
            'ML Confidence': prediction_result.get('ensemble_confidence', 0),
            'Sentiment Signal': 'HOLD',
            'Feature Count': prediction_result.get('feature_count', 0)
        }
        
        print(f"\n📋 Dashboard Row Data:")
        for key, value in ml_row.items():
            print(f"   {key}: {value}")
    else:
        print(f"❌ Error in prediction: {prediction_result}")

if __name__ == '__main__':
    test_prediction_conversion()
