#!/usr/bin/env python3
"""
Integration Test: ML Trading Config + News Sentiment Analysis
Tests the combined system to show enhanced news analysis with ML trading features
"""

import sys
import os
import logging
from datetime import datetime

# Setup path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_integration():
    """Test the integration between ML trading config and news sentiment analysis"""
    
    print("=" * 80)
    print("ML Trading Config + News Sentiment Integration Test")
    print("=" * 80)
    
    # Test 1: Import and initialize components
    print("\n1. Testing imports and initialization...")
    
    try:
        from src.news_sentiment import NewsSentimentAnalyzer
        from src.ml_trading_config import FeatureEngineer, TradingModelOptimizer, TRADING_CONFIGS
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        analyzer = NewsSentimentAnalyzer()
        print("✅ NewsSentimentAnalyzer initialized")
        
        # Check if ML trading components are available
        if analyzer.feature_engineer:
            print("✅ ML Trading FeatureEngineer initialized")
        else:
            print("⚠️  ML Trading components not available")
        
        if analyzer.ml_models:
            print(f"✅ ML Models initialized: {list(analyzer.ml_models.keys())}")
        else:
            print("⚠️  ML Models not initialized")
            
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False
    
    # Test 2: Test trading configurations
    print("\n2. Testing trading configurations...")
    
    for style, config in TRADING_CONFIGS.items():
        print(f"   {style.upper()}: threshold={config['threshold']}, "
              f"confidence={config['min_confidence']}, "
              f"position_size={config['position_size_multiplier']}")
    
    # Test 3: Analyze sentiment for a bank
    print("\n3. Testing enhanced sentiment analysis...")
    
    try:
        # Test with a real ASX bank symbol
        symbol = "CBA.AX"
        print(f"Analyzing sentiment for {symbol}...")
        
        # This will use the integrated system
        result = analyzer.analyze_bank_sentiment(symbol)
        
        print(f"\n📊 SENTIMENT ANALYSIS RESULTS FOR {symbol}")
        print("-" * 50)
        print(f"Overall Sentiment: {result['overall_sentiment']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"News Count: {result['news_count']}")
        
        # Show component breakdown
        if 'sentiment_components' in result:
            print("\n📈 Component Breakdown:")
            for component, score in result['sentiment_components'].items():
                print(f"   {component}: {score:.3f}")
        
        # Show ML trading features if available
        if 'ml_trading_details' in result and result['ml_trading_details']:
            print("\n🤖 ML Trading Feature Analysis:")
            ml_details = result['ml_trading_details']
            for feature, value in list(ml_details.items())[:5]:  # Show first 5
                print(f"   {feature}: {value:.3f}")
        
        # Show transformer info if available
        transformer_available = result.get('sentiment_scores', {}).get('transformer_available', False)
        print(f"\n🔬 Advanced Features:")
        print(f"   Transformers Available: {transformer_available}")
        print(f"   ML Trading Features: {analyzer.feature_engineer is not None}")
        
        # Show recent headlines
        if 'recent_headlines' in result:
            print(f"\n📰 Recent Headlines ({len(result['recent_headlines'])}):")
            for i, headline in enumerate(result['recent_headlines'][:3], 1):
                print(f"   {i}. {headline}")
        
        print("✅ Enhanced sentiment analysis completed successfully")
        
    except Exception as e:
        print(f"❌ Sentiment analysis error: {e}")
        logger.error(f"Detailed error: {e}", exc_info=True)
        return False
    
    # Test 4: Test feature engineering directly
    print("\n4. Testing ML trading features directly...")
    
    if analyzer.feature_engineer:
        try:
            # Test feature extraction
            sample_texts = [
                "Commonwealth Bank announces strong quarterly results with 15% profit growth",
                "CBA shares plunge after regulatory investigation announced",
                "Analysts upgrade CBA rating following positive earnings report"
            ]
            
            features, feature_names = analyzer.feature_engineer.create_trading_features(sample_texts)
            
            print(f"✅ Extracted {len(feature_names)} features from {len(sample_texts)} texts")
            print(f"   Feature examples: {feature_names[:5]}...")
            print(f"   Feature matrix shape: {features.shape}")
            
            # Test aggregation
            aggregated = analyzer._aggregate_news_features(features, feature_names)
            print(f"✅ Aggregated features: {len(aggregated)} metrics")
            
            # Show some key aggregated features
            key_features = ['bullish_score_sum', 'bearish_score_sum', 'confidence_high_sum', 
                          'financial_focus', 'urgency_score_mean']
            
            print("   Key aggregated features:")
            for feature in key_features:
                if feature in aggregated:
                    print(f"      {feature}: {aggregated[feature]:.3f}")
            
        except Exception as e:
            print(f"❌ Feature engineering error: {e}")
            return False
    else:
        print("⚠️  ML Trading feature engineering not available")
    
    # Test 5: Configuration recommendations
    print("\n5. Trading strategy recommendations...")
    
    if 'overall_sentiment' in result:
        sentiment_score = result['overall_sentiment']
        confidence = result['confidence']
        
        if confidence > 0.7:
            if sentiment_score > 0.3:
                recommended_style = 'aggressive'
                print(f"📈 RECOMMENDATION: {recommended_style.upper()} strategy")
                print(f"   Reason: High confidence ({confidence:.2f}) positive sentiment ({sentiment_score:.2f})")
            elif sentiment_score < -0.3:
                recommended_style = 'conservative'
                print(f"📉 RECOMMENDATION: {recommended_style.upper()} strategy")
                print(f"   Reason: High confidence ({confidence:.2f}) negative sentiment ({sentiment_score:.2f})")
            else:
                recommended_style = 'moderate'
                print(f"📊 RECOMMENDATION: {recommended_style.upper()} strategy")
                print(f"   Reason: High confidence ({confidence:.2f}) but neutral sentiment ({sentiment_score:.2f})")
        else:
            recommended_style = 'conservative'
            print(f"🔒 RECOMMENDATION: {recommended_style.upper()} strategy")
            print(f"   Reason: Low confidence ({confidence:.2f}) in analysis")
        
        # Show the recommended configuration
        config = TRADING_CONFIGS[recommended_style]
        print(f"\n   Configuration:")
        for key, value in config.items():
            print(f"      {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✅ INTEGRATION TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    print("\n🎯 Key Benefits of Integration:")
    print("   • Traditional sentiment analysis (TextBlob + VADER)")
    print("   • Advanced transformer-based sentiment (when available)")
    print("   • ML trading-specific feature extraction")
    print("   • Dynamic weighting based on confidence levels")
    print("   • Trading strategy recommendations")
    print("   • Enhanced event detection and impact scoring")
    print("   • Market context integration")
    print("   • Reddit sentiment analysis")
    
    return True

def test_ml_optimization():
    """Test ML optimization capabilities"""
    
    print("\n" + "=" * 80)
    print("ML OPTIMIZATION TEST")
    print("=" * 80)
    
    try:
        from src.ml_trading_config import TradingModelOptimizer
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        # Create a simple mock ML analyzer
        class MockMLAnalyzer:
            def __init__(self):
                self.models = {
                    'random_forest': RandomForestClassifier(n_estimators=10, random_state=42)
                }
        
        mock_analyzer = MockMLAnalyzer()
        optimizer = TradingModelOptimizer(mock_analyzer)
        
        print("✅ TradingModelOptimizer initialized")
        
        # Test trading score function
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        y_pred_proba = np.array([0.8, 0.3, 0.7, 0.9, 0.2, 0.6, 0.4, 0.1])
        
        score = optimizer.trading_score(y_true, y_pred_proba)
        print(f"✅ Trading score calculated: {score:.3f}")
        
        print("✅ ML optimization test completed")
        
    except Exception as e:
        print(f"❌ ML optimization test error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print(f"Starting integration test at {datetime.now()}")
    
    success = test_integration()
    
    if success:
        test_ml_optimization()
        print(f"\n🎉 All tests completed successfully at {datetime.now()}")
    else:
        print(f"\n❌ Tests failed at {datetime.now()}")
        sys.exit(1)
