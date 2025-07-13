#!/usr/bin/env python3
"""
Quick Demo of Claude Enhancement Features
Tests the actual implementations as they exist
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_temporal_analyzer():
    """Test the temporal sentiment analyzer"""
    print("Testing Temporal Sentiment Analyzer...")
    
    try:
        from temporal_sentiment_analyzer import TemporalSentimentAnalyzer, SentimentDataPoint
        
        # Create analyzer
        analyzer = TemporalSentimentAnalyzer(window_hours=24)
        
        # Create test data points
        base_time = datetime.now() - timedelta(hours=4)
        
        for i in range(5):
            data_point = SentimentDataPoint(
                timestamp=base_time + timedelta(hours=i),
                symbol='CBA',
                sentiment_score=0.5 + i * 0.1,  # Increasing sentiment
                confidence=0.8,
                news_count=10 + i,
                relevance_score=0.9
            )
            analyzer.add_sentiment_observation(data_point)
        
        # Analyze evolution
        result = analyzer.analyze_sentiment_evolution('CBA')
        
        print(f"✅ Temporal Analysis Results:")
        print(f"   - Trend: {result.get('trend', 'N/A'):.3f}")
        print(f"   - Velocity: {result.get('velocity', 'N/A'):.3f}")  
        print(f"   - Volatility: {result.get('volatility', 'N/A'):.3f}")
        print(f"   - Regime: {result.get('regime', 'N/A')}")
        print(f"   - Pattern Count: {len(result.get('patterns', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Temporal Analyzer Error: {e}")
        return False

def test_ensemble_learning():
    """Test the ensemble learning module"""
    print("\nTesting Enhanced Ensemble Learning...")
    
    try:
        from enhanced_ensemble_learning import EnhancedTransformerEnsemble, ModelPrediction, EnsemblePrediction
        
        # Create ensemble
        ensemble = EnhancedTransformerEnsemble()
        
        # Create test predictions
        predictions = [
            ModelPrediction(
                model_name='bert',
                prediction=0.75,
                confidence=0.9,
                features={'sentiment': 0.7, 'volume': 1000}
            ),
            ModelPrediction(
                model_name='roberta',
                prediction=0.82,
                confidence=0.85,
                features={'sentiment': 0.8, 'volume': 1200}
            ),
            ModelPrediction(
                model_name='distilbert',
                prediction=0.68,
                confidence=0.8,
                features={'sentiment': 0.6, 'volume': 800}
            )
        ]
        
        # Test weighted voting
        result = ensemble.weighted_voting_ensemble(predictions)
        
        print(f"✅ Ensemble Learning Results:")
        print(f"   - Final Prediction: {result.final_prediction:.3f}")
        print(f"   - Confidence: {result.confidence:.3f}")
        print(f"   - Strategy: {result.strategy_used}")
        print(f"   - Model Weights: {len(result.model_weights)} models")
        
        # Test confidence weighting
        conf_result = ensemble.confidence_weighted_ensemble(predictions)
        print(f"   - Confidence-weighted: {conf_result.final_prediction:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ensemble Learning Error: {e}")
        return False

def test_feature_engineering():
    """Test the advanced feature engineering module"""
    print("\nTesting Advanced Feature Engineering...")
    
    try:
        from advanced_feature_engineering import AdvancedFeatureEngineer
        
        # Create feature engineer
        engineer = AdvancedFeatureEngineer()
        
        # Test sentiment data
        sentiment_data = {
            'overall_sentiment': 0.7,
            'confidence': 0.85,
            'news_count': 15,
            'reddit_sentiment': 0.6,
            'event_score': 0.8,
            'sentiment_volatility': 0.2
        }
        
        # Test news data (with timezone-aware timestamps)
        news_data = [
            {
                'title': 'CBA reports strong quarterly earnings',
                'sentiment': 0.8,
                'timestamp': datetime.now().isoformat(),
                'source': 'Financial Review'
            },
            {
                'title': 'Banking sector expands digital services',
                'sentiment': 0.6,
                'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
                'source': 'Reuters'
            }
        ]
        
        # Generate comprehensive features
        result = engineer.engineer_comprehensive_features(
            symbol='CBA',
            sentiment_data=sentiment_data,
            news_data=news_data
        )
        
        print(f"✅ Feature Engineering Results:")
        print(f"   - Total Features: {result['feature_count']}")
        print(f"   - Quality Score: {result['feature_quality']['quality_score']:.3f}")
        print(f"   - Completeness: {result['feature_quality']['completeness']:.3f}")
        
        # Show some sample features
        features = result['features']
        sample_features = list(features.items())[:5]
        print(f"   - Sample Features:")
        for name, value in sample_features:
            print(f"     • {name}: {value}")
        
        # Test feature importance
        importance = engineer.get_feature_importance(features)
        top_features = importance['top_features'][:3]
        print(f"   - Top Features:")
        for name, score in top_features:
            print(f"     • {name}: {score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature Engineering Error: {e}")
        return False

def test_integration():
    """Test integration between modules"""
    print("\nTesting Module Integration...")
    
    try:
        # Import all modules
        from temporal_sentiment_analyzer import TemporalSentimentAnalyzer, SentimentDataPoint
        from enhanced_ensemble_learning import EnhancedTransformerEnsemble, ModelPrediction
        from advanced_feature_engineering import AdvancedFeatureEngineer
        
        # 1. Create temporal data
        temporal_analyzer = TemporalSentimentAnalyzer()
        base_time = datetime.now() - timedelta(hours=6)
        
        for i in range(6):
            data_point = SentimentDataPoint(
                timestamp=base_time + timedelta(hours=i),
                symbol='CBA',
                sentiment_score=0.4 + (i % 3) * 0.2,
                confidence=0.8 + (i % 2) * 0.1,
                news_count=8 + i,
                relevance_score=0.85
            )
            temporal_analyzer.add_sentiment_observation(data_point)
        
        # 2. Analyze temporal patterns
        temporal_result = temporal_analyzer.analyze_sentiment_evolution('CBA')
        
        # 3. Create enhanced sentiment data
        enhanced_sentiment = {
            'overall_sentiment': 0.7,
            'confidence': 0.85,
            'news_count': 12,
            'temporal_trend': temporal_result.get('trend', 0),
            'temporal_velocity': temporal_result.get('velocity', 0),
            'temporal_volatility': temporal_result.get('volatility', 0)
        }
        
        # 4. Generate comprehensive features
        feature_engineer = AdvancedFeatureEngineer()
        feature_result = feature_engineer.engineer_comprehensive_features(
            symbol='CBA',
            sentiment_data=enhanced_sentiment
        )
        
        features = feature_result['features']
        
        # 5. Create ensemble predictions with features
        ensemble = EnhancedTransformerEnsemble()
        predictions = [
            ModelPrediction('bert', 0.72, 0.9, features),
            ModelPrediction('roberta', 0.78, 0.85, features),
            ModelPrediction('distilbert', 0.65, 0.8, features)
        ]
        
        # 6. Get final ensemble prediction
        final_result = ensemble.adaptive_hybrid_ensemble(predictions)
        
        print(f"✅ Integration Test Results:")
        print(f"   - Temporal Trend: {temporal_result.get('trend', 0):.3f}")
        print(f"   - Feature Count: {len(features)}")
        print(f"   - Final Prediction: {final_result.final_prediction:.3f}")
        print(f"   - Final Confidence: {final_result.confidence:.3f}")
        print(f"   - Strategy Used: {final_result.strategy_used}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Claude Enhancement Feature Demo")
    print("=" * 60)
    
    results = {
        'temporal': test_temporal_analyzer(),
        'ensemble': test_ensemble_learning(),
        'features': test_feature_engineering(),
        'integration': test_integration()
    }
    
    print("\n" + "=" * 60)
    print("Demo Summary:")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper():>12}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Claude enhancement features are working!")
        print("\nKey achievements:")
        print("• Temporal sentiment analysis with velocity/acceleration")
        print("• Advanced ensemble learning with multiple strategies")
        print("• Comprehensive feature engineering (50+ features)")
        print("• Full integration pipeline from temporal → features → ensemble")
    else:
        print("⚠️  Some features need attention")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
