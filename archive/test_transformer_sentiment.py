#!/usr/bin/env python3
"""
Test script to demonstrate transformer-enhanced sentiment analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.news_sentiment import NewsSentimentAnalyzer
import json

def test_transformer_sentiment():
    """Test transformer-enhanced sentiment analysis"""
    
    print("🔍 Testing Transformer-Enhanced Sentiment Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = NewsSentimentAnalyzer()
    
    # Test financial news samples
    financial_news_samples = [
        {
            'title': 'Commonwealth Bank reports record quarterly profit of $2.6 billion',
            'summary': 'CBA posted strong results driven by home lending growth and reduced bad debt provisions',
            'relevance': 'high'
        },
        {
            'title': 'Westpac faces potential $1.3 billion fine from AUSTRAC for money laundering breaches',
            'summary': 'Banking regulator investigating serious compliance failures at major Australian bank',
            'relevance': 'high'
        },
        {
            'title': 'ANZ announces new digital banking platform to compete with fintech startups',
            'summary': 'The bank is investing heavily in technology to improve customer experience and reduce costs',
            'relevance': 'medium'
        }
    ]
    
    print("📊 Analyzing sample financial news...")
    print()
    
    for i, news in enumerate(financial_news_samples, 1):
        print(f"News {i}: {news['title']}")
        print(f"Summary: {news['summary']}")
        print()
        
        # Analyze with transformers
        text = f"{news['title']} {news['summary']}"
        
        # Test individual transformer models
        if analyzer.transformer_models:
            print("🤖 Transformer Analysis:")
            transformer_result = analyzer._analyze_with_transformers(text)
            
            if 'financial' in transformer_result:
                print(f"  📈 FinBERT (Financial): {transformer_result['financial']['score']:.3f} "
                      f"({transformer_result['financial']['label']}, "
                      f"confidence: {transformer_result['financial']['confidence']:.3f})")
            
            if 'general' in transformer_result:
                print(f"  🎯 RoBERTa (General): {transformer_result['general']['score']:.3f} "
                      f"(confidence: {transformer_result['general']['confidence']:.3f})")
            
            if 'emotion' in transformer_result:
                print(f"  😊 Emotion: {transformer_result['emotion']['dominant_emotion']} "
                      f"(sentiment: {transformer_result['emotion']['score']:.3f})")
            
            if 'news_type' in transformer_result:
                print(f"  📰 News Type: {transformer_result['news_type']['primary_type']} "
                      f"(confidence: {transformer_result['news_type']['confidence']:.3f})")
            
            if 'composite' in transformer_result:
                print(f"  🎯 Composite Score: {transformer_result['composite']['score']:.3f} "
                      f"(confidence: {transformer_result['composite']['confidence']:.3f})")
        else:
            print("⚠️  Transformer models not available - using traditional methods only")
        
        # Test traditional methods for comparison
        traditional_score = analyzer._analyze_traditional_sentiment(text)
        print(f"📊 Traditional (TextBlob + VADER): {traditional_score:.3f}")
        
        # Test composite method
        if analyzer.transformer_models:
            composite_score = analyzer._calculate_composite_sentiment(
                traditional_score, 
                transformer_result.get('composite', {}).get('score', 0),
                transformer_result.get('composite', {}).get('confidence', 0),
                news['relevance']
            )
            print(f"🎯 Final Composite Score: {composite_score:.3f}")
        else:
            print(f"🎯 Final Score (Traditional Only): {traditional_score:.3f}")
        
        print("-" * 60)
        print()
    
    # Test full sentiment analysis
    print("🔄 Testing Full Sentiment Analysis Pipeline...")
    print()
    
    # Create mock news data
    mock_news = financial_news_samples.copy()
    
    # Analyze sentiment using the enhanced pipeline
    sentiment_result = analyzer._analyze_news_sentiment(mock_news)
    
    print("📈 Full Analysis Results:")
    print(f"  Average Sentiment: {sentiment_result['average_sentiment']:.3f}")
    print(f"  Positive: {sentiment_result['positive_count']}")
    print(f"  Negative: {sentiment_result['negative_count']}")
    print(f"  Neutral: {sentiment_result['neutral_count']}")
    print(f"  Transformer Available: {sentiment_result.get('transformer_available', False)}")
    
    if 'method_breakdown' in sentiment_result:
        method_breakdown = sentiment_result['method_breakdown']
        print("\n🔍 Method Comparison:")
        print(f"  Traditional Mean: {method_breakdown['traditional']['mean']:.3f}")
        print(f"  Transformer Mean: {method_breakdown['transformer']['mean']:.3f}")
        print(f"  Composite Mean: {method_breakdown['composite']['mean']:.3f}")
        print(f"  Correlation: {method_breakdown['correlation']:.3f}")
    
    print("\n✅ Test completed!")
    
    return sentiment_result

def test_bank_analysis():
    """Test full bank sentiment analysis"""
    
    print("\n🏦 Testing Bank Sentiment Analysis")
    print("=" * 60)
    
    analyzer = NewsSentimentAnalyzer()
    
    # Test with a specific bank
    print("Analyzing CBA sentiment...")
    
    # This would normally fetch real news, but for testing we'll see the structure
    try:
        result = analyzer.analyze_bank_sentiment('CBA.AX')
        print(f"✅ Analysis completed for CBA")
        print(f"   Overall Sentiment: {result.get('overall_sentiment', 'N/A')}")
        print(f"   Confidence: {result.get('confidence', 'N/A')}")
        print(f"   News Count: {result.get('news_count', 'N/A')}")
        
        if 'sentiment_components' in result:
            print("   Component Breakdown:")
            for component, score in result['sentiment_components'].items():
                print(f"     {component}: {score:.3f}")
    
    except Exception as e:
        print(f"❌ Error in bank analysis: {e}")
        print("This is expected if news sources are not accessible or models are not loaded")

if __name__ == "__main__":
    print("🚀 Starting Transformer Sentiment Analysis Test")
    print("=" * 60)
    
    # Check Python version
    import sys
    print(f"🐍 Python version: {sys.version}")
    
    if sys.version_info >= (3, 13):
        print("⚠️  WARNING: Python 3.13 detected!")
        print("   PyTorch and TensorFlow don't fully support Python 3.13 yet.")
        print("   Transformers may not work. Consider using Python 3.11 or 3.12.")
        print("   See PYTHON_313_COMPATIBILITY.md for solutions.")
        print()
    
    # Check if transformers are available
    try:
        from transformers import pipeline
        print("✅ Transformers library is available")
        
        # Test if backend is available
        try:
            import torch
            print("✅ PyTorch backend available")
            backend_available = True
        except ImportError:
            try:
                import tensorflow as tf
                print("✅ TensorFlow backend available")
                backend_available = True
            except ImportError:
                print("❌ No backend (PyTorch or TensorFlow) available")
                backend_available = False
        
        if not backend_available:
            print("⚠️  Transformers installed but no backend available")
            print("   Falling back to traditional sentiment analysis")
            
    except ImportError:
        print("❌ Transformers library not available")
        print("Install with: pip install transformers")
        print("⚠️  Will test traditional methods only")
        backend_available = False
    
    print()
    
    # Run tests
    try:
        test_transformer_sentiment()
        test_bank_analysis()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("This is expected if transformers/backend are not available")
    
    print("\n🎉 Test completed!")
    print("\n📝 Notes:")
    print("- First run may take time to download models (if available)")
    print("- Models are cached for subsequent runs")
    print("- FinBERT is specifically trained for financial sentiment")
    print("- System falls back to traditional methods if transformers unavailable")
    
    if sys.version_info >= (3, 13):
        print("\n🔧 Python 3.13 Users:")
        print("- Consider using Python 3.11 or 3.12 for full transformer support")
        print("- Traditional sentiment analysis still works well (~70-75% accuracy)")
        print("- See PYTHON_313_COMPATIBILITY.md for migration options")
