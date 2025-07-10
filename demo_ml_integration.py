#!/usr/bin/env python3
"""
ML Integration Demo Script
Demonstrates the new ML capabilities in the trading analysis system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from news_trading_analyzer import NewsTradingAnalyzer
from src.ml_training_pipeline import MLTrainingPipeline
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("🚀 ML Integration Demo - Trading Analysis System")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing ML Training Pipeline...")
    ml_pipeline = MLTrainingPipeline()
    print("✅ ML Pipeline initialized")
    
    print("\n2. Initializing News Trading Analyzer...")
    analyzer = NewsTradingAnalyzer()
    print("✅ News Trading Analyzer initialized")
    
    # Check ML integration
    print("\n3. Checking ML Integration...")
    has_ml_pipeline = hasattr(analyzer.sentiment_analyzer, 'ml_pipeline')
    has_outcome_tracker = analyzer.outcome_tracker is not None
    
    print(f"   ML Pipeline integrated: {'✅' if has_ml_pipeline else '❌'}")
    print(f"   Outcome Tracker available: {'✅' if has_outcome_tracker else '❌'}")
    
    # Test analyze_and_track method
    print("\n4. Testing analyze_and_track method...")
    try:
        # Test with a sample bank
        symbol = "CBA.AX"
        print(f"   Analyzing {symbol}...")
        
        result = analyzer.analyze_and_track(symbol)
        
        if 'error' in result:
            print(f"   ⚠️  Analysis completed with warnings: {result['error']}")
        else:
            print(f"   ✅ Analysis completed successfully")
            print(f"   Signal: {result.get('signal', 'Unknown')}")
            print(f"   Sentiment Score: {result.get('sentiment_score', 0):.3f}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            
            # Check if trade was tracked
            if 'trade_id' in result:
                print(f"   📝 Trade tracked with ID: {result['trade_id']}")
            else:
                print(f"   ℹ️  No trade recorded (signal was {result.get('signal', 'Unknown')})")
                
    except Exception as e:
        print(f"   ❌ Error during analysis: {e}")
    
    # Check database status
    print("\n5. Checking ML Database Status...")
    try:
        import sqlite3
        conn = sqlite3.connect(ml_pipeline.db_path)
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"   Database tables: {[t[0] for t in tables]}")
        
        # Check data counts
        cursor.execute("SELECT COUNT(*) FROM sentiment_features")
        feature_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM trading_outcomes")
        outcome_count = cursor.fetchone()[0]
        
        print(f"   Sentiment features: {feature_count}")
        print(f"   Trading outcomes: {outcome_count}")
        
        conn.close()
        
    except Exception as e:
        print(f"   ❌ Database error: {e}")
    
    # Show next steps
    print("\n6. Next Steps:")
    print("   📊 Run the system regularly to collect training data")
    print("   🎯 After collecting 100+ samples, train your first ML model:")
    print("      python scripts/retrain_ml_models.py --min-samples 100")
    print("   🚀 Set up automated training:")
    print("      python scripts/schedule_ml_training.py")
    print("   📈 View ML predictions in the dashboard:")
    print("      python news_analysis_dashboard.py")
    
    print("\n" + "=" * 60)
    print("✅ ML Integration Demo Complete!")

if __name__ == "__main__":
    main()
