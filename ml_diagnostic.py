#!/usr/bin/env python3
"""
ML System Diagnostic Tool
Diagnoses why ML predictions aren't showing in the dashboard
"""

import os
import json
import sys
from datetime import datetime

def check_file_structure():
    """Check if all required files exist"""
    print("🔍 Checking File Structure...")
    
    files_to_check = [
        "data/ml_models/",
        "data/ml_models/models/current_model.pkl",
        "data/ml_models/models/current_metadata.json", 
        "data/ml_models/models/feature_scaler.pkl",
        "data/ml_models/training_data.db",
        "src/ml_training_pipeline.py",
        "core/smart_collector.py"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                print(f"✅ Directory exists: {file_path}")
            else:
                size = os.path.getsize(file_path)
                print(f"✅ File exists: {file_path} ({size} bytes)")
        else:
            print(f"❌ Missing: {file_path}")
    
    return True

def check_model_metadata():
    """Check model metadata and version"""
    print("\n📊 Checking Model Metadata...")
    
    metadata_path = "data/ml_models/models/current_metadata.json"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"✅ Model Type: {metadata.get('model_type', 'Unknown')}")
            print(f"✅ Version: {metadata.get('version', 'Unknown')}")
            print(f"✅ Training Date: {metadata.get('training_date', 'Unknown')}")
            print(f"✅ Feature Count: {len(metadata.get('feature_columns', []))}")
            
            performance = metadata.get('performance', {})
            if performance:
                print(f"✅ CV Score: {performance.get('avg_cv_score', 'N/A')}")
                
            feature_columns = metadata.get('feature_columns', [])
            if feature_columns:
                print(f"✅ Features: {', '.join(feature_columns[:5])}{'...' if len(feature_columns) > 5 else ''}")
            
        except Exception as e:
            print(f"❌ Error reading metadata: {e}")
    else:
        print("❌ Metadata file not found")

def check_training_data():
    """Check training data availability"""
    print("\n📈 Checking Training Data...")
    
    try:
        # Add the src directory to the path
        sys.path.append('src')
        from ml_training_pipeline import MLTrainingPipeline
        
        ml_pipeline = MLTrainingPipeline()
        X, y = ml_pipeline.prepare_training_dataset(min_samples=1)
        
        if X is not None:
            print(f"✅ Training samples: {len(X)}")
            print(f"✅ Feature columns: {len(X.columns) if hasattr(X, 'columns') else 'Unknown'}")
            print(f"✅ Positive outcomes: {sum(y)} / {len(y)} ({sum(y)/len(y)*100:.1f}%)")
            
            if len(X) >= 50:
                print("✅ Sufficient data for training")
            else:
                print(f"⚠️  Need more data (have {len(X)}, need 50+)")
        else:
            print("❌ No training data available")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running from the project root directory")
    except Exception as e:
        print(f"❌ Error checking training data: {e}")

def check_model_loading():
    """Test model loading"""
    print("\n🤖 Testing Model Loading...")
    
    try:
        import joblib
        import pandas as pd
        import numpy as np
        
        model_path = "data/ml_models/models/current_model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✅ Model loaded successfully: {type(model).__name__}")
            
            # Test prediction with dummy data
            test_features = pd.DataFrame([{
                'sentiment_score': 0.5,
                'confidence': 0.8,
                'news_count': 10,
                'reddit_sentiment': 0.3,
                'event_score': 1,
                'sentiment_confidence_interaction': 0.4,
                'news_volume_category': 1,
                'hour': 12,
                'day_of_week': 1,
                'is_market_hours': 1
            }])
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(test_features)
                print(f"✅ Test prediction successful: {probabilities[0]}")
            else:
                prediction = model.predict(test_features)
                print(f"✅ Test prediction successful: {prediction[0]}")
                
        else:
            print("❌ Model file not found")
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")

def check_sentiment_data():
    """Check recent sentiment analysis data"""
    print("\n📰 Checking Sentiment Data...")
    
    sentiment_dir = "data/sentiment_history"
    if os.path.exists(sentiment_dir):
        files = [f for f in os.listdir(sentiment_dir) if f.endswith('_history.json')]
        print(f"✅ Found {len(files)} sentiment history files")
        
        if files:
            # Check the first file
            sample_file = os.path.join(sentiment_dir, files[0])
            try:
                with open(sample_file, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list) and data:
                    latest = data[-1]
                    print(f"✅ Sample sentiment data structure:")
                    print(f"   - overall_sentiment: {latest.get('overall_sentiment', 'Missing')}")
                    print(f"   - confidence: {latest.get('confidence', 'Missing')}")
                    print(f"   - news_count: {latest.get('news_count', 'Missing')}")
                    print(f"   - ml_prediction: {'Present' if 'ml_prediction' in latest else 'Missing'}")
                    
                    if 'ml_prediction' in latest:
                        ml_pred = latest['ml_prediction']
                        print(f"     * prediction: {ml_pred.get('prediction', 'N/A')}")
                        print(f"     * probability: {ml_pred.get('probability', 'N/A')}")
                else:
                    print("⚠️  Sentiment data format unexpected")
                    
            except Exception as e:
                print(f"❌ Error reading sentiment data: {e}")
    else:
        print("❌ Sentiment history directory not found")

def run_full_diagnostic():
    """Run complete diagnostic"""
    print("🚀 ML System Diagnostic Tool")
    print("=" * 50)
    
    check_file_structure()
    check_model_metadata()
    check_training_data()
    check_model_loading()
    check_sentiment_data()
    
    print("\n" + "=" * 50)
    print("📋 RECOMMENDATIONS:")
    
    # Check what needs to be done
    model_exists = os.path.exists("data/ml_models/models/current_model.pkl")
    metadata_exists = os.path.exists("data/ml_models/models/current_metadata.json")
    
    if not model_exists or not metadata_exists:
        print("1. 🔨 Train ML models:")
        print("   python scripts/retrain_ml_models.py")
        print("   python daily_manager.py test")
    
    try:
        sys.path.append('src')
        from ml_training_pipeline import MLTrainingPipeline
        ml_pipeline = MLTrainingPipeline()
        X, y = ml_pipeline.prepare_training_dataset(min_samples=1)
        if X is None or len(X) < 50:
            print("2. 📊 Collect more training data:")
            print("   python core/smart_collector.py --once")
            print("   python daily_manager.py morning")
    except:
        print("2. ⚠️  Check training data availability")
    
    print("3. 🔄 Restart dashboard:")
    print("   python tools/launch_dashboard_auto.py")
    
    print("4. 🔧 Apply the dashboard fix from the artifacts above")

if __name__ == "__main__":
    run_full_diagnostic()
