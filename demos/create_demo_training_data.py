#!/usr/bin/env python3
"""
Create demo training data with both positive and negative outcomes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ml_training_pipeline import MLTrainingPipeline
from datetime import datetime, timedelta
import random

def create_demo_training_data():
    """Create realistic training data for ML testing"""
    
    print("🚀 Creating Demo Training Data for ML Testing")
    print("=" * 60)
    
    pipeline = MLTrainingPipeline()
    
    # Create diverse sentiment scenarios
    scenarios = [
        # Positive sentiment scenarios
        {
            'sentiment_data': {
                'symbol': 'CBA.AX',
                'timestamp': (datetime.now() - timedelta(days=10)).isoformat(),
                'overall_sentiment': 0.65,
                'confidence': 0.85,
                'news_count': 12,
                'reddit_sentiment': {'average_sentiment': 0.5},
                'sentiment_components': {'events': 0.3}
            },
            'outcome': {
                'entry_price': 95.50,
                'exit_price': 98.20,
                'profitable': True
            }
        },
        {
            'sentiment_data': {
                'symbol': 'WBC.AX',
                'timestamp': (datetime.now() - timedelta(days=9)).isoformat(),
                'overall_sentiment': 0.45,
                'confidence': 0.75,
                'news_count': 8,
                'reddit_sentiment': {'average_sentiment': 0.3},
                'sentiment_components': {'events': 0.2}
            },
            'outcome': {
                'entry_price': 22.80,
                'exit_price': 23.45,
                'profitable': True
            }
        },
        # Negative sentiment scenarios
        {
            'sentiment_data': {
                'symbol': 'ANZ.AX',
                'timestamp': (datetime.now() - timedelta(days=8)).isoformat(),
                'overall_sentiment': -0.45,
                'confidence': 0.80,
                'news_count': 15,
                'reddit_sentiment': {'average_sentiment': -0.6},
                'sentiment_components': {'events': -0.4}
            },
            'outcome': {
                'entry_price': 25.60,
                'exit_price': 24.30,
                'profitable': False
            }
        },
        {
            'sentiment_data': {
                'symbol': 'NAB.AX',
                'timestamp': (datetime.now() - timedelta(days=7)).isoformat(),
                'overall_sentiment': -0.25,
                'confidence': 0.70,
                'news_count': 6,
                'reddit_sentiment': {'average_sentiment': -0.3},
                'sentiment_components': {'events': -0.1}
            },
            'outcome': {
                'entry_price': 32.10,
                'exit_price': 31.45,
                'profitable': False
            }
        },
        # Mixed scenarios
        {
            'sentiment_data': {
                'symbol': 'CBA.AX',
                'timestamp': (datetime.now() - timedelta(days=6)).isoformat(),
                'overall_sentiment': 0.15,
                'confidence': 0.60,
                'news_count': 5,
                'reddit_sentiment': {'average_sentiment': 0.1},
                'sentiment_components': {'events': 0.05}
            },
            'outcome': {
                'entry_price': 96.80,
                'exit_price': 96.25,
                'profitable': False
            }
        },
        {
            'sentiment_data': {
                'symbol': 'WBC.AX',
                'timestamp': (datetime.now() - timedelta(days=5)).isoformat(),
                'overall_sentiment': 0.35,
                'confidence': 0.90,
                'news_count': 18,
                'reddit_sentiment': {'average_sentiment': 0.4},
                'sentiment_components': {'events': 0.25}
            },
            'outcome': {
                'entry_price': 23.15,
                'exit_price': 24.10,
                'profitable': True
            }
        },
        # More diverse examples
        {
            'sentiment_data': {
                'symbol': 'MQG.AX',
                'timestamp': (datetime.now() - timedelta(days=4)).isoformat(),
                'overall_sentiment': -0.55,
                'confidence': 0.85,
                'news_count': 20,
                'reddit_sentiment': {'average_sentiment': -0.7},
                'sentiment_components': {'events': -0.6}
            },
            'outcome': {
                'entry_price': 180.50,
                'exit_price': 175.20,
                'profitable': False
            }
        },
        {
            'sentiment_data': {
                'symbol': 'ANZ.AX',
                'timestamp': (datetime.now() - timedelta(days=3)).isoformat(),
                'overall_sentiment': 0.75,
                'confidence': 0.95,
                'news_count': 25,
                'reddit_sentiment': {'average_sentiment': 0.8},
                'sentiment_components': {'events': 0.7}
            },
            'outcome': {
                'entry_price': 25.80,
                'exit_price': 27.45,
                'profitable': True
            }
        }
    ]
    
    print(f"Creating {len(scenarios)} training samples...")
    
    # Insert the training data
    feature_ids = []
    for i, scenario in enumerate(scenarios):
        # Collect sentiment features
        feature_id = pipeline.collect_training_data(
            scenario['sentiment_data'], 
            scenario['sentiment_data']['symbol']
        )
        feature_ids.append(feature_id)
        
        # Record trading outcome
        outcome_data = {
            'symbol': scenario['sentiment_data']['symbol'],
            'signal_timestamp': scenario['sentiment_data']['timestamp'],
            'signal_type': 'BUY' if scenario['outcome']['profitable'] else 'SELL',
            'entry_price': scenario['outcome']['entry_price'],
            'exit_price': scenario['outcome']['exit_price'],
            'exit_timestamp': (datetime.now() - timedelta(days=i)).isoformat()
        }
        
        pipeline.record_trading_outcome(feature_id, outcome_data)
        
        print(f"  ✅ Sample {i+1}: {scenario['sentiment_data']['symbol']} - "
              f"Sentiment: {scenario['sentiment_data']['overall_sentiment']:.2f} - "
              f"Outcome: {'Profitable' if scenario['outcome']['profitable'] else 'Loss'}")
    
    # Check final database status
    print(f"\n📊 Final Database Status:")
    import sqlite3
    conn = sqlite3.connect(pipeline.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM sentiment_features")
    feature_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM trading_outcomes")
    outcome_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT outcome_label, COUNT(*) FROM trading_outcomes GROUP BY outcome_label")
    class_distribution = dict(cursor.fetchall())
    
    print(f"  Sentiment features: {feature_count}")
    print(f"  Trading outcomes: {outcome_count}")
    print(f"  Class distribution: {class_distribution}")
    
    conn.close()
    
    print(f"\n✅ Demo training data created successfully!")
    print(f"📈 Ready to train ML models with: python scripts/retrain_ml_models.py --min-samples 5")

if __name__ == "__main__":
    create_demo_training_data()
