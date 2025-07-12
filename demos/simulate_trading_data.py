#!/usr/bin/env python3
"""
ML Training Data Simulation
Simulates trading outcomes for existing sentiment features to test ML training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ml_training_pipeline import MLTrainingPipeline
from datetime import datetime, timedelta
import random
import sqlite3

def simulate_trading_outcomes():
    """Simulate trading outcomes for existing sentiment features"""
    
    print("🎲 ML Training Data Simulation")
    print("=" * 50)
    
    pipeline = MLTrainingPipeline()
    conn = sqlite3.connect(pipeline.db_path)
    cursor = conn.cursor()
    
    # Get existing sentiment features
    cursor.execute('SELECT id, symbol, timestamp, sentiment_score, confidence FROM sentiment_features')
    features = cursor.fetchall()
    
    print(f"Found {len(features)} sentiment features without outcomes")
    
    if len(features) == 0:
        print("❌ No sentiment features found. Run the analyzer first!")
        return
    
    print("\n📊 Simulating trading outcomes...")
    
    for feature_id, symbol, timestamp, sentiment_score, confidence in features:
        # Check if outcome already exists
        cursor.execute('SELECT id FROM trading_outcomes WHERE feature_id = ?', (feature_id,))
        if cursor.fetchone():
            print(f"⏭️  Feature {feature_id} already has outcome")
            continue
            
        # Simulate trading outcome based on sentiment
        # Higher sentiment score = higher chance of profit
        base_probability = 0.5  # 50% base chance
        sentiment_boost = sentiment_score * 0.3  # Sentiment can add up to 30%
        confidence_boost = (confidence - 0.5) * 0.2  # Confidence adds/subtracts up to 10%
        
        profit_probability = base_probability + sentiment_boost + confidence_boost
        profit_probability = max(0.1, min(0.9, profit_probability))  # Clamp between 10-90%
        
        is_profitable = random.random() < profit_probability
        
        # Simulate trade details
        entry_price = 95.0 + random.uniform(-5, 5)
        if is_profitable:
            return_pct = random.uniform(0.5, 5.0)  # 0.5% to 5% profit
            exit_price = entry_price * (1 + return_pct/100)
        else:
            return_pct = random.uniform(-3.0, -0.2)  # 0.2% to 3% loss
            exit_price = entry_price * (1 + return_pct/100)
        
        # Calculate outcome label (1 for profit after fees, 0 for loss)
        trading_fees = 0.2  # 0.2% round-trip fees
        net_return = return_pct - trading_fees
        outcome_label = 1 if net_return > 0 else 0
        
        # Insert simulated outcome
        signal_timestamp = timestamp
        exit_timestamp = (datetime.fromisoformat(timestamp) + timedelta(days=random.randint(1, 7))).isoformat()
        
        cursor.execute('''
            INSERT INTO trading_outcomes 
            (feature_id, symbol, signal_timestamp, signal_type, entry_price, exit_price, 
             exit_timestamp, return_pct, outcome_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feature_id, symbol, signal_timestamp, 
            'BUY' if sentiment_score > 0 else 'SELL',
            entry_price, exit_price, exit_timestamp, return_pct, outcome_label
        ))
        
        result_text = "PROFIT" if outcome_label == 1 else "LOSS"
        print(f"  📈 Feature {feature_id}: {result_text} ({return_pct:+.2f}% return, {net_return:+.2f}% after fees)")
    
    conn.commit()
    
    # Check final counts
    cursor.execute('SELECT COUNT(*) FROM sentiment_features')
    feature_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM trading_outcomes')
    outcome_count = cursor.fetchone()[0]
    
    cursor.execute('''
        SELECT COUNT(*) 
        FROM sentiment_features sf
        INNER JOIN trading_outcomes tro ON sf.id = tro.feature_id
        WHERE tro.outcome_label IS NOT NULL
    ''')
    paired_count = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"\n📊 Final Data Summary:")
    print(f"   Sentiment Features: {feature_count}")
    print(f"   Trading Outcomes: {outcome_count}")
    print(f"   Complete Pairs: {paired_count}")
    
    if paired_count >= 10:
        print(f"\n🎯 Ready for ML Training!")
        print(f"   Run: python scripts/retrain_ml_models.py --min-samples {paired_count}")
    else:
        print(f"\n⏳ Need more data...")
        print(f"   Run analyzer more times to collect more features")
        print(f"   Or run this simulation again after collecting more features")

if __name__ == "__main__":
    simulate_trading_outcomes()
