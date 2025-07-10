#!/usr/bin/env python3
"""
Automated ML training scheduler
Set up as cron job for regular model updates
"""

import schedule
import time
import subprocess
import logging
from datetime import datetime
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training():
    """Run ML model training"""
    try:
        logger.info(f"Starting ML training at {datetime.now()}")
        
        # Run training script
        result = subprocess.run(
            ['python', 'scripts/retrain_ml_models.py', '--min-samples', '500'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        if result.returncode == 0:
            logger.info("Training completed successfully")
            logger.info(result.stdout)
        else:
            logger.error(f"Training failed: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Error running training: {e}")

def check_data_quality():
    """Check if we have enough quality data for training"""
    try:
        from src.ml_training_pipeline import MLTrainingPipeline
        pipeline = MLTrainingPipeline()
        
        X, y = pipeline.prepare_training_dataset(min_samples=100)
        
        if X is not None:
            logger.info(f"Data check passed: {len(X)} samples available")
            
            # Check class balance
            class_balance = y.value_counts().to_dict()
            logger.info(f"Class balance: {class_balance}")
            
            # Warn if imbalanced
            if min(class_balance.values()) / max(class_balance.values()) < 0.3:
                logger.warning("Warning: Class imbalance detected")
        else:
            logger.warning("Insufficient data for training")
            
    except Exception as e:
        logger.error(f"Error checking data quality: {e}")

# Schedule tasks
schedule.every().sunday.at("02:00").do(run_training)
schedule.every().day.at("06:00").do(check_data_quality)

if __name__ == "__main__":
    logger.info("ML training scheduler started")
    
    # Run initial check
    check_data_quality()
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute
