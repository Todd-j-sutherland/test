#!/usr/bin/env python3
"""
Script to retrain ML models with collected data
Run this periodically (e.g., weekly via cron)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml_training_pipeline import MLTrainingPipeline
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Retrain ML models')
    parser.add_argument('--min-samples', type=int, default=50,
                      help='Minimum samples required for training')
    parser.add_argument('--evaluate-only', action='store_true',
                      help='Only evaluate current model, don\'t retrain')
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MLTrainingPipeline()
    
    # Prepare dataset
    logger.info("Preparing training dataset...")
    X, y = pipeline.prepare_training_dataset(min_samples=args.min_samples)
    
    if X is None:
        logger.error("Insufficient data for training")
        return
    
    logger.info(f"Dataset prepared: {len(X)} samples")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    
    if args.evaluate_only:
        # Just evaluate current model
        logger.info("Evaluation mode - skipping training")
        # Add evaluation code here
        return
    
    # Train models
    logger.info("Training models...")
    results = pipeline.train_models(X, y)
    
    logger.info("Training completed!")
    logger.info(f"Best model: {results['best_model']}")
    logger.info("Model scores:")
    for model, scores in results['model_scores'].items():
        logger.info(f"  {model}: {scores['avg_cv_score']:.4f}")

if __name__ == "__main__":
    main()
