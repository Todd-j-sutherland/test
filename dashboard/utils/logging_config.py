"""
Logging configuration for the dashboard system
Provides consistent logging across all dashboard components
"""

import logging
import os
from datetime import datetime
from typing import Optional

def setup_dashboard_logger(
    name: str, 
    log_level: str = "INFO", 
    log_to_file: bool = True,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Set up a logger for dashboard components
    
    Args:
        name: Logger name (usually __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file in addition to console
        log_dir: Directory for log files
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        try:
            os.makedirs(log_dir, exist_ok=True)
            
            # Create log filename with date
            log_filename = f"dashboard_{datetime.now().strftime('%Y%m%d')}.log"
            log_filepath = os.path.join(log_dir, log_filename)
            
            file_handler = logging.FileHandler(log_filepath)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"Dashboard logging initialized - Log file: {log_filepath}")
            
        except Exception as e:
            logger.warning(f"Could not set up file logging: {e}")
    
    return logger

def log_data_loading_stats(logger: logging.Logger, symbol: str, data_count: int, file_path: str):
    """Log data loading statistics"""
    logger.info(f"Data Loading - Symbol: {symbol}, Records: {data_count}, File: {file_path}")

def log_technical_analysis_request(logger: logging.Logger, symbol: str, period: str = "3mo"):
    """Log technical analysis requests"""
    logger.debug(f"Technical Analysis Request - Symbol: {symbol}, Period: {period}")

def log_ml_prediction_request(logger: logging.Logger, symbol: str, features_count: int):
    """Log ML prediction requests"""
    logger.info(f"ML Prediction Request - Symbol: {symbol}, Features: {features_count}")

def log_chart_generation(logger: logging.Logger, chart_type: str, symbol: str, data_points: int):
    """Log chart generation activities"""
    logger.debug(f"Chart Generation - Type: {chart_type}, Symbol: {symbol}, Data Points: {data_points}")

def log_position_risk_assessment(logger: logging.Logger, symbol: str, position_type: str, entry_price: float):
    """Log position risk assessment activities"""
    logger.info(f"Position Risk Assessment - Symbol: {symbol}, Type: {position_type}, Entry: ${entry_price:.2f}")

def log_error_with_context(logger: logging.Logger, error: Exception, context: str, **kwargs):
    """Log errors with additional context"""
    error_msg = f"Error in {context}: {str(error)}"
    if kwargs:
        error_msg += f" | Context: {kwargs}"
    logger.error(error_msg, exc_info=True)

def log_performance_metrics(logger: logging.Logger, operation: str, execution_time: float, success: bool = True):
    """Log performance metrics for operations"""
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"Performance - {operation}: {execution_time:.3f}s [{status}]")

# Create a default dashboard logger
dashboard_logger = setup_dashboard_logger("dashboard")
