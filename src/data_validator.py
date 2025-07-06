# src/data_validator.py
"""
Data validation pipeline for ensuring data quality and preventing bad trading decisions
Implements comprehensive market data validation with multiple quality checks
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good" 
    POOR = "poor"
    INVALID = "invalid"

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    quality: DataQuality
    errors: List[str]
    warnings: List[str]
    confidence_score: float
    
    def __str__(self):
        return f"Valid: {self.is_valid}, Quality: {self.quality.value}, Confidence: {self.confidence_score:.2f}"

class DataValidator:
    """Comprehensive market data validator"""
    
    def __init__(self):
        self.validation_rules = {
            'price_change_threshold': 0.50,  # 50% single-day change
            'volume_threshold': 1000,        # Minimum volume
            'missing_data_threshold': 0.05,  # 5% missing data max
            'zero_volume_threshold': 0.02,   # 2% zero volume days max
            'min_data_points': 20,           # Minimum data points needed
            'outlier_threshold': 0.05        # 5% outlier tolerance
        }
    
    def validate_market_data(self, data: pd.DataFrame, symbol: str) -> ValidationResult:
        """Comprehensive market data validation"""
        errors = []
        warnings = []
        confidence_score = 1.0
        
        # 1. Structure validation
        if not self._validate_structure(data, errors, warnings):
            return ValidationResult(False, DataQuality.INVALID, errors, warnings, 0.0)
        
        # 2. Data quality checks
        self._check_price_integrity(data, errors, warnings)
        self._check_volume_integrity(data, errors, warnings) 
        self._check_missing_data(data, errors, warnings)
        self._check_outliers(data, symbol, errors, warnings)
        self._check_data_completeness(data, errors, warnings)
        self._check_temporal_consistency(data, errors, warnings)
        
        # 3. Calculate confidence score
        confidence_score = self._calculate_confidence(data, len(errors), len(warnings))
        
        # 4. Determine overall quality
        quality = self._determine_quality(errors, warnings, confidence_score)
        is_valid = len(errors) == 0 and quality != DataQuality.INVALID
        
        return ValidationResult(is_valid, quality, errors, warnings, confidence_score)
    
    def _validate_structure(self, data: pd.DataFrame, errors: List[str], warnings: List[str]) -> bool:
        """Validate basic data structure"""
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return False
        
        if len(data) == 0:
            errors.append("Empty dataset")
            return False
        
        if len(data) < self.validation_rules['min_data_points']:
            warnings.append(f"Limited data: only {len(data)} points")
        
        return True
    
    def _check_price_integrity(self, data: pd.DataFrame, errors: List[str], warnings: List[str]):
        """Check for price anomalies and integrity"""
        # Check for negative or zero prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if (data[col] <= 0).any():
                errors.append(f"Invalid {col.lower()} prices detected (negative or zero)")
        
        # Check OHLC relationships
        invalid_ohlc = (
            (data['High'] < data['Low']) |
            (data['High'] < data['Open']) |
            (data['High'] < data['Close']) |
            (data['Low'] > data['Open']) |
            (data['Low'] > data['Close'])
        )
        
        if invalid_ohlc.any():
            count = invalid_ohlc.sum()
            errors.append(f"Invalid OHLC relationships in {count} rows ({count/len(data):.1%})")
        
        # Check for extreme price movements
        price_changes = data['Close'].pct_change().abs()
        extreme_moves = price_changes > self.validation_rules['price_change_threshold']
        
        if extreme_moves.any():
            max_change = price_changes.max()
            count = extreme_moves.sum()
            if count > 1:
                errors.append(f"Multiple extreme price movements detected (max: {max_change:.1%})")
            else:
                warnings.append(f"Extreme price movement detected: {max_change:.1%}")
        
        # Check for price gaps
        gaps = abs(data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        large_gaps = gaps > 0.10  # 10% gap
        if large_gaps.any():
            max_gap = gaps.max()
            warnings.append(f"Large price gaps detected (max: {max_gap:.1%})")
    
    def _check_volume_integrity(self, data: pd.DataFrame, errors: List[str], warnings: List[str]):
        """Check volume data quality"""
        if 'Volume' not in data.columns:
            warnings.append("Volume data not available")
            return
        
        # Check for negative volume
        if (data['Volume'] < 0).any():
            errors.append("Negative volume detected")
        
        # Check for zero volume ratio
        zero_volume_ratio = (data['Volume'] == 0).sum() / len(data)
        if zero_volume_ratio > self.validation_rules['zero_volume_threshold']:
            warnings.append(f"High zero-volume ratio: {zero_volume_ratio:.1%}")
        
        # Check volume consistency
        avg_volume = data['Volume'].mean()
        if avg_volume < self.validation_rules['volume_threshold']:
            warnings.append(f"Low average volume: {avg_volume:,.0f}")
        
        # Check for volume spikes
        volume_changes = data['Volume'].pct_change().abs()
        volume_spikes = volume_changes > 10  # 1000% volume spike
        if volume_spikes.any():
            count = volume_spikes.sum()
            warnings.append(f"Volume spikes detected: {count} instances")
    
    def _check_missing_data(self, data: pd.DataFrame, errors: List[str], warnings: List[str]):
        """Check for missing data patterns"""
        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        missing_ratio = missing_cells / total_cells
        
        if missing_ratio > self.validation_rules['missing_data_threshold']:
            errors.append(f"Too much missing data: {missing_ratio:.1%}")
        elif missing_ratio > 0:
            warnings.append(f"Some missing data: {missing_ratio:.1%}")
        
        # Check for missing data patterns
        for col in data.columns:
            col_missing = data[col].isnull().sum()
            if col_missing > 0:
                warnings.append(f"Missing {col} data: {col_missing} points")
    
    def _check_outliers(self, data: pd.DataFrame, symbol: str, errors: List[str], warnings: List[str]):
        """Check for statistical outliers"""
        # Use IQR method for outlier detection
        price_cols = ['Open', 'High', 'Low', 'Close']
        
        for col in price_cols:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:  # Avoid division by zero
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                    outlier_ratio = outliers / len(data)
                    
                    if outlier_ratio > self.validation_rules['outlier_threshold']:
                        warnings.append(f"High outlier count in {col}: {outliers} ({outlier_ratio:.1%})")
    
    def _check_data_completeness(self, data: pd.DataFrame, errors: List[str], warnings: List[str]):
        """Check data completeness and coverage"""
        # Check for recent data
        if not data.index.empty:
            latest_date = data.index.max()
            
            # Convert to timezone-naive if necessary
            if hasattr(latest_date, 'tz') and latest_date.tz is not None:
                latest_date = latest_date.tz_localize(None)
            
            current_time = datetime.now()
            days_old = (current_time - latest_date).days
            
            if days_old > 7:
                warnings.append(f"Data is {days_old} days old")
            elif days_old > 30:
                errors.append(f"Data is too old: {days_old} days")
        
        # Check for data frequency consistency
        if len(data) > 1:
            date_diffs = data.index.to_series().diff().dt.days
            if date_diffs.std() > 2:  # More than 2 days standard deviation
                warnings.append("Inconsistent data frequency detected")
    
    def _check_temporal_consistency(self, data: pd.DataFrame, errors: List[str], warnings: List[str]):
        """Check for temporal consistency"""
        if len(data) < 2:
            return
        
        # Check for duplicate dates
        if data.index.duplicated().any():
            count = data.index.duplicated().sum()
            errors.append(f"Duplicate dates detected: {count}")
        
        # Check for chronological order
        if not data.index.is_monotonic_increasing:
            errors.append("Data is not in chronological order")
    
    def _calculate_confidence(self, data: pd.DataFrame, error_count: int, warning_count: int) -> float:
        """Calculate data confidence score"""
        base_confidence = 1.0
        
        # Penalize errors heavily
        base_confidence -= error_count * 0.3
        
        # Penalize warnings moderately  
        base_confidence -= warning_count * 0.1
        
        # Bonus for data completeness
        if len(data) > 0:
            completeness = 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            base_confidence *= completeness
        
        # Bonus for sufficient data points
        if len(data) >= 252:  # 1 year of data
            base_confidence *= 1.1
        elif len(data) >= 60:  # 3 months
            base_confidence *= 1.05
        elif len(data) < 20:  # Less than 1 month
            base_confidence *= 0.8
        
        # Bonus for recent data
        if not data.index.empty:
            latest_date = data.index.max()
            
            # Convert to timezone-naive if necessary
            if hasattr(latest_date, 'tz') and latest_date.tz is not None:
                latest_date = latest_date.tz_localize(None)
            
            current_time = datetime.now()
            days_old = (current_time - latest_date).days
            if days_old <= 1:
                base_confidence *= 1.05
            elif days_old > 7:
                base_confidence *= 0.95
        
        return max(0.0, min(1.0, base_confidence))
    
    def _determine_quality(self, errors: List[str], warnings: List[str], confidence: float) -> DataQuality:
        """Determine overall data quality"""
        if errors:
            return DataQuality.INVALID
        elif confidence >= 0.9 and len(warnings) == 0:
            return DataQuality.EXCELLENT
        elif confidence >= 0.7 and len(warnings) <= 2:
            return DataQuality.GOOD
        else:
            return DataQuality.POOR

class EnhancedDataFeed:
    """Enhanced data feed with validation and multiple sources"""
    
    def __init__(self, base_data_feed):
        self.base_data_feed = base_data_feed
        self.validator = DataValidator()
        self.fallback_sources = []  # Can be extended with additional sources
    
    def get_validated_data(self, symbol: str, period: str = '1y') -> Tuple[pd.DataFrame, ValidationResult]:
        """Get data with validation"""
        # Try primary source first
        try:
            data = self.base_data_feed.get_historical_data(symbol, period)
            if data is not None and not data.empty:
                validation = self.validator.validate_market_data(data, symbol)
                
                if validation.is_valid:
                    logger.info(f"Data validation for {symbol}: {validation}")
                    return data, validation
                else:
                    logger.warning(f"Data validation failed for {symbol}: {validation}")
        
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
        
        # If primary source fails, return empty with invalid validation
        empty_validation = ValidationResult(
            False, DataQuality.INVALID, 
            ["No valid data available"], [], 0.0
        )
        return pd.DataFrame(), empty_validation
    
    def validate_existing_data(self, data: pd.DataFrame, symbol: str) -> ValidationResult:
        """Validate existing data"""
        return self.validator.validate_market_data(data, symbol)
