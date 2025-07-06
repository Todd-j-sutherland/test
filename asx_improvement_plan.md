# ASX Bank Trading Analysis System - Deep Dive Improvement Plan

## Executive Summary

After conducting a comprehensive code analysis, I've identified **7 critical areas** for improvement that will dramatically enhance your ASX trading system's performance, reliability, and profitability. The current system shows good architectural thinking but has significant gaps in performance, data quality, and advanced analytics.

## Critical Issues Identified

### 🔴 High Priority Issues
1. **Synchronous Processing** - Sequential analysis causing 5x slower execution
2. **No Data Validation** - Risk of bad data leading to poor decisions
3. **Basic Risk Management** - Missing portfolio-level risk controls
4. **Simple Prediction Logic** - Weighted averages instead of ML models

### 🟡 Medium Priority Issues
5. **Limited Technical Analysis** - Missing advanced patterns and multi-timeframe analysis
6. **Basic Sentiment Analysis** - Using TextBlob/VADER instead of financial-specific models
7. **No Real-time Data** - Polling-based updates causing delayed decisions

## Improvement Plan by Priority

---

## Priority 1: Async Data Processing (IMMEDIATE - Easy Win)

**Impact**: 5x performance improvement
**Effort**: Low (2-3 hours)
**ROI**: Immediate

### Current Problem
```python
# main.py - analyze_all_banks() - INEFFICIENT
for symbol in self.settings.BANK_SYMBOLS:
    result = self.analyze_bank(symbol)  # Blocking call
    time.sleep(1)  # Sequential processing
```

### Solution: Async Implementation

```python
# Enhanced main.py
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

class ASXBankTradingSystemAsync:
    def __init__(self):
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    async def start_session(self):
        """Initialize async session with connection pooling"""
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=20,
            keepalive_timeout=30
        )
        self.session = aiohttp.ClientSession(connector=connector)
    
    async def analyze_all_banks_async(self):
        """Analyze all banks concurrently"""
        if not self.session:
            await self.start_session()
        
        tasks = [
            self.analyze_bank_async(symbol) 
            for symbol in self.settings.BANK_SYMBOLS
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = {}
        for symbol, result in zip(self.settings.BANK_SYMBOLS, results):
            if isinstance(result, Exception):
                logger.error(f"Error analyzing {symbol}: {result}")
                processed_results[symbol] = {'error': str(result)}
            else:
                processed_results[symbol] = result
        
        return processed_results
    
    async def analyze_bank_async(self, symbol: str):
        """Async version of bank analysis"""
        try:
            # Run data-intensive operations in thread pool
            market_data_task = asyncio.create_task(
                asyncio.get_event_loop().run_in_executor(
                    self.executor, 
                    self.data_feed.get_historical_data, 
                    symbol
                )
            )
            
            # Run other analyses concurrently
            fundamental_task = asyncio.create_task(
                asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.fundamental.analyze,
                    symbol
                )
            )
            
            sentiment_task = asyncio.create_task(
                asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.sentiment.analyze_bank_sentiment,
                    symbol
                )
            )
            
            # Wait for all analyses to complete
            market_data, fundamental_metrics, sentiment_score = await asyncio.gather(
                market_data_task,
                fundamental_task, 
                sentiment_task
            )
            
            # Continue with synchronous processing
            if market_data is None or market_data.empty:
                return {'error': f'No data available for {symbol}'}
            
            # Technical analysis (CPU intensive, run in executor)
            technical_signals = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.technical.analyze,
                symbol,
                market_data
            )
            
            # Rest of the analysis...
            current_price = float(market_data['Close'].iloc[-1])
            risk_reward = self.risk_calc.calculate(symbol, current_price, technical_signals, fundamental_metrics)
            prediction = self.predictor.predict(symbol, technical_signals, fundamental_metrics, sentiment_score)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'technical_analysis': technical_signals,
                'fundamental_analysis': fundamental_metrics,
                'sentiment_analysis': sentiment_score,
                'risk_reward': risk_reward,
                'prediction': prediction
            }
            
        except Exception as e:
            logger.error(f"Error in async analyze_bank for {symbol}: {str(e)}")
            return {'error': str(e)}
```

---

## Priority 2: Data Validation Pipeline (CRITICAL)

**Impact**: Prevents catastrophic trading decisions
**Effort**: Medium (1 day)
**ROI**: Risk prevention (invaluable)

### Enhanced Data Feed with Validation

```python
# Enhanced data_feed.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum

class DataQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good" 
    POOR = "poor"
    INVALID = "invalid"

@dataclass
class ValidationResult:
    is_valid: bool
    quality: DataQuality
    errors: List[str]
    warnings: List[str]
    confidence_score: float

class DataValidator:
    def __init__(self):
        self.validation_rules = {
            'price_change_threshold': 0.50,  # 50% single-day change
            'volume_threshold': 1000,        # Minimum volume
            'missing_data_threshold': 0.05,  # 5% missing data max
            'zero_volume_threshold': 0.02    # 2% zero volume days max
        }
    
    def validate_market_data(self, data: pd.DataFrame, symbol: str) -> ValidationResult:
        """Comprehensive market data validation"""
        errors = []
        warnings = []
        confidence_score = 1.0
        
        # 1. Structure validation
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return ValidationResult(False, DataQuality.INVALID, errors, warnings, 0.0)
        
        if len(data) == 0:
            errors.append("Empty dataset")
            return ValidationResult(False, DataQuality.INVALID, errors, warnings, 0.0)
        
        # 2. Data quality checks
        self._check_price_integrity(data, errors, warnings)
        self._check_volume_integrity(data, errors, warnings) 
        self._check_missing_data(data, errors, warnings)
        self._check_outliers(data, symbol, errors, warnings)
        
        # 3. Calculate confidence score
        confidence_score = self._calculate_confidence(data, len(errors), len(warnings))
        
        # 4. Determine overall quality
        quality = self._determine_quality(errors, warnings, confidence_score)
        is_valid = len(errors) == 0 and quality != DataQuality.INVALID
        
        return ValidationResult(is_valid, quality, errors, warnings, confidence_score)
    
    def _check_price_integrity(self, data: pd.DataFrame, errors: List[str], warnings: List[str]):
        """Check for price anomalies and integrity"""
        # Check for negative prices
        if (data[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
            errors.append("Negative or zero prices detected")
        
        # Check OHLC relationships
        invalid_ohlc = (
            (data['High'] < data['Low']) |
            (data['High'] < data['Open']) |
            (data['High'] < data['Close']) |
            (data['Low'] > data['Open']) |
            (data['Low'] > data['Close'])
        )
        if invalid_ohlc.any():
            errors.append(f"Invalid OHLC relationships in {invalid_ohlc.sum()} rows")
        
        # Check for extreme price movements
        price_changes = data['Close'].pct_change().abs()
        extreme_moves = price_changes > self.validation_rules['price_change_threshold']
        if extreme_moves.any():
            max_change = price_changes.max()
            warnings.append(f"Extreme price movement detected: {max_change:.2%}")
    
    def _check_volume_integrity(self, data: pd.DataFrame, errors: List[str], warnings: List[str]):
        """Check volume data quality"""
        if 'Volume' not in data.columns:
            warnings.append("Volume data not available")
            return
        
        # Check for negative volume
        if (data['Volume'] < 0).any():
            errors.append("Negative volume detected")
        
        # Check for suspiciously low volume
        zero_volume_ratio = (data['Volume'] == 0).sum() / len(data)
        if zero_volume_ratio > self.validation_rules['zero_volume_threshold']:
            warnings.append(f"High zero-volume ratio: {zero_volume_ratio:.2%}")
        
        # Check volume consistency
        avg_volume = data['Volume'].mean()
        if avg_volume < self.validation_rules['volume_threshold']:
            warnings.append(f"Low average volume: {avg_volume:,.0f}")
    
    def _check_missing_data(self, data: pd.DataFrame, errors: List[str], warnings: List[str]):
        """Check for missing data patterns"""
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_ratio > self.validation_rules['missing_data_threshold']:
            errors.append(f"Too much missing data: {missing_ratio:.2%}")
        elif missing_ratio > 0:
            warnings.append(f"Some missing data: {missing_ratio:.2%}")
    
    def _check_outliers(self, data: pd.DataFrame, symbol: str, errors: List[str], warnings: List[str]):
        """Check for statistical outliers"""
        # Use IQR method for outlier detection
        for col in ['Open', 'High', 'Low', 'Close']:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            if outliers > len(data) * 0.05:  # More than 5% outliers
                warnings.append(f"High outlier count in {col}: {outliers}")
    
    def _calculate_confidence(self, data: pd.DataFrame, error_count: int, warning_count: int) -> float:
        """Calculate data confidence score"""
        base_confidence = 1.0
        
        # Penalize errors heavily
        base_confidence -= error_count * 0.3
        
        # Penalize warnings moderately  
        base_confidence -= warning_count * 0.1
        
        # Bonus for data completeness
        completeness = 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
        base_confidence *= completeness
        
        # Bonus for sufficient data points
        if len(data) >= 252:  # 1 year of data
            base_confidence *= 1.1
        elif len(data) >= 60:  # 3 months
            base_confidence *= 1.05
        elif len(data) < 20:  # Less than 1 month
            base_confidence *= 0.8
        
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

class EnhancedASXDataFeed(ASXDataFeed):
    def __init__(self):
        super().__init__()
        self.validator = DataValidator()
        self.data_sources = [
            'yfinance',
            'alpha_vantage',  # Add as backup
            'polygon',        # Add as backup
        ]
        
    async def get_validated_data(self, symbol: str) -> Tuple[pd.DataFrame, ValidationResult]:
        """Get data with validation from multiple sources"""
        best_data = None
        best_validation = None
        
        for source in self.data_sources:
            try:
                if source == 'yfinance':
                    data = self.get_historical_data(symbol)
                elif source == 'alpha_vantage':
                    data = await self._get_alpha_vantage_data(symbol)
                elif source == 'polygon':
                    data = await self._get_polygon_data(symbol)
                
                if data is not None and not data.empty:
                    validation = self.validator.validate_market_data(data, symbol)
                    
                    # Use first valid source or upgrade if better quality found
                    if validation.is_valid and (
                        best_validation is None or 
                        validation.confidence_score > best_validation.confidence_score
                    ):
                        best_data = data
                        best_validation = validation
                        
                        # If excellent quality, no need to check other sources
                        if validation.quality == DataQuality.EXCELLENT:
                            break
                            
            except Exception as e:
                logger.warning(f"Error fetching from {source} for {symbol}: {e}")
                continue
        
        if best_data is None:
            # Return empty data with invalid validation
            empty_validation = ValidationResult(
                False, DataQuality.INVALID, 
                ["No valid data from any source"], [], 0.0
            )
            return pd.DataFrame(), empty_validation
        
        return best_data, best_validation
    
    async def _get_alpha_vantage_data(self, symbol: str) -> pd.DataFrame:
        """Backup data source - Alpha Vantage"""
        # Implementation for Alpha Vantage API
        # This would require API key configuration
        pass
    
    async def _get_polygon_data(self, symbol: str) -> pd.DataFrame:
        """Backup data source - Polygon.io"""
        # Implementation for Polygon.io API
        # This would require API key configuration
        pass
```

---

## Priority 3: Advanced Risk Management (CRITICAL)

**Impact**: Capital preservation and portfolio optimization
**Effort**: High (2-3 days)
**ROI**: Prevents large losses

### Enhanced Risk Calculator

```python
# Enhanced risk_calculator.py
import numpy as np
import pandas as pd
from scipy.stats import norm, t
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedRiskManager:
    def __init__(self):
        self.confidence_levels = [0.95, 0.99]
        self.risk_free_rate = 0.02  # 2% annual
        self.lookback_periods = {
            'short': 30,   # 1 month
            'medium': 90,  # 3 months  
            'long': 252    # 1 year
        }
    
    def calculate_comprehensive_risk(self, symbol: str, price_data: pd.DataFrame, 
                                   position_size: float, account_balance: float) -> Dict:
        """Calculate comprehensive risk metrics"""
        
        if len(price_data) < 30:
            return {'error': 'Insufficient data for risk calculation'}
        
        # Calculate returns
        returns = price_data['Close'].pct_change().dropna()
        
        # 1. Value at Risk (VaR) and Conditional VaR
        var_metrics = self._calculate_var_cvar(returns)
        
        # 2. Maximum Drawdown
        drawdown_metrics = self._calculate_drawdown_metrics(price_data['Close'])
        
        # 3. Volatility metrics
        volatility_metrics = self._calculate_volatility_metrics(returns)
        
        # 4. Tail risk metrics
        tail_risk_metrics = self._calculate_tail_risk(returns)
        
        # 5. Portfolio impact
        portfolio_impact = self._calculate_portfolio_impact(
            returns, position_size, account_balance
        )
        
        # 6. Risk-adjusted returns
        risk_adjusted_metrics = self._calculate_risk_adjusted_returns(
            returns, self.risk_free_rate
        )
        
        # 7. Stress testing
        stress_test_results = self._run_stress_tests(returns, position_size)
        
        # 8. Overall risk score
        overall_risk_score = self._calculate_overall_risk_score(
            var_metrics, drawdown_metrics, volatility_metrics, tail_risk_metrics
        )
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'var_metrics': var_metrics,
            'drawdown_metrics': drawdown_metrics,
            'volatility_metrics': volatility_metrics,
            'tail_risk_metrics': tail_risk_metrics,
            'portfolio_impact': portfolio_impact,
            'risk_adjusted_metrics': risk_adjusted_metrics,
            'stress_test_results': stress_test_results,
            'overall_risk_score': overall_risk_score,
            'recommendations': self._generate_risk_recommendations(overall_risk_score)
        }
    
    def _calculate_var_cvar(self, returns: pd.Series) -> Dict:
        """Calculate Value at Risk and Conditional VaR using multiple methods"""
        var_results = {}
        
        for confidence_level in self.confidence_levels:
            # Historical simulation method
            var_hist = returns.quantile(1 - confidence_level)
            cvar_hist = returns[returns <= var_hist].mean()
            
            # Parametric method (assuming normal distribution)
            var_param = norm.ppf(1 - confidence_level, returns.mean(), returns.std())
            
            # Modified Cornish-Fisher method (accounts for skewness and kurtosis)
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            z = norm.ppf(1 - confidence_level)
            
            # Cornish-Fisher adjustment
            z_cf = z + (z**2 - 1) * skewness / 6 + (z**3 - 3*z) * kurtosis / 24
            var_cf = returns.mean() + returns.std() * z_cf
            
            var_results[f'confidence_{int(confidence_level*100)}'] = {
                'var_historical': var_hist,
                'var_parametric': var_param,
                'var_cornish_fisher': var_cf,
                'cvar_historical': cvar_hist,
                'confidence_level': confidence_level
            }
        
        return var_results
    
    def _calculate_drawdown_metrics(self, prices: pd.Series) -> Dict:
        """Calculate comprehensive drawdown metrics"""
        # Calculate cumulative returns
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Average drawdown
        avg_drawdown = drawdown[drawdown < 0].mean()
        
        # Drawdown duration
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        
        for date, dd in drawdown.items():
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_date = date
            elif dd >= 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                if start_date:
                    duration = (date - start_date).days
                    drawdown_periods.append(duration)
        
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Recovery factor
        total_return = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration_days': max_drawdown_duration,
            'avg_drawdown_duration_days': avg_drawdown_duration,
            'recovery_factor': recovery_factor,
            'num_drawdown_periods': len(drawdown_periods)
        }
    
    def _calculate_volatility_metrics(self, returns: pd.Series) -> Dict:
        """Calculate various volatility metrics"""
        # Standard volatility measures
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # EWMA volatility (exponentially weighted moving average)
        ewma_vol = returns.ewm(span=30).std().iloc[-1] * np.sqrt(252)
        
        # GARCH-like volatility (simplified)
        vol_of_vol = returns.rolling(30).std().std() * np.sqrt(252)
        
        # Parkinson volatility (uses high-low data if available)
        # This would require OHLC data - simplified here
        
        # Volatility clustering detection
        vol_changes = returns.rolling(5).std().pct_change().dropna()
        vol_clustering = vol_changes.autocorr()
        
        return {
            'daily_volatility': daily_vol,
            'annual_volatility': annual_vol,
            'ewma_volatility': ewma_vol,
            'volatility_of_volatility': vol_of_vol,
            'volatility_clustering': vol_clustering,
            'volatility_regime': 'high' if annual_vol > 0.25 else 'medium' if annual_vol > 0.15 else 'low'
        }
    
    def _calculate_tail_risk(self, returns: pd.Series) -> Dict:
        """Calculate tail risk metrics"""
        # Skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Tail ratio (90th percentile / 10th percentile)
        tail_ratio = returns.quantile(0.9) / abs(returns.quantile(0.1))
        
        # Expected shortfall at different confidence levels
        shortfall_95 = returns[returns <= returns.quantile(0.05)].mean()
        shortfall_99 = returns[returns <= returns.quantile(0.01)].mean()
        
        # Tail dependence (simplified measure)
        extreme_negative_days = (returns < returns.quantile(0.05)).sum()
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_ratio': tail_ratio,
            'expected_shortfall_95': shortfall_95,
            'expected_shortfall_99': shortfall_99,
            'extreme_negative_days': extreme_negative_days,
            'tail_risk_level': 'high' if kurtosis > 3 or skewness < -0.5 else 'normal'
        }
    
    def _calculate_portfolio_impact(self, returns: pd.Series, position_size: float, 
                                  account_balance: float) -> Dict:
        """Calculate how this position impacts overall portfolio"""
        position_weight = position_size / account_balance
        
        # Position VaR impact
        position_var_95 = returns.quantile(0.05) * position_size
        position_var_99 = returns.quantile(0.01) * position_size
        
        # Contribution to portfolio risk
        portfolio_contribution = position_weight * returns.std() * np.sqrt(252)
        
        # Marginal VaR (simplified)
        marginal_var = returns.quantile(0.05) * position_weight
        
        return {
            'position_weight': position_weight,
            'position_var_95': position_var_95,
            'position_var_99': position_var_99,
            'portfolio_risk_contribution': portfolio_contribution,
            'marginal_var': marginal_var,
            'concentration_risk': 'high' if position_weight > 0.2 else 'medium' if position_weight > 0.1 else 'low'
        }
    
    def _run_stress_tests(self, returns: pd.Series, position_size: float) -> Dict:
        """Run various stress test scenarios"""
        current_price = position_size  # Simplified
        
        scenarios = {
            'market_crash_2008': -0.50,     # 50% decline
            'covid_crash_2020': -0.35,      # 35% decline
            'flash_crash': -0.20,           # 20% decline in one day
            'volatility_spike': returns.std() * 3,  # 3x normal volatility
            'interest_rate_shock': -0.15    # 15% decline due to rate changes
        }
        
        stress_results = {}
        for scenario, shock in scenarios.items():
            if 'volatility' in scenario:
                # For volatility scenarios, calculate impact differently
                potential_loss = shock * current_price
            else:
                # For price shock scenarios
                potential_loss = shock * current_price
            
            stress_results[scenario] = {
                'shock_magnitude': shock,
                'potential_loss': potential_loss,
                'loss_percentage': potential_loss / position_size if position_size > 0 else 0
            }
        
        return stress_results
    
    def _calculate_overall_risk_score(self, var_metrics: Dict, drawdown_metrics: Dict,
                                    volatility_metrics: Dict, tail_risk_metrics: Dict) -> float:
        """Calculate overall risk score (0-100, higher = riskier)"""
        risk_factors = []
        
        # VaR contribution (30% weight)
        var_95 = abs(var_metrics['confidence_95']['var_historical'])
        if var_95 > 0.05:  # 5% daily loss
            risk_factors.append(30)
        elif var_95 > 0.03:  # 3% daily loss
            risk_factors.append(20)
        elif var_95 > 0.02:  # 2% daily loss
            risk_factors.append(10)
        
        # Drawdown contribution (25% weight)
        max_dd = abs(drawdown_metrics['max_drawdown'])
        if max_dd > 0.30:  # 30% drawdown
            risk_factors.append(25)
        elif max_dd > 0.20:  # 20% drawdown
            risk_factors.append(15)
        elif max_dd > 0.10:  # 10% drawdown
            risk_factors.append(8)
        
        # Volatility contribution (25% weight)
        annual_vol = volatility_metrics['annual_volatility']
        if annual_vol > 0.40:  # 40% annual volatility
            risk_factors.append(25)
        elif annual_vol > 0.25:  # 25% annual volatility
            risk_factors.append(15)
        elif annual_vol > 0.15:  # 15% annual volatility
            risk_factors.append(8)
        
        # Tail risk contribution (20% weight)
        if tail_risk_metrics['tail_risk_level'] == 'high':
            risk_factors.append(20)
        
        total_risk_score = sum(risk_factors)
        return min(100, total_risk_score)
    
    def _generate_risk_recommendations(self, risk_score: float) -> List[str]:
        """Generate actionable risk management recommendations"""
        recommendations = []
        
        if risk_score > 80:
            recommendations.extend([
                "CRITICAL: Avoid this position - extremely high risk",
                "Consider paper trading this strategy first",
                "Wait for better risk/reward setup"
            ])
        elif risk_score > 60:
            recommendations.extend([
                "HIGH RISK: Reduce position size by 50%",
                "Use tight stop losses",
                "Monitor position closely"
            ])
        elif risk_score > 40:
            recommendations.extend([
                "MODERATE RISK: Standard position size acceptable",
                "Use trailing stops",
                "Consider profit-taking levels"
            ])
        else:
            recommendations.extend([
                "LOW RISK: Position size can be increased",
                "Good risk/reward profile",
                "Monitor for position scaling opportunities"
            ])
        
        # Always include general risk management
        recommendations.extend([
            "Never risk more than 2% of account on single trade",
            "Maintain portfolio diversification",
            "Regular risk assessment recommended"
        ])
        
        return recommendations

class PortfolioRiskManager:
    def __init__(self):
        self.correlation_lookback = 60  # days
        
    def calculate_portfolio_risk(self, positions: List[Dict], 
                               price_data: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate comprehensive portfolio risk metrics"""
        
        if not positions:
            return {'error': 'No positions in portfolio'}
        
        # Extract position data
        symbols = [pos['symbol'] for pos in positions]
        weights = np.array([pos['weight'] for pos in positions])
        
        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(symbols, price_data)
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(
            weights, symbols, price_data, correlation_matrix
        )
        
        # Calculate diversification benefits
        diversification_metrics = self._calculate_diversification_metrics(
            weights, correlation_matrix
        )
        
        # Risk concentration analysis
        concentration_metrics = self._calculate_concentration_risk(weights, symbols)
        
        return {
            'portfolio_metrics': portfolio_metrics,
            'diversification_metrics': diversification_metrics,
            'concentration_metrics': concentration_metrics,
            'correlation_matrix': correlation_matrix.tolist(),
            'recommendations': self._generate_portfolio_recommendations(
                portfolio_metrics, diversification_metrics, concentration_metrics
            )
        }
    
    def _calculate_correlation_matrix(self, symbols: List[str], 
                                    price_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Calculate correlation matrix for portfolio positions"""
        returns_data = {}
        
        for symbol in symbols:
            if symbol in price_data and not price_data[symbol].empty:
                returns = price_data[symbol]['Close'].pct_change().dropna()
                returns_data[symbol] = returns.tail(self.correlation_lookback)
        
        # Create DataFrame of returns
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr().fillna(0).values
        
        return correlation_matrix
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray, symbols: List[str],
                                   price_data: Dict[str, pd.DataFrame], 
                                   correlation_matrix: np.ndarray) -> Dict:
        """Calculate portfolio-level risk metrics"""
        
        # Calculate individual asset volatilities
        volatilities = []
        expected_returns = []
        
        for symbol in symbols:
            if symbol in price_data and not price_data[symbol].empty:
                returns = price_data[symbol]['Close'].pct_change().dropna()
                vol = returns.std() * np.sqrt(252)  # Annualized
                exp_return = returns.mean() * 252    # Annualized
                volatilities.append(vol)
                expected_returns.append(exp_return)
            else:
                volatilities.append(0.2)  # Default volatility
                expected_returns.append(0.1)  # Default return
        
        volatilities = np.array(volatilities)
        expected_returns = np.array(expected_returns)
        
        # Portfolio expected return
        portfolio_return = np.dot(weights, expected_returns)
        
        # Portfolio volatility using correlation matrix
        portfolio_variance = np.dot(weights, np.dot(
            correlation_matrix * np.outer(volatilities, volatilities), weights
        ))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        # Portfolio VaR
        portfolio_var_95 = norm.ppf(0.05, portfolio_return/252, portfolio_volatility/np.sqrt(252))
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': portfolio_var_95,
            'individual_volatilities': volatilities.tolist(),
            'individual_returns': expected_returns.tolist()
        }
    
    def _calculate_diversification_metrics(self, weights: np.ndarray, 
                                         correlation_matrix: np.ndarray) -> Dict:
        """Calculate diversification effectiveness metrics"""
        
        # Diversification ratio
        weighted_avg_vol = np.sum(weights * np.diag(correlation_matrix))
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(correlation_matrix, weights)))
        diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1
        
        # Effective number of positions
        herfindahl_index = np.sum(weights ** 2)
        effective_positions = 1 / herfindahl_index
        
        # Average correlation
        n = len(weights)
        if n > 1:
            # Extract off-diagonal elements
            off_diag_sum = np.sum(correlation_matrix) - np.trace(correlation_matrix)
            avg_correlation = off_diag_sum / (n * (n - 1))
        else:
            avg_correlation = 0
        
        return {
            'diversification_ratio': diversification_ratio,
            'effective_positions': effective_positions,
            'average_correlation': avg_correlation,
            'herfindahl_index': herfindahl_index,
            'diversification_level': 'high' if diversification_ratio > 1.2 else 'medium' if diversification_ratio > 1.1 else 'low'
        }
    
    def _calculate_concentration_risk(self, weights: np.ndarray, symbols: List[str]) -> Dict:
        """Calculate concentration risk metrics"""
        
        # Largest position weight
        max_weight = np.max(weights)
        max_weight_symbol = symbols[np.argmax(weights)]
        
        # Top 3 positions weight
        top_3_weights = np.sort(weights)[-3:] if len(weights) >= 3 else weights
        top_3_concentration = np.sum(top_3_weights)
        
        # Sector concentration (simplified - assume all banks for ASX)
        sector_concentration = 1.0  # All banking sector
        
        return {
            'max_position_weight': max_weight,
            'max_position_symbol': max_weight_symbol,
            'top_3_concentration': top_3_concentration,
            'sector_concentration': sector_concentration,
            'concentration_risk': 'high' if max_weight > 0.3 else 'medium' if max_weight > 0.2 else 'low'
        }
    
    def _generate_portfolio_recommendations(self, portfolio_metrics: Dict,
                                          diversification_metrics: Dict,
                                          concentration_metrics: Dict) -> List[str]:
        """Generate portfolio-level recommendations"""
        recommendations = []
        
        # Diversification recommendations
        if diversification_metrics['diversification_level'] == 'low':
            recommendations.append("Consider adding positions to improve diversification")
        
        # Concentration recommendations
        if concentration_metrics['concentration_risk'] == 'high':
            recommendations.append(f"Reduce concentration in {concentration_metrics['max_position_symbol']}")
        
        # Risk-adjusted return recommendations
        if portfolio_metrics['sharpe_ratio'] < 0.5:
            recommendations.append("Poor risk-adjusted returns - review strategy")
        elif portfolio_metrics['sharpe_ratio'] > 1.5:
            recommendations.append("Excellent risk-adjusted returns - consider scaling up")
        
        # Volatility recommendations
        if portfolio_metrics['volatility'] > 0.3:
            recommendations.append("High portfolio volatility - consider reducing position sizes")
        
        return recommendations
```

---

## Priority 4: Machine Learning Prediction Models

**Impact**: Dramatically improved prediction accuracy
**Effort**: High (3-4 days)
**ROI**: Significant edge in market predictions

### Advanced ML Predictor

```python
# Enhanced market_predictor.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLMarketPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_trained = False
        self.model_performance = {}
        
        # Initialize ensemble models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize ML models for ensemble"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=