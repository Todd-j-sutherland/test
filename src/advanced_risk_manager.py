# src/advanced_risk_manager.py
"""
Advanced Risk Management System for ASX Bank Trading
Implements comprehensive risk metrics, VaR calculations, and portfolio-level risk management
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics container"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    overall_risk_score: float
    risk_level: RiskLevel
    recommendations: List[str]

class AdvancedRiskManager:
    """Advanced risk management with comprehensive metrics"""
    
    def __init__(self):
        self.confidence_levels = [0.95, 0.99]
        self.risk_free_rate = 0.02  # 2% annual
        self.lookback_periods = {
            'short': 30,   # 1 month
            'medium': 90,  # 3 months  
            'long': 252    # 1 year
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            'var_95_high': 0.05,      # 5% daily VaR
            'var_95_extreme': 0.08,   # 8% daily VaR
            'max_drawdown_high': 0.20, # 20% max drawdown
            'max_drawdown_extreme': 0.30, # 30% max drawdown
            'volatility_high': 0.25,  # 25% annual volatility
            'volatility_extreme': 0.40 # 40% annual volatility
        }
    
    def calculate_comprehensive_risk(self, symbol: str, price_data: pd.DataFrame, 
                                   position_size: float = 10000, 
                                   account_balance: float = 100000) -> Dict:
        """Calculate comprehensive risk metrics"""
        
        if len(price_data) < 30:
            return {'error': 'Insufficient data for risk calculation (minimum 30 days required)'}
        
        try:
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
            
            # 8. Overall risk score and level
            overall_risk_score = self._calculate_overall_risk_score(
                var_metrics, drawdown_metrics, volatility_metrics, tail_risk_metrics
            )
            
            risk_level = self._determine_risk_level(overall_risk_score, var_metrics, 
                                                  drawdown_metrics, volatility_metrics)
            
            # 9. Generate recommendations
            recommendations = self._generate_risk_recommendations(
                risk_level, overall_risk_score, var_metrics, drawdown_metrics, 
                volatility_metrics, portfolio_impact
            )
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'analysis_period_days': len(price_data),
                'var_metrics': var_metrics,
                'drawdown_metrics': drawdown_metrics,
                'volatility_metrics': volatility_metrics,
                'tail_risk_metrics': tail_risk_metrics,
                'portfolio_impact': portfolio_impact,
                'risk_adjusted_metrics': risk_adjusted_metrics,
                'stress_test_results': stress_test_results,
                'overall_risk_score': overall_risk_score,
                'risk_level': risk_level.value,
                'recommendations': recommendations,
                'risk_summary': self._create_risk_summary(
                    risk_level, overall_risk_score, var_metrics, drawdown_metrics
                )
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive risk calculation for {symbol}: {str(e)}")
            return {'error': str(e)}
    
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
            if not np.isnan(skewness) and not np.isnan(kurtosis):
                z_cf = z + (z**2 - 1) * skewness / 6 + (z**3 - 3*z) * kurtosis / 24
                var_cf = returns.mean() + returns.std() * z_cf
            else:
                var_cf = var_param
            
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
        returns = prices.pct_change().fillna(0)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        max_drawdown_idx = drawdown.idxmin()
        
        # Average drawdown
        avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        
        # Drawdown duration analysis
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        
        for date, dd in drawdown.items():
            if dd < -0.01 and not in_drawdown:  # Start of significant drawdown (>1%)
                in_drawdown = True
                start_date = date
            elif dd >= -0.005 and in_drawdown:  # End of drawdown (recovery to <0.5%)
                in_drawdown = False
                if start_date is not None:
                    try:
                        # Handle both datetime and integer indices
                        if hasattr(date, 'days') and hasattr(start_date, 'days'):
                            # Both are datetime-like
                            duration = (date - start_date).days
                        elif hasattr(date, 'to_pydatetime') and hasattr(start_date, 'to_pydatetime'):
                            # Both are pandas timestamps
                            duration = (date.to_pydatetime() - start_date.to_pydatetime()).days
                        else:
                            # Fallback - assume they're positions and estimate days
                            duration = abs(int(date) - int(start_date))
                    except (TypeError, AttributeError):
                        # Final fallback - estimate based on position difference
                        duration = abs(int(date) - int(start_date))
                    drawdown_periods.append(duration)
        
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Recovery factor
        total_return = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Ulcer Index (measure of downside risk)
        ulcer_index = np.sqrt(np.mean(drawdown**2))
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_date': max_drawdown_idx.strftime('%Y-%m-%d') if hasattr(max_drawdown_idx, 'strftime') else str(max_drawdown_idx),
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration_days': max_drawdown_duration,
            'avg_drawdown_duration_days': avg_drawdown_duration,
            'recovery_factor': recovery_factor,
            'num_drawdown_periods': len(drawdown_periods),
            'ulcer_index': ulcer_index,
            'current_drawdown': drawdown.iloc[-1] if not drawdown.empty else 0
        }
    
    def _calculate_volatility_metrics(self, returns: pd.Series) -> Dict:
        """Calculate various volatility metrics"""
        # Standard volatility measures
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Rolling volatility (30-day)
        rolling_vol = returns.rolling(30).std() * np.sqrt(252)
        current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else annual_vol
        
        # EWMA volatility (exponentially weighted moving average)
        ewma_vol = returns.ewm(span=30).std().iloc[-1] * np.sqrt(252) if len(returns) > 30 else annual_vol
        
        # Volatility of volatility
        vol_of_vol = rolling_vol.std() if len(rolling_vol) > 10 else 0
        
        # Volatility clustering (autocorrelation of squared returns)
        squared_returns = returns**2
        vol_clustering = squared_returns.autocorr(lag=1) if len(squared_returns) > 1 else 0
        
        # Volatility regime detection
        vol_percentiles = rolling_vol.quantile([0.25, 0.75]) if len(rolling_vol) > 10 else [annual_vol, annual_vol]
        
        if current_vol > vol_percentiles.iloc[1]:
            vol_regime = 'high'
        elif current_vol < vol_percentiles.iloc[0]:
            vol_regime = 'low'
        else:
            vol_regime = 'normal'
        
        return {
            'daily_volatility': daily_vol,
            'annual_volatility': annual_vol,
            'current_volatility': current_vol,
            'ewma_volatility': ewma_vol,
            'volatility_of_volatility': vol_of_vol,
            'volatility_clustering': vol_clustering,
            'volatility_regime': vol_regime,
            'vol_25th_percentile': vol_percentiles.iloc[0] if len(vol_percentiles) > 0 else annual_vol,
            'vol_75th_percentile': vol_percentiles.iloc[1] if len(vol_percentiles) > 0 else annual_vol
        }
    
    def _calculate_tail_risk(self, returns: pd.Series) -> Dict:
        """Calculate tail risk metrics"""
        # Skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Tail ratio (90th percentile / 10th percentile)
        p90 = returns.quantile(0.9)
        p10 = returns.quantile(0.1)
        tail_ratio = p90 / abs(p10) if p10 != 0 else 0
        
        # Expected shortfall at different confidence levels
        shortfall_95 = returns[returns <= returns.quantile(0.05)].mean()
        shortfall_99 = returns[returns <= returns.quantile(0.01)].mean()
        
        # Extreme value metrics
        extreme_negative_days = (returns < returns.quantile(0.05)).sum()
        extreme_positive_days = (returns > returns.quantile(0.95)).sum()
        
        # Maximum single-day loss
        max_single_day_loss = returns.min()
        max_single_day_gain = returns.max()
        
        # Tail risk classification
        if kurtosis > 3 and skewness < -0.5:
            tail_risk_level = 'high'
        elif kurtosis > 2 or skewness < -0.3:
            tail_risk_level = 'medium'
        else:
            tail_risk_level = 'low'
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_ratio': tail_ratio,
            'expected_shortfall_95': shortfall_95,
            'expected_shortfall_99': shortfall_99,
            'extreme_negative_days': extreme_negative_days,
            'extreme_positive_days': extreme_positive_days,
            'max_single_day_loss': max_single_day_loss,
            'max_single_day_gain': max_single_day_gain,
            'tail_risk_level': tail_risk_level
        }
    
    def _calculate_portfolio_impact(self, returns: pd.Series, position_size: float, 
                                  account_balance: float) -> Dict:
        """Calculate how this position impacts overall portfolio"""
        position_weight = position_size / account_balance
        
        # Position VaR impact (dollar amounts)
        position_var_95 = returns.quantile(0.05) * position_size
        position_var_99 = returns.quantile(0.01) * position_size
        
        # Account percentage at risk
        account_risk_95 = abs(position_var_95) / account_balance
        account_risk_99 = abs(position_var_99) / account_balance
        
        # Risk contribution to portfolio
        portfolio_risk_contribution = position_weight * returns.std() * np.sqrt(252)
        
        # Position sizing recommendations
        kelly_fraction = self._calculate_kelly_fraction(returns)
        optimal_position_size = kelly_fraction * account_balance
        
        # Concentration risk assessment
        if position_weight > 0.25:
            concentration_risk = 'extreme'
        elif position_weight > 0.15:
            concentration_risk = 'high'
        elif position_weight > 0.10:
            concentration_risk = 'medium'
        else:
            concentration_risk = 'low'
        
        return {
            'position_size': position_size,
            'position_weight': position_weight,
            'position_var_95_dollars': position_var_95,
            'position_var_99_dollars': position_var_99,
            'account_risk_95_percent': account_risk_95,
            'account_risk_99_percent': account_risk_99,
            'portfolio_risk_contribution': portfolio_risk_contribution,
            'kelly_fraction': kelly_fraction,
            'optimal_position_size': optimal_position_size,
            'concentration_risk': concentration_risk,
            'recommended_max_position': min(optimal_position_size, account_balance * 0.20)
        }
    
    def _calculate_kelly_fraction(self, returns: pd.Series) -> float:
        """Calculate Kelly fraction for optimal position sizing"""
        try:
            mean_return = returns.mean()
            variance = returns.var()
            
            if variance > 0 and mean_return > 0:
                kelly = mean_return / variance
                # Cap Kelly fraction at 25% for safety
                return min(kelly, 0.25)
            else:
                return 0.05  # Conservative default
        except:
            return 0.05
    
    def _calculate_risk_adjusted_returns(self, returns: pd.Series, risk_free_rate: float) -> Dict:
        """Calculate risk-adjusted return metrics"""
        # Annualized returns
        mean_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = (mean_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Calmar ratio (return to max drawdown)
        # This requires drawdown calculation, simplified here
        max_drawdown = abs(returns.cumsum().expanding().max() - returns.cumsum()).max()
        calmar_ratio = mean_return / max_drawdown if max_drawdown > 0 else 0
        
        # Information ratio (simplified)
        tracking_error = returns.std() * np.sqrt(252)
        information_ratio = mean_return / tracking_error if tracking_error > 0 else 0
        
        return {
            'annualized_return': mean_return,
            'annualized_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio
        }
    
    def _run_stress_tests(self, returns: pd.Series, position_size: float) -> Dict:
        """Run various stress test scenarios"""
        scenarios = {
            'market_crash_2008': -0.50,     # 50% decline
            'covid_crash_2020': -0.35,      # 35% decline
            'flash_crash': -0.20,           # 20% decline in one day
            'gradual_decline': -0.25,       # 25% decline over time
            'interest_rate_shock': -0.15,   # 15% decline due to rate changes
            'volatility_spike': returns.std() * 3  # 3x normal volatility
        }
        
        stress_results = {}
        for scenario, shock in scenarios.items():
            if scenario == 'volatility_spike':
                # For volatility scenarios, calculate potential loss range
                potential_loss_range = {
                    'low': shock * 0.5 * position_size,
                    'high': shock * 1.5 * position_size
                }
                stress_results[scenario] = {
                    'shock_type': 'volatility',
                    'shock_magnitude': shock,
                    'potential_loss_range': potential_loss_range,
                    'scenario_probability': 0.05  # 5% probability
                }
            else:
                # For price shock scenarios
                potential_loss = shock * position_size
                stress_results[scenario] = {
                    'shock_type': 'price',
                    'shock_magnitude': shock,
                    'potential_loss': potential_loss,
                    'loss_percentage': potential_loss / position_size if position_size > 0 else 0,
                    'scenario_probability': 0.02 if 'crash' in scenario else 0.10
                }
        
        return stress_results
    
    def _calculate_overall_risk_score(self, var_metrics: Dict, drawdown_metrics: Dict,
                                    volatility_metrics: Dict, tail_risk_metrics: Dict) -> float:
        """Calculate overall risk score (0-100, higher = riskier)"""
        risk_factors = []
        
        # VaR contribution (30% weight)
        var_95 = abs(var_metrics['confidence_95']['var_historical'])
        if var_95 > self.risk_thresholds['var_95_extreme']:
            risk_factors.append(30)
        elif var_95 > self.risk_thresholds['var_95_high']:
            risk_factors.append(20)
        elif var_95 > 0.03:  # 3% daily loss
            risk_factors.append(10)
        else:
            risk_factors.append(5)
        
        # Drawdown contribution (25% weight)
        max_dd = abs(drawdown_metrics['max_drawdown'])
        if max_dd > self.risk_thresholds['max_drawdown_extreme']:
            risk_factors.append(25)
        elif max_dd > self.risk_thresholds['max_drawdown_high']:
            risk_factors.append(15)
        elif max_dd > 0.10:  # 10% drawdown
            risk_factors.append(8)
        else:
            risk_factors.append(3)
        
        # Volatility contribution (25% weight)
        annual_vol = volatility_metrics['annual_volatility']
        if annual_vol > self.risk_thresholds['volatility_extreme']:
            risk_factors.append(25)
        elif annual_vol > self.risk_thresholds['volatility_high']:
            risk_factors.append(15)
        elif annual_vol > 0.15:  # 15% annual volatility
            risk_factors.append(8)
        else:
            risk_factors.append(3)
        
        # Tail risk contribution (20% weight)
        if tail_risk_metrics['tail_risk_level'] == 'high':
            risk_factors.append(20)
        elif tail_risk_metrics['tail_risk_level'] == 'medium':
            risk_factors.append(10)
        else:
            risk_factors.append(5)
        
        total_risk_score = sum(risk_factors)
        return min(100, total_risk_score)
    
    def _determine_risk_level(self, risk_score: float, var_metrics: Dict, 
                            drawdown_metrics: Dict, volatility_metrics: Dict) -> RiskLevel:
        """Determine overall risk level"""
        if risk_score >= 80:
            return RiskLevel.EXTREME
        elif risk_score >= 60:
            return RiskLevel.HIGH
        elif risk_score >= 40:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_risk_recommendations(self, risk_level: RiskLevel, risk_score: float,
                                     var_metrics: Dict, drawdown_metrics: Dict,
                                     volatility_metrics: Dict, portfolio_impact: Dict) -> List[str]:
        """Generate actionable risk management recommendations"""
        recommendations = []
        
        # Risk level specific recommendations
        if risk_level == RiskLevel.EXTREME:
            recommendations.extend([
                "⚠️ EXTREME RISK: Consider avoiding this position entirely",
                "If trading, use maximum 1-2% of account balance",
                "Set very tight stop losses (2-3% maximum)",
                "Consider hedging strategies or protective puts",
                "Monitor position continuously during market hours"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "🔴 HIGH RISK: Use reduced position sizing (2-5% of account)",
                "Implement strict stop losses (3-5% maximum)",
                "Consider profit-taking at 5-10% gains",
                "Avoid leveraged positions",
                "Monitor closely for deteriorating conditions"
            ])
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "🟡 MEDIUM RISK: Standard position sizing (5-10% of account)",
                "Use trailing stops to protect profits",
                "Consider partial profit-taking at resistance levels",
                "Monitor for trend changes"
            ])
        else:
            recommendations.extend([
                "🟢 LOW RISK: Can use standard to higher position sizing",
                "Consider scaling in on weakness",
                "Use wider stops to avoid whipsaws"
            ])
        
        # Specific metric-based recommendations
        var_95 = abs(var_metrics['confidence_95']['var_historical'])
        if var_95 > 0.05:
            recommendations.append(f"Daily VaR is high ({var_95:.1%}) - consider reducing position size")
        
        if abs(drawdown_metrics['max_drawdown']) > 0.20:
            recommendations.append("High historical drawdown - expect potential significant losses")
        
        if volatility_metrics['volatility_regime'] == 'high':
            recommendations.append("Currently in high volatility regime - expect larger price swings")
        
        if portfolio_impact['concentration_risk'] in ['high', 'extreme']:
            recommendations.append(f"Position creates {portfolio_impact['concentration_risk']} concentration risk - consider reducing size")
        
        # Always include general risk management
        recommendations.extend([
            "Never risk more than 2% of account on a single trade",
            "Use proper position sizing based on volatility",
            "Set stop losses before entering positions",
            "Regular portfolio rebalancing recommended",
            "Consider correlation with other holdings"
        ])
        
        return recommendations
    
    def _create_risk_summary(self, risk_level: RiskLevel, risk_score: float,
                           var_metrics: Dict, drawdown_metrics: Dict) -> str:
        """Create a concise risk summary"""
        var_95 = abs(var_metrics['confidence_95']['var_historical'])
        max_dd = abs(drawdown_metrics['max_drawdown'])
        
        return (f"Risk Level: {risk_level.value.upper()} (Score: {risk_score:.0f}/100) | "
                f"Daily VaR (95%): {var_95:.1%} | Max Drawdown: {max_dd:.1%}")

# Portfolio-level risk management
class PortfolioRiskManager:
    """Portfolio-level risk management and optimization"""
    
    def __init__(self):
        self.correlation_lookback = 60  # days
        self.max_portfolio_var = 0.02   # 2% daily portfolio VaR limit
        self.max_concentration = 0.25   # 25% maximum position size
        
    def calculate_portfolio_risk(self, positions: List[Dict], 
                               price_data: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate comprehensive portfolio risk metrics"""
        
        if not positions:
            return {'error': 'No positions in portfolio'}
        
        try:
            # Extract position data
            symbols = [pos['symbol'] for pos in positions]
            weights = np.array([pos.get('weight', 0) for pos in positions])
            
            # Ensure weights sum to 1
            if weights.sum() > 0:
                weights = weights / weights.sum()
            
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
            
            # Generate portfolio recommendations
            recommendations = self._generate_portfolio_recommendations(
                portfolio_metrics, diversification_metrics, concentration_metrics
            )
            
            return {
                'timestamp': datetime.now().isoformat(),
                'portfolio_metrics': portfolio_metrics,
                'diversification_metrics': diversification_metrics,
                'concentration_metrics': concentration_metrics,
                'recommendations': recommendations,
                'portfolio_summary': self._create_portfolio_summary(
                    portfolio_metrics, diversification_metrics, concentration_metrics
                )
            }
            
        except Exception as e:
            logger.error(f"Error in portfolio risk calculation: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_correlation_matrix(self, symbols: List[str], 
                                    price_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Calculate correlation matrix for portfolio positions"""
        returns_data = {}
        
        for symbol in symbols:
            if symbol in price_data and not price_data[symbol].empty:
                returns = price_data[symbol]['Close'].pct_change().dropna()
                if len(returns) > 0:
                    returns_data[symbol] = returns
        
        if not returns_data:
            # Return identity matrix if no data
            return np.eye(len(symbols))
        
        # Create DataFrame of returns
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr().fillna(0).values
        
        # Ensure positive semi-definite
        eigenvals = np.linalg.eigvals(correlation_matrix)
        if np.any(eigenvals < 0):
            correlation_matrix = correlation_matrix + np.eye(len(symbols)) * 0.01
        
        return correlation_matrix
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray, symbols: List[str],
                                   price_data: Dict[str, pd.DataFrame], 
                                   correlation_matrix: np.ndarray) -> Dict:
        """Calculate portfolio-level risk metrics"""
        
        # Calculate individual asset metrics
        volatilities = []
        expected_returns = []
        
        for symbol in symbols:
            if symbol in price_data and not price_data[symbol].empty:
                returns = price_data[symbol]['Close'].pct_change().dropna()
                if len(returns) > 0:
                    vol = returns.std() * np.sqrt(252)
                    ret = returns.mean() * 252
                    volatilities.append(vol)
                    expected_returns.append(ret)
                else:
                    volatilities.append(0.20)  # Default volatility
                    expected_returns.append(0.10)  # Default return
            else:
                volatilities.append(0.20)  # Default volatility
                expected_returns.append(0.10)  # Default return
        
        volatilities = np.array(volatilities)
        expected_returns = np.array(expected_returns)
        
        # Portfolio expected return
        portfolio_return = np.dot(weights, expected_returns)
        
        # Portfolio volatility using correlation matrix
        portfolio_variance = np.dot(weights, np.dot(
            correlation_matrix * np.outer(volatilities, volatilities), weights
        ))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Portfolio VaR (95% confidence)
        portfolio_var_95 = norm.ppf(0.05, portfolio_return/252, portfolio_volatility/np.sqrt(252))
        portfolio_var_99 = norm.ppf(0.01, portfolio_return/252, portfolio_volatility/np.sqrt(252))
        
        # Risk-adjusted metrics
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'var_95': portfolio_var_95,
            'var_99': portfolio_var_99,
            'sharpe_ratio': sharpe_ratio,
            'individual_volatilities': volatilities.tolist(),
            'individual_returns': expected_returns.tolist(),
            'portfolio_size': len(symbols)
        }
    
    def _calculate_diversification_metrics(self, weights: np.ndarray, 
                                         correlation_matrix: np.ndarray) -> Dict:
        """Calculate diversification effectiveness metrics"""
        
        # Diversification ratio
        weighted_avg_vol = np.sum(weights * np.sqrt(np.diag(correlation_matrix)))
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(correlation_matrix, weights)))
        diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1
        
        # Effective number of positions (inverse of Herfindahl index)
        herfindahl_index = np.sum(weights ** 2)
        effective_positions = 1 / herfindahl_index if herfindahl_index > 0 else 1
        
        # Average correlation
        n = len(weights)
        if n > 1:
            # Extract upper triangular elements (excluding diagonal)
            upper_tri = np.triu(correlation_matrix, k=1)
            avg_correlation = np.sum(upper_tri) / (n * (n - 1) / 2)
        else:
            avg_correlation = 0
        
        # Diversification level
        if diversification_ratio > 1.3:
            diversification_level = 'excellent'
        elif diversification_ratio > 1.2:
            diversification_level = 'good'
        elif diversification_ratio > 1.1:
            diversification_level = 'fair'
        else:
            diversification_level = 'poor'
        
        return {
            'diversification_ratio': diversification_ratio,
            'effective_positions': effective_positions,
            'avg_correlation': avg_correlation,
            'diversification_level': diversification_level,
            'herfindahl_index': herfindahl_index
        }
    
    def _calculate_concentration_risk(self, weights: np.ndarray, symbols: List[str]) -> Dict:
        """Calculate concentration risk metrics"""
        
        # Largest position weight
        max_weight = np.max(weights)
        max_weight_idx = np.argmax(weights)
        max_weight_symbol = symbols[max_weight_idx] if max_weight_idx < len(symbols) else 'Unknown'
        
        # Top 3 positions weight
        top_3_weights = np.sort(weights)[-3:] if len(weights) >= 3 else weights
        top_3_concentration = np.sum(top_3_weights)
        
        # Concentration risk level
        if max_weight > 0.30:
            concentration_risk = 'extreme'
        elif max_weight > 0.20:
            concentration_risk = 'high'
        elif max_weight > 0.15:
            concentration_risk = 'medium'
        else:
            concentration_risk = 'low'
        
        return {
            'max_position_weight': max_weight,
            'max_position_symbol': max_weight_symbol,
            'top_3_concentration': top_3_concentration,
            'concentration_risk': concentration_risk,
            'num_positions': len(weights)
        }
    
    def _generate_portfolio_recommendations(self, portfolio_metrics: Dict,
                                          diversification_metrics: Dict,
                                          concentration_metrics: Dict) -> List[str]:
        """Generate portfolio-level recommendations"""
        recommendations = []
        
        # Diversification recommendations
        if diversification_metrics['diversification_level'] == 'poor':
            recommendations.append("🔴 Poor diversification - consider adding uncorrelated positions")
        elif diversification_metrics['diversification_level'] == 'fair':
            recommendations.append("🟡 Fair diversification - room for improvement")
        elif diversification_metrics['diversification_level'] == 'excellent':
            recommendations.append("🟢 Excellent diversification benefits")
        
        # Concentration recommendations
        if concentration_metrics['concentration_risk'] == 'extreme':
            recommendations.append(f"🔴 EXTREME concentration risk in {concentration_metrics['max_position_symbol']} ({concentration_metrics['max_position_weight']:.1%})")
        elif concentration_metrics['concentration_risk'] == 'high':
            recommendations.append(f"🟡 High concentration in {concentration_metrics['max_position_symbol']} - consider reducing")
        
        # Risk-adjusted return recommendations
        if portfolio_metrics['sharpe_ratio'] < 0.5:
            recommendations.append("🔴 Poor risk-adjusted returns - review strategy")
        elif portfolio_metrics['sharpe_ratio'] > 1.5:
            recommendations.append("🟢 Excellent risk-adjusted returns")
        
        # Volatility recommendations
        if portfolio_metrics['volatility'] > 0.25:
            recommendations.append("🔴 High portfolio volatility - consider reducing position sizes")
        elif portfolio_metrics['volatility'] < 0.10:
            recommendations.append("🟢 Low portfolio volatility - conservative positioning")
        
        # Portfolio VaR recommendations
        if abs(portfolio_metrics['var_95']) > 0.02:
            recommendations.append("🔴 High portfolio VaR - consider risk reduction")
        
        return recommendations
    
    def _create_portfolio_summary(self, portfolio_metrics: Dict, diversification_metrics: Dict,
                                 concentration_metrics: Dict) -> str:
        """Create portfolio risk summary"""
        return (f"Portfolio: {portfolio_metrics['portfolio_size']} positions | "
                f"Volatility: {portfolio_metrics['volatility']:.1%} | "
                f"VaR (95%): {abs(portfolio_metrics['var_95']):.1%} | "
                f"Sharpe: {portfolio_metrics['sharpe_ratio']:.2f} | "
                f"Diversification: {diversification_metrics['diversification_level']}")
