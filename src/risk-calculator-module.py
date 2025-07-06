# src/risk_calculator.py
"""
Risk and reward calculator for position sizing and risk management
Implements professional risk management strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
from datetime import datetime, timedelta

# Import settings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Settings

logger = logging.getLogger(__name__)

class RiskRewardCalculator:
    """Calculates risk metrics and position sizing"""
    
    def __init__(self):
        self.settings = Settings()
        self.risk_params = self.settings.RISK_PARAMETERS
        self.position_sizing = self.settings.POSITION_SIZING
    
    def calculate(self, symbol: str, current_price: float, technical: Dict, 
                  fundamental: Dict, account_balance: Optional[float] = None) -> Dict:
        """Calculate comprehensive risk/reward metrics"""
        
        if account_balance is None:
            account_balance = self.position_sizing['account_size']
        
        try:
            # Get ATR for volatility-based calculations
            atr = technical.get('indicators', {}).get('atr', current_price * 0.02)
            
            # Calculate stop loss levels
            stop_loss_levels = self._calculate_stop_loss(
                current_price, 
                atr, 
                technical.get('support_resistance', {})
            )
            
            # Calculate take profit levels
            take_profit_levels = self._calculate_take_profit(
                current_price,
                atr,
                technical.get('support_resistance', {}),
                technical.get('overall_signal', 0)
            )
            
            # Calculate position size
            position_size = self._calculate_position_size(
                account_balance,
                current_price,
                stop_loss_levels['recommended']
            )
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                current_price,
                stop_loss_levels,
                take_profit_levels,
                position_size,
                technical,
                fundamental
            )
            
            # Calculate Kelly Criterion
            kelly_percentage = self._calculate_kelly_criterion(
                technical.get('overall_signal', 0),
                risk_metrics['risk_reward_ratio']
            )
            
            # Overall risk score
            risk_score = self._calculate_overall_risk_score(
                technical,
                fundamental,
                risk_metrics
            )
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'stop_loss': stop_loss_levels,
                'take_profit': take_profit_levels,
                'position_size': position_size,
                'risk_metrics': risk_metrics,
                'kelly_percentage': kelly_percentage,
                'risk_score': risk_score,
                'risk_reward_ratio': risk_metrics['risk_reward_ratio'],
                'recommendations': self._generate_recommendations(
                    risk_score,
                    risk_metrics,
                    kelly_percentage
                )
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk/reward for {symbol}: {str(e)}")
            return self._default_risk_calculation()
    
    def _calculate_stop_loss(self, price: float, atr: float, support_resistance: Dict) -> Dict:
        """Calculate multiple stop loss levels"""
        
        # ATR-based stop loss
        atr_stop = price - (atr * self.risk_params['stop_loss_atr_multiplier'])
        
        # Percentage-based stop loss
        risk_level = self.settings.get_risk_level()
        percentage_stop = price * (1 - risk_level['stop_loss'])
        
        # Support-based stop loss
        support_levels = support_resistance.get('support', [])
        if support_levels:
            # Use the highest support level that's below current price
            valid_supports = [s for s in support_levels if s < price]
            support_stop = max(valid_supports) * 0.99 if valid_supports else atr_stop
        else:
            support_stop = atr_stop
        
        # Trailing stop loss
        trailing_stop = price * 0.95  # 5% trailing stop
        
        # Recommended stop (most conservative)
        recommended_stop = max(atr_stop, percentage_stop, support_stop)
        
        return {
            'atr_based': round(atr_stop, 2),
            'percentage_based': round(percentage_stop, 2),
            'support_based': round(support_stop, 2),
            'trailing': round(trailing_stop, 2),
            'recommended': round(recommended_stop, 2),
            'stop_distance': round(price - recommended_stop, 2),
            'stop_percentage': round(((price - recommended_stop) / price) * 100, 2)
        }
    
    def _calculate_take_profit(self, price: float, atr: float, 
                              support_resistance: Dict, signal_strength: float) -> Dict:
        """Calculate multiple take profit levels"""
        
        # Risk/reward based targets
        min_rr_ratio = self.risk_params['take_profit_ratio']
        
        # ATR-based targets
        target_1 = price + (atr * min_rr_ratio)
        target_2 = price + (atr * min_rr_ratio * 1.5)
        target_3 = price + (atr * min_rr_ratio * 2)
        
        # Resistance-based targets
        resistance_levels = support_resistance.get('resistance', [])
        if resistance_levels:
            # Filter resistances above current price
            valid_resistances = [r for r in resistance_levels if r > price]
            if valid_resistances:
                resistance_targets = valid_resistances[:3]
            else:
                resistance_targets = [target_1, target_2, target_3]
        else:
            resistance_targets = [target_1, target_2, target_3]
        
        # Fibonacci extension targets
        fib_target_1 = price * 1.0236  # 2.36% extension
        fib_target_2 = price * 1.0382  # 3.82% extension
        fib_target_3 = price * 1.0618  # 6.18% extension
        
        # Adjust based on signal strength
        if abs(signal_strength) > 70:
            multiplier = 1.2
        elif abs(signal_strength) > 50:
            multiplier = 1.0
        else:
            multiplier = 0.8
        
        return {
            'target_1': round(min(target_1 * multiplier, resistance_targets[0] if resistance_targets else target_1), 2),
            'target_2': round(target_2 * multiplier, 2),
            'target_3': round(target_3 * multiplier, 2),
            'resistance_based': [round(r, 2) for r in resistance_targets],
            'fibonacci_based': [round(fib_target_1, 2), round(fib_target_2, 2), round(fib_target_3, 2)],
            'profit_potential': round(((target_1 - price) / price) * 100, 2)
        }
    
    def _calculate_position_size(self, account_balance: float, entry_price: float, 
                                stop_loss: float) -> Dict:
        """Calculate position size using multiple methods"""
        
        # Risk-based position sizing
        risk_amount = account_balance * self.position_sizing['risk_per_trade']
        stop_distance = abs(entry_price - stop_loss)
        risk_based_shares = int(risk_amount / stop_distance) if stop_distance > 0 else 0
        risk_based_value = risk_based_shares * entry_price
        
        # Fixed percentage position sizing
        max_position_value = account_balance * self.risk_params['max_position_size']
        fixed_shares = int(max_position_value / entry_price)
        fixed_value = fixed_shares * entry_price
        
        # Volatility-based position sizing (using ATR)
        volatility_adjusted_size = min(risk_based_value, max_position_value)
        volatility_shares = int(volatility_adjusted_size / entry_price)
        
        # Recommended position size (most conservative)
        recommended_shares = min(risk_based_shares, fixed_shares, volatility_shares)
        recommended_value = recommended_shares * entry_price
        
        return {
            'recommended_shares': recommended_shares,
            'recommended_value': round(recommended_value, 2),
            'risk_based_shares': risk_based_shares,
            'fixed_percentage_shares': fixed_shares,
            'volatility_adjusted_shares': volatility_shares,
            'position_percentage': round((recommended_value / account_balance) * 100, 2),
            'max_loss_amount': round(recommended_shares * stop_distance, 2),
            'max_loss_percentage': round((recommended_shares * stop_distance / account_balance) * 100, 2)
        }
    
    def _calculate_risk_metrics(self, price: float, stop_loss: Dict, 
                               take_profit: Dict, position_size: Dict,
                               technical: Dict, fundamental: Dict) -> Dict:
        """Calculate comprehensive risk metrics"""
        
        # Basic risk/reward ratio
        stop_distance = price - stop_loss['recommended']
        profit_distance = take_profit['target_1'] - price
        risk_reward_ratio = profit_distance / stop_distance if stop_distance > 0 else 0
        
        # Sharpe ratio approximation
        expected_return = (profit_distance / price) * 100
        volatility = technical.get('indicators', {}).get('atr', price * 0.02) / price * 100
        sharpe_ratio = (expected_return - 2.0) / volatility if volatility > 0 else 0  # 2% risk-free rate
        
        # Win probability estimation based on technical signals
        signal_strength = abs(technical.get('overall_signal', 0))
        base_win_rate = 0.5
        
        if signal_strength > 70:
            win_probability = base_win_rate + 0.2
        elif signal_strength > 50:
            win_probability = base_win_rate + 0.1
        elif signal_strength > 30:
            win_probability = base_win_rate + 0.05
        else:
            win_probability = base_win_rate
        
        # Expected value calculation
        expected_win = profit_distance * win_probability
        expected_loss = stop_distance * (1 - win_probability)
        expected_value = expected_win - expected_loss
        
        # Risk of ruin calculation
        risk_of_ruin = self._calculate_risk_of_ruin(
            win_probability,
            risk_reward_ratio,
            position_size['max_loss_percentage'] / 100
        )
        
        return {
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'win_probability': round(win_probability * 100, 2),
            'expected_value': round(expected_value, 2),
            'expected_value_percentage': round((expected_value / price) * 100, 2),
            'risk_of_ruin': round(risk_of_ruin * 100, 2),
            'max_drawdown_expected': round(self._estimate_max_drawdown(win_probability), 2)
        }
    
    def _calculate_kelly_criterion(self, signal_strength: float, risk_reward_ratio: float) -> float:
        """Calculate Kelly Criterion for optimal position sizing"""
        
        # Estimate win probability based on signal strength
        win_prob = 0.5 + (abs(signal_strength) / 100 * 0.3)  # Max 80% win probability
        win_prob = min(0.8, max(0.2, win_prob))
        
        # Kelly formula: f = p - q/b
        # f = fraction of capital to wager
        # p = probability of winning
        # q = probability of losing (1-p)
        # b = ratio of win to loss (risk/reward ratio)
        
        if risk_reward_ratio > 0:
            kelly_percentage = (win_prob - (1 - win_prob) / risk_reward_ratio) * 100
            
            # Apply Kelly fraction (usually 25% of full Kelly for safety)
            safe_kelly = kelly_percentage * 0.25
            
            # Cap at maximum position size
            max_position = self.risk_params['max_position_size'] * 100
            return min(max(0, safe_kelly), max_position)
        
        return 0
    
    def _calculate_overall_risk_score(self, technical: Dict, fundamental: Dict, 
                                     risk_metrics: Dict) -> float:
        """Calculate overall risk score (0-100, higher = riskier)"""
        
        risk_factors = []
        
        # Technical risk factors
        trend = technical.get('trend', {})
        if trend.get('primary') == 'bearish':
            risk_factors.append(20)
        elif trend.get('primary') == 'neutral':
            risk_factors.append(10)
        
        # Volatility risk
        atr = technical.get('indicators', {}).get('atr', 0)
        if atr > 2:  # High volatility
            risk_factors.append(15)
        
        # Fundamental risk factors
        pe_ratio = fundamental.get('pe_ratio', 15)
        if pe_ratio > 20:  # Overvalued
            risk_factors.append(15)
        elif pe_ratio < 10:  # Possibly distressed
            risk_factors.append(10)
        
        debt_to_equity = fundamental.get('debt_to_equity', 1)
        if debt_to_equity > 2:  # High leverage
            risk_factors.append(20)
        
        # Position risk
        if risk_metrics['risk_reward_ratio'] < 1.5:
            risk_factors.append(20)
        
        # Probability risk
        if risk_metrics['win_probability'] < 50:
            risk_factors.append(15)
        
        # Calculate total risk score
        base_risk = 20  # Base risk for any trade
        total_risk = base_risk + sum(risk_factors)
        
        return min(100, total_risk)
    
    def _calculate_risk_of_ruin(self, win_prob: float, risk_reward: float, 
                                risk_per_trade: float) -> float:
        """Calculate probability of account ruin"""
        
        if win_prob >= 1 or risk_reward <= 0:
            return 0
        
        # Simplified risk of ruin formula
        # R = ((1-p)/p)^(C/A)
        # p = win probability
        # C = capital units
        # A = average win in units
        
        loss_prob = 1 - win_prob
        
        if win_prob > 0:
            risk_ratio = loss_prob / win_prob
            capital_units = 1 / risk_per_trade
            
            try:
                risk_of_ruin = pow(risk_ratio, capital_units / risk_reward)
                return min(1, risk_of_ruin)
            except:
                return 0.5
        
        return 1.0
    
    def _estimate_max_drawdown(self, win_probability: float) -> float:
        """Estimate maximum expected drawdown"""
        
        # Based on win probability and consecutive losses
        losing_streak_prob = pow(1 - win_probability, 5)  # 5 losses in a row
        
        # Expected max drawdown
        if win_probability > 0.6:
            return 10.0  # 10% max drawdown
        elif win_probability > 0.5:
            return 15.0  # 15% max drawdown
        elif win_probability > 0.4:
            return 20.0  # 20% max drawdown
        else:
            return 30.0  # 30% max drawdown
    
    def _generate_recommendations(self, risk_score: float, risk_metrics: Dict, 
                                 kelly_percentage: float) -> Dict:
        """Generate risk-based recommendations"""
        
        recommendations = {
            'action': '',
            'confidence': '',
            'position_sizing': '',
            'risk_management': [],
            'warnings': []
        }
        
        # Determine action based on risk score and metrics
        if risk_score > 80:
            recommendations['action'] = 'AVOID'
            recommendations['confidence'] = 'HIGH'
            recommendations['warnings'].append('Very high risk - consider avoiding this trade')
        elif risk_score > 60:
            recommendations['action'] = 'PROCEED_WITH_CAUTION'
            recommendations['confidence'] = 'LOW'
            recommendations['warnings'].append('High risk - reduce position size')
        elif risk_score < 40 and risk_metrics['risk_reward_ratio'] > 2:
            recommendations['action'] = 'FAVORABLE'
            recommendations['confidence'] = 'HIGH'
        else:
            recommendations['action'] = 'NEUTRAL'
            recommendations['confidence'] = 'MEDIUM'
        
        # Position sizing recommendation
        if kelly_percentage > 0:
            recommendations['position_sizing'] = f"Optimal position size: {kelly_percentage:.1f}% of capital"
        else:
            recommendations['position_sizing'] = "Signal too weak for position"
        
        # Risk management recommendations
        if risk_metrics['risk_reward_ratio'] < 1.5:
            recommendations['risk_management'].append('Poor risk/reward ratio - consider different entry/exit levels')
        
        if risk_metrics['win_probability'] < 50:
            recommendations['risk_management'].append('Low win probability - wait for stronger signal')
        
        if risk_metrics['risk_of_ruin'] > 5:
            recommendations['risk_management'].append('High risk of ruin - reduce position size')
        
        recommendations['risk_management'].append('Use trailing stop loss once in profit')
        recommendations['risk_management'].append('Consider scaling out at multiple targets')
        
        return recommendations
    
    def _default_risk_calculation(self) -> Dict:
        """Return default risk calculation when errors occur"""
        return {
            'symbol': '',
            'current_price': 0,
            'stop_loss': {
                'recommended': 0,
                'stop_percentage': 5.0
            },
            'take_profit': {
                'target_1': 0,
                'profit_potential': 10.0
            },
            'position_size': {
                'recommended_shares': 0,
                'recommended_value': 0,
                'position_percentage': 0
            },
            'risk_metrics': {
                'risk_reward_ratio': 0,
                'win_probability': 50,
                'expected_value': 0
            },
            'kelly_percentage': 0,
            'risk_score': 100,
            'recommendations': {
                'action': 'AVOID',
                'warnings': ['Unable to calculate risk metrics']
            }
        }
    
    def calculate_portfolio_risk(self, positions: List[Dict]) -> Dict:
        """Calculate overall portfolio risk metrics"""
        
        if not positions:
            return {
                'total_exposure': 0,
                'risk_concentration': 0,
                'correlation_risk': 'low',
                'recommendations': []
            }
        
        total_value = sum(p.get('value', 0) for p in positions)
        total_risk = sum(p.get('risk_amount', 0) for p in positions)
        
        # Check concentration risk
        largest_position = max(positions, key=lambda x: x.get('value', 0))
        concentration = largest_position.get('value', 0) / total_value if total_value > 0 else 0
        
        # Since all ASX banks are correlated, high correlation risk
        correlation_risk = 'high' if len(positions) > 2 else 'medium'
        
        recommendations = []
        if concentration > 0.3:
            recommendations.append('High concentration risk - diversify positions')
        
        if len(positions) >= self.risk_params['max_open_positions']:
            recommendations.append('Maximum positions reached - close before opening new')
        
        if correlation_risk == 'high':
            recommendations.append('High correlation between bank stocks - consider other sectors')
        
        return {
            'total_exposure': total_value,
            'total_risk_amount': total_risk,
            'risk_percentage': (total_risk / total_value * 100) if total_value > 0 else 0,
            'position_count': len(positions),
            'concentration_risk': concentration * 100,
            'correlation_risk': correlation_risk,
            'recommendations': recommendations
        }
    
    def validate_trade(self, entry_price: float, stop_loss: float, 
                      take_profit: float, account_balance: float) -> Dict:
        """Validate a potential trade against risk rules"""
        
        validations = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check risk/reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        if rr_ratio < self.risk_params['take_profit_ratio']:
            validations['errors'].append(f'Risk/reward ratio {rr_ratio:.2f} below minimum {self.risk_params["take_profit_ratio"]}')
            validations['is_valid'] = False
        
        # Check position size
        position_value = entry_price * 100  # Assume 100 shares
        if position_value > account_balance * self.risk_params['max_position_size']:
            validations['errors'].append('Position size exceeds maximum allowed')
            validations['is_valid'] = False
        
        # Check stop loss distance
        stop_percentage = abs(entry_price - stop_loss) / entry_price
        if stop_percentage > 0.1:  # 10% stop loss
            validations['warnings'].append('Stop loss distance greater than 10%')
        
        return validations