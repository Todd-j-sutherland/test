# backtesting_system.py - Historical Performance Validation System
"""
Backtesting system to validate the accuracy of ASX Bank Trading System predictions
against historical data and actual market performance.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
import logging
import asyncio
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.technical_analysis import TechnicalAnalyzer
from src.fundamental_analysis import FundamentalAnalyzer
from src.market_predictor import MarketPredictor
from src.advanced_risk_manager import AdvancedRiskManager
from src.data_feed import ASXDataFeed
from src.data_validator import DataValidator
from src.async_main import ASXBankTradingSystemAsync
from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Container for backtest results"""
    date: str
    symbol: str
    predicted_direction: str
    predicted_confidence: float
    actual_return_1d: float
    actual_return_5d: float
    actual_return_20d: float
    prediction_accuracy_1d: bool
    prediction_accuracy_5d: bool
    prediction_accuracy_20d: bool
    risk_level: str
    var_predicted: float
    actual_max_loss: float
    var_accuracy: bool
    trade_profitable: bool
    trade_return: float
    # Enhanced analysis fields
    analysis_type: str = 'legacy'  # 'enhanced' or 'legacy'
    data_quality: str = 'unknown'  # 'excellent', 'validated', 'basic', etc.
    data_confidence: float = 0.0   # 0-1 confidence score from data validation

class HistoricalBacktester:
    """
    Backtesting system for validating trading system performance
    """
    
    def __init__(self):
        self.settings = Settings()
        self.data_feed = ASXDataFeed()
        self.data_validator = DataValidator()
        self.technical_analyzer = TechnicalAnalyzer()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.market_predictor = MarketPredictor()
        self.risk_manager = AdvancedRiskManager()
        
        # Backtesting parameters
        self.confidence_threshold = 0.3  # Minimum confidence to take a position
        self.max_position_size = 0.1     # Maximum 10% per position
        self.stop_loss_pct = 0.05        # 5% stop loss
        self.take_profit_pct = 0.10      # 10% take profit
        
    def get_historical_data_for_backtest(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical data for backtesting period"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval="1d")
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def simulate_analysis_at_date(self, symbol: str, analysis_date: datetime, lookback_days: int = 252) -> Dict:
        """
        Simulate running the analysis system at a specific historical date
        Uses only data available up to that date
        """
        try:
            # Get data available up to analysis date (simulate real-time conditions)
            end_date = analysis_date.strftime('%Y-%m-%d')
            start_date = (analysis_date - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            # Get historical market data up to analysis date
            historical_data = self.get_historical_data_for_backtest(symbol, start_date, end_date)
            
            if historical_data.empty or len(historical_data) < 30:
                return {'error': 'Insufficient historical data'}
            
            # Run technical analysis with data available at that time
            technical_analysis = self.technical_analyzer.analyze(symbol, historical_data)
            
            # Run fundamental analysis (this is less time-sensitive)
            # Note: For true backtesting, you'd need historical fundamental data
            fundamental_analysis = self.fundamental_analyzer.analyze(symbol)
            
            # Simulate sentiment analysis (simplified - no historical news available)
            # For backtesting, we'll use a neutral sentiment score
            sentiment_analysis = {
                'overall_sentiment': 0.0,
                'confidence': 0.5,
                'score': 0.0,
                'classification': 'neutral',
                'news_count': 0,
                'simulated': True  # Flag to indicate this is simulated
            }
            
            # Run market prediction based on available data
            prediction = self.market_predictor.predict(
                symbol, technical_analysis, fundamental_analysis, sentiment_analysis
            )
            
            # Calculate risk metrics
            risk_analysis = self.risk_manager.calculate_comprehensive_risk(
                symbol, historical_data, position_size=10000, account_balance=100000
            )
            
            return {
                'symbol': symbol,
                'analysis_date': analysis_date.isoformat(),
                'prediction': prediction,
                'technical': technical_analysis,
                'fundamental': fundamental_analysis,
                'sentiment': sentiment_analysis,
                'risk': risk_analysis,
                'data_points': len(historical_data)
            }
            
        except Exception as e:
            logger.error(f"Error in simulated analysis for {symbol} on {analysis_date}: {str(e)}")
            return {'error': str(e)}
    
    def calculate_actual_returns(self, symbol: str, entry_date: datetime, periods: List[int] = [1, 5, 20]) -> Dict:
        """Calculate actual returns for different periods after entry date"""
        try:
            # Get price data for the period after entry
            start_date = entry_date.strftime('%Y-%m-%d')
            end_date = (entry_date + timedelta(days=max(periods) + 10)).strftime('%Y-%m-%d')
            
            price_data = self.get_historical_data_for_backtest(symbol, start_date, end_date)
            
            if price_data.empty:
                return {'error': 'No price data available'}
            
            entry_price = price_data['Close'].iloc[0]
            returns = {}
            
            for period in periods:
                if len(price_data) > period:
                    exit_price = price_data['Close'].iloc[period]
                    period_return = (exit_price - entry_price) / entry_price
                    returns[f'return_{period}d'] = period_return
                else:
                    returns[f'return_{period}d'] = None
            
            # Calculate maximum drawdown during the period
            if len(price_data) > 1:
                prices = price_data['Close'].iloc[1:min(21, len(price_data))]  # First 20 days
                max_loss = ((prices.min() - entry_price) / entry_price) if not prices.empty else 0
                returns['max_loss_20d'] = max_loss
            else:
                returns['max_loss_20d'] = 0
                
            return returns
            
        except Exception as e:
            logger.error(f"Error calculating actual returns for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def simulate_trade(self, prediction: Dict, actual_returns: Dict, position_size: float = 0.05) -> Dict:
        """
        Simulate a trade based on prediction and calculate performance
        """
        try:
            direction = prediction.get('direction', 'neutral')
            confidence = prediction.get('confidence', 0)
            
            # Only trade if confidence is above threshold
            if confidence < self.confidence_threshold:
                return {
                    'trade_taken': False,
                    'reason': 'Confidence below threshold',
                    'return': 0
                }
            
            # Only trade on non-neutral predictions
            if direction == 'neutral':
                return {
                    'trade_taken': False,
                    'reason': 'Neutral prediction',
                    'return': 0
                }
            
            # Calculate position size based on confidence and risk
            adjusted_position_size = min(position_size * confidence, self.max_position_size)
            
            # Get actual 5-day return (typical holding period)
            actual_return = actual_returns.get('return_5d', 0)
            if actual_return is None:
                return {
                    'trade_taken': False,
                    'reason': 'No return data available',
                    'return': 0
                }
            
            # Calculate trade return based on direction
            if direction == 'bullish':
                trade_return = actual_return * adjusted_position_size
            elif direction == 'bearish':
                # Short position (inverse return)
                trade_return = -actual_return * adjusted_position_size
            else:
                trade_return = 0
            
            # Apply stop loss and take profit
            if trade_return < -self.stop_loss_pct * adjusted_position_size:
                trade_return = -self.stop_loss_pct * adjusted_position_size
            elif trade_return > self.take_profit_pct * adjusted_position_size:
                trade_return = self.take_profit_pct * adjusted_position_size
            
            return {
                'trade_taken': True,
                'direction': direction,
                'confidence': confidence,
                'position_size': adjusted_position_size,
                'actual_return': actual_return,
                'trade_return': trade_return,
                'profitable': trade_return > 0
            }
            
        except Exception as e:
            logger.error(f"Error simulating trade: {str(e)}")
            return {'trade_taken': False, 'error': str(e), 'return': 0}
    
    async def run_enhanced_backtest(self, symbols: List[str], start_date: str, end_date: str, frequency_days: int = 7) -> List[BacktestResult]:
        """
        Run comprehensive backtest using the ENHANCED analysis system
        """
        logger.info(f"Starting ENHANCED backtest from {start_date} to {end_date}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Analysis frequency: every {frequency_days} days")
        
        results = []
        
        # Generate test dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        test_dates = []
        current_date = start_dt
        while current_date < end_dt:
            # Add all weekdays, skip weekends
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                test_dates.append(current_date)
                # Skip ahead by frequency_days after adding a valid date
                current_date += timedelta(days=frequency_days)
            else:
                # If weekend, just move to next day
                current_date += timedelta(days=1)
        
        logger.info(f"Generated {len(test_dates)} test dates")
        
        total_tests = len(symbols) * len(test_dates)
        completed_tests = 0
        
        for symbol in symbols:
            logger.info(f"Testing {symbol} with ENHANCED analysis...")
            
            for test_date in test_dates:
                try:
                    # Skip if too close to end date (need time for return calculation)
                    if (end_dt - test_date).days < 25:
                        continue
                    
                    # Run enhanced analysis at historical date
                    analysis = await self.simulate_enhanced_analysis_at_date(symbol, test_date)
                    
                    if 'error' in analysis:
                        completed_tests += 1
                        continue
                    
                    # Calculate actual returns after the analysis date
                    actual_returns = self.calculate_actual_returns(symbol, test_date)
                    
                    if 'error' in actual_returns:
                        completed_tests += 1
                        continue
                    
                    # Extract analysis results (handle both enhanced and legacy formats)
                    if 'enhanced_analysis' in analysis:
                        # Enhanced analysis format
                        enhanced_result = analysis['enhanced_analysis']
                        prediction = enhanced_result.get('prediction', {})
                        risk = enhanced_result.get('enhanced_risk', enhanced_result.get('risk', {}))
                        data_validation = analysis.get('data_validation', {})
                    else:
                        # Legacy analysis format (fallback)
                        prediction = analysis.get('prediction', {})
                        risk = analysis.get('risk', {})
                        data_validation = {'is_valid': True, 'quality': 'basic', 'confidence_score': 0.5}
                    
                    # Simulate trade
                    trade_result = self.simulate_trade(prediction, actual_returns)
                    
                    # Calculate prediction accuracy
                    predicted_direction = prediction.get('direction', 'neutral')
                    actual_1d = actual_returns.get('return_1d', 0) or 0
                    actual_5d = actual_returns.get('return_5d', 0) or 0
                    actual_20d = actual_returns.get('return_20d', 0) or 0
                    
                    # Accuracy calculation
                    def check_accuracy(predicted_dir, actual_return):
                        if predicted_dir == 'bullish':
                            return actual_return > 0
                        elif predicted_dir == 'bearish':
                            return actual_return < 0
                        else:  # neutral
                            return abs(actual_return) < 0.02  # Within 2%
                    
                    # VaR accuracy check
                    var_predicted = 0
                    if 'var_metrics' in risk and 'confidence_95' in risk['var_metrics']:
                        var_predicted = abs(risk['var_metrics']['confidence_95'].get('var_historical', 0))
                    
                    actual_max_loss = abs(actual_returns.get('max_loss_20d', 0))
                    var_accuracy = actual_max_loss <= var_predicted if var_predicted > 0 else True
                    
                    # Create backtest result with enhanced data
                    result = BacktestResult(
                        date=test_date.strftime('%Y-%m-%d'),
                        symbol=symbol,
                        predicted_direction=predicted_direction,
                        predicted_confidence=prediction.get('confidence', 0),
                        actual_return_1d=actual_1d,
                        actual_return_5d=actual_5d,
                        actual_return_20d=actual_20d,
                        prediction_accuracy_1d=check_accuracy(predicted_direction, actual_1d),
                        prediction_accuracy_5d=check_accuracy(predicted_direction, actual_5d),
                        prediction_accuracy_20d=check_accuracy(predicted_direction, actual_20d),
                        risk_level=risk.get('risk_level', 'unknown'),
                        var_predicted=var_predicted,
                        actual_max_loss=actual_max_loss,
                        var_accuracy=var_accuracy,
                        trade_profitable=trade_result.get('profitable', False),
                        trade_return=trade_result.get('trade_return', 0)
                    )
                    
                    # Add enhanced analysis metadata
                    result.analysis_type = analysis.get('analysis_type', 'enhanced')
                    result.data_quality = data_validation.get('quality', 'unknown')
                    result.data_confidence = data_validation.get('confidence_score', 0)
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error in enhanced backtest for {symbol} on {test_date}: {str(e)}")
                
                completed_tests += 1
                if completed_tests % 10 == 0:
                    logger.info(f"Progress: {completed_tests}/{total_tests} ({completed_tests/total_tests*100:.1f}%)")
        
        logger.info(f"Enhanced backtest completed. Generated {len(results)} results.")
        return results
    
    def run_backtest(self, symbols: List[str], start_date: str, end_date: str, frequency_days: int = 7) -> List[BacktestResult]:
        """
        Run comprehensive backtest using LEGACY analysis system (for comparison)
        """
        logger.info(f"Starting LEGACY backtest from {start_date} to {end_date}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Analysis frequency: every {frequency_days} days")
        
        results = []
        
        # Generate test dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        test_dates = []
        current_date = start_dt
        while current_date < end_dt:
            # Add all weekdays, skip weekends
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                test_dates.append(current_date)
                # Skip ahead by frequency_days after adding a valid date
                current_date += timedelta(days=frequency_days)
            else:
                # If weekend, just move to next day
                current_date += timedelta(days=1)
        
        logger.info(f"Generated {len(test_dates)} test dates")
        
        total_tests = len(symbols) * len(test_dates)
        completed_tests = 0
        
        for symbol in symbols:
            logger.info(f"Testing {symbol} with LEGACY analysis...")
            
            for test_date in test_dates:
                try:
                    # Skip if too close to end date (need time for return calculation)
                    if (end_dt - test_date).days < 25:
                        continue
                    
                    # Run legacy analysis at historical date
                    analysis = self.simulate_analysis_at_date(symbol, test_date)
                    
                    if 'error' in analysis:
                        completed_tests += 1
                        continue
                    
                    # Calculate actual returns after the analysis date
                    actual_returns = self.calculate_actual_returns(symbol, test_date)
                    
                    if 'error' in actual_returns:
                        completed_tests += 1
                        continue
                    
                    # Simulate trade
                    trade_result = self.simulate_trade(analysis['prediction'], actual_returns)
                    
                    # Extract prediction details
                    prediction = analysis['prediction']
                    risk = analysis.get('risk', {})
                    
                    # Calculate prediction accuracy
                    predicted_direction = prediction.get('direction', 'neutral')
                    actual_1d = actual_returns.get('return_1d', 0) or 0
                    actual_5d = actual_returns.get('return_5d', 0) or 0
                    actual_20d = actual_returns.get('return_20d', 0) or 0
                    
                    # Accuracy calculation
                    def check_accuracy(predicted_dir, actual_return):
                        if predicted_dir == 'bullish':
                            return actual_return > 0
                        elif predicted_dir == 'bearish':
                            return actual_return < 0
                        else:  # neutral
                            return abs(actual_return) < 0.02  # Within 2%
                    
                    # VaR accuracy check
                    var_predicted = 0
                    if 'var_metrics' in risk and 'confidence_95' in risk['var_metrics']:
                        var_predicted = abs(risk['var_metrics']['confidence_95'].get('var_historical', 0))
                    
                    actual_max_loss = abs(actual_returns.get('max_loss_20d', 0))
                    var_accuracy = actual_max_loss <= var_predicted if var_predicted > 0 else True
                    
                    # Create backtest result
                    result = BacktestResult(
                        date=test_date.strftime('%Y-%m-%d'),
                        symbol=symbol,
                        predicted_direction=predicted_direction,
                        predicted_confidence=prediction.get('confidence', 0),
                        actual_return_1d=actual_1d,
                        actual_return_5d=actual_5d,
                        actual_return_20d=actual_20d,
                        prediction_accuracy_1d=check_accuracy(predicted_direction, actual_1d),
                        prediction_accuracy_5d=check_accuracy(predicted_direction, actual_5d),
                        prediction_accuracy_20d=check_accuracy(predicted_direction, actual_20d),
                        risk_level=risk.get('risk_level', 'unknown'),
                        var_predicted=var_predicted,
                        actual_max_loss=actual_max_loss,
                        var_accuracy=var_accuracy,
                        trade_profitable=trade_result.get('profitable', False),
                        trade_return=trade_result.get('trade_return', 0),
                        analysis_type='legacy',
                        data_quality='basic',
                        data_confidence=0.5
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error in legacy backtest for {symbol} on {test_date}: {str(e)}")
                
                completed_tests += 1
                if completed_tests % 10 == 0:
                    logger.info(f"Progress: {completed_tests}/{total_tests} ({completed_tests/total_tests*100:.1f}%)")
        
        logger.info(f"Legacy backtest completed. Generated {len(results)} results.")
        return results
    
    def analyze_backtest_results(self, results: List[BacktestResult]) -> Dict:
        """Analyze and summarize backtest results"""
        if not results:
            return {'error': 'No results to analyze'}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([vars(r) for r in results])
        
        # Overall statistics
        total_tests = len(df)
        
        # Prediction accuracy
        accuracy_1d = df['prediction_accuracy_1d'].mean()
        accuracy_5d = df['prediction_accuracy_5d'].mean()
        accuracy_20d = df['prediction_accuracy_20d'].mean()
        
        # VaR accuracy
        var_accuracy = df['var_accuracy'].mean()
        
        # Trading performance
        trades_taken = df['trade_profitable'].count()
        profitable_trades = df['trade_profitable'].sum()
        win_rate = profitable_trades / trades_taken if trades_taken > 0 else 0
        
        average_return = df['trade_return'].mean()
        total_return = df['trade_return'].sum()
        
        # By symbol analysis
        symbol_stats = df.groupby('symbol').agg({
            'prediction_accuracy_5d': 'mean',
            'var_accuracy': 'mean',
            'trade_return': ['count', 'sum', 'mean'],
            'trade_profitable': 'sum'
        }).round(4)
        
        # By confidence level
        df['confidence_bucket'] = pd.cut(df['predicted_confidence'], 
                                       bins=[0, 0.3, 0.5, 0.7, 1.0], 
                                       labels=['Low', 'Medium', 'High', 'Very High'])
        
        confidence_stats = df.groupby('confidence_bucket').agg({
            'prediction_accuracy_5d': 'mean',
            'trade_return': 'mean'
        }).round(4)
        
        return {
            'total_tests': total_tests,
            'prediction_accuracy': {
                '1_day': accuracy_1d,
                '5_day': accuracy_5d,
                '20_day': accuracy_20d
            },
            'var_accuracy': var_accuracy,
            'trading_performance': {
                'trades_taken': trades_taken,
                'win_rate': win_rate,
                'average_return': average_return,
                'total_return': total_return,
                'profitable_trades': profitable_trades
            },
            'symbol_performance': symbol_stats.to_dict(),
            'confidence_analysis': confidence_stats.to_dict()
        }
    
    def generate_backtest_report(self, results: List[BacktestResult], output_file: str = None):
        """Generate comprehensive backtest report"""
        analysis = self.analyze_backtest_results(results)
        
        if output_file is None:
            output_file = f"reports/backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        os.makedirs('reports', exist_ok=True)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ASX Trading System Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .good {{ color: green; font-weight: bold; }}
                .bad {{ color: red; font-weight: bold; }}
                .neutral {{ color: orange; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>🧪 ASX Trading System Backtest Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>📊 Overall Performance Summary</h2>
                <div class="metric">
                    <h3>Total Tests</h3>
                    <div>{analysis['total_tests']}</div>
                </div>
                <div class="metric">
                    <h3>5-Day Accuracy</h3>
                    <div class="{'good' if analysis['prediction_accuracy']['5_day'] > 0.6 else 'bad' if analysis['prediction_accuracy']['5_day'] < 0.4 else 'neutral'}">{analysis['prediction_accuracy']['5_day']:.1%}</div>
                </div>
                <div class="metric">
                    <h3>VaR Accuracy</h3>
                    <div class="{'good' if analysis['var_accuracy'] > 0.9 else 'bad' if analysis['var_accuracy'] < 0.8 else 'neutral'}">{analysis['var_accuracy']:.1%}</div>
                </div>
                <div class="metric">
                    <h3>Win Rate</h3>
                    <div class="{'good' if analysis['trading_performance']['win_rate'] > 0.55 else 'bad' if analysis['trading_performance']['win_rate'] < 0.45 else 'neutral'}">{analysis['trading_performance']['win_rate']:.1%}</div>
                </div>
                <div class="metric">
                    <h3>Total Return</h3>
                    <div class="{'good' if analysis['trading_performance']['total_return'] > 0 else 'bad'}">{analysis['trading_performance']['total_return']:.2%}</div>
                </div>
            </div>
            
            <h2>🎯 Prediction Accuracy by Time Horizon</h2>
            <table>
                <tr><th>Time Horizon</th><th>Accuracy</th><th>Assessment</th></tr>
                <tr><td>1 Day</td><td>{analysis['prediction_accuracy']['1_day']:.1%}</td><td>{'Good' if analysis['prediction_accuracy']['1_day'] > 0.5 else 'Needs Improvement'}</td></tr>
                <tr><td>5 Days</td><td>{analysis['prediction_accuracy']['5_day']:.1%}</td><td>{'Good' if analysis['prediction_accuracy']['5_day'] > 0.5 else 'Needs Improvement'}</td></tr>
                <tr><td>20 Days</td><td>{analysis['prediction_accuracy']['20_day']:.1%}</td><td>{'Good' if analysis['prediction_accuracy']['20_day'] > 0.5 else 'Needs Improvement'}</td></tr>
            </table>
            
            <h2>💰 Trading Performance Details</h2>
            <table>
                <tr><th>Metric</th><th>Value</th><th>Benchmark</th><th>Status</th></tr>
                <tr><td>Win Rate</td><td>{analysis['trading_performance']['win_rate']:.1%}</td><td>>55%</td><td>{'✅' if analysis['trading_performance']['win_rate'] > 0.55 else '❌'}</td></tr>
                <tr><td>Average Return per Trade</td><td>{analysis['trading_performance']['average_return']:.2%}</td><td>>0%</td><td>{'✅' if analysis['trading_performance']['average_return'] > 0 else '❌'}</td></tr>
                <tr><td>Total Return</td><td>{analysis['trading_performance']['total_return']:.2%}</td><td>>0%</td><td>{'✅' if analysis['trading_performance']['total_return'] > 0 else '❌'}</td></tr>
                <tr><td>Trades Taken</td><td>{analysis['trading_performance']['trades_taken']}</td><td>-</td><td>-</td></tr>
            </table>
            
            <h2>⚠️ Risk Management Assessment</h2>
            <p><strong>VaR Model Accuracy:</strong> {analysis['var_accuracy']:.1%}</p>
            <p>The VaR (Value at Risk) model should be accurate ~95% of the time. Current accuracy: {'✅ Good' if analysis['var_accuracy'] > 0.9 else '⚠️ Needs Calibration' if analysis['var_accuracy'] > 0.8 else '❌ Poor - Requires Attention'}</p>
            
            <h2>🏆 Recommendations</h2>
            <ul>
        """
        
        # Add recommendations based on performance
        if analysis['prediction_accuracy']['5_day'] < 0.5:
            html_content += "<li>❌ Prediction accuracy is below 50%. Consider refining the ML model or adding more features.</li>"
        else:
            html_content += "<li>✅ Prediction accuracy is acceptable. Continue monitoring.</li>"
        
        if analysis['trading_performance']['win_rate'] < 0.45:
            html_content += "<li>❌ Win rate is low. Consider increasing confidence threshold or improving entry signals.</li>"
        elif analysis['trading_performance']['win_rate'] > 0.55:
            html_content += "<li>✅ Good win rate. Trading strategy is performing well.</li>"
        
        if analysis['var_accuracy'] < 0.9:
            html_content += "<li>⚠️ VaR model needs recalibration. Risk estimates may be inaccurate.</li>"
        else:
            html_content += "<li>✅ Risk management models are well-calibrated.</li>"
        
        html_content += """
            </ul>
            
            <h2>📈 Next Steps</h2>
            <ol>
                <li>If accuracy is low, consider adding more technical indicators or improving the ML model</li>
                <li>If win rate is poor, adjust confidence thresholds or position sizing</li>
                <li>If VaR accuracy is off, recalibrate risk models with more recent data</li>
                <li>Consider running longer backtests for more robust statistics</li>
            </ol>
            
            <p><em>This backtest uses simulated sentiment data due to historical news limitations. 
            For production use, consider integrating historical news feeds for more accurate sentiment analysis.</em></p>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Backtest report saved to: {output_file}")
        return output_file

    async def simulate_enhanced_analysis_at_date(self, symbol: str, analysis_date: datetime, lookback_days: int = 252) -> Dict:
        """
        Simulate running the ENHANCED analysis system at a specific historical date
        Uses the new enhanced async system with advanced data validation and risk management
        """
        try:
            # Get data available up to analysis date (simulate real-time conditions)
            end_date = analysis_date.strftime('%Y-%m-%d')
            start_date = (analysis_date - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            # Get historical market data up to analysis date
            historical_data = self.get_historical_data_for_backtest(symbol, start_date, end_date)
            
            if historical_data.empty or len(historical_data) < 30:
                return {'error': 'Insufficient historical data'}
            
            # Validate data using the advanced data validator
            validation_result = self.data_validator.validate_market_data(historical_data, symbol)
            
            # Use the enhanced async analysis system (but in a point-in-time simulation)
            try:
                async with ASXBankTradingSystemAsync() as enhanced_system:
                    # Store original method
                    original_get_data = enhanced_system.data_feed.get_historical_data
                    
                    # Create a patched version that returns our historical data for this symbol
                    def backtest_get_data(requested_symbol, period="1mo", interval="1d"):
                        if requested_symbol == symbol:
                            return historical_data
                        # For other symbols, use original method
                        return original_get_data(requested_symbol, period=period, interval=interval)
                    
                    # Temporarily replace the method
                    enhanced_system.data_feed.get_historical_data = backtest_get_data
                    
                    try:
                        # Run enhanced analysis for this specific symbol
                        enhanced_result = await enhanced_system.analyze_bank_async(symbol)
                    finally:
                        # Always restore original method
                        enhanced_system.data_feed.get_historical_data = original_get_data
                    
                    if 'error' in enhanced_result:
                        logger.warning(f"Enhanced analysis failed for {symbol}: {enhanced_result['error']}")
                        return await self._fallback_to_legacy_analysis(symbol, analysis_date, historical_data)
                    
                    # Add enhanced risk management
                    comprehensive_risk = self.risk_manager.calculate_comprehensive_risk(
                        symbol, historical_data, position_size=10000, account_balance=100000
                    )
                    enhanced_result['enhanced_risk'] = comprehensive_risk
                    
                    # Add data validation info
                    enhanced_result['data_validation'] = {
                        'is_valid': validation_result.is_valid,
                        'quality': validation_result.quality.value,
                        'confidence_score': validation_result.confidence_score,
                        'warnings': validation_result.warnings,
                        'errors': validation_result.errors
                    }
                    
                    return {
                        'symbol': symbol,
                        'analysis_date': analysis_date.isoformat(),
                        'enhanced_analysis': enhanced_result,
                        'data_points': len(historical_data),
                        'analysis_type': 'enhanced'
                    }
                    
            except Exception as e:
                logger.warning(f"Enhanced analysis failed for {symbol}, falling back to legacy: {str(e)}")
                # Fallback to legacy analysis
                return await self._fallback_to_legacy_analysis(symbol, analysis_date, historical_data)
                
        except Exception as e:
            logger.error(f"Error in enhanced analysis for {symbol} on {analysis_date}: {str(e)}")
            return {'error': str(e)}
    
    async def _fallback_to_legacy_analysis(self, symbol: str, analysis_date: datetime, historical_data: pd.DataFrame) -> Dict:
        """Fallback to the original analysis method if enhanced fails"""
        # Run technical analysis with data available at that time
        technical_analysis = self.technical_analyzer.analyze(symbol, historical_data)
        
        # Run fundamental analysis (this is less time-sensitive)
        fundamental_analysis = self.fundamental_analyzer.analyze(symbol)
        
        # Simulate sentiment analysis (simplified - no historical news available)
        sentiment_analysis = {
            'overall_sentiment': 0.0,
            'confidence': 0.5,
            'score': 0.0,
            'classification': 'neutral',
            'news_count': 0,
            'simulated': True
        }
        
        # Run market prediction based on available data
        prediction = self.market_predictor.predict(
            symbol, technical_analysis, fundamental_analysis, sentiment_analysis
        )
        
        # Calculate risk metrics
        risk_analysis = self.risk_manager.calculate_comprehensive_risk(
            symbol, historical_data, position_size=10000, account_balance=100000
        )
        
        return {
            'symbol': symbol,
            'analysis_date': analysis_date.isoformat(),
            'prediction': prediction,
            'technical': technical_analysis,
            'fundamental': fundamental_analysis,
            'sentiment': sentiment_analysis,
            'risk': risk_analysis,
            'data_points': len(historical_data),
            'analysis_type': 'legacy'
        }
    

def main():
    """Run backtest on ASX bank stocks"""
    print("🧪 ASX Trading System Backtester")
    print("=" * 50)
    
    # Initialize backtester
    backtester = HistoricalBacktester()
    
    # Configuration
    symbols = ['CBA.AX', 'WBC.AX', 'ANZ.AX', 'NAB.AX']  # Reduced set for demo
    start_date = '2024-01-01'  # Start of backtest period
    end_date = '2024-06-01'    # End of backtest period (need buffer for return calculation)
    frequency_days = 14        # Test every 2 weeks
    
    print(f"📊 Configuration:")
    print(f"   Symbols: {symbols}")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   Frequency: Every {frequency_days} days")
    print(f"   Confidence threshold: {backtester.confidence_threshold}")
    print()
    
    # Ask user which analysis to run
    print("Choose analysis type:")
    print("1. Enhanced Analysis (NEW - includes advanced data validation & risk management)")
    print("2. Legacy Analysis (original system)")
    print("3. Both (compare enhanced vs legacy)")
    
    try:
        choice = input("Enter choice (1-3) [default: 1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        choice = "1"  # Default for automated runs
    
    if choice == "1":
        # Run enhanced backtest
        print("🚀 Starting ENHANCED backtest...")
        results = asyncio.run(backtester.run_enhanced_backtest(symbols, start_date, end_date, frequency_days))
        report_suffix = "_enhanced"
    elif choice == "2":
        # Run legacy backtest
        print("🚀 Starting LEGACY backtest...")
        results = backtester.run_backtest(symbols, start_date, end_date, frequency_days)
        report_suffix = "_legacy"
    else:
        # Run both and compare
        print("🚀 Starting COMPARISON backtest (Enhanced vs Legacy)...")
        
        print("\n--- Running Enhanced Analysis ---")
        enhanced_results = asyncio.run(backtester.run_enhanced_backtest(symbols, start_date, end_date, frequency_days))
        
        print("\n--- Running Legacy Analysis ---")
        legacy_results = backtester.run_backtest(symbols, start_date, end_date, frequency_days)
        
        # Combine results for comparison
        results = enhanced_results + legacy_results
        report_suffix = "_comparison"
        
        # Print comparison summary
        if enhanced_results and legacy_results:
            enhanced_analysis = backtester.analyze_backtest_results(enhanced_results)
            legacy_analysis = backtester.analyze_backtest_results(legacy_results)
            
            print(f"\n📊 COMPARISON RESULTS:")
            print("=" * 50)
            print(f"ENHANCED - 5-Day Accuracy: {enhanced_analysis['prediction_accuracy']['5_day']:.1%}")
            print(f"LEGACY   - 5-Day Accuracy: {legacy_analysis['prediction_accuracy']['5_day']:.1%}")
            print(f"ENHANCED - VaR Accuracy: {enhanced_analysis['var_accuracy']:.1%}")
            print(f"LEGACY   - VaR Accuracy: {legacy_analysis['var_accuracy']:.1%}")
    
    if not results:
        print("❌ No results generated. Check the date range and data availability.")
        return
    
    # Analyze results
    print("📊 Analyzing results...")
    analysis = backtester.analyze_backtest_results(results)
    
    # Print summary
    print(f"\n🎯 BACKTEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total tests: {analysis['total_tests']}")
    print(f"Prediction accuracy (5-day): {analysis['prediction_accuracy']['5_day']:.1%}")
    print(f"VaR model accuracy: {analysis['var_accuracy']:.1%}")
    print(f"Trading win rate: {analysis['trading_performance']['win_rate']:.1%}")
    print(f"Total return: {analysis['trading_performance']['total_return']:.2%}")
    print(f"Average return per trade: {analysis['trading_performance']['average_return']:.2%}")
    
    # Generate detailed report
    print("\n📋 Generating detailed report...")
    report_file = backtester.generate_backtest_report(results, 
        output_file=f"reports/backtest_report{report_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    print(f"📁 Report saved to: {report_file}")
    
    # Save raw results
    results_file = f"reports/backtest_results{report_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump([vars(r) for r in results], f, indent=2, default=str)
    print(f"📁 Raw results saved to: {results_file}")
    
    print("\n✅ Backtest completed successfully!")

if __name__ == "__main__":
    main()
