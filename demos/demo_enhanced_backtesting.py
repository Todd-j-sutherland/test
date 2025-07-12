#!/usr/bin/env python3
"""
Enhanced Backtesting Demo (Simplified)
Demonstrates the enhanced backtesting capabilities without plotting dependencies
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
import logging
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import settings
from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedBacktester:
    """
    Simplified version of the backtesting system for demonstration
    """
    
    def __init__(self):
        self.confidence_threshold = 0.3
        self.max_position_size = 0.1
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.10
        
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical data for backtesting period"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval="1d")
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_simple_technical_signals(self, data: pd.DataFrame) -> Dict:
        """Calculate simple technical analysis signals"""
        if len(data) < 20:
            return {'signal': 'neutral', 'confidence': 0.0}
        
        # Simple moving averages
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Get current values
        current_price = data['Close'].iloc[-1]
        sma_10 = data['SMA_10'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        
        # Generate signals
        signals = []
        
        # Moving average signals
        if current_price > sma_10 > sma_20:
            signals.append(('buy', 0.6))
        elif current_price < sma_10 < sma_20:
            signals.append(('sell', 0.6))
        
        # RSI signals
        if rsi < 30:
            signals.append(('buy', 0.7))
        elif rsi > 70:
            signals.append(('sell', 0.7))
        
        # Determine overall signal
        if not signals:
            return {'signal': 'neutral', 'confidence': 0.0}
        
        # Count signals
        buy_signals = [s for s in signals if s[0] == 'buy']
        sell_signals = [s for s in signals if s[0] == 'sell']
        
        if len(buy_signals) > len(sell_signals):
            confidence = np.mean([s[1] for s in buy_signals])
            return {'signal': 'buy', 'confidence': confidence}
        elif len(sell_signals) > len(buy_signals):
            confidence = np.mean([s[1] for s in sell_signals])
            return {'signal': 'sell', 'confidence': confidence}
        else:
            return {'signal': 'neutral', 'confidence': 0.0}
    
    def simulate_analysis_at_date(self, symbol: str, analysis_date: datetime) -> Dict:
        """Simulate analysis at a specific date"""
        try:
            # Get historical data up to analysis date
            end_date = analysis_date.strftime('%Y-%m-%d')
            start_date = (analysis_date - timedelta(days=252)).strftime('%Y-%m-%d')
            
            historical_data = self.get_historical_data(symbol, start_date, end_date)
            
            if historical_data.empty or len(historical_data) < 30:
                return {'error': 'Insufficient historical data'}
            
            # Calculate technical signals
            technical_signals = self.calculate_simple_technical_signals(historical_data)
            
            # Simulate sentiment (neutral for historical periods)
            sentiment_score = 0.0
            sentiment_confidence = 0.5
            
            # Combine signals for prediction
            if technical_signals['signal'] == 'buy' and technical_signals['confidence'] > 0.5:
                prediction = 'bullish'
                confidence = technical_signals['confidence']
            elif technical_signals['signal'] == 'sell' and technical_signals['confidence'] > 0.5:
                prediction = 'bearish'
                confidence = technical_signals['confidence']
            else:
                prediction = 'neutral'
                confidence = 0.5
            
            return {
                'symbol': symbol,
                'analysis_date': analysis_date.isoformat(),
                'prediction': prediction,
                'confidence': confidence,
                'technical_signals': technical_signals,
                'sentiment_score': sentiment_score,
                'sentiment_confidence': sentiment_confidence,
                'data_points': len(historical_data)
            }
            
        except Exception as e:
            logger.error(f"Error in simulated analysis: {str(e)}")
            return {'error': str(e)}
    
    def calculate_actual_returns(self, symbol: str, entry_date: datetime, periods: List[int] = [1, 5, 20]) -> Dict:
        """Calculate actual returns after entry date"""
        try:
            start_date = entry_date.strftime('%Y-%m-%d')
            end_date = (entry_date + timedelta(days=max(periods) + 10)).strftime('%Y-%m-%d')
            
            price_data = self.get_historical_data(symbol, start_date, end_date)
            
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
            
            return returns
            
        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
            return {'error': str(e)}
    
    def run_simplified_backtest(self, symbols: List[str], start_date: str, end_date: str, frequency_days: int = 7) -> List[Dict]:
        """Run simplified backtest"""
        results = []
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        for symbol in symbols:
            logger.info(f"Backtesting {symbol}...")
            
            current_date = start_dt
            while current_date <= end_dt:
                # Run analysis simulation
                analysis_result = self.simulate_analysis_at_date(symbol, current_date)
                
                if 'error' not in analysis_result:
                    # Calculate actual returns
                    actual_returns = self.calculate_actual_returns(current_date, [1, 5, 20])
                    
                    # Check prediction accuracy
                    predicted_direction = analysis_result['prediction']
                    
                    accuracy_1d = None
                    accuracy_5d = None
                    accuracy_20d = None
                    
                    if 'return_1d' in actual_returns and actual_returns['return_1d'] is not None:
                        actual_1d = actual_returns['return_1d']
                        if predicted_direction == 'bullish':
                            accuracy_1d = actual_1d > 0
                        elif predicted_direction == 'bearish':
                            accuracy_1d = actual_1d < 0
                        else:
                            accuracy_1d = abs(actual_1d) < 0.02
                    
                    if 'return_5d' in actual_returns and actual_returns['return_5d'] is not None:
                        actual_5d = actual_returns['return_5d']
                        if predicted_direction == 'bullish':
                            accuracy_5d = actual_5d > 0
                        elif predicted_direction == 'bearish':
                            accuracy_5d = actual_5d < 0
                        else:
                            accuracy_5d = abs(actual_5d) < 0.02
                    
                    result = {
                        'date': current_date.strftime('%Y-%m-%d'),
                        'symbol': symbol,
                        'predicted_direction': predicted_direction,
                        'predicted_confidence': analysis_result['confidence'],
                        'actual_return_1d': actual_returns.get('return_1d'),
                        'actual_return_5d': actual_returns.get('return_5d'),
                        'actual_return_20d': actual_returns.get('return_20d'),
                        'prediction_accuracy_1d': accuracy_1d,
                        'prediction_accuracy_5d': accuracy_5d,
                        'data_points': analysis_result['data_points']
                    }
                    
                    results.append(result)
                
                current_date += timedelta(days=frequency_days)
        
        return results
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze backtest results"""
        if not results:
            return {'error': 'No results to analyze'}
        
        # Filter valid results
        valid_results = [r for r in results if r.get('prediction_accuracy_1d') is not None]
        
        if not valid_results:
            return {'error': 'No valid results for analysis'}
        
        # Calculate accuracy metrics
        accuracy_1d = np.mean([r['prediction_accuracy_1d'] for r in valid_results])
        accuracy_5d = np.mean([r['prediction_accuracy_5d'] for r in valid_results if r.get('prediction_accuracy_5d') is not None])
        
        # Calculate returns
        returns_1d = [r['actual_return_1d'] for r in valid_results if r.get('actual_return_1d') is not None]
        returns_5d = [r['actual_return_5d'] for r in valid_results if r.get('actual_return_5d') is not None]
        
        analysis = {
            'total_tests': len(results),
            'valid_tests': len(valid_results),
            'prediction_accuracy': {
                '1_day': accuracy_1d,
                '5_day': accuracy_5d if not np.isnan(accuracy_5d) else 0
            },
            'returns': {
                'avg_1d': np.mean(returns_1d) if returns_1d else 0,
                'avg_5d': np.mean(returns_5d) if returns_5d else 0,
                'std_1d': np.std(returns_1d) if returns_1d else 0,
                'std_5d': np.std(returns_5d) if returns_5d else 0
            },
            'symbol_breakdown': {}
        }
        
        # Symbol-wise analysis
        for symbol in set(r['symbol'] for r in valid_results):
            symbol_results = [r for r in valid_results if r['symbol'] == symbol]
            symbol_accuracy = np.mean([r['prediction_accuracy_1d'] for r in symbol_results])
            symbol_returns = [r['actual_return_1d'] for r in symbol_results if r.get('actual_return_1d') is not None]
            
            analysis['symbol_breakdown'][symbol] = {
                'tests': len(symbol_results),
                'accuracy': symbol_accuracy,
                'avg_return': np.mean(symbol_returns) if symbol_returns else 0
            }
        
        return analysis

def main():
    print("🚀 Enhanced Backtesting Demo (Simplified)")
    print("=" * 60)
    
    # Initialize backtester
    backtester = SimplifiedBacktester()
    
    # Define backtest parameters
    settings = Settings()
    symbols = settings.BANK_SYMBOLS[:2]  # Use first two symbols for demo
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    frequency_days = 14  # Test every 2 weeks
    
    print(f"📊 Backtest Parameters:")
    print(f"   Symbols: {symbols}")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   Frequency: Every {frequency_days} days")
    
    # Run backtest
    print(f"\n🔍 Running Enhanced Backtest...")
    results = backtester.run_simplified_backtest(symbols, start_date, end_date, frequency_days)
    
    if not results:
        print("❌ No results generated. Check data availability.")
        return
    
    # Analyze results
    print(f"\n📈 Analyzing Results...")
    analysis = backtester.analyze_results(results)
    
    if 'error' in analysis:
        print(f"❌ Analysis error: {analysis['error']}")
        return
    
    # Display results
    print(f"\n🎯 BACKTEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total tests: {analysis['total_tests']}")
    print(f"Valid tests: {analysis['valid_tests']}")
    print(f"Prediction accuracy (1-day): {analysis['prediction_accuracy']['1_day']:.1%}")
    print(f"Prediction accuracy (5-day): {analysis['prediction_accuracy']['5_day']:.1%}")
    print(f"Average 1-day return: {analysis['returns']['avg_1d']:.2%}")
    print(f"Average 5-day return: {analysis['returns']['avg_5d']:.2%}")
    print(f"Return volatility (1-day): {analysis['returns']['std_1d']:.2%}")
    
    # Symbol breakdown
    print(f"\n📊 Symbol Performance:")
    for symbol, stats in analysis['symbol_breakdown'].items():
        print(f"   {symbol}: {stats['tests']} tests, {stats['accuracy']:.1%} accuracy, {stats['avg_return']:.2%} avg return")
    
    print(f"\n💡 Key Insights:")
    print(f"   - Technical analysis shows {analysis['prediction_accuracy']['1_day']:.1%} directional accuracy")
    print(f"   - Average returns are {analysis['returns']['avg_1d']:.2%} per day")
    print(f"   - System processed {analysis['valid_tests']} valid trading scenarios")
    
    print(f"\n🔧 System Capabilities Demonstrated:")
    print(f"   ✅ Historical data retrieval and processing")
    print(f"   ✅ Technical analysis signal generation")
    print(f"   ✅ Prediction accuracy calculation")
    print(f"   ✅ Return and risk analysis")
    print(f"   ✅ Multi-symbol backtesting")
    
    print(f"\n🎯 For Enhanced Backtesting:")
    print(f"   📈 Full system: python archive/legacy_root_files/backtesting_system.py")
    print(f"   📊 ML backtesting: python demo_ml_backtesting.py")
    print(f"   🤖 Current analysis: python news_trading_analyzer.py --technical")
    
    print(f"\n============================================================")
    print(f"✅ Enhanced Backtesting Demo Complete!")

if __name__ == "__main__":
    main()
