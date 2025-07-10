#!/usr/bin/env python3
"""
ML Backtesting Demo - Historical Trading Analysis
Shows how to use the ML backtesting system to evaluate historical performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ml_backtester import MLBacktester
from src.ml_training_pipeline import MLTrainingPipeline
from src.data_feed import ASXDataFeed
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("🚀 ML Backtesting Demo - Historical Trading Analysis")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing ML Backtesting System...")
    ml_pipeline = MLTrainingPipeline()
    data_feed = ASXDataFeed()
    backtester = MLBacktester(ml_pipeline, data_feed)
    print("✅ ML Backtesting System initialized")
    
    # Check available historical data
    print("\n2. Checking Historical Data Availability...")
    import sqlite3
    conn = sqlite3.connect(ml_pipeline.db_path)
    cursor = conn.cursor()
    
    # Get date range of available data
    cursor.execute("""
        SELECT 
            MIN(timestamp) as earliest_date,
            MAX(timestamp) as latest_date,
            COUNT(*) as total_records,
            COUNT(DISTINCT symbol) as unique_symbols
        FROM sentiment_features
    """)
    
    result = cursor.fetchone()
    if result and result[0]:
        earliest_date, latest_date, total_records, unique_symbols = result
        print(f"   📊 Historical Data Available:")
        print(f"   - Date Range: {earliest_date} to {latest_date}")
        print(f"   - Total Records: {total_records}")
        print(f"   - Unique Symbols: {unique_symbols}")
        
        # Show symbols with data
        cursor.execute("""
            SELECT symbol, COUNT(*) as count
            FROM sentiment_features
            GROUP BY symbol
            ORDER BY count DESC
        """)
        symbols_data = cursor.fetchall()
        print(f"   - Symbols with data: {dict(symbols_data)}")
        
    else:
        print("   ⚠️  No historical sentiment data available")
        print("   💡 Run the analyzer regularly to collect data:")
        print("      python news_trading_analyzer.py --symbols CBA.AX,WBC.AX")
        conn.close()
        return
    
    conn.close()
    
    # Run backtesting demo
    print("\n3. Running ML Backtesting Demo...")
    
    # Use the symbol with most data
    if symbols_data:
        symbol = symbols_data[0][0]  # Symbol with most data
        print(f"   📈 Testing symbol: {symbol}")
        
        # Define backtest period (last 30 days of available data)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"   📅 Backtest Period: {start_date} to {end_date}")
        
        # Run backtest
        try:
            backtest_results = backtester.backtest_predictions(symbol, start_date, end_date)
            
            if 'error' in backtest_results:
                print(f"   ⚠️  Backtest completed with limitation: {backtest_results['error']}")
                print("   💡 This is normal - need more historical data for comprehensive backtesting")
            else:
                print(f"   ✅ Backtest completed successfully!")
                
                # Display results
                metrics = backtest_results['metrics']
                print(f"\n   📊 Backtest Results:")
                print(f"   - Total Trades: {metrics['total_trades']}")
                print(f"   - Win Rate: {metrics['win_rate']:.1%}")
                print(f"   - Avg Return per Trade: {metrics['avg_return_per_trade']:.2%}")
                print(f"   - Best Trade: {metrics['best_trade']:.2%}")
                print(f"   - Worst Trade: {metrics['worst_trade']:.2%}")
                print(f"   - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"   - Max Drawdown: {metrics['max_drawdown']:.2%}")
                print(f"   - Total Return: {backtest_results['total_return']:.2%}")
                
                # Show individual trades
                if backtest_results['trades']:
                    print(f"\n   🔍 Individual Trades:")
                    for i, trade in enumerate(backtest_results['trades'][:5]):  # Show first 5
                        print(f"   {i+1}. {trade['entry_date']} → {trade['exit_date']}: {trade['return']:.2%}")
                    
                    if len(backtest_results['trades']) > 5:
                        print(f"   ... and {len(backtest_results['trades']) - 5} more trades")
                
        except Exception as e:
            print(f"   ❌ Error during backtesting: {e}")
            print("   💡 This might be due to insufficient historical data")
    
    # Show system capabilities
    print("\n4. ML Backtesting System Capabilities:")
    print("   🎯 Historical Performance Analysis")
    print("   - Tests ML predictions against actual price movements")
    print("   - Calculates win rate, returns, and risk metrics")
    print("   - Identifies best and worst performing predictions")
    print("   - Provides Sharpe ratio and drawdown analysis")
    
    print("\n   📊 Advanced Features Available:")
    print("   - Multi-timeframe analysis (1h, 4h, daily bars)")
    print("   - Feature importance analysis")
    print("   - Model performance over time")
    print("   - Risk-adjusted returns")
    print("   - Portfolio-level backtesting")
    
    print("\n   🔮 Enhanced Backtesting (archive/legacy_root_files/backtesting_system.py):")
    print("   - Comprehensive historical analysis")
    print("   - Technical analysis backtesting")
    print("   - Advanced risk management")
    print("   - HTML report generation")
    print("   - Legacy vs Enhanced system comparison")
    
    # Show how to improve backtesting
    print("\n5. How to Improve Backtesting Results:")
    print("   📈 Collect More Data:")
    print("      python news_trading_analyzer.py --all  # Run daily")
    print("   🤖 Train Better Models:")
    print("      python scripts/retrain_ml_models.py --min-samples 100")
    print("   📊 Run Comprehensive Backtests:")
    print("      python archive/legacy_root_files/backtesting_system.py")
    
    print("\n   💡 For Production Use:")
    print("   - Integrate historical news feeds for realistic sentiment")
    print("   - Use multiple timeframes for entry/exit timing")
    print("   - Implement proper risk management")
    print("   - Consider transaction costs and slippage")
    print("   - Use walk-forward analysis for model validation")
    
    print("\n============================================================")
    print("✅ ML Backtesting Demo Complete!")
    print("🎯 Your system is ready for historical analysis and ML validation!")

if __name__ == "__main__":
    main()
