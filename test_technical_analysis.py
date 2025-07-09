#!/usr/bin/env python3
"""
Test script for technical analysis integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.technical_analysis import TechnicalAnalyzer, get_market_data

def test_technical_analysis():
    """Test the technical analysis functionality"""
    print("🔍 Testing Technical Analysis Module")
    print("=" * 50)
    
    analyzer = TechnicalAnalyzer()
    
    # Test with Australian bank symbols
    test_symbols = ['CBA.AX', 'WBC.AX', 'ANZ.AX']
    
    for symbol in test_symbols:
        print(f"\n📈 Analyzing {symbol}...")
        
        try:
            # Get market data
            data = get_market_data(symbol, period='3mo')
            
            if data.empty:
                print(f"❌ No data available for {symbol}")
                continue
            
            # Perform technical analysis
            analysis = analyzer.analyze(symbol, data)
            
            # Display results
            print(f"✅ Analysis completed for {symbol}")
            print(f"   Current Price: ${analysis['current_price']:.2f}")
            print(f"   RSI: {analysis['indicators'].get('rsi', 0):.2f}")
            print(f"   Momentum Score: {analysis['momentum']['score']:.2f}")
            print(f"   Momentum Strength: {analysis['momentum']['strength']}")
            print(f"   Trend: {analysis['trend']['direction']}")
            print(f"   Overall Signal: {analysis['overall_signal']:.2f}")
            print(f"   Recommendation: {analysis['recommendation']}")
            
            # Show some momentum details
            momentum = analysis['momentum']
            print(f"   📊 Momentum Details:")
            print(f"      1-Day Change: {momentum['price_change_1d']:.2f}%")
            print(f"      5-Day Change: {momentum['price_change_5d']:.2f}%")
            print(f"      Volume Momentum: {momentum['volume_momentum']}")
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {str(e)}")
    
    print("\n🎯 Technical Analysis Test Complete!")
    print("\nNote: Make sure you have 'yfinance' installed:")
    print("pip install yfinance")
    print("\nTo run the full dashboard with technical analysis:")
    print("cd /Users/toddsutherland/Repos/trading_analysis && source .venv/bin/activate && python launch_dashboard_auto.py")

if __name__ == "__main__":
    test_technical_analysis()
