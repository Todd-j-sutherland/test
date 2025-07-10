#!/usr/bin/env python3
"""
Test script to verify technical analysis integration in dashboard
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.technical_analysis import TechnicalAnalyzer, get_market_data

def test_dashboard_integration():
    """Test the technical analysis integration for dashboard"""
    print("🔧 Testing Dashboard Technical Analysis Integration")
    print("=" * 60)
    
    analyzer = TechnicalAnalyzer()
    
    # Test with multiple symbols
    test_symbols = ['CBA.AX', 'WBC.AX', 'ANZ.AX']
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol} integration...")
    
    success_count = 0
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol} integration...")
        
        try:
            # Get market data
            data = get_market_data(symbol, period='3mo')
            
            if data.empty:
                print(f"❌ No data available for {symbol}")
                continue
            
            # Perform technical analysis
            analysis = analyzer.analyze(symbol, data)
            
            # Test dashboard component calculations
            if 'momentum' in analysis:
                momentum_data = analysis['momentum']
                
                # Convert momentum score to 0-100 scale (as done in dashboard)
                momentum_score = max(0, min(100, (momentum_data['score'] + 100) / 2))
                
                # Volume score based on volume momentum
                if momentum_data['volume_momentum'] == 'high':
                    volume_score = 75
                elif momentum_data['volume_momentum'] == 'normal':
                    volume_score = 50
                else:
                    volume_score = 25
                
                # ML Trading score based on overall signal
                ml_trading_score = max(0, min(100, (analysis['overall_signal'] + 100) / 2))
                
                print(f"✅ Dashboard Component Calculations:")
                print(f"   Raw Momentum Score: {momentum_data['score']:.2f}")
                print(f"   Dashboard Momentum Score: {momentum_score:.1f}")
                print(f"   Volume Momentum: {momentum_data['volume_momentum']}")
                print(f"   Dashboard Volume Score: {volume_score:.1f}")
                print(f"   Raw Overall Signal: {analysis['overall_signal']:.2f}")
                print(f"   Dashboard ML Trading Score: {ml_trading_score:.1f}")
                
                success_count += 1
                
            else:
                print(f"❌ Momentum data not found in analysis")
                
        except Exception as e:
            print(f"❌ Error in dashboard integration test: {str(e)}")
    
    return success_count > 0

if __name__ == "__main__":
    success = test_dashboard_integration()
    if success:
        print("\n🎉 Dashboard integration test passed!")
        print("The Volume, Momentum, and ML Trading components should now show non-zero values.")
    else:
        print("\n❌ Dashboard integration test failed!")
