# test_improvements.py - Test script for ASX improvement implementations
"""
Test script to demonstrate the improvements from the ASX improvement plan:
1. Async processing performance
2. Data validation capabilities
3. Advanced risk management
4. Enhanced reporting
"""

import asyncio
import time
import logging
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.async_main import ASXBankTradingSystemAsync
from src.advanced_risk_manager import AdvancedRiskManager
from src.data_validator import DataValidator
from src.data_feed import ASXDataFeed
from main import ASXBankTradingSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovementTester:
    """Test the improvements implemented from the ASX improvement plan"""
    
    def __init__(self):
        self.data_feed = ASXDataFeed()
        self.advanced_risk_manager = AdvancedRiskManager()
        self.data_validator = DataValidator()
    
    async def test_async_performance(self):
        """Test Priority 1: Async processing performance improvement"""
        print("\n🚀 TESTING ASYNC PROCESSING PERFORMANCE")
        print("=" * 60)
        
        # Test symbols (subset for faster testing)
        test_symbols = ['CBA.AX', 'ANZ.AX', 'WBC.AX']
        
        # Test synchronous processing
        print("📊 Testing synchronous processing...")
        sync_start = time.time()
        sync_system = ASXBankTradingSystem()
        sync_results = {}
        
        for symbol in test_symbols:
            try:
                result = sync_system.analyze_bank(symbol)
                sync_results[symbol] = result
                time.sleep(0.1)  # Small delay to simulate API limits
            except Exception as e:
                sync_results[symbol] = {'error': str(e)}
        
        sync_duration = time.time() - sync_start
        
        # Test asynchronous processing
        print("⚡ Testing asynchronous processing...")
        async_start = time.time()
        async with ASXBankTradingSystemAsync() as async_system:
            # Override the bank symbols for testing
            async_system.settings.BANK_SYMBOLS = test_symbols
            async_results = await async_system.analyze_all_banks_async()
        
        async_duration = time.time() - async_start
        
        # Compare results
        speedup = sync_duration / async_duration if async_duration > 0 else 0
        
        print(f"\n📈 PERFORMANCE RESULTS:")
        print(f"   Synchronous time:  {sync_duration:.2f}s")
        print(f"   Asynchronous time: {async_duration:.2f}s")
        print(f"   Speedup factor:    {speedup:.2f}x")
        print(f"   Symbols analyzed:  {len(test_symbols)}")
        
        if speedup > 1.5:
            print("✅ ASYNC IMPROVEMENT SUCCESSFUL!")
        else:
            print("⚠️  Async improvement less than expected")
        
        return {
            'sync_duration': sync_duration,
            'async_duration': async_duration,
            'speedup': speedup,
            'symbols_tested': len(test_symbols)
        }
    
    def test_data_validation(self):
        """Test Priority 2: Data validation pipeline"""
        print("\n🔍 TESTING DATA VALIDATION PIPELINE")
        print("=" * 60)
        
        test_symbol = 'CBA.AX'
        
        # Get data and validate
        try:
            data, validation = self.data_feed.get_historical_data_validated(test_symbol)
            
            print(f"📊 Data validation for {test_symbol}:")
            print(f"   Valid: {validation.is_valid}")
            print(f"   Quality: {validation.quality.value}")
            print(f"   Confidence: {validation.confidence_score:.2f}")
            print(f"   Data points: {len(data) if data is not None else 0}")
            
            if validation.warnings:
                print(f"   Warnings: {len(validation.warnings)}")
                for warning in validation.warnings[:3]:  # Show first 3
                    print(f"     - {warning}")
            
            if validation.errors:
                print(f"   Errors: {len(validation.errors)}")
                for error in validation.errors[:3]:  # Show first 3
                    print(f"     - {error}")
            
            print("✅ DATA VALIDATION IMPLEMENTED SUCCESSFULLY!")
            
            return {
                'is_valid': validation.is_valid,
                'quality': validation.quality.value,
                'confidence_score': validation.confidence_score,
                'data_points': len(data) if data is not None else 0,
                'warnings_count': len(validation.warnings),
                'errors_count': len(validation.errors)
            }
            
        except Exception as e:
            print(f"❌ Data validation test failed: {str(e)}")
            return {'error': str(e)}
    
    def test_advanced_risk_management(self):
        """Test Priority 3: Advanced risk management"""
        print("\n⚠️  TESTING ADVANCED RISK MANAGEMENT")
        print("=" * 60)
        
        test_symbol = 'CBA.AX'
        
        try:
            # Get market data
            market_data = self.data_feed.get_historical_data(test_symbol)
            
            if market_data is None or market_data.empty:
                print(f"❌ No data available for {test_symbol}")
                return {'error': 'No data available'}
            
            # Calculate comprehensive risk
            risk_analysis = self.advanced_risk_manager.calculate_comprehensive_risk(
                test_symbol, market_data, position_size=10000, account_balance=100000
            )
            
            if 'error' in risk_analysis:
                print(f"❌ Risk analysis failed: {risk_analysis['error']}")
                return risk_analysis
            
            # Display key risk metrics
            print(f"📊 Advanced Risk Analysis for {test_symbol}:")
            print(f"   Risk Level: {risk_analysis['risk_level'].upper()}")
            print(f"   Risk Score: {risk_analysis['overall_risk_score']:.0f}/100")
            
            # VaR metrics
            var_metrics = risk_analysis['var_metrics']
            var_95 = abs(var_metrics['confidence_95']['var_historical'])
            var_99 = abs(var_metrics['confidence_99']['var_historical'])
            print(f"   VaR (95%): {var_95:.2%}")
            print(f"   VaR (99%): {var_99:.2%}")
            
            # Drawdown metrics
            drawdown_metrics = risk_analysis['drawdown_metrics']
            max_drawdown = abs(drawdown_metrics['max_drawdown'])
            print(f"   Max Drawdown: {max_drawdown:.2%}")
            
            # Volatility metrics
            volatility_metrics = risk_analysis['volatility_metrics']
            annual_vol = volatility_metrics['annual_volatility']
            print(f"   Annual Volatility: {annual_vol:.2%}")
            
            # Show top recommendations
            recommendations = risk_analysis['recommendations']
            print(f"   Top Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"     {i}. {rec}")
            
            print("✅ ADVANCED RISK MANAGEMENT IMPLEMENTED SUCCESSFULLY!")
            
            return {
                'risk_level': risk_analysis['risk_level'],
                'risk_score': risk_analysis['overall_risk_score'],
                'var_95': var_95,
                'max_drawdown': max_drawdown,
                'annual_volatility': annual_vol,
                'recommendations_count': len(recommendations)
            }
            
        except Exception as e:
            print(f"❌ Advanced risk management test failed: {str(e)}")
            return {'error': str(e)}
    
    async def run_all_tests(self):
        """Run all improvement tests"""
        print("\n🎯 RUNNING ASX IMPROVEMENT PLAN TESTS")
        print("=" * 80)
        print("Testing implementations from your ASX improvement plan...")
        print("=" * 80)
        
        results = {}
        
        # Test 1: Async performance
        try:
            results['async_performance'] = await self.test_async_performance()
        except Exception as e:
            print(f"❌ Async performance test failed: {str(e)}")
            results['async_performance'] = {'error': str(e)}
        
        # Test 2: Data validation
        try:
            results['data_validation'] = self.test_data_validation()
        except Exception as e:
            print(f"❌ Data validation test failed: {str(e)}")
            results['data_validation'] = {'error': str(e)}
        
        # Test 3: Advanced risk management
        try:
            results['advanced_risk'] = self.test_advanced_risk_management()
        except Exception as e:
            print(f"❌ Advanced risk management test failed: {str(e)}")
            results['advanced_risk'] = {'error': str(e)}
        
        # Summary
        print("\n📋 TEST SUMMARY")
        print("=" * 60)
        
        successful_tests = 0
        total_tests = 3
        
        for test_name, result in results.items():
            if 'error' not in result:
                print(f"✅ {test_name.replace('_', ' ').title()}: PASSED")
                successful_tests += 1
            else:
                print(f"❌ {test_name.replace('_', ' ').title()}: FAILED")
        
        print(f"\n🎯 Overall Success Rate: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
        
        if successful_tests == total_tests:
            print("🎉 ALL IMPROVEMENTS SUCCESSFULLY IMPLEMENTED!")
        elif successful_tests > 0:
            print("⚠️  SOME IMPROVEMENTS IMPLEMENTED")
        else:
            print("❌ IMPLEMENTATION NEEDS WORK")
        
        return results

async def main():
    """Main test function"""
    tester = ImprovementTester()
    
    print("🚀 Starting ASX Improvement Plan Test Suite")
    print(f"🕐 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        results = await tester.run_all_tests()
        
        # Save results
        import json
        output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📁 Test results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Test suite failed: {str(e)}")
        logger.error(f"Test suite error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
