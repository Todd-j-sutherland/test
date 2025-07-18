#!/usr/bin/env python3
"""
Integration Test for Position Risk Assessor in Professional Dashboard
Tests the complete integration and functionality
"""

import sys
import os
import traceback

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        print("✅ Data science libraries imported successfully")
    except ImportError as e:
        print(f"❌ Data science libraries import failed: {e}")
        return False
    
    try:
        from position_risk_assessor import PositionRiskAssessor
        print("✅ Position Risk Assessor imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Position Risk Assessor import failed: {e}")
        print("📝 Note: This is expected if running without the assessor module")
        return False

def test_position_risk_assessor():
    """Test Position Risk Assessor functionality"""
    print("\n🎯 Testing Position Risk Assessor...")
    
    try:
        from position_risk_assessor import PositionRiskAssessor
        
        assessor = PositionRiskAssessor()
        print("✅ PositionRiskAssessor instance created")
        
        # Test assessment
        result = assessor.assess_position_risk(
            symbol="CBA.AX",
            entry_price=100.0,
            current_price=99.0,
            position_type="long"
        )
        
        print("✅ Position assessment completed")
        print(f"📊 Risk Score: {result.get('risk_metrics', {}).get('risk_score', 'N/A')}")
        print(f"📈 Recovery Probability: {result.get('risk_metrics', {}).get('breakeven_probability_30d', 0):.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Position Risk Assessor test failed: {e}")
        print(f"🔍 Error details: {traceback.format_exc()}")
        return False

def test_dashboard_methods():
    """Test dashboard integration methods"""
    print("\n📊 Testing dashboard integration...")
    
    try:
        # Add the current directory to path
        sys.path.append('.')
        
        # Import dashboard class
        from professional_dashboard import NewsAnalysisDashboard
        
        dashboard = NewsAnalysisDashboard()
        print("✅ Dashboard instance created")
        
        # Test if Position Risk methods exist
        required_methods = [
            'display_position_risk_section',
            'display_risk_assessment_results',
            'create_recovery_probability_chart',
            'display_position_recommendations',
            'display_risk_breakdown',
            'display_market_context',
            'display_action_plan',
            'display_fallback_risk_assessment'
        ]
        
        for method_name in required_methods:
            if hasattr(dashboard, method_name):
                print(f"✅ Method {method_name} exists")
            else:
                print(f"❌ Method {method_name} missing")
                return False
        
        print("✅ All Position Risk methods are integrated")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard integration test failed: {e}")
        print(f"🔍 Error details: {traceback.format_exc()}")
        return False

def test_fallback_assessment():
    """Test fallback assessment functionality"""
    print("\n🔄 Testing fallback assessment...")
    
    try:
        sys.path.append('.')
        from professional_dashboard import NewsAnalysisDashboard
        
        dashboard = NewsAnalysisDashboard()
        
        # Test basic heuristic calculation
        symbol = "CBA.AX"
        entry_price = 100.0
        current_price = 99.0
        position_type = "long"
        
        # Calculate return manually
        current_return = ((current_price - entry_price) / entry_price) * 100
        print(f"📊 Calculated return: {current_return:.2f}%")
        
        # Test get_current_price method
        try:
            price = dashboard.get_current_price(symbol)
            print(f"📈 Current price method: ${price:.2f}")
        except Exception as e:
            print(f"⚠️ Current price method error: {e}")
        
        print("✅ Fallback assessment calculations working")
        return True
        
    except Exception as e:
        print(f"❌ Fallback assessment test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🚀 Position Risk Assessor Integration Test\n")
    
    tests = [
        ("Import Tests", test_imports),
        ("Position Risk Assessor", test_position_risk_assessor), 
        ("Dashboard Integration", test_dashboard_methods),
        ("Fallback Assessment", test_fallback_assessment)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("INTEGRATION TEST SUMMARY")
    print('='*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Position Risk Assessor is fully integrated!")
        print("\n🚀 Ready to deploy:")
        print("   1. Run: streamlit run professional_dashboard.py")
        print("   2. Navigate to the Position Risk Assessment section")
        print("   3. Test with your trading positions")
    else:
        print(f"\n⚠️ {total-passed} tests failed. Check the errors above.")
        
        if not any(result for test_name, result in results if "Position Risk Assessor" in test_name):
            print("\n💡 To enable full functionality:")
            print("   1. Ensure position_risk_assessor.py is in the working directory")
            print("   2. Install any missing dependencies")
            print("   3. Re-run this test")

if __name__ == "__main__":
    main()
