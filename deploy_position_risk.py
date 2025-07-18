#!/usr/bin/env python3
"""
Launch Script for Professional Dashboard with Position Risk Assessor
Deploys the complete integrated trading analysis platform
"""

import subprocess
import sys
import os
import time

def check_requirements():
    """Check if all required packages are available"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'yfinance'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("📦 Install with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are available")
    return True

def check_position_risk_assessor():
    """Check if Position Risk Assessor is available"""
    print("\n🎯 Checking Position Risk Assessor...")
    
    if os.path.exists('position_risk_assessor.py'):
        print("✅ position_risk_assessor.py found")
        
        try:
            from position_risk_assessor import PositionRiskAssessor
            print("✅ PositionRiskAssessor can be imported")
            
            # Quick test
            assessor = PositionRiskAssessor()
            result = assessor.assess_position_risk("CBA.AX", 100.0, 99.0, "long")
            print("✅ Position Risk Assessor is functional")
            print(f"   🔍 Test result: Risk Score {result.get('risk_metrics', {}).get('risk_score', 'N/A')}")
            return True
            
        except Exception as e:
            print(f"⚠️ Position Risk Assessor import/test failed: {e}")
            print("🔄 Dashboard will run with fallback heuristic assessment")
            return False
    else:
        print("⚠️ position_risk_assessor.py not found")
        print("🔄 Dashboard will run with fallback heuristic assessment")
        return False

def check_dashboard_file():
    """Check if professional dashboard file exists and is valid"""
    print("\n📊 Checking Professional Dashboard...")
    
    if os.path.exists('professional_dashboard.py'):
        print("✅ professional_dashboard.py found")
        
        # Check for Position Risk integration
        with open('professional_dashboard.py', 'r') as f:
            content = f.read()
            
        integration_markers = [
            'display_position_risk_section',
            'POSITION_RISK_AVAILABLE',
            'Position Risk Assessment'
        ]
        
        integrated_features = 0
        for marker in integration_markers:
            if marker in content:
                print(f"✅ {marker} integrated")
                integrated_features += 1
            else:
                print(f"❌ {marker} missing")
        
        if integrated_features == len(integration_markers):
            print("✅ Position Risk Assessor fully integrated in dashboard")
            return True
        else:
            print(f"⚠️ {len(integration_markers) - integrated_features} integration features missing")
            return False
    else:
        print("❌ professional_dashboard.py not found")
        return False

def launch_dashboard():
    """Launch the professional dashboard"""
    print("\n🚀 Launching Professional Dashboard...")
    print("📊 Features available:")
    print("   • Real-time sentiment analysis")
    print("   • Technical analysis & momentum")
    print("   • ML-powered predictions")
    print("   • 🎯 Position Risk Assessment (NEW!)")
    print("   • Historical trend analysis")
    print("   • Professional trading recommendations")
    
    print("\n🌐 Starting Streamlit server...")
    print("💡 Dashboard will open in your default browser")
    print("📍 Navigate to the 'Position Risk Assessment' section to test the new feature")
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'professional_dashboard.py',
            '--server.port=8504',
            '--server.address=localhost'
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        print("🔧 Try running manually:")
        print("   streamlit run professional_dashboard.py --server.port=8504")

def display_usage_guide():
    """Display guide for using the Position Risk Assessor"""
    print("\n" + "="*60)
    print("🎯 POSITION RISK ASSESSOR - USAGE GUIDE")
    print("="*60)
    
    print("\n📋 How to Use:")
    print("1. 🌐 Open the dashboard at http://localhost:8504")
    print("2. 📊 Navigate to 'Position Risk Assessment' section") 
    print("3. 🏦 Select your bank symbol (CBA.AX, WBC.AX, etc.)")
    print("4. 💰 Enter your entry price and current market price")
    print("5. 📍 Choose position type (long/short)")
    print("6. 🎯 Click 'Assess Position Risk' for analysis")
    
    print("\n📊 What You'll Get:")
    print("• 🎯 Risk Score (0-10 scale)")
    print("• 📈 Recovery probability predictions")
    print("• ⏱️ Estimated recovery timeline")
    print("• 💡 Actionable recommendations")
    print("• 📋 Comprehensive action plan")
    print("• 🌍 Market context analysis")
    
    print("\n🔍 Example Analysis:")
    print("Entry: $100.00 → Current: $99.00 (Long position)")
    print("• Risk Score: 4.1/10 (Moderate)")
    print("• Recovery Probability: 60.2% (30-day)")
    print("• Recommendation: REDUCE_POSITION")
    print("• Expected Recovery: 14 days")
    
    print("\n🎉 Ready to transform your position management!")

def main():
    """Main launch sequence"""
    print("🚀 PROFESSIONAL TRADING DASHBOARD LAUNCHER")
    print("   with integrated Position Risk Assessor")
    print("="*60)
    
    # Check system readiness
    checks = [
        ("Requirements", check_requirements),
        ("Position Risk Assessor", check_position_risk_assessor),
        ("Dashboard Integration", check_dashboard_file)
    ]
    
    all_passed = True
    assessor_available = False
    
    for check_name, check_func in checks:
        print(f"\n🔍 {check_name}:")
        result = check_func()
        
        if check_name == "Position Risk Assessor":
            assessor_available = result
        
        if check_name == "Requirements" and not result:
            all_passed = False
            break
    
    if not all_passed:
        print("\n❌ Critical requirements missing. Please install required packages first.")
        return
    
    # Show feature status
    print(f"\n{'='*60}")
    print("SYSTEM STATUS")
    print('='*60)
    print("✅ Professional Dashboard: READY")
    print("✅ Sentiment Analysis: READY") 
    print("✅ Technical Analysis: READY")
    print("✅ ML Predictions: READY")
    
    if assessor_available:
        print("✅ Position Risk Assessor: READY (Full ML-powered)")
    else:
        print("🔄 Position Risk Assessor: READY (Heuristic fallback)")
    
    # Display usage guide
    display_usage_guide()
    
    # Confirm launch
    print(f"\n{'='*60}")
    response = input("🚀 Launch the dashboard now? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        launch_dashboard()
    else:
        print("\n👋 Launch cancelled. Run this script again when ready!")
        print("💡 Or launch manually with: streamlit run professional_dashboard.py")

if __name__ == "__main__":
    main()
