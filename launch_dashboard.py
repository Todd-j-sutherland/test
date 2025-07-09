#!/usr/bin/env python3
"""
Launch script for News Analysis Dashboard with Technical Analysis
Ensures proper environment activation and dependency installation
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_virtual_env():
    """Check if we're in a virtual environment"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['yfinance', 'streamlit', 'plotly', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(missing_packages):
    """Install missing dependencies"""
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)

def test_technical_analysis():
    """Test technical analysis functionality"""
    try:
        from src.technical_analysis import TechnicalAnalyzer, get_market_data
        
        print("🔍 Testing technical analysis...")
        analyzer = TechnicalAnalyzer()
        
        # Quick test with CBA
        data = get_market_data('CBA.AX', period='1mo')
        if not data.empty:
            analysis = analyzer.analyze('CBA.AX', data)
            print(f"✅ Technical analysis working - CBA Price: ${analysis['current_price']:.2f}")
            return True
        else:
            print("⚠️ No market data available")
            return False
    except Exception as e:
        print(f"❌ Technical analysis test failed: {e}")
        return False

def run_dashboard():
    """Run the news analysis dashboard"""
    try:
        import streamlit.web.cli as stcli
        import sys
        
        # Set up the Streamlit command
        sys.argv = ["streamlit", "run", "news_analysis_dashboard.py", "--server.port=8501"]
        stcli.main()
    except ImportError:
        print("❌ Streamlit not available. Installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'streamlit'], check=True)
        run_dashboard()
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        print("\nTry running manually:")
        print("streamlit run news_analysis_dashboard.py")

def main():
    """Main launcher function"""
    print("🚀 ASX Bank News Analysis Dashboard Launcher")
    print("=" * 50)
    
    # Check current directory
    if not os.path.exists('news_analysis_dashboard.py'):
        print("❌ Please run this script from the trading_analysis directory")
        return
    
    # Check if we're in virtual environment
    if not check_virtual_env():
        print("⚠️ Not in virtual environment. Checking for .venv...")
        if os.path.exists('.venv'):
            print("📁 Found .venv directory. Please activate it first:")
            print("source .venv/bin/activate")
            return
        else:
            print("⚠️ No virtual environment found. Proceeding with system Python...")
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"📦 Installing missing packages: {', '.join(missing)}")
        try:
            install_dependencies(missing)
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("Please install manually:")
            print(f"pip install {' '.join(missing)}")
            return
    else:
        print("✅ All dependencies are installed")
    
    # Test technical analysis
    print("\n🔧 Testing technical analysis...")
    if test_technical_analysis():
        print("✅ Technical analysis is working")
    else:
        print("⚠️ Technical analysis may not work properly")
    
    # Launch dashboard
    print("\n🌐 Launching dashboard...")
    print("📊 Dashboard will open at: http://localhost:8501")
    print("⏹️ Press Ctrl+C to stop the dashboard")
    
    try:
        run_dashboard()
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nAlternative launch command:")
        print("streamlit run news_analysis_dashboard.py")

if __name__ == "__main__":
    main()
