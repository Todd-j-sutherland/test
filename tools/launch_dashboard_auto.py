#!/usr/bin/env python3
"""
Automated launcher for the ASX Bank News Analysis Dashboard
This script bypasses the Streamlit email prompt automatically
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'yfinance', 'plotly', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def kill_existing_streamlit():
    """Kill any existing Streamlit processes"""
    try:
        subprocess.run(['pkill', '-f', 'streamlit'], check=False)
        time.sleep(2)
    except:
        pass

def launch_dashboard():
    """Launch the dashboard with automatic email prompt bypass"""
    print("🚀 Launching ASX Bank News Analysis Dashboard")
    print("=" * 50)
    
    # Check if we're in the right directory
    dashboard_path = '../core/news_analysis_dashboard.py'
    if not os.path.exists(dashboard_path):
        # Try current directory (in case run from root)
        dashboard_path = 'core/news_analysis_dashboard.py'
        if not os.path.exists(dashboard_path):
            print("❌ Dashboard file not found. Please run from the trading_analysis directory.")
            return False
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Kill any existing Streamlit processes
    kill_existing_streamlit()
    
    # Set environment variables to bypass Streamlit config
    env = os.environ.copy()
    env['STREAMLIT_SERVER_HEADLESS'] = 'true'
    env['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    env['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    
    print("📊 Starting dashboard server...")
    print("🌐 Dashboard will be available at: http://localhost:8501")
    print("⏳ Please wait for the server to start...")
    print("\n" + "="*50)
    
    try:
        # Start Streamlit with automatic email bypass
        process = subprocess.Popen(
            [sys.executable, '-m', 'streamlit', 'run', dashboard_path, '--server.headless', 'true'],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Send empty line to bypass email prompt
        process.stdin.write('\n')
        process.stdin.flush()
        
        # Monitor output
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
            if 'Local URL:' in line:
                print("\n✅ Dashboard is ready!")
                print("🌐 Open your browser to: http://localhost:8501")
                break
        
        # Wait for process to finish
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user")
        process.terminate()
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = launch_dashboard()
    if not success:
        sys.exit(1)
