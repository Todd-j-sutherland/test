#!/usr/bin/env python3
"""
Run Analysis and Generate Dashboard
Complete workflow to analyze banks and generate dashboard
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print("Output:", result.stdout[:500])  # First 500 chars
        else:
            print(f"❌ {description} failed")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False
    
    return True

def main():
    """Main workflow"""
    print("🏦 ASX Bank News Analysis & Dashboard Generation")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Run news analysis for all banks
    success = run_command(
        "python news_trading_analyzer.py --all",
        "Analyzing all banks"
    )
    
    if not success:
        print("\n⚠️ Analysis failed, but continuing with dashboard generation...")
    
    # Step 2: Generate HTML dashboard
    success = run_command(
        "python generate_dashboard.py",
        "Generating HTML dashboard"
    )
    
    if success:
        print("\n🎉 Complete workflow finished successfully!")
        print("\n📊 Your dashboard should now be open in your browser")
        print("📁 Dashboard file: news_analysis_dashboard.html")
    else:
        print("\n❌ Dashboard generation failed")
    
    # Step 3: Optionally run Streamlit dashboard
    print(f"\n🌐 Optional: Run interactive Streamlit dashboard:")
    print("   streamlit run news_analysis_dashboard.py")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
