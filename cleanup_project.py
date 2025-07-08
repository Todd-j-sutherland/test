#!/usr/bin/env python3
"""
Project Cleanup Script
Organizes files and removes redundancies for news-focused trading analysis
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """Clean up project structure for news trading analysis focus"""
    
    print("🧹 Starting project cleanup...")
    
    # Create archive directory for moved files
    archive_dir = Path("archive")
    archive_dir.mkdir(exist_ok=True)
    
    # Files to move to archive (redundant or non-essential for news analysis)
    files_to_archive = [
        # Redundant main files
        "main.py",
        "enhanced_main.py", 
        "simple_news_analysis.py",
        "simple_enhanced_demo.py",
        "demo_improvements.py",
        "test_improvements.py",
        "trading_dashboard.py",
        "simple_dashboard.py",
        "run_trading_system.py",
        
        # Cleanup/maintenance scripts
        "perform_cleanup.py",
        
        # Shell scripts (keep but move)
        "run.sh",
        "run.bat",
        
        # Test files (move to tests directory)
        "test_ml_integration.py",
        "test_transformer_sentiment.py",
    ]
    
    # Src files to archive (trading infrastructure not needed for news analysis)
    src_files_to_archive = [
        "src/async_main.py",
        "src/trading_orchestrator.py", 
        "src/advanced_risk_manager.py",
        "src/data_validator.py",
        "src/fundamental_analysis.py",
        "src/technical_analysis.py",
        "src/market_predictor.py",
        "src/alert_system.py",
        "src/report_generator.py",
        "src/risk_calculator.py",
    ]
    
    # Essential files for news analysis (keep in src/)
    essential_src_files = [
        "src/news_sentiment.py",           # Core news sentiment analysis
        "src/ml_trading_config.py",       # ML features for trading
        "src/news_impact_analyzer.py",    # News impact analysis
        "src/sentiment_history.py",       # Sentiment tracking
        "src/data_feed.py",              # Basic data fetching
        "src/__init__.py"                 # Package marker
    ]
    
    moved_count = 0
    
    # Move redundant root files
    for file_path in files_to_archive:
        if os.path.exists(file_path):
            try:
                dest = archive_dir / file_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(file_path, dest)
                print(f"📦 Moved {file_path} to archive/")
                moved_count += 1
            except Exception as e:
                print(f"❌ Error moving {file_path}: {e}")
    
    # Move non-essential src files
    for file_path in src_files_to_archive:
        if os.path.exists(file_path):
            try:
                dest = archive_dir / file_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(file_path, dest)
                print(f"📦 Moved {file_path} to archive/")
                moved_count += 1
            except Exception as e:
                print(f"❌ Error moving {file_path}: {e}")
    
    # Create essential directories
    essential_dirs = ["logs", "reports", "data/cache", "tests"]
    for dir_path in essential_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Ensured directory exists: {dir_path}")
    
    # Move test files to tests directory
    test_files = ["test_ml_integration.py", "test_transformer_sentiment.py"]
    for test_file in test_files:
        if os.path.exists(test_file):
            dest = Path("tests") / test_file
            try:
                shutil.move(test_file, dest)
                print(f"🧪 Moved {test_file} to tests/")
                moved_count += 1
            except Exception as e:
                print(f"❌ Error moving {test_file}: {e}")
    
    print(f"\n✅ Cleanup complete! Moved {moved_count} files to archive.")
    print("\n📁 Current project structure:")
    print("├── news_trading_analyzer.py  (🎯 MAIN ENTRY POINT)")
    print("├── src/")
    print("│   ├── news_sentiment.py")
    print("│   ├── ml_trading_config.py") 
    print("│   ├── news_impact_analyzer.py")
    print("│   ├── sentiment_history.py")
    print("│   └── data_feed.py")
    print("├── config/")
    print("├── utils/")
    print("├── tests/")
    print("├── archive/  (📦 moved files)")
    print("├── reports/  (📊 output)")
    print("└── logs/     (📝 logging)")
    
    print("\n🎯 Focus: News sentiment analysis for trading decisions")
    print("🚀 Use: python news_trading_analyzer.py --help")

def verify_essential_files():
    """Verify all essential files are present"""
    
    essential_files = [
        "news_trading_analyzer.py",
        "src/news_sentiment.py", 
        "src/ml_trading_config.py",
        "config/settings.py",
        "utils/cache_manager.py"
    ]
    
    missing_files = []
    for file_path in essential_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing essential files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease ensure these files exist before running the analyzer.")
        return False
    else:
        print("\n✅ All essential files present")
        return True

if __name__ == "__main__":
    print("Project Cleanup for News Trading Analysis")
    print("=" * 50)
    
    # Ask for confirmation
    response = input("This will move files to archive/ directory. Continue? (y/N): ")
    if response.lower() in ['y', 'yes']:
        cleanup_project()
        verify_essential_files()
        
        print("\n" + "=" * 50)
        print("🎉 Project is now organized for news trading analysis!")
        print("🎯 Primary focus: Analyze news sources for trading sentiment")
        print("🚀 Get started: python news_trading_analyzer.py --all")
    else:
        print("Cleanup cancelled.")
