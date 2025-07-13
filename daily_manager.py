#!/usr/bin/env python3
"""
Daily Trading System Manager
Simplifies daily operations with one-command workflows
"""

import sys
import subprocess
import time
import os
from datetime import datetime

class TradingSystemManager:
    def __init__(self):
        self.base_dir = "/root/test"
        os.chdir(self.base_dir)
    
    def run_command(self, command, description=""):
        """Run a command and show status"""
        if description:
            print(f"\n🔄 {description}...")
        
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"✅ Success")
                if result.stdout.strip():
                    print(result.stdout.strip())
            else:
                print(f"❌ Error: {result.stderr.strip()}")
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout after 30 seconds")
            return False
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False
    
    def morning_routine(self):
        """Complete morning startup routine"""
        print("🌅 MORNING ROUTINE - Starting Trading System")
        print("=" * 50)
        
        # 1. System status check
        self.run_command("python tools/comprehensive_analyzer.py", "Checking system status")
        
        # 2. Start smart collector (background)
        print("\n🔄 Starting smart collector (background)...")
        subprocess.Popen(["python", "core/smart_collector.py"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        print("✅ Smart collector started")
        
        # 3. Launch dashboard (background)
        print("\n🔄 Launching dashboard (background)...")
        subprocess.Popen(["python", "tools/launch_dashboard_auto.py"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        print("✅ Dashboard launched")
        
        print("\n🎯 MORNING ROUTINE COMPLETE!")
        print("📊 Dashboard: http://localhost:8501")
        print("📈 Smart collector running in background")
        print("⏰ Next check recommended in 2-3 hours")
    
    def evening_routine(self):
        """Complete evening analysis routine"""
        print("🌆 EVENING ROUTINE - Daily Analysis")
        print("=" * 50)
        
        # 1. Daily collection report
        self.run_command("python core/advanced_daily_collection.py", "Generating daily report")
        
        # 2. Paper trading status
        self.run_command("python core/advanced_paper_trading.py --mode status", "Checking trading performance")
        
        # 3. Quick system health
        self.run_command("python tools/comprehensive_analyzer.py", "Final system health check")
        
        print("\n🎯 EVENING ROUTINE COMPLETE!")
        print("📊 Check reports/ folder for detailed analysis")
        print("💤 System ready for overnight")
    
    def quick_status(self):
        """Quick system status check"""
        print("📊 QUICK STATUS CHECK")
        print("=" * 30)
        
        # Sample count
        sample_cmd = """python -c "
from src.ml_training_pipeline import MLTrainingPipeline
try:
    p = MLTrainingPipeline()
    X, y = p.prepare_training_dataset(min_samples=1)
    if X is not None:
        print(f'Training Samples: {len(X)}')
    else:
        print('Training Samples: 0')
except Exception as e:
    print(f'Sample check failed: {e}')
" """
        self.run_command(sample_cmd)
        
        # Model performance
        perf_cmd = """python -c "
import sys; sys.path.append('core')
from advanced_paper_trading import AdvancedPaperTrader
try:
    apt = AdvancedPaperTrader()
    if hasattr(apt, 'performance_metrics') and apt.performance_metrics:
        win_rate = apt.performance_metrics.get('win_rate', 0)
        total_return = apt.performance_metrics.get('total_return', 0)
        print(f'Win Rate: {win_rate:.1%}')
        print(f'Total Return: {total_return:.1%}')
    else:
        print('Performance: No trades yet')
except Exception as e:
    print(f'Performance check: {e}')
" """
        self.run_command(perf_cmd)
        
        # Collection progress
        progress_cmd = """python -c "
import json, os
try:
    if os.path.exists('data/ml_models/collection_progress.json'):
        with open('data/ml_models/collection_progress.json', 'r') as f:
            progress = json.load(f)
        print(f'Signals Today: {progress.get(\"signals_today\", 0)}')
    else:
        print('No collection progress data')
except Exception as e:
    print(f'Progress check failed: {e}')
" """
        self.run_command(progress_cmd)
    
    def weekly_maintenance(self):
        """Weekly optimization routine"""
        print("📅 WEEKLY MAINTENANCE - System Optimization")
        print("=" * 50)
        
        # 1. Retrain models
        self.run_command("python scripts/retrain_ml_models.py", "Retraining ML models")
        
        # 2. Generate comprehensive analysis
        self.run_command("python tools/comprehensive_analyzer.py", "Running comprehensive system analysis")
        
        # 3. Weekly performance report
        self.run_command("python core/advanced_paper_trading.py --report-only", "Generating weekly performance report")
        
        # 4. Trading pattern analysis
        self.run_command("python tools/analyze_trading_patterns.py", "Analyzing trading patterns and improvements")
        
        # 5. Data quality check
        data_check_cmd = """python -c "
from src.ml_training_pipeline import MLTrainingPipeline
try:
    p = MLTrainingPipeline()
    X, y = p.prepare_training_dataset(min_samples=1)
    if X is not None:
        print(f'Training dataset: {len(X)} samples, {X.shape[1]} features')
        print(f'Class balance: {y.mean():.3f}')
        print('✅ Data quality check passed')
    else:
        print('❌ No training data available')
except Exception as e:
    print(f'Data check failed: {e}')
" """
        self.run_command(data_check_cmd, "Checking data quality")
        
        print("\n🎯 WEEKLY MAINTENANCE COMPLETE!")
        print("📊 System optimized for next week")
        print("📈 Check reports/ folder for detailed analysis")
    
    def emergency_restart(self):
        """Emergency system restart"""
        print("🚨 EMERGENCY RESTART")
        print("=" * 30)
        
        # Stop all processes
        print("🔄 Stopping all trading processes...")
        subprocess.run("pkill -f 'smart_collector\\|launch_dashboard\\|news_trading'", shell=True)
        time.sleep(2)
        print("✅ Processes stopped")
        
        # Restart
        print("\n🔄 Restarting system...")
        subprocess.Popen(["python", "core/smart_collector.py"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        time.sleep(1)
        subprocess.Popen(["python", "tools/launch_dashboard_auto.py"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        print("✅ System restarted")
        
        print("\n🎯 EMERGENCY RESTART COMPLETE!")

def main():
    if len(sys.argv) < 2:
        print("""
🎯 Trading System Manager - Usage Guide

Available Commands:
  python daily_manager.py morning     # Complete morning startup routine
  python daily_manager.py evening     # Complete evening analysis routine  
  python daily_manager.py status      # Quick status check
  python daily_manager.py weekly      # Weekly optimization routine
  python daily_manager.py restart     # Emergency restart all systems

Examples:
  python daily_manager.py morning     # Start your trading day
  python daily_manager.py status      # Quick health check
  python daily_manager.py evening     # End of day analysis
  
🔥 Pro Tip: Add aliases to your shell:
  alias tm-start="python daily_manager.py morning"
  alias tm-status="python daily_manager.py status"  
  alias tm-end="python daily_manager.py evening"
        """)
        return
    
    manager = TradingSystemManager()
    command = sys.argv[1].lower()
    
    if command == "morning":
        manager.morning_routine()
    elif command == "evening":
        manager.evening_routine()
    elif command == "status":
        manager.quick_status()
    elif command == "weekly":
        manager.weekly_maintenance()
    elif command == "restart":
        manager.emergency_restart()
    else:
        print(f"❌ Unknown command: {command}")
        print("Available: morning, evening, status, weekly, restart")

if __name__ == "__main__":
    main()
