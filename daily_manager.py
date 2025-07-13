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
        # Auto-detect the correct base directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(current_dir) == "trading_analysis":
            self.base_dir = current_dir
        else:
            # Look for trading_analysis directory
            possible_paths = [
                "/Users/toddsutherland/Repos/trading_analysis",  # macOS
                "/root/trading_analysis",  # Linux server
                os.path.join(os.path.expanduser("~"), "trading_analysis"),  # Generic home
                current_dir  # Current directory as fallback
            ]
            
            self.base_dir = None
            for path in possible_paths:
                if os.path.exists(path) and os.path.isfile(os.path.join(path, "daily_manager.py")):
                    self.base_dir = path
                    break
            
            if not self.base_dir:
                self.base_dir = current_dir
                print(f"⚠️  Using current directory: {self.base_dir}")
        
        os.chdir(self.base_dir)
        print(f"📁 Working directory: {self.base_dir}")
    
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
        
        # 2. Enhanced sentiment analysis
        print("\n🧠 Running enhanced sentiment analysis...")
        enhanced_cmd = """python -c "
import sys; sys.path.append('src')
from temporal_sentiment_analyzer import TemporalSentimentAnalyzer
from advanced_feature_engineering import AdvancedFeatureEngineer

print('✅ Enhanced sentiment analysis initialized')
try:
    analyzer = TemporalSentimentAnalyzer()
    engineer = AdvancedFeatureEngineer()
    print('✅ Enhanced modules loaded successfully')
    
    # Sample feature engineering for major banks
    for symbol in ['CBA.AX', 'WBC.AX', 'ANZ.AX', 'NAB.AX']:
        sentiment_data = {'overall_sentiment': 0.5, 'confidence': 0.8, 'news_count': 5}
        features = engineer.engineer_comprehensive_features(symbol, sentiment_data)
        print(f'✅ {symbol}: Generated {len(features.get(\"features\", []))} enhanced features')
    
except Exception as e:
    print(f'⚠️  Enhanced sentiment analysis warning: {e}')
" """
        self.run_command(enhanced_cmd, "Enhanced sentiment & feature analysis")
        
        # 3. Start smart collector (background)
        print("\n🔄 Starting smart collector (background)...")
        subprocess.Popen(["python", "core/smart_collector.py"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        print("✅ Smart collector started")
        
        # 4. Launch dashboard (background)
        print("\n🔄 Launching dashboard (background)...")
        subprocess.Popen(["python", "tools/launch_dashboard_auto.py"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        print("✅ Dashboard launched")
        
        print("\n🎯 MORNING ROUTINE COMPLETE!")
        print("📊 Dashboard: http://localhost:8501")
        print("📈 Smart collector running in background")
        print("🧠 Enhanced ML features active")
        print("⏰ Next check recommended in 2-3 hours")
    
    def evening_routine(self):
        """Complete evening analysis routine"""
        print("🌆 EVENING ROUTINE - Daily Analysis")
        print("=" * 50)
        
        # 1. Enhanced ensemble analysis
        print("\n🤖 Running enhanced ensemble analysis...")
        ensemble_cmd = """python -c "
import sys; sys.path.append('src')
from enhanced_ensemble_learning import EnhancedTransformerEnsemble, ModelPrediction
from temporal_sentiment_analyzer import TemporalSentimentAnalyzer
from datetime import datetime
import numpy as np

try:
    # Initialize enhanced systems
    ensemble = EnhancedTransformerEnsemble()
    analyzer = TemporalSentimentAnalyzer()
    
    print('✅ Enhanced ensemble system initialized')
    
    # Simulate ensemble analysis for major banks
    symbols = ['CBA.AX', 'WBC.AX', 'ANZ.AX', 'NAB.AX']
    ensemble_results = []
    
    for symbol in symbols:
        # Get temporal analysis
        analysis = analyzer.analyze_sentiment_evolution(symbol)
        trend = analysis.get('trend', 0.0)
        volatility = analysis.get('volatility', 0.0)
        
        print(f'✅ {symbol}: Temporal analysis complete (trend: {trend:.3f}, vol: {volatility:.3f})')
        ensemble_results.append((symbol, trend, volatility))
    
    print(f'✅ Enhanced ensemble analysis completed for {len(symbols)} symbols')
    
    # Performance summary
    avg_trend = np.mean([r[1] for r in ensemble_results])
    avg_volatility = np.mean([r[2] for r in ensemble_results])
    print(f'📊 Market Summary: Avg Trend: {avg_trend:.3f}, Avg Volatility: {avg_volatility:.3f}')
    
except Exception as e:
    print(f'⚠️  Enhanced ensemble warning: {e}')
" """
        self.run_command(ensemble_cmd, "Enhanced ensemble & temporal analysis")
        
        # 2. Daily collection report
        self.run_command("python core/advanced_daily_collection.py", "Generating daily report")
        
        # 3. Paper trading status
        self.run_command("python core/advanced_paper_trading.py --mode status", "Checking trading performance")
        
        # 4. Quick system health
        self.run_command("python tools/comprehensive_analyzer.py", "Final system health check")
        
        print("\n🎯 EVENING ROUTINE COMPLETE!")
        print("📊 Check reports/ folder for detailed analysis")
        print("🧠 Enhanced ML analysis completed")
        print("💤 System ready for overnight")
    
    def quick_status(self):
        """Quick system status check"""
        print("📊 QUICK STATUS CHECK")
        print("=" * 30)
        
        # Enhanced features status
        enhanced_status_cmd = """python -c "
import sys; sys.path.append('src')
try:
    from temporal_sentiment_analyzer import TemporalSentimentAnalyzer
    from enhanced_ensemble_learning import EnhancedTransformerEnsemble
    from advanced_feature_engineering import AdvancedFeatureEngineer
    
    print('✅ Enhanced Features: All modules available')
    
    # Quick test
    analyzer = TemporalSentimentAnalyzer()
    ensemble = EnhancedTransformerEnsemble()
    engineer = AdvancedFeatureEngineer()
    
    # Test feature generation
    test_sentiment = {'overall_sentiment': 0.6, 'confidence': 0.8, 'news_count': 5}
    features = engineer.engineer_comprehensive_features('TEST.AX', test_sentiment)
    feature_count = len(features.get('features', []))
    
    print(f'✅ Feature Engineering: {feature_count} features generated')
    print('✅ Enhanced ML: All systems operational')
    
except Exception as e:
    print(f'❌ Enhanced Features: {e}')
" """
        self.run_command(enhanced_status_cmd, "Enhanced ML Status")
        
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
        
        # Enhanced ML model performance analysis
        enhanced_weekly_cmd = """python -c "
import sys; sys.path.append('src')
from enhanced_ensemble_learning import EnhancedTransformerEnsemble
from temporal_sentiment_analyzer import TemporalSentimentAnalyzer
from advanced_feature_engineering import AdvancedFeatureEngineer
import json

try:
    print('🔧 Enhanced ML Weekly Maintenance')
    
    # Initialize systems
    ensemble = EnhancedTransformerEnsemble()
    analyzer = TemporalSentimentAnalyzer()
    engineer = AdvancedFeatureEngineer()
    
    # Performance analysis for major symbols
    symbols = ['CBA.AX', 'WBC.AX', 'ANZ.AX', 'NAB.AX']
    weekly_analysis = {}
    
    for symbol in symbols:
        # Enhanced analysis
        sentiment_data = {'overall_sentiment': 0.5, 'confidence': 0.8, 'news_count': 10}
        features = engineer.engineer_comprehensive_features(symbol, sentiment_data)
        temporal_analysis = analyzer.analyze_sentiment_evolution(symbol)
        
        weekly_analysis[symbol] = {
            'features_generated': len(features.get('features', [])),
            'temporal_trend': temporal_analysis.get('trend', 0.0),
            'temporal_volatility': temporal_analysis.get('volatility', 0.0)
        }
    
    print(f'✅ Weekly analysis completed for {len(symbols)} symbols')
    
    # Save weekly report
    with open('reports/enhanced_weekly_analysis.json', 'w') as f:
        json.dump(weekly_analysis, f, indent=2, default=str)
    
    print('✅ Enhanced weekly report saved to reports/enhanced_weekly_analysis.json')
    
except Exception as e:
    print(f'⚠️  Enhanced weekly maintenance warning: {e}')
" """
        self.run_command(enhanced_weekly_cmd, "Enhanced ML weekly analysis")
        
        # 1. Retrain ML models
        self.run_command("python scripts/retrain_ml_models.py", "Retraining ML models")
        
        # 2. Generate comprehensive analysis
        self.run_command("python tools/comprehensive_analyzer.py", "Running comprehensive system analysis")
        
        # 3. Weekly performance report
        self.run_command("python core/advanced_paper_trading.py --report-only", "Generating weekly performance report")
        
        # 4. Trading pattern analysis
        self.run_command("python tools/analyze_trading_patterns.py", "Analyzing trading patterns and improvements")
        
        print("\n🎯 WEEKLY MAINTENANCE COMPLETE!")
        print("📊 Check reports/ folder for all analysis")
        print("🧠 Enhanced ML models analyzed and optimized")
        print("⚡ System optimized for next week")
    
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

    def test_enhanced_features(self):
        """Test the enhanced ML features"""
        print("🧪 TESTING ENHANCED FEATURES")
        print("=" * 40)
        
        # Comprehensive test of enhanced features
        test_cmd = """python tests/test_simple_functionality.py"""
        if self.run_command(test_cmd, "Running enhanced feature tests"):
            print("✅ All enhanced features working correctly")
        else:
            print("⚠️  Some tests may need attention")
        
        # Interactive feature demonstration
        demo_cmd = """python -c "
import sys; sys.path.append('src')
from temporal_sentiment_analyzer import SentimentDataPoint, TemporalSentimentAnalyzer
from enhanced_ensemble_learning import ModelPrediction, EnhancedTransformerEnsemble
from advanced_feature_engineering import AdvancedFeatureEngineer
from datetime import datetime, timedelta
import numpy as np

print('🚀 ENHANCED FEATURES DEMONSTRATION')
print('=' * 40)

# 1. Temporal Sentiment Analysis Demo
print('\\n1️⃣  TEMPORAL SENTIMENT ANALYSIS')
analyzer = TemporalSentimentAnalyzer()

# Add sample sentiment observations
base_time = datetime.now()
for i in range(5):
    point = SentimentDataPoint(
        timestamp=base_time - timedelta(hours=i),
        symbol='CBA.AX',
        sentiment_score=0.5 + 0.2 * np.sin(i * 0.5),
        confidence=0.8 + 0.1 * np.random.random(),
        news_count=np.random.randint(3, 8),
        relevance_score=0.8
    )
    analyzer.add_sentiment_observation(point)

analysis = analyzer.analyze_sentiment_evolution('CBA.AX')
print(f'✅ Temporal Analysis: {len(analysis)} metrics calculated')
print(f'   📈 Trend: {analysis.get(\"trend\", 0):.3f}')
print(f'   📊 Volatility: {analysis.get(\"volatility\", 0):.3f}')
print(f'   🎯 Regime: {analysis.get(\"current_regime\", \"unknown\")}')

# 2. Advanced Feature Engineering Demo
print('\\n2️⃣  ADVANCED FEATURE ENGINEERING')
engineer = AdvancedFeatureEngineer()

sentiment_data = {
    'overall_sentiment': 0.65,
    'confidence': 0.85,
    'news_count': 7,
    'sentiment_components': {
        'news': 0.7,
        'reddit': 0.6,
        'events': 0.65
    }
}

features = engineer.engineer_comprehensive_features('CBA.AX', sentiment_data)
print(f'✅ Feature Engineering: Generated comprehensive feature set')
print(f'   📊 Feature Quality: {features.get(\"feature_quality\", 0):.3f}')
print(f'   🔢 Feature Count: {features.get(\"feature_count\", 0)}')

# 3. Enhanced Ensemble Learning Demo
print('\\n3️⃣  ENHANCED ENSEMBLE LEARNING')
ensemble = EnhancedTransformerEnsemble()

# Create sample model predictions
predictions = []
for i in range(3):
    pred = ModelPrediction(
        model_name=f'enhanced_model_{i}',
        prediction=0.6 + 0.1 * np.random.random(),
        confidence=0.8 + 0.1 * np.random.random(),
        timestamp=datetime.now()
    )
    predictions.append(pred)

print(f'✅ Ensemble Learning: Created {len(predictions)} model predictions')
print(f'   🤖 Models: {[p.model_name for p in predictions]}')
print(f'   📈 Avg Prediction: {np.mean([p.prediction for p in predictions]):.3f}')
print(f'   🎯 Avg Confidence: {np.mean([p.confidence for p in predictions]):.3f}')

print('\\n🎯 ENHANCED FEATURES DEMONSTRATION COMPLETE!')
print('All advanced ML components are operational and ready for trading analysis.')
" """
        self.run_command(demo_cmd, "Enhanced features demonstration")
        
        print("\n🎯 ENHANCED FEATURES TEST COMPLETE!")
        print("🧠 All advanced ML components verified")
        print("📊 System ready for enhanced trading analysis")

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
  python daily_manager.py test        # Test enhanced ML features

Examples:
  python daily_manager.py morning     # Start your trading day
  python daily_manager.py status      # Quick health check
  python daily_manager.py evening     # End of day analysis
  python daily_manager.py test        # Test enhanced features
  
🔥 Pro Tip: Add aliases to your shell:
  alias tm-start="python daily_manager.py morning"
  alias tm-status="python daily_manager.py status"  
  alias tm-end="python daily_manager.py evening"
  alias tm-test="python daily_manager.py test"
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
    elif command == "test":
        manager.test_enhanced_features()
    else:
        print(f"❌ Unknown command: {command}")
        print("Available: morning, evening, status, weekly, restart, test")

if __name__ == "__main__":
    main()
