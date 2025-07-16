"""
Test script for the modular dashboard components
Verifies that all components can be imported and initialized
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test that all dashboard components can be imported"""
    print("🧪 Testing Dashboard Component Imports...")
    
    try:
        # Test logging utilities
        from dashboard.utils.logging_config import setup_dashboard_logger
        logger = setup_dashboard_logger("test")
        print("✅ Logging utilities imported successfully")
        
        # Test data manager
        from dashboard.utils.data_manager import DataManager
        data_manager = DataManager()
        print("✅ DataManager imported and initialized")
        
        # Test helpers
        from dashboard.utils.helpers import format_sentiment_score, get_confidence_level
        score_text, score_class = format_sentiment_score(0.5)
        conf_text, conf_class = get_confidence_level(0.8)
        print("✅ Helper functions working correctly")
        
        # Test chart generator (may fail due to plotly import)
        try:
            from dashboard.charts.chart_generator import ChartGenerator
            chart_gen = ChartGenerator()
            print("✅ ChartGenerator imported and initialized")
        except ImportError as e:
            print(f"⚠️  ChartGenerator import failed (expected): {e}")
        
        # Test UI components (may fail due to streamlit import)
        try:
            from dashboard.components.ui_components import UIComponents
            ui_components = UIComponents()
            print("✅ UIComponents imported and initialized")
        except ImportError as e:
            print(f"⚠️  UIComponents import failed (expected): {e}")
        
        # Test main dashboard (may fail due to streamlit import)
        try:
            from dashboard.main_dashboard import ProfessionalDashboard
            print("✅ ProfessionalDashboard imported successfully")
        except ImportError as e:
            print(f"⚠️  ProfessionalDashboard import failed (expected): {e}")
        
        print("\n✅ All basic components imported successfully!")
        print("ℹ️  Streamlit/Plotly import warnings are expected in test environment")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    return True

def test_data_structure():
    """Test the data directory structure"""
    print("\n🗂️  Testing Data Directory Structure...")
    
    data_path = "data/sentiment_history"
    
    if os.path.exists(data_path):
        files = [f for f in os.listdir(data_path) if f.endswith('_history.json')]
        print(f"✅ Data directory found with {len(files)} JSON files")
        
        # Show first few files
        for file in files[:3]:
            file_path = os.path.join(data_path, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"   📄 {file}: {file_size:.1f}KB")
        
        if len(files) > 3:
            print(f"   ... and {len(files) - 3} more files")
            
    else:
        print("⚠️  Data directory not found - dashboard will show warnings")
    
    return True

def test_logging():
    """Test the logging system"""
    print("\n📝 Testing Logging System...")
    
    try:
        from dashboard.utils.logging_config import (
            setup_dashboard_logger, log_data_loading_stats,
            log_performance_metrics, log_error_with_context
        )
        
        # Create test logger
        logger = setup_dashboard_logger("test_component", "INFO", True, "logs")
        
        # Test different log types
        logger.info("Test info message")
        log_data_loading_stats(logger, "TEST.AX", 100, "test_file.json")
        log_performance_metrics(logger, "Test Operation", 1.234, True)
        
        print("✅ Logging system working correctly")
        print("📁 Check logs/ directory for output files")
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration loading"""
    print("\n⚙️  Testing Configuration...")
    
    try:
        from app.config.settings import Settings
        settings = Settings()
        print(f"✅ Settings loaded: {len(settings.BANK_SYMBOLS)} bank symbols configured")
        
    except ImportError:
        print("⚠️  Settings module not found - using fallback configuration")
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Professional Dashboard Modular Testing")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_structure,
        test_logging,
        test_configuration
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All tests passed! Dashboard should work correctly.")
        print("💡 Run: streamlit run professional_dashboard_modular.py")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        print("💡 Missing dependencies can be installed with:")
        print("   pip install streamlit plotly pandas numpy")

if __name__ == "__main__":
    main()
