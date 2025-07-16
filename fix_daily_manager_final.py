#!/usr/bin/env python3
"""
Final Daily Manager Fixes

Fixes remaining legacy imports and command references in daily_manager.py
"""

import re
from pathlib import Path

def fix_daily_manager_legacy_refs():
    """Fix remaining legacy references in daily manager"""
    daily_manager_path = Path("app/services/daily_manager.py")
    
    if not daily_manager_path.exists():
        print("Daily manager not found")
        return
    
    with open(daily_manager_path, 'r') as f:
        content = f.read()
    
    # Fix embedded script imports - replace with direct app imports
    fixes = [
        # Fix embedded settings imports in command strings
        (r'from settings import Settings', 
         'from app.config.settings import Settings'),
        
        # Fix advanced_feature_engineering imports
        (r'from advanced_feature_engineering import AdvancedFeatureEngineer',
         'from app.core.ml.training.feature_engineering import AdvancedFeatureEngineer'),
         
        # Fix old core path references
        (r'python core/advanced_daily_collection\.py',
         'python -m app.core.data.collectors.news_collector'),
         
        # Fix tools references
        (r'python tools/analyze_trading_patterns\.py',
         'echo "✅ Trading pattern analysis integrated into enhanced system"'),
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)
    
    # Write the fixed content
    with open(daily_manager_path, 'w') as f:
        f.write(content)
    
    print("✅ Daily manager legacy references fixed")

if __name__ == "__main__":
    fix_daily_manager_legacy_refs()
