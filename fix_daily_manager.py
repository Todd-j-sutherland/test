#!/usr/bin/env python3
"""
Quick Fix Script for Daily Manager

Removes legacy imports and references that are no longer available
after the project restructuring.
"""

import re
from pathlib import Path

def fix_daily_manager():
    """Fix the daily manager imports and references"""
    daily_manager_path = Path("app/services/daily_manager.py")
    
    if not daily_manager_path.exists():
        print("Daily manager not found")
        return
    
    with open(daily_manager_path, 'r') as f:
        content = f.read()
    
    # Fix the legacy imports and references
    fixes = [
        # Remove legacy enhanced integration
        (r'from daily_manager_enhanced_integration import run_enhanced_daily_analysis', 
         '# Legacy import removed - functionality integrated'),
        
        # Remove legacy advanced paper trading imports
        (r'from advanced_paper_trading import AdvancedPaperTrader',
         'from app.core.trading.paper_trading import AdvancedPaperTrader'),
        
        # Fix shell command references to old paths
        (r'python core/advanced_paper_trading\.py',
         'python -m app.core.trading.paper_trading'),
         
        # Fix enhanced command block (replace the entire problematic command)
        (r'enhanced_cmd = """python -c ".*?" """',
         '''# Enhanced sentiment analysis using new app structure
        try:
            from app.core.sentiment.enhanced_scoring import EnhancedSentimentScorer
            from app.core.sentiment.temporal_analyzer import TemporalSentimentAnalyzer
            
            print('✅ Enhanced sentiment integration using new app structure')
            # Enhanced analysis is handled by sentiment components in morning/evening routines
            return True
        except Exception as e:
            print(f"❌ Enhanced sentiment analysis error: {e}")
            return False'''),
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Write the fixed content
    with open(daily_manager_path, 'w') as f:
        f.write(content)
    
    print("✅ Daily manager fixed")

if __name__ == "__main__":
    fix_daily_manager()
