#!/usr/bin/env python3
"""
Import path fixer for restructured project
Updates all import statements to use new package structure
"""

import os
import re
from pathlib import Path

class ImportFixer:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.app_dir = self.base_dir / 'app'
        
        # Define import mapping patterns
        self.import_patterns = [
            # Old src imports to new app structure
            (r'from src\.([^.\s]+)', r'from app.core.\1'),
            (r'import src\.([^.\s]+)', r'import app.core.\1'),
            
            # Specific module mappings
            (r'from app\.core\.ml_training_pipeline', r'from app.core.ml.training.pipeline'),
            (r'from app\.core\.enhanced_ensemble_learning', r'from app.core.ml.ensemble.enhanced_ensemble'),
            (r'from app\.core\.enhanced_sentiment_integration', r'from app.core.sentiment.integration'),
            (r'from app\.core\.news_sentiment', r'from app.core.sentiment.news_analyzer'),
            (r'from app\.core\.temporal_sentiment_analyzer', r'from app.core.sentiment.temporal_analyzer'),
            (r'from app\.core\.position_risk_assessor', r'from app.core.trading.risk_management'),
            (r'from app\.core\.advanced_feature_engineering', r'from app.core.ml.training.feature_engineering'),
            (r'from app\.core\.technical_analysis', r'from app.core.analysis.technical'),
            (r'from app\.core\.news_impact_analyzer', r'from app.core.analysis.news_impact'),
            (r'from app\.core\.data_feed', r'from app.core.data.collectors.market_data'),
            (r'from app\.core\.bank_symbols', r'from app.utils.constants'),
            (r'from app\.core\.bank_keywords', r'from app.utils.keywords'),
            (r'from app\.core\.sentiment_history', r'from app.core.sentiment.history'),
            (r'from app\.core\.trading_outcome_tracker', r'from app.core.trading.position_tracker'),
            
            # Import variations
            (r'import app\.core\.ml_training_pipeline', r'import app.core.ml.training.pipeline'),
            (r'import app\.core\.enhanced_ensemble_learning', r'import app.core.ml.ensemble.enhanced_ensemble'),
            
            # Config imports
            (r'from config\.', r'from app.config.'),
            (r'import config\.', r'import app.config.'),
            
            # Relative imports within app
            (r'from \.\.\.config\.', r'from ...config.'),
            (r'from \.\.config\.', r'from ..config.'),
        ]
    
    def fix_file_imports(self, file_path: Path):
        """Fix imports in a single file"""
        try:
            content = file_path.read_text()
            original_content = content
            
            # Apply all import pattern fixes
            for old_pattern, new_pattern in self.import_patterns:
                content = re.sub(old_pattern, new_pattern, content)
            
            # Write back if changed
            if content != original_content:
                file_path.write_text(content)
                print(f"✅ Fixed imports in: {file_path.relative_to(self.base_dir)}")
                return True
            return False
                
        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")
            return False
    
    def fix_all_imports(self):
        """Fix imports in all Python files"""
        print("🔧 Fixing import statements in all Python files...")
        
        fixed_count = 0
        
        # Fix imports in app directory
        for root, dirs, files in os.walk(self.app_dir):
            # Skip __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('__pycache__')]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    if self.fix_file_imports(file_path):
                        fixed_count += 1
        
        # Also fix any remaining root files that might have imports
        root_python_files = list(self.base_dir.glob('*.py'))
        for file_path in root_python_files:
            if file_path.name not in ['migrate_structure.py', 'fix_imports.py']:
                if self.fix_file_imports(file_path):
                    fixed_count += 1
        
        # Fix imports in scripts directory
        scripts_dir = self.base_dir / 'scripts'
        if scripts_dir.exists():
            for file_path in scripts_dir.glob('*.py'):
                if self.fix_file_imports(file_path):
                    fixed_count += 1
        
        print(f"📊 Fixed imports in {fixed_count} files")
    
    def create_missing_modules(self):
        """Create any missing modules that are being imported"""
        print("🔧 Creating missing modules...")
        
        # Modules that might be missing
        missing_modules = [
            'app/core/sentiment/history.py',
            'app/core/data/validators/__init__.py',
            'app/api/__init__.py',
            'app/api/routes/__init__.py',
            'app/api/schemas/__init__.py',
        ]
        
        for module_path in missing_modules:
            full_path = self.base_dir / module_path
            if not full_path.exists():
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create basic module content
                if module_path.endswith('history.py'):
                    content = '''"""Sentiment history management"""

class SentimentHistoryManager:
    """Manages sentiment history data"""
    
    def __init__(self):
        self.history = []
    
    def add_sentiment_record(self, record):
        """Add a sentiment record to history"""
        self.history.append(record)
'''
                else:
                    content = '"""Module placeholder"""'
                
                full_path.write_text(content)
                print(f"✅ Created: {module_path}")

def main():
    fixer = ImportFixer()
    fixer.create_missing_modules()
    fixer.fix_all_imports()
    print("🎯 Import fixing complete!")

if __name__ == '__main__':
    main()
