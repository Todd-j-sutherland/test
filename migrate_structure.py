#!/usr/bin/env python3
"""
Migration script to reorganize trading analysis system
Moves files from old structure to new industry-standard structure
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple

class ProjectRestructure:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.app_dir = self.base_dir / 'app'
        self.backup_dir = self.base_dir / 'migration_backup'
        
    def create_backup(self):
        """Create backup of current structure"""
        print("🔄 Creating backup of current structure...")
        
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        # Copy current files to backup
        shutil.copytree(self.base_dir, self.backup_dir, 
                       ignore=shutil.ignore_patterns('*.git*', '*__pycache__*', 
                                                   '*.venv*', 'migration_backup'))
        print(f"✅ Backup created at {self.backup_dir}")
    
    def get_migration_map(self) -> List[Tuple[str, str]]:
        """Define file migration mapping"""
        return [
            # Configuration files
            ('config/settings.py', 'app/config/settings.py'),
            ('config/ml_config.yaml', 'app/config/ml_config.yaml'),
            
            # Core sentiment components (already moved some)
            ('enhanced_sentiment_scoring.py', 'app/core/sentiment/enhanced_scoring.py'),
            ('src/temporal_sentiment_analyzer.py', 'app/core/sentiment/temporal_analyzer.py'),
            ('src/news_sentiment.py', 'app/core/sentiment/news_analyzer.py'),
            ('src/enhanced_sentiment_integration.py', 'app/core/sentiment/integration.py'),
            
            # ML components
            ('src/enhanced_ensemble_learning.py', 'app/core/ml/ensemble/enhanced_ensemble.py'),
            ('src/enhanced_transformer_ensemble.py', 'app/core/ml/ensemble/transformer_ensemble.py'),
            ('src/ml_training_pipeline.py', 'app/core/ml/training/pipeline.py'),
            ('src/advanced_feature_engineering.py', 'app/core/ml/training/feature_engineering.py'),
            ('src/ml_backtester.py', 'app/core/ml/prediction/backtester.py'),
            
            # Trading components
            ('src/position_risk_assessor.py', 'app/core/trading/risk_management.py'),
            ('src/trading_outcome_tracker.py', 'app/core/trading/position_tracker.py'),
            ('core/advanced_paper_trading.py', 'app/core/trading/paper_trading.py'),
            
            # Data components
            ('src/data_feed.py', 'app/core/data/collectors/market_data.py'),
            ('core/smart_collector.py', 'app/core/data/collectors/news_collector.py'),
            ('core/news_trading_analyzer.py', 'app/core/data/processors/news_processor.py'),
            
            # Analysis components
            ('src/technical_analysis.py', 'app/core/analysis/technical.py'),
            ('src/news_impact_analyzer.py', 'app/core/analysis/news_impact.py'),
            
            # Services
            ('daily_manager.py', 'app/services/daily_manager.py'),
            ('core/advanced_daily_collection.py', 'app/services/data_collector.py'),
            
            # Dashboard components
            ('dashboard/main_dashboard.py', 'app/dashboard/main.py'),
            ('dashboard/charts/chart_generator.py', 'app/dashboard/components/charts.py'),
            ('dashboard/utils/data_manager.py', 'app/dashboard/utils/data_manager.py'),
            ('dashboard/utils/helpers.py', 'app/dashboard/utils/helpers.py'),
            ('professional_dashboard.py', 'app/dashboard/pages/professional.py'),
            
            # Scripts
            ('scripts/retrain_ml_models.py', 'scripts/retrain_models.py'),
            ('scripts/schedule_ml_training.py', 'scripts/schedule_training.py'),
            
            # Utilities
            ('src/bank_symbols.py', 'app/utils/constants.py'),
            ('src/bank_keywords.py', 'app/utils/keywords.py'),
        ]
    
    def migrate_files(self):
        """Migrate files to new structure"""
        print("🔄 Migrating files to new structure...")
        
        migration_map = self.get_migration_map()
        migrated_count = 0
        skipped_count = 0
        
        for old_path, new_path in migration_map:
            old_file = self.base_dir / old_path
            new_file = self.base_dir / new_path
            
            if old_file.exists():
                # Create directory if it doesn't exist
                new_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file (don't move yet, in case we need rollback)
                if not new_file.exists():
                    shutil.copy2(old_file, new_file)
                    print(f"✅ Migrated: {old_path} -> {new_path}")
                    migrated_count += 1
                else:
                    print(f"⚠️  Skipped (exists): {new_path}")
                    skipped_count += 1
            else:
                print(f"❌ Missing source: {old_path}")
                skipped_count += 1
        
        print(f"📊 Migration summary: {migrated_count} migrated, {skipped_count} skipped")
    
    def create_missing_init_files(self):
        """Create __init__.py files for all packages"""
        print("🔄 Creating missing __init__.py files...")
        
        # Find all directories that should be packages
        for root, dirs, files in os.walk(self.app_dir):
            root_path = Path(root)
            
            # Skip __pycache__ and .git directories
            dirs[:] = [d for d in dirs if not d.startswith('__pycache__') and not d.startswith('.')]
            
            # Create __init__.py if directory contains .py files or subdirectories
            python_files = [f for f in files if f.endswith('.py')]
            has_subdirs = any(Path(root_path / d).is_dir() for d in dirs)
            
            if python_files or has_subdirs:
                init_file = root_path / '__init__.py'
                if not init_file.exists():
                    init_file.write_text('"""Package initialization"""')
                    print(f"✅ Created: {init_file.relative_to(self.base_dir)}")
    
    def update_imports(self):
        """Update import statements in migrated files"""
        print("🔄 Updating import statements...")
        
        # Define import mapping
        import_mapping = {
            'from src.': 'from app.core.',
            'from config.': 'from app.config.',
            'import src.': 'import app.core.',
            'from enhanced_sentiment_scoring': 'from app.core.sentiment.enhanced_scoring',
            'from temporal_sentiment_analyzer': 'from app.core.sentiment.temporal_analyzer',
            'from enhanced_ensemble_learning': 'from app.core.ml.ensemble.enhanced_ensemble',
        }
        
        # Update files in app directory
        for root, dirs, files in os.walk(self.app_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    self._update_file_imports(file_path, import_mapping)
    
    def _update_file_imports(self, file_path: Path, import_mapping: dict):
        """Update imports in a single file"""
        try:
            content = file_path.read_text()
            original_content = content
            
            # Apply import mappings
            for old_import, new_import in import_mapping.items():
                content = content.replace(old_import, new_import)
            
            # Write back if changed
            if content != original_content:
                file_path.write_text(content)
                print(f"✅ Updated imports: {file_path.relative_to(self.base_dir)}")
                
        except Exception as e:
            print(f"❌ Error updating {file_path}: {e}")
    
    def create_new_tests_structure(self):
        """Reorganize tests into new structure"""
        print("🔄 Reorganizing tests...")
        
        new_tests_dir = self.base_dir / 'tests'
        
        # Create test structure
        test_dirs = ['unit', 'integration', 'e2e', 'fixtures']
        for test_dir in test_dirs:
            (new_tests_dir / test_dir).mkdir(parents=True, exist_ok=True)
        
        # Move existing tests
        test_migrations = [
            ('tests/test_enhanced_sentiment_scoring.py', 'tests/unit/test_sentiment_scoring.py'),
            ('tests/test_enhanced_sentiment_integration.py', 'tests/integration/test_sentiment_integration.py'),
            ('tests/test_daily_manager.py', 'tests/integration/test_daily_manager.py'),
            ('tests/test_system_integration.py', 'tests/integration/test_system.py'),
        ]
        
        for old_test, new_test in test_migrations:
            old_file = self.base_dir / old_test
            new_file = self.base_dir / new_test
            
            if old_file.exists() and not new_file.exists():
                new_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old_file, new_file)
                print(f"✅ Migrated test: {old_test} -> {new_test}")
    
    def cleanup_old_structure(self):
        """Clean up old files (after confirmation)"""
        print("🧹 Ready for cleanup phase...")
        print("⚠️  This will remove old files. Make sure new structure is working first!")
        
        response = input("Do you want to proceed with cleanup? (y/N): ")
        if response.lower() != 'y':
            print("Cleanup cancelled.")
            return
        
        # Files and directories to remove after migration
        cleanup_items = [
            'src/',
            'config/',
            'core/',
            'enhanced_sentiment_scoring.py',
            'professional_dashboard.py',
            'daily_manager.py',
        ]
        
        for item in cleanup_items:
            item_path = self.base_dir / item
            if item_path.exists():
                if item_path.is_dir():
                    shutil.rmtree(item_path)
                else:
                    item_path.unlink()
                print(f"🗑️  Removed: {item}")
    
    def validate_migration(self):
        """Validate the migration was successful"""
        print("🔍 Validating migration...")
        
        # Check key files exist
        key_files = [
            'app/__init__.py',
            'app/config/settings.py',
            'app/core/sentiment/enhanced_scoring.py',
            'app/core/ml/ensemble/enhanced_ensemble.py',
            'app/services/daily_manager.py',
            'pyproject.toml',
        ]
        
        all_good = True
        for file_path in key_files:
            full_path = self.base_dir / file_path
            if full_path.exists():
                print(f"✅ Found: {file_path}")
            else:
                print(f"❌ Missing: {file_path}")
                all_good = False
        
        if all_good:
            print("🎯 Migration validation passed!")
        else:
            print("⚠️  Migration validation failed - some files missing")
        
        return all_good
    
    def run_full_migration(self):
        """Run complete migration process"""
        print("🚀 Starting Trading Analysis System Restructure")
        print("=" * 60)
        
        try:
            # Step 1: Backup
            self.create_backup()
            
            # Step 2: Migrate files
            self.migrate_files()
            
            # Step 3: Create package structure
            self.create_missing_init_files()
            
            # Step 4: Update imports
            self.update_imports()
            
            # Step 5: Reorganize tests
            self.create_new_tests_structure()
            
            # Step 6: Validate
            if self.validate_migration():
                print("\n🎯 MIGRATION COMPLETED SUCCESSFULLY!")
                print("Next steps:")
                print("1. Test the new structure: python -m app.main status")
                print("2. Run tests: python -m pytest tests/")
                print("3. If everything works, run cleanup: python migrate_structure.py --cleanup")
            else:
                print("\n⚠️  MIGRATION COMPLETED WITH ISSUES")
                print("Please review the missing files and fix before proceeding")
                
        except Exception as e:
            print(f"\n❌ MIGRATION FAILED: {e}")
            print(f"Backup available at: {self.backup_dir}")

def main():
    import sys
    
    restructure = ProjectRestructure()
    
    if '--cleanup' in sys.argv:
        restructure.cleanup_old_structure()
    elif '--validate' in sys.argv:
        restructure.validate_migration()
    else:
        restructure.run_full_migration()

if __name__ == '__main__':
    main()
