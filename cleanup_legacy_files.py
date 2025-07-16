#!/usr/bin/env python3
"""
Legacy File Cleanup Script

This script identifies and archives or removes legacy files that are no longer 
needed after the project restructuring into the app/ package structure.
"""

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Files that are actively used in the new structure
ACTIVE_FILES = {
    # New app structure
    'app/',
    'pyproject.toml',
    'requirements.txt',
    '.gitignore',
    'LICENSE',
    'README.md',
    
    # Data directories (keep)
    'data/',
    'logs/',
    'reports/',
    
    # Tests (keep)
    'tests/',
    
    # Migration scripts (keep temporarily)
    'migrate_structure.py',
    'fix_imports.py',
    
    # Documentation (selective keep)
    'RESTRUCTURE_COMPLETE.md',
    'TESTING_COMPLETE_SUMMARY.md',
    'FILE_ORGANIZATION.md',
    'ORGANIZATION_COMPLETE.md',
}

# Files/directories to archive (move to archive/)
ARCHIVE_CANDIDATES = [
    # Legacy root files that have been moved to app/
    'daily_manager.py',  # Now in app/services/
    'enhanced_sentiment_scoring.py',  # Now in app/core/sentiment/
    'professional_dashboard.py',  # Legacy version
    'professional_dashboard_modular.py',  # Legacy version  
    'professional_dashboard_backup.py',  # Backup file
    'original_professional_dashboard.py',  # Legacy version
    
    # Demo files
    'demo_claude_enhancements.py',
    'demo_position_risk_assessment.py', 
    'demo_position_risk_working.py',
    'demo_dashboard_integration.py',
    'demo_enhanced_sentiment.py',
    'position_risk_dashboard_demo.py',
    
    # Setup/deploy scripts that may be outdated
    'deploy_position_risk.py',
    'setup_position_tracking.py',
    'launch_professional_dashboard.sh',
    
    # Test files that haven't been migrated to tests/
    'test_integration.py',
    'test_modular_dashboard.py',
    'ml_diagnostic.py',
    
    # Legacy source directories (already moved to app/core/)
    'src/',
    'core/',
    'config/',  # Old config, now in app/config/
    
    # Old dashboard structure (now in app/dashboard/)
    'dashboard/',
    'utils/',  # Old utils, now in app/utils/
    
    # Scripts that may be outdated
    'scripts/',
    'tools/',
    
    # Documentation that's outdated or superseded
    'DASHBOARD_ENHANCEMENTS_SUMMARY.md',
    'DASHBOARD_MODULAR_GUIDE.md', 
    'POSITION_RISK_COMPLETE_SUMMARY.md',
    'POSITION_RISK_ASSESSOR_ANALYSIS.md',
    'POSITION_RISK_IMPLEMENTATION_ANALYSIS.md',
    'ENHANCED_INTEGRATION_SUMMARY.md',
    'ENHANCED_SENTIMENT_INTEGRATION_GUIDE.md',
    'IMPLEMENTATION_SUMMARY.md',
    'DIGITALOCEAN_QUICK_GUIDE.md',
    'QUICK_DROPLET_CHECK.md',
    'RESTRUCTURE_PLAN.md',
    
    # Legacy test files
    'tests_files/',
    
    # Verification scripts
    'verify_droplet.sh',
    'claude_doc.txt',
    
    # Old requirements
    'requirements_simple.txt',
    
    # Demo files in demos/ directory
    'demos/',
    
    # Old documentation
    'docs/',
    'documentation/',
    
    # Templates (if not used)
    'templates/',
]

# Files to completely remove (not archive)
DELETE_CANDIDATES = [
    '__pycache__/',
    '*.pyc',
    '*.pyo',
    '.pytest_cache/',
    '*.log',
    '.DS_Store',
    'Thumbs.db',
]

def create_archive_directory():
    """Create timestamped archive directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path(f"archive/cleanup_{timestamp}")
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir

def is_file_referenced(file_path, project_root):
    """Check if a file is referenced in the new app structure"""
    try:
        # Skip checking for references in files we're about to archive
        file_name = Path(file_path).name
        
        # Search for imports or references in app/ directory
        for root, dirs, files in os.walk(project_root / "app"):
            for file in files:
                if file.endswith('.py'):
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            content = f.read()
                            if file_name.replace('.py', '') in content:
                                return True, f"Referenced in {os.path.join(root, file)}"
                    except:
                        continue
        
        # Check tests directory
        tests_dir = project_root / "tests"
        if tests_dir.exists():
            for root, dirs, files in os.walk(tests_dir):
                for file in files:
                    if file.endswith('.py'):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                content = f.read()
                                if file_name.replace('.py', '') in content:
                                    return True, f"Referenced in {os.path.join(root, file)}"
                        except:
                            continue
                            
        return False, "No references found"
    except Exception as e:
        return False, f"Error checking references: {e}"

def analyze_project_structure(project_root):
    """Analyze the current project structure and categorize files"""
    project_root = Path(project_root)
    
    active_files = []
    archive_candidates = []
    delete_candidates = []
    
    print("🔍 Analyzing project structure...")
    print(f"Project root: {project_root}")
    
    for item in project_root.iterdir():
        if item.name.startswith('.') and item.name not in ['.gitignore', '.env.example']:
            continue
            
        relative_path = item.relative_to(project_root)
        
        # Check if it's an active file/directory
        if any(str(relative_path).startswith(active) for active in ACTIVE_FILES):
            active_files.append(relative_path)
        elif str(relative_path) in ARCHIVE_CANDIDATES or item.name in ARCHIVE_CANDIDATES:
            archive_candidates.append(relative_path)
        elif any(str(relative_path).endswith(pattern.replace('*', '')) for pattern in DELETE_CANDIDATES):
            delete_candidates.append(relative_path)
        else:
            # Check if it's referenced in the new structure
            is_ref, ref_info = is_file_referenced(item, project_root)
            if is_ref:
                print(f"⚠️  {relative_path} - {ref_info}")
                active_files.append(relative_path)
            else:
                archive_candidates.append(relative_path)
    
    return active_files, archive_candidates, delete_candidates

def main():
    project_root = Path.cwd()
    
    print("🧹 Legacy File Cleanup Tool")
    print("=" * 50)
    
    # Analyze structure
    active_files, archive_candidates, delete_candidates = analyze_project_structure(project_root)
    
    print(f"\n📊 Analysis Results:")
    print(f"   Active files/dirs: {len(active_files)}")
    print(f"   Archive candidates: {len(archive_candidates)}")
    print(f"   Delete candidates: {len(delete_candidates)}")
    
    print(f"\n📁 Files/directories to keep active:")
    for file in sorted(active_files)[:10]:  # Show first 10
        print(f"   ✅ {file}")
    if len(active_files) > 10:
        print(f"   ... and {len(active_files) - 10} more")
    
    print(f"\n📦 Files/directories to archive:")
    for file in sorted(archive_candidates):
        size = "unknown"
        try:
            if (project_root / file).is_file():
                size = f"{(project_root / file).stat().st_size / 1024:.1f}KB"
            elif (project_root / file).is_dir():
                size = "directory"
        except:
            pass
        print(f"   📦 {file} ({size})")
    
    print(f"\n🗑️  Files/directories to delete:")
    for file in sorted(delete_candidates):
        print(f"   🗑️  {file}")
    
    if not archive_candidates and not delete_candidates:
        print("\n✅ No legacy files found to clean up!")
        return
    
    # Ask for confirmation
    print(f"\n⚠️  This will:")
    print(f"   - Archive {len(archive_candidates)} items to archive/cleanup_[timestamp]/")
    print(f"   - Delete {len(delete_candidates)} items permanently")
    
    response = input(f"\n🤔 Proceed with cleanup? [y/N]: ").strip().lower()
    
    if response == 'y':
        # Create archive directory
        archive_dir = create_archive_directory()
        print(f"\n📦 Created archive directory: {archive_dir}")
        
        # Archive files
        archived_count = 0
        for item in archive_candidates:
            src = project_root / item
            dst = archive_dir / item
            
            try:
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if src.is_dir():
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                        shutil.rmtree(src)
                    else:
                        shutil.copy2(src, dst)
                        src.unlink()
                    print(f"   📦 Archived: {item}")
                    archived_count += 1
            except Exception as e:
                print(f"   ❌ Failed to archive {item}: {e}")
        
        # Delete files
        deleted_count = 0
        for item in delete_candidates:
            src = project_root / item
            try:
                if src.exists():
                    if src.is_dir():
                        shutil.rmtree(src)
                    else:
                        src.unlink()
                    print(f"   🗑️  Deleted: {item}")
                    deleted_count += 1
            except Exception as e:
                print(f"   ❌ Failed to delete {item}: {e}")
        
        print(f"\n✅ Cleanup complete!")
        print(f"   📦 Archived: {archived_count} items")
        print(f"   🗑️  Deleted: {deleted_count} items")
        print(f"   📁 Archive location: {archive_dir}")
        
        # Create cleanup report
        report_file = archive_dir / "cleanup_report.txt"
        with open(report_file, 'w') as f:
            f.write(f"Legacy File Cleanup Report\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Project: trading_analysis\n\n")
            f.write(f"Archived Items ({archived_count}):\n")
            for item in archive_candidates:
                f.write(f"  - {item}\n")
            f.write(f"\nDeleted Items ({deleted_count}):\n")
            for item in delete_candidates:
                f.write(f"  - {item}\n")
        
        print(f"   📄 Report saved: {report_file}")
        
    else:
        print("❌ Cleanup cancelled.")

if __name__ == "__main__":
    main()
