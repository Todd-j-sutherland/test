#!/usr/bin/env python3
"""
Automated cleanup script for ASX Trading Analysis
This script will remove unnecessary files and simplify the project structure
for a news sentiment-focused analysis system.
"""

import os
import shutil
import sys
from pathlib import Path
import argparse

def get_files_to_remove():
    """Get comprehensive list of files/directories to remove"""
    
    files_to_remove = [
        # Complex trading system components
        'src/trading_orchestrator.py',
        'src/advanced_risk_manager.py',
        'src/technical_analysis.py',
        'src/fundamental_analysis.py',
        'src/market_predictor.py',
        'src/data_validator.py',
        'src/risk_calculator.py',
        'src/news_impact_analyzer.py',
        
        # Multiple dashboard systems
        'trading_dashboard.py',
        'simple_dashboard.py',
        
        # Complex main systems
        'enhanced_main.py',
        'src/async_main.py',
        'run_trading_system.py',
        'main.py',  # Original complex main
        
        # Backtesting system
        'backtesting_system.py',
        
        # Demo and test files
        'demo_improvements.py',
        'test_improvements.py',
        'simple_enhanced_demo.py',
        
        # Complex alert system
        'src/alert_system.py',
        
        # Complex report generator
        'src/report_generator.py',
        
        # Multiple run scripts
        'run.sh',
        'run.bat',
        
        # Documentation for removed features
        'TRADING_ORCHESTRATOR_README.md',
        'DECISION_MAKING_GUIDE.md',
        'COMPLETE_METRICS_GUIDE.md',
        'PERFORMANCE_TEST_RESULTS.md',
        'DATA_FEED_FIX_SUMMARY.md',
        'PROJECT_CLEANUP_SUMMARY.md',
        'IMPROVEMENTS_README.md',
        'asx_improvement_plan.md',
        'NEWS_SENTIMENT_ANALYSIS.md',
        
        # Cleanup analysis script (no longer needed after cleanup)
        'cleanup_analysis.py',
        
        # Complex templates and tests
        'templates/',
        'tests/',
        
        # Cache and logs (can be regenerated)
        'data/cache/',
        'data/historical/',
        'data/impact_analysis/',
        'data/sentiment_history/',
        'logs/',
        'reports/',
        
        # Python cache
        '__pycache__/',
        'src/__pycache__/',
        'config/__pycache__/',
    ]
    
    return files_to_remove

def get_dependencies_to_remove():
    """Get list of dependencies that can be removed from requirements.txt"""
    
    # Dependencies that are no longer needed for sentiment-only analysis
    dependencies_to_remove = [
        'plotly',
        'dash',
        'dash-bootstrap-components',
        'dash-core-components',
        'dash-html-components',
        'dash-table',
        'flask',
        'flask-cors',
        'werkzeug',
        'jinja2',
        'markupsafe',
        'itsdangerous',
        'click',
        'blinker',
        'ta-lib',
        'scikit-learn',
        'joblib',
        'threadpoolctl',
        'scipy',
        'matplotlib',
        'cycler',
        'fonttools',
        'kiwisolver',
        'pillow',
        'pyparsing',
        'packaging',
        'contourpy',
        'aiohttp',
        'aiosignal',
        'async-timeout',
        'attrs',
        'frozenlist',
        'multidict',
        'yarl',
        'asyncio',
        'websockets',
        'APScheduler',
        'tzlocal',
        'pytz-deprecation-shim',
        'six',
        'uvloop',
        'schedule',
        'psutil',
        'memory-profiler',
        'line-profiler',
        'py-spy',
        'snakeviz',
        'streamlit',
        'altair',
        'gitpython',
        'pyarrow',
        'pydeck',
        'tornado',
        'tenacity',
        'toml',
        'typing-extensions',
        'watchdog',
        'pympler',
        'openpyxl',
        'xlsxwriter',
        'seaborn',
        'statsmodels',
        'patsy',
        'rpy2',
        'jupyter',
        'notebook',
        'ipykernel',
        'ipython',
        'ipywidgets',
        'bokeh',
        'holoviews',
        'panel',
        'param',
        'pyviz-comms',
        'markdown',
        'pygments',
        'colorama',
        'tabulate',
        'rich',
        'typer',
        'fastapi',
        'uvicorn',
        'pydantic',
        'starlette',
        'anyio',
        'sniffio',
        'h11',
        'httptools',
        'python-multipart',
        'watchfiles',
        'websockets',
        'email-validator',
        'dnspython',
        'orjson',
        'ujson',
        'httpx',
        'httpcore',
        'h2',
        'hpack',
        'hyperframe',
        'rfc3986',
        'sniffio',
        'socksio',
        'trio',
        'outcome',
        'sortedcontainers',
        'exceptiongroup',
        'trio-websocket',
        'wsproto',
        'flask-socketio',
        'python-socketio',
        'python-engineio',
        'bidict',
        'eventlet',
        'greenlet',
        'monotonic',
        'dnspython',
        'enum34',
        'futures',
        'gevent',
        'gevent-websocket',
        'gunicorn',
        'celery',
        'redis',
        'kombu',
        'billiard',
        'pytz',
        'vine',
        'amqp',
        'pymongo',
        'gridfs',
        'bson',
        'dnspython',
        'pymongo',
        'motor',
        'aiomongo',
        'mongoengine',
        'pymongo',
        'elasticsearch',
        'elasticsearch-dsl',
        'certifi',
        'urllib3',
        'requests-oauthlib',
        'oauthlib',
        'requests-toolbelt',
        'requests-cache',
        'requests-html',
        'pyquery',
        'cssselect',
        'lxml',
        'html5lib',
        'webencodings',
        'bleach',
        'tinycss2',
        'cssutils',
        'premailer',
        'cachetools',
        'google-auth',
        'google-auth-oauthlib',
        'google-auth-httplib2',
        'google-api-python-client',
        'google-cloud-storage',
        'google-cloud-core',
        'google-resumable-media',
        'googleapis-common-protos',
        'grpcio',
        'grpcio-status',
        'protobuf',
        'pyasn1',
        'pyasn1-modules',
        'rsa',
        'cachetools',
        'google-crc32c',
        'google-cloud-bigquery',
        'db-dtypes',
        'pydata-google-auth',
        'google-cloud-bigquery-storage',
        'grpcio-gcp',
        'google-cloud-bigtable',
        'grpc-google-iam-v1',
        'google-cloud-datastore',
        'google-cloud-firestore',
        'google-cloud-logging',
        'google-cloud-monitoring',
        'google-cloud-pubsub',
        'google-cloud-storage',
        'google-cloud-tasks',
        'google-cloud-translate',
        'google-cloud-vision',
        'google-cloud-speech',
        'google-cloud-texttospeech',
        'google-cloud-dialogflow',
        'google-cloud-videointelligence',
        'google-cloud-automl',
        'google-cloud-language',
        'google-cloud-dataflow',
        'apache-beam',
        'avro-python3',
        'fastavro',
        'hdfs3',
        'pyarrow',
        'fastparquet',
        'dask',
        'distributed',
        'cloudpickle',
        'msgpack',
        'psutil',
        'sortedcontainers',
        'tblib',
        'tornado',
        'zict',
        'locket',
        'partd',
        'toolz',
        'cytoolz',
        'heapdict',
        'click',
        'pyyaml',
        'jinja2',
        'fsspec',
        'aiofiles',
        'aiofiles'
    ]
    
    return dependencies_to_remove

def backup_important_files():
    """Create backups of important files before cleanup"""
    
    backup_dir = Path('backup_before_cleanup')
    backup_dir.mkdir(exist_ok=True)
    
    important_files = [
        'requirements.txt',
        'README.md',
        'config/settings.py',
        'src/news_sentiment.py',
        'src/data_feed.py',
        'simple_news_analysis.py'
    ]
    
    print("Creating backups of important files...")
    for file_path in important_files:
        if Path(file_path).exists():
            shutil.copy2(file_path, backup_dir / Path(file_path).name)
            print(f"  ✓ Backed up {file_path}")

def remove_files_and_directories(dry_run=True):
    """Remove files and directories"""
    
    files_to_remove = get_files_to_remove()
    
    print(f"\n{'DRY RUN - ' if dry_run else ''}Removing files and directories...")
    
    removed_count = 0
    for item in files_to_remove:
        path = Path(item)
        
        if path.exists():
            if dry_run:
                if path.is_dir():
                    print(f"  Would remove directory: {item}")
                else:
                    print(f"  Would remove file: {item}")
            else:
                try:
                    if path.is_dir():
                        shutil.rmtree(path)
                        print(f"  ✓ Removed directory: {item}")
                    else:
                        path.unlink()
                        print(f"  ✓ Removed file: {item}")
                    removed_count += 1
                except Exception as e:
                    print(f"  ✗ Error removing {item}: {e}")
        else:
            if dry_run:
                print(f"  (Not found): {item}")
    
    if not dry_run:
        print(f"\nRemoved {removed_count} files/directories")
    
    return removed_count

def clean_requirements():
    """Create a new clean requirements.txt"""
    
    print("\nCreating clean requirements.txt...")
    
    # Use our minimal requirements as the base
    if Path('requirements_simple.txt').exists():
        shutil.copy2('requirements_simple.txt', 'requirements.txt')
        print("  ✓ Replaced requirements.txt with simplified version")
    else:
        print("  ✗ requirements_simple.txt not found")

def create_new_readme():
    """Create a new simplified README"""
    
    readme_content = """# ASX Bank News Sentiment Analysis

A streamlined tool for analyzing news sentiment of ASX bank stocks using advanced NLP techniques.

## Features

- **News Sentiment Analysis**: Advanced sentiment analysis using transformers, VADER, and TextBlob
- **Multi-source Data**: Analyzes news from multiple sources including financial news sites
- **Reddit Sentiment**: Tracks sentiment from Australian finance subreddits
- **Real-time Analysis**: Get current sentiment scores and price context
- **Clean Reports**: Generate clear, actionable sentiment reports

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Analysis**:
   ```bash
   # Analyze a single bank
   python simple_news_analysis.py --symbol CBA
   
   # Analyze all banks
   python simple_news_analysis.py
   
   # Save results to file
   python simple_news_analysis.py --save
   ```

## Configuration

Edit `config/settings.py` to customize:
- Bank symbols to analyze
- News sources
- Analysis parameters
- Output settings

## Dependencies

Core dependencies (see `requirements.txt`):
- `transformers` - Advanced NLP models
- `textblob` - Sentiment analysis
- `vaderSentiment` - Financial sentiment analysis
- `yfinance` - Stock price data
- `requests` - Web scraping
- `beautifulsoup4` - HTML parsing
- `praw` - Reddit API

## Output

The system generates:
- Console output with sentiment scores and recommendations
- JSON files with detailed analysis (when using `--save`)
- Logs in `logs/news_sentiment.log`

## Architecture

```
src/
├── news_sentiment.py    # Main sentiment analysis engine
├── data_feed.py         # Data collection and feeds
└── __init__.py

config/
├── settings.py          # Configuration
└── __init__.py

simple_news_analysis.py  # Main entry point
requirements.txt         # Dependencies
```

## Analysis Methods

The system uses a multi-layered approach:

1. **Enhanced Sentiment Analysis**: Combines multiple methods for accuracy
2. **Financial Keyword Scoring**: Weighted analysis of financial terms
3. **Reddit Sentiment**: Community sentiment from Australian finance forums
4. **Event Detection**: Identifies significant corporate events
5. **Confidence Scoring**: Provides confidence levels for recommendations

## Contributing

This is a simplified, focused tool. For feature requests or improvements:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

See LICENSE file for details.
"""
    
    print("\nCreating new README.md...")
    with open('README.md', 'w') as f:
        f.write(readme_content)
    print("  ✓ Created new simplified README.md")

def main():
    """Main cleanup function"""
    
    parser = argparse.ArgumentParser(description='Cleanup ASX Trading Analysis project')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be removed without actually removing')
    parser.add_argument('--skip-backup', action='store_true', help='Skip creating backups')
    
    args = parser.parse_args()
    
    print("🧹 ASX Trading Analysis Cleanup Tool")
    print("=" * 50)
    
    if not args.dry_run:
        confirm = input("\nThis will permanently remove files. Are you sure? (y/N): ")
        if confirm.lower() != 'y':
            print("Cleanup cancelled.")
            return
    
    # Create backups
    if not args.skip_backup and not args.dry_run:
        backup_important_files()
    
    # Remove files and directories
    remove_files_and_directories(dry_run=args.dry_run)
    
    if not args.dry_run:
        # Clean requirements
        clean_requirements()
        
        # Create new README
        create_new_readme()
        
        print("\n🎉 Cleanup complete!")
        print("\nNext steps:")
        print("1. Review the changes")
        print("2. Test the simplified system: python simple_news_analysis.py --symbol CBA")
        print("3. Update any remaining configuration if needed")
        print("4. Remove the backup_before_cleanup directory when satisfied")
    else:
        print(f"\n📋 Dry run complete!")
        print("Run without --dry-run to perform actual cleanup")

if __name__ == "__main__":
    main()
