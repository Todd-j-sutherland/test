#!/usr/bin/env python3
"""
Main application entry point for Trading Analysis System
"""

import sys
import argparse
import logging
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))

from app.config.settings import Settings
from app.config.logging import setup_logging
from app.services.daily_manager import TradingSystemManager

def setup_cli():
    """Setup command line interface"""
    parser = argparse.ArgumentParser(
        description="Trading Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.main morning              # Run morning routine
  python -m app.main evening              # Run evening routine
  python -m app.main status               # Quick status check
  python -m app.main dashboard            # Launch dashboard
  python -m app.main news                 # Run news sentiment analysis
  python -m app.main --config custom.yml  # Use custom config
        """
    )
    
    parser.add_argument(
        'command',
        choices=['morning', 'evening', 'status', 'weekly', 'restart', 'test', 'dashboard', 'news'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in dry-run mode (no actual trading operations)'
    )
    
    return parser

def main():
    """Main application entry point"""
    parser = setup_cli()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else getattr(logging, args.log_level)
    setup_logging(level=log_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Trading Analysis System - Command: {args.command}")
    
    try:
        # Initialize system manager
        manager = TradingSystemManager(
            config_path=args.config,
            dry_run=args.dry_run
        )
        
        # Execute command
        if args.command == 'morning':
            manager.morning_routine()
        elif args.command == 'evening':
            manager.evening_routine()
        elif args.command == 'status':
            manager.quick_status()
        elif args.command == 'weekly':
            manager.weekly_maintenance()
        elif args.command == 'restart':
            manager.emergency_restart()
        elif args.command == 'test':
            manager.test_enhanced_features()
        elif args.command == 'dashboard':
            from app.dashboard.main import run_dashboard
            run_dashboard()
        elif args.command == 'news':
            manager.news_analysis()
        
        logger.info(f"Command '{args.command}' completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()
