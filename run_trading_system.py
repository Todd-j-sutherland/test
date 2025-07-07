#!/usr/bin/env python3
# run_trading_system.py
"""
Command-line interface for the ASX Bank Trading System
Provides easy access to all system components.
"""

import asyncio
import argparse
import sys
import os
import signal
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.trading_orchestrator import TradingOrchestrator
from enhanced_main import main as run_analysis
from backtesting_system import main as run_backtest

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n🛑 Received interrupt signal. Shutting down gracefully...")
    sys.exit(0)

def print_banner():
    """Print system banner"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║           🏛️  ASX BANK TRADING SYSTEM v2.0                  ║
║                                                              ║
║  Advanced algorithmic trading system for ASX bank stocks    ║
║  with real-time analysis, risk management, and automation   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

async def run_live_trading(args):
    """Run the live trading orchestrator"""
    print("🚀 Starting Live Trading System...")
    print(f"📊 Mode: {'Paper Trading' if args.paper else 'Live Trading'}")
    print(f"⏰ Analysis Interval: {args.interval} seconds")
    
    # Create orchestrator
    orchestrator = TradingOrchestrator(paper_trading=args.paper)
    
    # Set parameters
    orchestrator.min_confidence = args.min_confidence
    orchestrator.max_position_size = args.max_position
    orchestrator.analysis_interval = args.interval
    
    try:
        # Start trading
        await orchestrator.start_trading()
    except KeyboardInterrupt:
        print("\n🛑 Stopping trading system...")
    finally:
        await orchestrator.stop_trading()

def run_dashboard(args):
    """Run the web dashboard"""
    print("🌐 Starting Trading Dashboard...")
    print(f"📊 Access at: http://localhost:{args.port}")
    
    # Try Flask dashboard first (more reliable)
    try:
        import subprocess
        subprocess.run([
            sys.executable, "simple_dashboard.py"
        ])
    except ImportError:
        # Fallback to Streamlit if Flask fails
        try:
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                "trading_dashboard.py", 
                "--server.port", str(args.port),
                "--server.address", "0.0.0.0"
            ])
        except ImportError:
            print("❌ Dashboard dependencies not installed.")
            print("Please install with: pip install flask streamlit")
            sys.exit(1)

def run_single_analysis(args):
    """Run a single analysis cycle"""
    print("🔍 Running Single Analysis...")
    
    # Import and run the enhanced analysis
    import enhanced_main
    enhanced_main.main()

def run_backtest_analysis(args):
    """Run backtesting analysis"""
    print("📈 Running Backtesting Analysis...")
    
    # Import and run backtesting
    import backtesting_system
    backtesting_system.main()

def show_status():
    """Show system status"""
    print("📊 System Status:")
    print("─" * 50)
    
    # Check if components are available
    components = {
        "Enhanced Analysis": "enhanced_main.py",
        "Backtesting System": "backtesting_system.py",
        "Trading Orchestrator": "src/trading_orchestrator.py",
        "Dashboard": "trading_dashboard.py"
    }
    
    for name, file_path in components.items():
        if os.path.exists(file_path):
            print(f"✅ {name}: Available")
        else:
            print(f"❌ {name}: Not found")
    
    # Check dependencies
    print("\n🔧 Dependencies:")
    print("─" * 50)
    
    required_packages = [
        "pandas", "numpy", "yfinance", "plotly", 
        "streamlit", "asyncio", "aiohttp"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: Installed")
        except ImportError:
            print(f"❌ {package}: Not installed")

def main():
    """Main command-line interface"""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Print banner
    print_banner()
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="ASX Bank Trading System - Command Line Interface",
        epilog="Example: python run_trading_system.py trade --paper --interval 300"
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Trade command
    trade_parser = subparsers.add_parser('trade', help='Start live trading')
    trade_parser.add_argument('--paper', action='store_true', 
                             help='Use paper trading mode (default: True)')
    trade_parser.add_argument('--live', action='store_true',
                             help='Use live trading mode (requires broker connection)')
    trade_parser.add_argument('--interval', type=int, default=300,
                             help='Analysis interval in seconds (default: 300)')
    trade_parser.add_argument('--min-confidence', type=float, default=0.3,
                             help='Minimum confidence for trades (default: 0.3)')
    trade_parser.add_argument('--max-position', type=float, default=0.1,
                             help='Maximum position size (default: 0.1)')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Start web dashboard')
    dashboard_parser.add_argument('--port', type=int, default=8501,
                                 help='Port for web dashboard (default: 8501)')
    
    # Analysis command
    analysis_parser = subparsers.add_parser('analyze', help='Run single analysis')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'trade':
        # Override paper trading if live is specified
        if args.live:
            args.paper = False
            print("⚠️  WARNING: Live trading mode selected!")
            print("⚠️  This will execute real trades with real money!")
            confirm = input("Are you sure you want to continue? (yes/no): ")
            if confirm.lower() != 'yes':
                print("Trading cancelled.")
                return
        else:
            args.paper = True
        
        # Run live trading
        asyncio.run(run_live_trading(args))
    
    elif args.command == 'dashboard':
        # Run dashboard
        run_dashboard(args)
    
    elif args.command == 'analyze':
        # Run single analysis
        run_single_analysis(args)
    
    elif args.command == 'backtest':
        # Run backtesting
        run_backtest_analysis(args)
    
    elif args.command == 'status':
        # Show status
        show_status()
    
    else:
        # Show help if no command specified
        parser.print_help()
        print("\n🎯 Quick Start Examples:")
        print("─" * 50)
        print("  python run_trading_system.py analyze       # Run single analysis")
        print("  python run_trading_system.py backtest      # Run backtesting")
        print("  python run_trading_system.py trade --paper # Start paper trading")
        print("  python run_trading_system.py dashboard     # Start web dashboard")
        print("  python run_trading_system.py status        # Check system status")

if __name__ == "__main__":
    main()
