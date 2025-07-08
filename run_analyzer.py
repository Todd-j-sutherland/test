#!/usr/bin/env python3
"""
News Trading Analyzer - Simple Runner
Run news sentiment analysis without transformer downloads
"""

import os
import sys

# Set environment variable to skip transformers
os.environ['SKIP_TRANSFORMERS'] = '1'

# Import and run the main analyzer
if __name__ == "__main__":
    from news_trading_analyzer import main
    main()
