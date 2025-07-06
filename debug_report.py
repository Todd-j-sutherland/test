#!/usr/bin/env python3
"""
Debug script to check what data is being passed to the report generator
"""

import sys
import os
sys.path.append('.')

from main import ASXBankTradingSystem
import json

def main():
    """Test the data flow"""
    
    # Create system
    system = ASXBankTradingSystem()
    
    # Analyze a single bank
    print("Analyzing CBA.AX...")
    result = system.analyze_bank('CBA.AX')
    
    print("\n=== ANALYSIS RESULT ===")
    print(json.dumps(result, indent=2, default=str))
    
    # Test with all banks
    print("\n\nAnalyzing all banks...")
    results = system.analyze_all_banks()
    
    print("\n=== ALL BANKS STRUCTURE ===")
    for symbol, data in results.items():
        print(f"\n{symbol}:")
        if 'error' in data:
            print(f"  ERROR: {data['error']}")
        else:
            print(f"  Keys: {list(data.keys())}")
            if 'technical_analysis' in data:
                print(f"  Technical keys: {list(data['technical_analysis'].keys())}")
            if 'fundamental_analysis' in data:
                print(f"  Fundamental keys: {list(data['fundamental_analysis'].keys())}")
                if 'metrics' in data['fundamental_analysis']:
                    metrics = data['fundamental_analysis']['metrics']
                    print(f"  PE Ratio: {metrics.get('pe_ratio', 'N/A')}")
                    print(f"  Dividend Yield: {metrics.get('dividend_yield', 'N/A')}")
            if 'risk_reward' in data:
                print(f"  Risk/Reward keys: {list(data['risk_reward'].keys())}")
                print(f"  Risk Score: {data['risk_reward'].get('risk_score', 'N/A')}")

if __name__ == "__main__":
    main()
