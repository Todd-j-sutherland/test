# main.py - ASX Bank Trading Analysis System
"""
Free ASX Bank Trading Analysis System
Analyzes Australian banking stocks with risk/reward metrics, market sentiment,
and bullish/bearish predictions using only free APIs and data sources.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
import schedule
import time
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_feed import ASXDataFeed
from src.technical_analysis import TechnicalAnalyzer
from src.fundamental_analysis import FundamentalAnalyzer
from src.news_sentiment import NewsSentimentAnalyzer
from src.risk_calculator import RiskRewardCalculator
from src.market_predictor import MarketPredictor
from src.alert_system import AlertSystem
from src.report_generator import ReportGenerator
from config.settings import Settings

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ASXBankTradingSystem:
    """Main orchestrator for the ASX bank trading analysis system"""
    
    def __init__(self):
        self.settings = Settings()
        self.data_feed = ASXDataFeed()
        self.technical = TechnicalAnalyzer()
        self.fundamental = FundamentalAnalyzer()
        self.sentiment = NewsSentimentAnalyzer()
        self.risk_calc = RiskRewardCalculator()
        self.predictor = MarketPredictor()
        self.alert_system = AlertSystem()
        self.report_gen = ReportGenerator()
        
    def analyze_all_banks(self):
        """Analyze all banks in the watchlist"""
        logger.info("Starting analysis of all ASX banks...")
        
        results = {}
        
        for symbol in self.settings.BANK_SYMBOLS:
            try:
                logger.info(f"Analyzing {symbol}...")
                result = self.analyze_bank(symbol)
                results[symbol] = result
                
                # Small delay to avoid overwhelming APIs
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                results[symbol] = {'error': str(e)}
                
        return results
    
    def analyze_bank(self, symbol):
        """Analyze a single bank stock"""
        try:
            # Get market data
            market_data = self.data_feed.get_historical_data(symbol, period=self.settings.DEFAULT_ANALYSIS_PERIOD)
            if market_data is None or market_data.empty:
                return {'error': f'No data available for {symbol}'}
            
            # Technical analysis
            technical_signals = self.technical.analyze(symbol, market_data)
            
            # Fundamental analysis
            fundamental_metrics = self.fundamental.analyze(symbol)
            
            # News sentiment
            sentiment_score = self.sentiment.analyze_bank_sentiment(symbol)
            
            # Risk/reward calculation
            current_price = float(market_data['Close'].iloc[-1])
            risk_reward = self.risk_calc.calculate(symbol, current_price, technical_signals, fundamental_metrics)
            
            # Market prediction
            prediction = self.predictor.predict(
                symbol,
                technical_signals, 
                fundamental_metrics, 
                sentiment_score
            )
            
            # Compile results
            analysis_result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': float(market_data['Close'].iloc[-1]),
                'technical': technical_signals,
                'fundamental': fundamental_metrics,
                'sentiment': sentiment_score,
                'risk_reward': risk_reward,
                'prediction': prediction
            }
            
            # Check for alerts
            alerts = self.alert_system.check_alert_thresholds(analysis_result)
            if alerts:
                for alert in alerts:
                    self.alert_system.send_alert(alert)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in analyze_bank for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def generate_report(self, results):
        """Generate analysis report"""
        try:
            report_path = self.report_gen.generate_daily_report(results, {})
            logger.info(f"Report generated: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return None
    
    def start_scheduler(self):
        """Start the automated scheduler"""
        logger.info("Starting automated scheduler...")
        
        # Schedule daily analysis at 9:30 AM (after market open)
        schedule.every().monday.at("09:30").do(self.run_daily_analysis)
        schedule.every().tuesday.at("09:30").do(self.run_daily_analysis)
        schedule.every().wednesday.at("09:30").do(self.run_daily_analysis)
        schedule.every().thursday.at("09:30").do(self.run_daily_analysis)
        schedule.every().friday.at("09:30").do(self.run_daily_analysis)
        
        # Schedule afternoon check at 3:00 PM (before market close)
        schedule.every().monday.at("15:00").do(self.run_afternoon_check)
        schedule.every().tuesday.at("15:00").do(self.run_afternoon_check)
        schedule.every().wednesday.at("15:00").do(self.run_afternoon_check)
        schedule.every().thursday.at("15:00").do(self.run_afternoon_check)
        schedule.every().friday.at("15:00").do(self.run_afternoon_check)
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def run_daily_analysis(self):
        """Run the daily analysis routine"""
        logger.info("Running daily analysis...")
        results = self.analyze_all_banks()
        self.generate_report(results)
        
    def run_afternoon_check(self):
        """Run afternoon check for any urgent signals"""
        logger.info("Running afternoon check...")
        results = self.analyze_all_banks()
        
        # Only send alerts for strong signals
        for symbol, result in results.items():
            if 'prediction' in result:
                prediction = result['prediction']
                if prediction.get('signal_strength', 0) > 70:
                    alert_data = {
                        'symbol': symbol,
                        'alert_type': 'strong_signal',
                        'signal_strength': prediction.get('signal_strength', 0),
                        'data': result
                    }
                    self.alert_system.send_alert(alert_data)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ASX Bank Trading Analysis System')
    parser.add_argument('command', choices=['analyze', 'bank', 'schedule', 'report'], 
                       help='Command to execute')
    parser.add_argument('--symbol', type=str, help='Bank symbol to analyze (for bank command)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    system = ASXBankTradingSystem()
    
    try:
        if args.command == 'analyze':
            # Analyze all banks
            results = system.analyze_all_banks()
            report_path = system.generate_report(results)
            
            print("\n" + "="*50)
            print("ASX BANK ANALYSIS COMPLETE")
            print("="*50)
            
            for symbol, result in results.items():
                if 'error' in result:
                    print(f"{symbol}: ERROR - {result['error']}")
                else:
                    prediction = result.get('prediction', {})
                    signal = prediction.get('signal', 'UNKNOWN')
                    confidence = prediction.get('confidence', 0)
                    print(f"{symbol}: {signal} (Confidence: {confidence}%)")
            
            if report_path:
                print(f"\nDetailed report saved to: {report_path}")
                
        elif args.command == 'bank':
            if not args.symbol:
                print("Please specify a bank symbol with --symbol")
                return
                
            symbol = args.symbol.upper()
            if not symbol.endswith('.AX'):
                symbol += '.AX'
                
            result = system.analyze_bank(symbol)
            
            if 'error' in result:
                print(f"Error analyzing {symbol}: {result['error']}")
            else:
                print(f"\nAnalysis for {symbol}:")
                print(f"Current Price: ${result['current_price']:.2f}")
                
                prediction = result.get('prediction', {})
                print(f"Signal: {prediction.get('signal', 'UNKNOWN')}")
                print(f"Confidence: {prediction.get('confidence', 0)}%")
                
                risk_reward = result.get('risk_reward', {})
                print(f"Risk Score: {risk_reward.get('risk_score', 'N/A')}")
                print(f"Potential Reward: {risk_reward.get('potential_reward', 'N/A')}")
                
        elif args.command == 'schedule':
            print("Starting automated scheduler...")
            system.start_scheduler()
            
        elif args.command == 'report':
            results = system.analyze_all_banks()
            report_path = system.generate_report(results)
            print(f"Report generated: {report_path}")
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        logger.info("System shutdown requested")
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
