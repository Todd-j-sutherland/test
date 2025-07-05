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
        self.alerts = AlertSystem()
        self.reporter = ReportGenerator()
        
        # Ensure directories exist
        Path("logs").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
        Path("data/cache").mkdir(parents=True, exist_ok=True)
        
    def analyze_single_bank(self, symbol):
        """Complete analysis for a single bank"""
        logger.info(f"Starting analysis for {symbol}")
        
        try:
            # 1. Fetch current and historical data
            current_data = self.data_feed.get_current_data(symbol)
            historical_data = self.data_feed.get_historical_data(symbol, period="3mo")
            
            # 2. Technical Analysis
            tech_analysis = self.technical.analyze(symbol, historical_data)
            
            # 3. Fundamental Analysis
            fundamental = self.fundamental.analyze(symbol)
            
            # 4. News Sentiment
            sentiment = self.sentiment.analyze_bank_sentiment(symbol)
            
            # 5. Risk/Reward Calculation
            risk_reward = self.risk_calc.calculate(
                symbol=symbol,
                current_price=current_data['price'],
                technical=tech_analysis,
                fundamental=fundamental
            )
            
            # 6. Market Prediction
            prediction = self.predictor.predict(
                symbol=symbol,
                technical=tech_analysis,
                fundamental=fundamental,
                sentiment=sentiment
            )
            
            # Compile results
            analysis_result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_data['price'],
                'technical_analysis': tech_analysis,
                'fundamental_analysis': fundamental,
                'sentiment_analysis': sentiment,
                'risk_reward': risk_reward,
                'prediction': prediction,
                'recommendation': self._generate_recommendation(
                    tech_analysis, fundamental, sentiment, risk_reward, prediction
                )
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return None
    
    def _generate_recommendation(self, technical, fundamental, sentiment, risk_reward, prediction):
        """Generate trading recommendation based on all analyses"""
        
        # Score each component
        scores = {
            'technical': technical['overall_signal'],
            'fundamental': fundamental['score'],
            'sentiment': sentiment['overall_sentiment'],
            'risk_reward': risk_reward['risk_reward_ratio'],
            'prediction': prediction['confidence']
        }
        
        # Weight the scores
        weights = {
            'technical': 0.3,
            'fundamental': 0.2,
            'sentiment': 0.15,
            'risk_reward': 0.2,
            'prediction': 0.15
        }
        
        # Calculate weighted score (-100 to 100)
        weighted_score = sum(scores[k] * weights[k] for k in scores)
        
        # Generate recommendation
        if weighted_score > 50:
            action = "STRONG BUY"
            confidence = "HIGH"
        elif weighted_score > 20:
            action = "BUY"
            confidence = "MEDIUM"
        elif weighted_score > -20:
            action = "HOLD"
            confidence = "MEDIUM"
        elif weighted_score > -50:
            action = "SELL"
            confidence = "MEDIUM"
        else:
            action = "STRONG SELL"
            confidence = "HIGH"
        
        return {
            'action': action,
            'confidence': confidence,
            'score': weighted_score,
            'reasons': self._get_recommendation_reasons(scores)
        }
    
    def _get_recommendation_reasons(self, scores):
        """Generate human-readable reasons for recommendation"""
        reasons = []
        
        if scores['technical'] > 50:
            reasons.append("Strong technical indicators (breakout/momentum)")
        elif scores['technical'] < -50:
            reasons.append("Weak technical indicators (breakdown/oversold)")
        
        if scores['fundamental'] > 50:
            reasons.append("Excellent fundamentals (undervalued)")
        elif scores['fundamental'] < -50:
            reasons.append("Poor fundamentals (overvalued)")
        
        if scores['sentiment'] > 0.5:
            reasons.append("Positive market sentiment")
        elif scores['sentiment'] < -0.5:
            reasons.append("Negative market sentiment")
        
        if scores['risk_reward'] > 3:
            reasons.append("Excellent risk/reward ratio")
        elif scores['risk_reward'] < 1:
            reasons.append("Poor risk/reward ratio")
        
        return reasons
    
    def run_market_analysis(self):
        """Analyze overall market conditions"""
        logger.info("Running market-wide analysis")
        
        # Analyze ASX 200 (XJO)
        market_data = self.data_feed.get_market_data()
        
        # Analyze sector performance
        sector_analysis = self.predictor.analyze_banking_sector()
        
        # RBA and economic indicators
        economic_indicators = self.fundamental.get_economic_indicators()
        
        return {
            'market_trend': market_data['trend'],
            'sector_performance': sector_analysis,
            'economic_indicators': economic_indicators,
            'market_sentiment': self.sentiment.get_market_sentiment()
        }
    
    def run_complete_analysis(self):
        """Run complete analysis for all banks"""
        logger.info("Starting complete analysis run")
        
        # Get market analysis first
        market_analysis = self.run_market_analysis()
        
        # Analyze each bank
        results = {}
        for symbol in self.settings.BANK_SYMBOLS:
            analysis = self.analyze_single_bank(symbol)
            if analysis:
                results[symbol] = analysis
        
        # Generate report
        report = self.reporter.generate_daily_report(results, market_analysis)
        
        # Send alerts for significant findings
        self._check_and_send_alerts(results, market_analysis)
        
        # Save results
        self._save_results(results, market_analysis)
        
        return results, market_analysis
    
    def _check_and_send_alerts(self, results, market_analysis):
        """Check for alert conditions and send notifications"""
        alerts_to_send = []
        
        for symbol, analysis in results.items():
            # Check for strong signals
            if analysis['recommendation']['action'] in ['STRONG BUY', 'STRONG SELL']:
                alerts_to_send.append({
                    'type': 'SIGNAL',
                    'symbol': symbol,
                    'action': analysis['recommendation']['action'],
                    'price': analysis['current_price'],
                    'reasons': analysis['recommendation']['reasons']
                })
            
            # Check for risk alerts
            if analysis['risk_reward']['risk_score'] > 80:
                alerts_to_send.append({
                    'type': 'RISK',
                    'symbol': symbol,
                    'message': f"High risk detected for {symbol}"
                })
        
        # Send alerts
        for alert in alerts_to_send:
            self.alerts.send_alert(alert)
    
    def _save_results(self, results, market_analysis):
        """Save analysis results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/analysis_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'market_analysis': market_analysis,
                'bank_analysis': results
            }, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")
    
    def start_scheduler(self):
        """Start the automated scheduler"""
        logger.info("Starting automated scheduler")
        
        # Schedule tasks
        schedule.every().day.at("08:30").do(self.run_morning_analysis)
        schedule.every().day.at("10:15").do(self.run_market_open_analysis)
        schedule.every().day.at("15:45").do(self.run_closing_analysis)
        schedule.every().monday.at("09:00").do(self.run_weekly_analysis)
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def run_morning_analysis(self):
        """Pre-market analysis"""
        logger.info("Running morning pre-market analysis")
        
        # Check overnight news
        overnight_news = self.sentiment.get_overnight_news()
        
        # Check US market impact
        us_impact = self.predictor.analyze_us_market_impact()
        
        # Generate morning report
        self.reporter.generate_morning_brief(overnight_news, us_impact)
    
    def run_market_open_analysis(self):
        """Analysis after market open"""
        logger.info("Running market open analysis")
        self.run_complete_analysis()
    
    def run_closing_analysis(self):
        """End of day analysis"""
        logger.info("Running closing analysis")
        
        # Get closing data
        results, market = self.run_complete_analysis()
        
        # Generate end of day report
        self.reporter.generate_eod_report(results, market)
    
    def run_weekly_analysis(self):
        """Weekly comprehensive analysis"""
        logger.info("Running weekly analysis")
        
        # Get weekly data
        weekly_data = {}
        for symbol in self.settings.BANK_SYMBOLS:
            weekly_data[symbol] = self.data_feed.get_historical_data(symbol, period="1mo")
        
        # Generate weekly report
        self.reporter.generate_weekly_report(weekly_data)

def main():
    """Main entry point"""
    system = ASXBankTradingSystem()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "analyze":
            # Run single analysis
            results, market = system.run_complete_analysis()
            print("\n=== Analysis Complete ===")
            print(json.dumps(results, indent=2, default=str))
        
        elif sys.argv[1] == "schedule":
            # Start scheduler
            system.start_scheduler()
        
        elif sys.argv[1] == "bank" and len(sys.argv) > 2:
            # Analyze specific bank
            symbol = sys.argv[2].upper()
            if not symbol.endswith('.AX'):
                symbol += '.AX'
            result = system.analyze_single_bank(symbol)
            print(json.dumps(result, indent=2, default=str))
        
        else:
            print("Usage:")
            print("  python main.py analyze     # Run complete analysis")
            print("  python main.py schedule    # Start automated scheduler")
            print("  python main.py bank CBA    # Analyze specific bank")
    
    else:
        # Default: run analysis
        results, market = system.run_complete_analysis()
        print("\n=== Analysis Complete ===")
        print("Check the reports folder for detailed analysis")

if __name__ == "__main__":
    main()
