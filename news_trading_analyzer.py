#!/usr/bin/env python3
"""
News Trading Analyzer - Primary Entry Point
Focuses on news sentiment analysis for trading decisions

Core Purpose: Analyze news sources and provide trading sentiment scores
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Core imports
from src.news_sentiment import NewsSentimentAnalyzer
from src.ml_trading_config import TRADING_CONFIGS
from config.settings import Settings
from src.trading_outcome_tracker import TradingOutcomeTracker

# Setup logging
def setup_logging(log_level='INFO'):
    """Setup logging configuration"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/news_trading_analyzer.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class NewsTradingAnalyzer:
    """
    Primary news trading analysis system
    
    Purpose: Analyze news sentiment to provide trading recommendations
    Features:
    - Multi-source news collection (RSS, Yahoo Finance, Google News, Reddit)
    - Advanced sentiment analysis (Traditional + Transformers + ML features)
    - Trading strategy recommendations
    - Risk-adjusted scoring
    """
    
    def __init__(self):
        logger.info("🚀 Initializing News Trading Analyzer...")
        self.settings = Settings()
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        
        # Initialize outcome tracker
        if hasattr(self.sentiment_analyzer, 'ml_pipeline'):
            self.outcome_tracker = TradingOutcomeTracker(
                self.sentiment_analyzer.ml_pipeline
            )
        else:
            self.outcome_tracker = None
            logger.warning("ML pipeline not available, outcome tracking disabled")
        
        # Default bank symbols to analyze
        self.bank_symbols = [
            'CBA.AX',  # Commonwealth Bank
            'WBC.AX',  # Westpac
            'ANZ.AX',  # ANZ Bank
            'NAB.AX',  # National Australia Bank
            'MQG.AX'   # Macquarie Group
        ]
        
        logger.info("✅ News Trading Analyzer initialized successfully")
    
    def analyze_single_bank(self, symbol: str, detailed: bool = False) -> dict:
        """
        Analyze news sentiment for a single bank
        
        Args:
            symbol: Bank symbol (e.g., 'CBA.AX')
            detailed: Whether to include detailed breakdown
            
        Returns:
            Dict with sentiment analysis and trading recommendation
        """
        logger.info(f"📊 Analyzing {symbol}...")
        
        try:
            # Get comprehensive sentiment analysis
            result = self.sentiment_analyzer.analyze_bank_sentiment(symbol)
            
            # Extract key metrics
            sentiment_score = result.get('overall_sentiment', 0)
            confidence = result.get('confidence', 0)
            news_count = result.get('news_count', 0)
            
            # Generate trading recommendation
            trading_recommendation = self._get_trading_recommendation(
                sentiment_score, confidence, news_count
            )
            
            # Prepare output
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'sentiment_score': float(sentiment_score),
                'confidence': float(confidence),
                'news_count': news_count,
                'trading_recommendation': trading_recommendation,
                'signal': self._get_trading_signal(sentiment_score, confidence)
            }
            
            if detailed:
                analysis['detailed_breakdown'] = {
                    'sentiment_components': result.get('sentiment_components', {}),
                    'recent_headlines': result.get('recent_headlines', []),
                    'significant_events': result.get('significant_events', {}),
                    'reddit_sentiment': result.get('reddit_sentiment', {}),
                    'ml_trading_details': result.get('ml_trading_details', {})
                }
            
            logger.info(f"✅ {symbol}: {analysis['signal']} "
                       f"(Score: {sentiment_score:.3f}, Confidence: {confidence:.3f})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def analyze_and_track(self, symbol: str) -> dict:
        """Analyze sentiment and track for ML training"""
        result = self.analyze_single_bank(symbol, detailed=True)
        
        # Record signal if it's actionable
        if self.outcome_tracker and result.get('signal') not in ['HOLD', None]:
            trade_id = self.outcome_tracker.record_signal(symbol, result)
            result['trade_id'] = trade_id
            logger.info(f"📝 Recorded trade signal: {trade_id}")
        
        return result
    
    def close_trade(self, trade_id: str, exit_price: float):
        """Close a trade and record outcome"""
        if self.outcome_tracker:
            exit_data = {
                'price': exit_price,
                'timestamp': datetime.now().isoformat()
            }
            self.outcome_tracker.close_trade(trade_id, exit_data)
            logger.info(f"🔚 Trade closed: {trade_id}")
        else:
            logger.warning("Outcome tracker not available")
    
    def analyze_all_banks(self, detailed: bool = False) -> dict:
        """
        Analyze all major Australian banks
        
        Args:
            detailed: Whether to include detailed breakdown for each bank
            
        Returns:
            Dict with analysis for all banks plus market overview
        """
        logger.info("🏦 Analyzing all major Australian banks...")
        
        results = {}
        all_scores = []
        
        for symbol in self.bank_symbols:
            analysis = self.analyze_single_bank(symbol, detailed)
            results[symbol] = analysis
            
            if 'sentiment_score' in analysis:
                all_scores.append(analysis['sentiment_score'])
        
        # Calculate market overview
        if all_scores:
            market_overview = {
                'average_sentiment': sum(all_scores) / len(all_scores),
                'most_bullish': max(results.items(), key=lambda x: x[1].get('sentiment_score', -999)),
                'most_bearish': min(results.items(), key=lambda x: x[1].get('sentiment_score', 999)),
                'high_confidence_count': sum(1 for r in results.values() 
                                           if r.get('confidence', 0) > 0.7),
                'total_analyzed': len(all_scores)
            }
        else:
            market_overview = {'error': 'No successful analyses'}
        
        return {
            'market_overview': market_overview,
            'individual_analysis': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_trading_recommendation(self, sentiment_score: float, 
                                  confidence: float, news_count: int) -> dict:
        """
        Generate trading strategy recommendation based on analysis
        
        Args:
            sentiment_score: Overall sentiment (-1 to 1)
            confidence: Analysis confidence (0 to 1)
            news_count: Number of news articles analyzed
            
        Returns:
            Dict with strategy recommendation and parameters
        """
        # Determine strategy based on sentiment and confidence
        if confidence > 0.7:
            if sentiment_score > 0.3:
                strategy_type = 'aggressive'
                action = 'BUY'
                reasoning = f"High confidence ({confidence:.2f}) positive sentiment ({sentiment_score:.2f})"
            elif sentiment_score < -0.3:
                strategy_type = 'conservative'
                action = 'SELL'
                reasoning = f"High confidence ({confidence:.2f}) negative sentiment ({sentiment_score:.2f})"
            else:
                strategy_type = 'moderate'
                action = 'HOLD'
                reasoning = f"High confidence ({confidence:.2f}) but neutral sentiment ({sentiment_score:.2f})"
        else:
            strategy_type = 'conservative'
            if sentiment_score > 0.1:
                action = 'WEAK_BUY'
            elif sentiment_score < -0.1:
                action = 'WEAK_SELL'
            else:
                action = 'HOLD'
            reasoning = f"Low confidence ({confidence:.2f}) in analysis"
        
        # Get strategy parameters
        config = TRADING_CONFIGS.get(strategy_type, TRADING_CONFIGS['moderate'])
        
        return {
            'action': action,
            'strategy_type': strategy_type,
            'reasoning': reasoning,
            'parameters': {
                'position_size_multiplier': config['position_size_multiplier'],
                'stop_loss_multiplier': config['stop_loss_multiplier'],
                'take_profit_multiplier': config['take_profit_multiplier'],
                'confidence_threshold': config['min_confidence']
            },
            'data_quality': {
                'news_articles': news_count,
                'confidence_level': confidence,
                'recommendation_strength': abs(sentiment_score) * confidence
            }
        }
    
    def _get_trading_signal(self, sentiment_score: float, confidence: float) -> str:
        """
        Get simple trading signal
        
        Returns:
            Signal string: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
        """
        strength = abs(sentiment_score) * confidence
        
        if sentiment_score > 0.3 and confidence > 0.7:
            return "STRONG_BUY"
        elif sentiment_score > 0.1 and confidence > 0.5:
            return "BUY"
        elif sentiment_score < -0.3 and confidence > 0.7:
            return "STRONG_SELL"
        elif sentiment_score < -0.1 and confidence > 0.5:
            return "SELL"
        else:
            return "HOLD"
    
    def export_analysis(self, analysis_result: dict, filename: str = None) -> str:
        """
        Export analysis results to JSON file
        
        Args:
            analysis_result: Analysis result from analyze_single_bank or analyze_all_banks
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"news_analysis_{timestamp}.json"
        
        # Ensure reports directory exists
        os.makedirs('reports', exist_ok=True)
        filepath = os.path.join('reports', filename)
        
        # Export with pretty formatting
        with open(filepath, 'w') as f:
            json.dump(analysis_result, f, indent=2, default=str)
        
        logger.info(f"📄 Analysis exported to: {filepath}")
        return filepath
    
    def get_enhanced_analysis(self, symbol: str) -> dict:
        """
        Get enhanced analysis including detailed keyword filtering insights
        
        Args:
            symbol: Bank symbol (e.g., 'CBA.AX')
            
        Returns:
            Dict with comprehensive analysis including filtering insights
        """
        logger.info(f"🔍 Running enhanced analysis for {symbol}...")
        
        try:
            # Get standard sentiment analysis
            standard_result = self.analyze_single_bank(symbol, detailed=True)
            
            # Get filtering insights
            filtering_summary = self.sentiment_analyzer.get_filtered_news_summary(symbol)
            
            # Combine results
            enhanced_result = standard_result.copy()
            enhanced_result['filtering_insights'] = filtering_summary
            
            # Add recommendation confidence based on filtering quality
            filtering_efficiency = filtering_summary.get('filtering_summary', {}).get('filtering_efficiency', 0)
            avg_relevance = filtering_summary.get('filtering_summary', {}).get('avg_relevance_score', 0)
            
            # Adjust confidence based on filtering quality
            original_confidence = enhanced_result.get('confidence', 0)
            filtering_boost = min(filtering_efficiency * avg_relevance * 0.2, 0.15)
            enhanced_result['enhanced_confidence'] = min(original_confidence + filtering_boost, 1.0)
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced analysis for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # ...existing code...

def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(description='News Trading Analyzer')
    parser.add_argument('--symbol', '-s', type=str, 
                       help='Analyze specific bank symbol (e.g., CBA.AX)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Analyze all major banks')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='Include detailed breakdown in results')
    parser.add_argument('--export', '-e', action='store_true',
                       help='Export results to JSON file')
    parser.add_argument('--enhanced', '-en', action='store_true',
                       help='Use enhanced keyword filtering and analysis')
    parser.add_argument('--filtering-test', '-ft', action='store_true',
                       help='Test the enhanced keyword filtering system')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level')
    
    args = parser.parse_args()
    
    # Setup logging with specified level
    global logger
    logger = setup_logging(args.log_level)
    
    # Initialize analyzer
    analyzer = NewsTradingAnalyzer()
    
    try:
        if args.filtering_test:
            # Test the enhanced filtering system
            print("\n🧪 Testing Enhanced Keyword Filtering System")
            print("=" * 60)
            
            from src.bank_keywords import BankNewsFilter
            filter_system = BankNewsFilter()
            
            test_titles = [
                "NAB announces record profit amid rising interest rates",
                "RBA holds cash rate steady at 4.35%",
                "Commonwealth Bank faces ASIC investigation over fees",
                "Westpac launches new digital banking platform",
                "Major scam warning for ANZ customers",
                "Tech stocks surge on Wall Street overnight"  # Should not match
            ]
            
            for title in test_titles:
                result = filter_system.is_relevant_banking_news(title)
                print(f"\nTitle: {title}")
                print(f"Relevant: {result['is_relevant']} (Score: {result['relevance_score']:.2f})")
                print(f"Categories: {', '.join(result['categories'])}")
                if result['matched_keywords']:
                    print(f"Keywords: {', '.join(result['matched_keywords'][:3])}")
            
            print(f"\n✅ Enhanced filtering system test complete!")
            return
        
        elif args.symbol:
            # Analyze single bank
            if args.enhanced:
                result = analyzer.get_enhanced_analysis(args.symbol)
                
                print(f"\n{'='*60}")
                print(f"ENHANCED NEWS TRADING ANALYSIS: {args.symbol}")
                print(f"{'='*60}")
                print(f"Sentiment Score: {result.get('sentiment_score', 'N/A'):.3f}")
                print(f"Standard Confidence: {result.get('confidence', 'N/A'):.3f}")
                print(f"Enhanced Confidence: {result.get('enhanced_confidence', 'N/A'):.3f}")
                print(f"Trading Signal: {result.get('signal', 'N/A')}")
                print(f"Recommendation: {result.get('trading_recommendation', {}).get('action', 'N/A')}")
                
                # Show filtering insights
                filtering = result.get('filtering_insights', {})
                if filtering and 'filtering_summary' in filtering:
                    fs = filtering['filtering_summary']
                    print(f"\n📊 Filtering Performance:")
                    print(f"  Articles Found: {fs.get('total_articles_found', 'N/A')}")
                    print(f"  Relevant Articles: {fs.get('relevant_articles', 'N/A')}")
                    print(f"  Filtering Efficiency: {fs.get('filtering_efficiency', 0):.1%}")
                    print(f"  Avg Relevance Score: {fs.get('avg_relevance_score', 0):.3f}")
                
                if filtering and 'high_priority_articles' in filtering:
                    print(f"\n🎯 Top Priority Articles:")
                    for i, article in enumerate(filtering['high_priority_articles'][:3], 1):
                        print(f"  {i}. {article['title'][:60]}...")
                        print(f"     Priority: {article['priority_score']:.3f} | Source: {article['source']}")
            else:
                result = analyzer.analyze_single_bank(args.symbol, args.detailed)
            
            print(f"\n{'='*60}")
            print(f"NEWS TRADING ANALYSIS: {args.symbol}")
            print(f"{'='*60}")
            print(f"Sentiment Score: {result.get('sentiment_score', 'N/A'):.3f}")
            print(f"Confidence: {result.get('confidence', 'N/A'):.3f}")
            print(f"Trading Signal: {result.get('signal', 'N/A')}")
            print(f"Recommendation: {result.get('trading_recommendation', {}).get('action', 'N/A')}")
            print(f"Strategy: {result.get('trading_recommendation', {}).get('strategy_type', 'N/A')}")
            print(f"News Articles: {result.get('news_count', 'N/A')}")
            
            if args.detailed:
                print(f"\nRecent Headlines:")
                headlines = result.get('detailed_breakdown', {}).get('recent_headlines', [])
                for i, headline in enumerate(headlines[:3], 1):
                    if headline:
                        print(f"  {i}. {headline}")
            
        elif args.all:
            # Analyze all banks
            result = analyzer.analyze_all_banks(args.detailed)
            
            print(f"\n{'='*60}")
            print(f"MARKET OVERVIEW - AUSTRALIAN BANKS")
            print(f"{'='*60}")
            
            market = result.get('market_overview', {})
            print(f"Average Sentiment: {market.get('average_sentiment', 'N/A'):.3f}")
            print(f"High Confidence Analyses: {market.get('high_confidence_count', 'N/A')}")
            
            most_bullish = market.get('most_bullish', ['N/A', {}])
            most_bearish = market.get('most_bearish', ['N/A', {}])
            print(f"Most Bullish: {most_bullish[0]} ({most_bullish[1].get('sentiment_score', 'N/A'):.3f})")
            print(f"Most Bearish: {most_bearish[0]} ({most_bearish[1].get('sentiment_score', 'N/A'):.3f})")
            
            print(f"\nIndividual Bank Analysis:")
            print("-" * 60)
            for symbol, analysis in result.get('individual_analysis', {}).items():
                signal = analysis.get('signal', 'N/A')
                score = analysis.get('sentiment_score', 'N/A')
                confidence = analysis.get('confidence', 'N/A')
                if isinstance(score, (int, float)) and isinstance(confidence, (int, float)):
                    print(f"{symbol:<8} | {signal:<12} | Score: {score:>6.3f} | Confidence: {confidence:>6.3f}")
                else:
                    print(f"{symbol:<8} | ERROR")
        
        else:
            # Default: quick analysis of CBA
            print("No specific symbol specified. Analyzing CBA.AX as example...")
            result = analyzer.analyze_single_bank('CBA.AX')
            
            print(f"\nQuick Analysis - CBA.AX:")
            print(f"Signal: {result.get('signal', 'N/A')}")
            print(f"Score: {result.get('sentiment_score', 'N/A'):.3f}")
            print(f"Confidence: {result.get('confidence', 'N/A'):.3f}")
            print(f"\nUse --help for more options")
        
        # Export if requested
        if args.export:
            filepath = analyzer.export_analysis(result)
            print(f"\n📄 Results exported to: {filepath}")
        
        print(f"\n{'='*60}")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
