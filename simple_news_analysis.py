#!/usr/bin/env python3
"""
Simplified ASX Bank News Sentiment Analysis
Focus on news sentiment analysis using transformers for better accuracy
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.news_sentiment import NewsSentimentAnalyzer
from src.data_feed import ASXDataFeed
from config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/news_sentiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleNewsAnalyzer:
    """Simplified news sentiment analyzer focused on transformers-based analysis"""
    
    def __init__(self):
        self.settings = Settings()
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        self.data_feed = ASXDataFeed()
        
    def analyze_bank_sentiment(self, symbol: str) -> dict:
        """Analyze sentiment for a specific bank with current price context"""
        
        logger.info(f"Analyzing sentiment for {symbol}...")
        
        # Get current price for context
        try:
            market_data = self.data_feed.get_historical_data(symbol, period='1d')
            if market_data is not None and not market_data.empty:
                current_price = float(market_data['Close'].iloc[-1])
                price_change = ((current_price - float(market_data['Close'].iloc[-2])) / float(market_data['Close'].iloc[-2])) * 100 if len(market_data) > 1 else 0
            else:
                current_price = None
                price_change = 0
        except Exception as e:
            logger.warning(f"Could not get price data for {symbol}: {e}")
            current_price = None
            price_change = 0
        
        # Analyze sentiment
        sentiment_result = self.sentiment_analyzer.analyze_bank_sentiment(symbol)
        
        # Simplify the result
        simplified_result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'current_price': current_price,
            'price_change_pct': price_change,
            'sentiment_score': sentiment_result.get('overall_sentiment', 0),
            'sentiment_confidence': sentiment_result.get('confidence', 0),
            'news_count': sentiment_result.get('news_count', 0),
            'sentiment_components': sentiment_result.get('sentiment_components', {}),
            'analysis_method': sentiment_result.get('sentiment_scores', {}).get('analysis_method', 'unknown'),
            'recent_headlines': sentiment_result.get('recent_headlines', [])[:3],  # Top 3 headlines
            'recommendation': self._generate_simple_recommendation(sentiment_result)
        }
        
        return simplified_result
    
    def analyze_all_banks(self) -> dict:
        """Analyze sentiment for all banks"""
        
        logger.info("Starting sentiment analysis for all ASX banks...")
        
        results = {}
        
        for symbol in self.settings.BANK_SYMBOLS:
            try:
                result = self.analyze_bank_sentiment(symbol)
                results[symbol] = result
                
                # Short delay to be respectful to data sources
                import time
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def _generate_simple_recommendation(self, sentiment_result: dict) -> str:
        """Generate a simple recommendation based on sentiment"""
        
        sentiment_score = sentiment_result.get('overall_sentiment', 0)
        confidence = sentiment_result.get('confidence', 0)
        news_count = sentiment_result.get('news_count', 0)
        
        # Low confidence or no news
        if confidence < 0.3 or news_count < 2:
            return "HOLD - Insufficient data for confident recommendation"
        
        # Strong sentiment with good confidence
        if sentiment_score > 0.3 and confidence > 0.6:
            return "BUY - Strong positive sentiment detected"
        elif sentiment_score > 0.1 and confidence > 0.5:
            return "WEAK BUY - Moderate positive sentiment"
        elif sentiment_score < -0.3 and confidence > 0.6:
            return "SELL - Strong negative sentiment detected"
        elif sentiment_score < -0.1 and confidence > 0.5:
            return "WEAK SELL - Moderate negative sentiment"
        else:
            return "HOLD - Neutral sentiment"
    
    def print_analysis_summary(self, results: dict):
        """Print a clean summary of the analysis"""
        
        print("\n" + "="*80)
        print("🏛️  ASX BANK NEWS SENTIMENT ANALYSIS")
        print("="*80)
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔍 Analysis Method: Transformers-based sentiment analysis")
        print(f"📊 Banks Analyzed: {len(results)}")
        
        print("\n📈 INDIVIDUAL BANK ANALYSIS:")
        print("-" * 50)
        
        for symbol, result in results.items():
            if 'error' in result:
                print(f"❌ {symbol}: ERROR - {result['error']}")
                continue
            
            price = result.get('current_price')
            price_change = result.get('price_change_pct', 0)
            sentiment_score = result.get('sentiment_score', 0)
            confidence = result.get('sentiment_confidence', 0)
            news_count = result.get('news_count', 0)
            method = result.get('analysis_method', 'unknown')
            recommendation = result.get('recommendation', 'HOLD')
            
            # Format price display
            price_display = f"${price:.2f}" if price else "N/A"
            change_display = f"({price_change:+.1f}%)" if price else ""
            
            # Sentiment emoji
            if sentiment_score > 0.2:
                sentiment_emoji = "😊"
            elif sentiment_score > 0:
                sentiment_emoji = "🙂"
            elif sentiment_score < -0.2:
                sentiment_emoji = "😟"
            elif sentiment_score < 0:
                sentiment_emoji = "😐"
            else:
                sentiment_emoji = "😐"
            
            print(f"{sentiment_emoji} {symbol}: {price_display} {change_display}")
            print(f"    📰 News: {news_count} articles | Method: {method.upper()}")
            print(f"    💭 Sentiment: {sentiment_score:.3f} (Confidence: {confidence:.1%})")
            print(f"    🎯 Recommendation: {recommendation}")
            
            # Show recent headlines
            headlines = result.get('recent_headlines', [])
            if headlines:
                print(f"    📑 Recent Headlines:")
                for i, headline in enumerate(headlines[:2], 1):
                    print(f"        {i}. {headline[:60]}...")
            print()
        
        # Summary statistics
        valid_results = [r for r in results.values() if 'error' not in r]
        if valid_results:
            avg_sentiment = sum(r.get('sentiment_score', 0) for r in valid_results) / len(valid_results)
            total_news = sum(r.get('news_count', 0) for r in valid_results)
            
            print("📊 SUMMARY STATISTICS:")
            print("-" * 50)
            print(f"📈 Average Sentiment: {avg_sentiment:.3f}")
            print(f"📰 Total News Articles: {total_news}")
            print(f"🎯 Positive Sentiment: {sum(1 for r in valid_results if r.get('sentiment_score', 0) > 0.1)}/{len(valid_results)} banks")
            print(f"🎯 Negative Sentiment: {sum(1 for r in valid_results if r.get('sentiment_score', 0) < -0.1)}/{len(valid_results)} banks")
        
        print("\n" + "="*80)
        print("✅ Analysis complete!")
        print("="*80)
    
    def save_results(self, results: dict, filename: str = None):
        """Save results to JSON file"""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"sentiment_analysis_{timestamp}.json"
        
        output_path = Path("reports") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
        return output_path


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='ASX Bank News Sentiment Analysis')
    parser.add_argument('--symbol', type=str, help='Analyze specific bank symbol (e.g., CBA.AX)')
    parser.add_argument('--save', action='store_true', help='Save results to JSON file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure directories exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    analyzer = SimpleNewsAnalyzer()
    
    try:
        if args.symbol:
            # Analyze single bank
            result = analyzer.analyze_bank_sentiment(args.symbol.upper())
            results = {args.symbol.upper(): result}
        else:
            # Analyze all banks
            results = analyzer.analyze_all_banks()
        
        # Print summary
        analyzer.print_analysis_summary(results)
        
        # Save results if requested
        if args.save:
            analyzer.save_results(results)
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
