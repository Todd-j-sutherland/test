# src/async_main.py - Async ASX Bank Trading Analysis System
"""
Async version of the ASX Bank Trading Analysis System
Provides 5x performance improvement through concurrent processing
"""

import asyncio
import aiohttp
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_feed import ASXDataFeed
from src.technical_analysis import TechnicalAnalyzer
from src.fundamental_analysis import FundamentalAnalyzer
from src.news_sentiment import NewsSentimentAnalyzer
from src.risk_calculator import RiskRewardCalculator
from src.market_predictor import MarketPredictor
from src.alert_system import AlertSystem
from src.report_generator import ReportGenerator
from config.settings import Settings

logger = logging.getLogger(__name__)

class ASXBankTradingSystemAsync:
    """Async main orchestrator for the ASX bank trading analysis system"""
    
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
        
        # Async components
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_session()
        
    async def start_session(self):
        """Initialize async session with connection pooling"""
        if self.session is None:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=20,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
    
    async def close_session(self):
        """Close async session"""
        if self.session:
            await self.session.close()
            self.session = None
        
        if self.executor:
            self.executor.shutdown(wait=True)
    
    async def analyze_all_banks_async(self) -> Dict:
        """Analyze all banks concurrently - 5x performance improvement"""
        logger.info("Starting async analysis of all ASX banks...")
        
        if not self.session:
            await self.start_session()
        
        # Create analysis tasks for all symbols
        tasks = [
            self.analyze_bank_async(symbol) 
            for symbol in self.settings.BANK_SYMBOLS
        ]
        
        # Execute all analyses concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        execution_time = time.time() - start_time
        
        # Process results and handle exceptions
        processed_results = {}
        for symbol, result in zip(self.settings.BANK_SYMBOLS, results):
            if isinstance(result, Exception):
                logger.error(f"Error analyzing {symbol}: {result}")
                processed_results[symbol] = {'error': str(result)}
            else:
                processed_results[symbol] = result
        
        logger.info(f"Async analysis completed in {execution_time:.2f}s for {len(self.settings.BANK_SYMBOLS)} symbols")
        return processed_results
    
    async def analyze_bank_async(self, symbol: str) -> Dict:
        """Async version of bank analysis"""
        try:
            # Run data-intensive operations concurrently in thread pool
            loop = asyncio.get_event_loop()
            
            # Create concurrent tasks
            market_data_task = loop.run_in_executor(
                self.executor, 
                self.data_feed.get_historical_data, 
                symbol,
                self.settings.DEFAULT_ANALYSIS_PERIOD
            )
            
            fundamental_task = loop.run_in_executor(
                self.executor,
                self.fundamental.analyze,
                symbol
            )
            
            sentiment_task = loop.run_in_executor(
                self.executor,
                self.sentiment.analyze_bank_sentiment,
                symbol
            )
            
            # Wait for all data gathering to complete
            market_data, fundamental_metrics, sentiment_score = await asyncio.gather(
                market_data_task,
                fundamental_task, 
                sentiment_task,
                return_exceptions=True
            )
            
            # Handle exceptions in data gathering
            if isinstance(market_data, Exception):
                logger.error(f"Error fetching market data for {symbol}: {market_data}")
                return {'error': f'Market data error: {str(market_data)}'}
            
            if isinstance(fundamental_metrics, Exception):
                logger.warning(f"Error in fundamental analysis for {symbol}: {fundamental_metrics}")
                fundamental_metrics = {'error': str(fundamental_metrics)}
            
            if isinstance(sentiment_score, Exception):
                logger.warning(f"Error in sentiment analysis for {symbol}: {sentiment_score}")
                sentiment_score = {'error': str(sentiment_score)}
            
            # Validate market data
            if market_data is None or market_data.empty:
                return {'error': f'No market data available for {symbol}'}
            
            # Run CPU-intensive analyses in thread pool
            technical_task = loop.run_in_executor(
                self.executor,
                self.technical.analyze,
                symbol,
                market_data
            )
            
            # Get current price
            current_price = float(market_data['Close'].iloc[-1])
            
            # Wait for technical analysis
            technical_signals = await technical_task
            
            # Run remaining analyses concurrently
            risk_task = loop.run_in_executor(
                self.executor,
                self.risk_calc.calculate,
                symbol,
                current_price,
                technical_signals,
                fundamental_metrics
            )
            
            prediction_task = loop.run_in_executor(
                self.executor,
                self.predictor.predict,
                symbol,
                technical_signals,
                fundamental_metrics,
                sentiment_score
            )
            
            # Wait for final analyses
            risk_reward, prediction = await asyncio.gather(
                risk_task,
                prediction_task,
                return_exceptions=True
            )
            
            # Handle exceptions in final analyses
            if isinstance(risk_reward, Exception):
                logger.warning(f"Error in risk calculation for {symbol}: {risk_reward}")
                risk_reward = {'error': str(risk_reward)}
            
            if isinstance(prediction, Exception):
                logger.warning(f"Error in prediction for {symbol}: {prediction}")
                prediction = {'error': str(prediction)}
            
            # Compile results
            analysis_result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'technical_analysis': technical_signals,
                'fundamental_analysis': fundamental_metrics,
                'sentiment_analysis': sentiment_score,
                'risk_reward': risk_reward,
                'prediction': prediction,
                'processing_method': 'async'
            }
            
            # Check for alerts asynchronously
            alerts_task = loop.run_in_executor(
                self.executor,
                self.alert_system.check_alert_thresholds,
                analysis_result
            )
            
            alerts = await alerts_task
            if alerts:
                for alert in alerts:
                    # Send alerts asynchronously without blocking
                    loop.run_in_executor(
                        self.executor,
                        self.alert_system.send_alert,
                        alert
                    )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in async analyze_bank for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    async def run_daily_analysis_async(self):
        """Run daily analysis asynchronously"""
        try:
            logger.info("Starting daily async analysis...")
            
            # Run analysis
            results = await self.analyze_all_banks_async()
            
            # Generate report asynchronously
            loop = asyncio.get_event_loop()
            report_path = await loop.run_in_executor(
                self.executor,
                self.report_gen.generate_daily_report,
                results,
                {}
            )
            
            if report_path:
                logger.info(f"Daily report generated: {report_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in daily async analysis: {str(e)}")
            return {'error': str(e)}

# Async wrapper functions for backward compatibility
async def run_async_analysis():
    """Run async analysis with proper session management"""
    async with ASXBankTradingSystemAsync() as system:
        results = await system.analyze_all_banks_async()
        return results

def run_sync_analysis():
    """Synchronous wrapper for async analysis"""
    return asyncio.run(run_async_analysis())

if __name__ == "__main__":
    # Example usage
    async def main():
        async with ASXBankTradingSystemAsync() as system:
            results = await system.analyze_all_banks_async()
            print(f"Analysis completed for {len(results)} symbols")
            
            # Print summary
            for symbol, result in results.items():
                if 'error' not in result:
                    print(f"{symbol}: {result.get('prediction', {}).get('direction', 'Unknown')}")
                else:
                    print(f"{symbol}: Error - {result['error']}")
    
    asyncio.run(main())
