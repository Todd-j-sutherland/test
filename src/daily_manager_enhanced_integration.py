#!/usr/bin/env python3
"""
Daily Manager Enhanced Sentiment Integration
Integrates the enhanced sentiment scoring with the existing daily_manager.py ML workflow
"""

import sys
import os
from typing import Dict, List, Optional
from datetime import datetime
import logging

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

try:
    from enhanced_sentiment_integration import SentimentIntegrationManager, get_enhanced_trading_signals
    from temporal_sentiment_analyzer import TemporalSentimentAnalyzer, SentimentDataPoint
    from enhanced_ensemble_learning import EnhancedTransformerEnsemble, ModelPrediction
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    INTEGRATION_AVAILABLE = False
    print(f"⚠️  Enhanced sentiment integration not available: {e}")

logger = logging.getLogger(__name__)

class DailyManagerEnhancedIntegration:
    """
    Integrates enhanced sentiment with the daily manager ML workflow
    """
    
    def __init__(self):
        if not INTEGRATION_AVAILABLE:
            raise ImportError("Required modules not available for enhanced integration")
            
        self.sentiment_manager = SentimentIntegrationManager()
        self.temporal_analyzer = TemporalSentimentAnalyzer()
        self.ensemble_learner = EnhancedTransformerEnsemble()
        
    def enhance_temporal_analysis(self, symbol: str, legacy_sentiment_data: Dict) -> Dict:
        """
        Enhance the temporal sentiment analysis with the enhanced sentiment system
        
        Args:
            symbol: Trading symbol
            legacy_sentiment_data: Raw sentiment data from existing system
            
        Returns:
            Enhanced temporal analysis results
        """
        
        # Get enhanced sentiment metrics
        enhanced_metrics = self.sentiment_manager.convert_legacy_to_enhanced(legacy_sentiment_data)
        
        # Create enhanced sentiment data point for temporal analysis
        enhanced_data_point = SentimentDataPoint(
            timestamp=datetime.now(),
            symbol=symbol,
            sentiment_score=enhanced_metrics.normalized_score / 100,  # Convert to -1 to 1 scale
            confidence=enhanced_metrics.confidence,
            news_count=legacy_sentiment_data.get('news_count', 0),
            relevance_score=enhanced_metrics.percentile_rank / 100,
            volume_impact=abs(enhanced_metrics.z_score) / 3.0,  # Normalize z-score impact
            source_credibility=enhanced_metrics.confidence
        )
        
        # Add to temporal analyzer
        if symbol not in self.temporal_analyzer.sentiment_history:
            self.temporal_analyzer.sentiment_history[symbol] = []
        
        self.temporal_analyzer.sentiment_history[symbol].append(enhanced_data_point)
        
        # Get temporal evolution analysis
        temporal_analysis = self.temporal_analyzer.analyze_sentiment_evolution(symbol)
        
        # Enhance with our sentiment metrics
        enhanced_temporal = {
            'symbol': symbol,
            'temporal_trend': temporal_analysis.get('trend', 0.0),
            'temporal_volatility': temporal_analysis.get('volatility', 0.0),
            'enhanced_score': enhanced_metrics.normalized_score,
            'enhanced_confidence': enhanced_metrics.confidence,
            'statistical_significance': enhanced_metrics.z_score,
            'historical_percentile': enhanced_metrics.percentile_rank,
            'strength_category': enhanced_metrics.strength_category.name,
            'market_adjusted': enhanced_metrics.market_adjusted_score != enhanced_metrics.raw_score,
            'volatility_adjusted': enhanced_metrics.volatility_adjusted_score != enhanced_metrics.raw_score,
            'regime_detection': temporal_analysis.get('regime', 'unknown'),
            'momentum_score': temporal_analysis.get('momentum', 0.0)
        }
        
        return enhanced_temporal
    
    def enhance_ensemble_prediction(self, symbol: str, enhanced_temporal: Dict) -> Dict:
        """
        Create enhanced ensemble predictions using the enhanced sentiment data
        
        Args:
            symbol: Trading symbol
            enhanced_temporal: Enhanced temporal analysis results
            
        Returns:
            Enhanced ensemble prediction
        """
        
        # Create model predictions based on enhanced sentiment
        predictions = []
        
        # Enhanced Sentiment Model
        sentiment_prediction = ModelPrediction(
            model_name="enhanced_sentiment",
            prediction=enhanced_temporal['enhanced_score'] / 100,  # Normalize to 0-1
            confidence=enhanced_temporal['enhanced_confidence'],
            probability_scores={
                'bullish': max(0, enhanced_temporal['enhanced_score'] - 50) / 50,
                'bearish': max(0, 50 - enhanced_temporal['enhanced_score']) / 50,
                'neutral': 1 - abs(enhanced_temporal['enhanced_score'] - 50) / 50
            },
            processing_time=0.001
        )
        predictions.append(sentiment_prediction)
        
        # Temporal Trend Model
        temporal_prediction = ModelPrediction(
            model_name="temporal_trend",
            prediction=(enhanced_temporal['temporal_trend'] + 1) / 2,  # Convert -1,1 to 0,1
            confidence=0.8,
            probability_scores={
                'bullish': max(0, enhanced_temporal['temporal_trend']),
                'bearish': max(0, -enhanced_temporal['temporal_trend']),
                'neutral': 1 - abs(enhanced_temporal['temporal_trend'])
            },
            processing_time=0.002
        )
        predictions.append(temporal_prediction)
        
        # Statistical Significance Model
        stat_significance = abs(enhanced_temporal['statistical_significance'])
        stat_prediction = ModelPrediction(
            model_name="statistical_significance",
            prediction=min(1.0, stat_significance / 2.0),  # Z-score importance
            confidence=min(1.0, stat_significance / 1.96),  # 95% confidence threshold
            probability_scores={
                'significant': min(1.0, stat_significance / 1.96),
                'moderate': max(0, 1 - stat_significance / 1.96),
                'weak': max(0, 1 - stat_significance / 1.0)
            },
            processing_time=0.001
        )
        predictions.append(stat_prediction)
        
        # Create ensemble prediction using the enhanced ensemble learner
        try:
            # This would normally use the actual ensemble learner
            # For now, we'll create a weighted average
            weights = {
                'enhanced_sentiment': 0.5,
                'temporal_trend': 0.3,
                'statistical_significance': 0.2
            }
            
            weighted_prediction = sum(
                pred.prediction * weights.get(pred.model_name, 0.33) 
                for pred in predictions
            )
            
            ensemble_confidence = sum(
                pred.confidence * weights.get(pred.model_name, 0.33) 
                for pred in predictions
            )
            
            ensemble_result = {
                'symbol': symbol,
                'ensemble_prediction': weighted_prediction,
                'ensemble_confidence': ensemble_confidence,
                'individual_predictions': [
                    {
                        'model': pred.model_name,
                        'prediction': pred.prediction,
                        'confidence': pred.confidence
                    } for pred in predictions
                ],
                'model_weights': weights,
                'ensemble_method': 'enhanced_weighted_average',
                'reasoning': [
                    f"Enhanced sentiment: {enhanced_temporal['enhanced_score']:.1f}/100",
                    f"Temporal trend: {enhanced_temporal['temporal_trend']:+.3f}",
                    f"Statistical significance: {enhanced_temporal['statistical_significance']:.2f}",
                    f"Historical percentile: {enhanced_temporal['historical_percentile']:.0f}%"
                ]
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            ensemble_result = {
                'symbol': symbol,
                'error': str(e),
                'fallback_prediction': enhanced_temporal['enhanced_score'] / 100
            }
        
        return ensemble_result
    
    def run_enhanced_morning_analysis(self, symbols: List[str]) -> Dict:
        """
        Run enhanced morning analysis for the daily manager
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            Complete enhanced analysis results
        """
        
        print("🚀 Enhanced Morning Analysis Starting...")
        results = {}
        
        for symbol in symbols:
            try:
                # Simulate legacy sentiment data (in real system, this comes from news analysis)
                legacy_sentiment = {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'news_count': 8,
                    'sentiment_scores': {'average_sentiment': 0.15},
                    'reddit_sentiment': {'average_sentiment': 0.05},
                    'significant_events': {'events_detected': []},
                    'overall_sentiment': 0.12,
                    'confidence': 0.68,
                    'recent_headlines': [
                        f'{symbol} quarterly update',
                        f'Banking sector analysis for {symbol}'
                    ]
                }
                
                # Run enhanced temporal analysis
                enhanced_temporal = self.enhance_temporal_analysis(symbol, legacy_sentiment)
                
                # Run enhanced ensemble prediction
                ensemble_result = self.enhance_ensemble_prediction(symbol, enhanced_temporal)
                
                # Get trading signals
                trading_signals = get_enhanced_trading_signals(legacy_sentiment)
                
                # Combine results
                results[symbol] = {
                    'enhanced_temporal': enhanced_temporal,
                    'ensemble_prediction': ensemble_result,
                    'trading_signals': trading_signals,
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
                print(f"✅ {symbol}: Enhanced analysis complete")
                
            except Exception as e:
                print(f"❌ {symbol}: Analysis error - {e}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def run_enhanced_evening_analysis(self, symbols: List[str]) -> Dict:
        """
        Run enhanced evening analysis for the daily manager
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            Complete enhanced evening analysis results
        """
        
        print("🌆 Enhanced Evening Analysis Starting...")
        
        # Get morning results (would normally be stored)
        morning_results = self.run_enhanced_morning_analysis(symbols)
        
        # Enhanced evening summary
        summary = {
            'analysis_date': datetime.now().date().isoformat(),
            'symbols_analyzed': len(symbols),
            'successful_analyses': len([r for r in morning_results.values() if 'error' not in r]),
            'average_enhanced_score': 0,
            'average_confidence': 0,
            'high_significance_signals': 0,
            'trading_recommendations': {}
        }
        
        valid_results = [r for r in morning_results.values() if 'error' not in r]
        
        if valid_results:
            # Calculate averages
            enhanced_scores = [r['enhanced_temporal']['enhanced_score'] for r in valid_results]
            confidences = [r['enhanced_temporal']['enhanced_confidence'] for r in valid_results]
            z_scores = [abs(r['enhanced_temporal']['statistical_significance']) for r in valid_results]
            
            summary['average_enhanced_score'] = sum(enhanced_scores) / len(enhanced_scores)
            summary['average_confidence'] = sum(confidences) / len(confidences)
            summary['high_significance_signals'] = len([z for z in z_scores if z > 1.96])
            
            # Trading recommendations
            for symbol, result in morning_results.items():
                if 'error' not in result:
                    signals = result['trading_signals']
                    summary['trading_recommendations'][symbol] = {
                        'conservative': signals['conservative']['signal'],
                        'moderate': signals['moderate']['signal'],
                        'aggressive': signals['aggressive']['signal']
                    }
        
        print(f"✅ Evening analysis complete - {summary['successful_analyses']}/{summary['symbols_analyzed']} symbols processed")
        
        return {
            'summary': summary,
            'detailed_results': morning_results
        }

# Integration functions for daily_manager.py
def run_enhanced_daily_analysis(symbols: List[str] = None) -> Dict:
    """
    Main function to be called from daily_manager.py
    
    Args:
        symbols: List of symbols to analyze (defaults to bank symbols)
        
    Returns:
        Enhanced analysis results
    """
    
    if symbols is None:
        # Default to major Australian banks
        symbols = ['CBA.AX', 'ANZ.AX', 'WBC.AX', 'NAB.AX']
    
    try:
        integrator = DailyManagerEnhancedIntegration()
        return integrator.run_enhanced_evening_analysis(symbols)
    except ImportError:
        return {
            'error': 'Enhanced sentiment integration not available',
            'fallback': 'Using basic sentiment analysis'
        }

if __name__ == "__main__":
    # Test the integration
    print("🧪 Testing Enhanced Daily Manager Integration")
    print("=" * 50)
    
    # Test with sample symbols
    test_symbols = ['CBA.AX', 'ANZ.AX']
    results = run_enhanced_daily_analysis(test_symbols)
    
    if 'error' not in results:
        summary = results['summary']
        print(f"✅ Analysis completed successfully")
        print(f"📊 Average Enhanced Score: {summary['average_enhanced_score']:.1f}/100")
        print(f"📊 Average Confidence: {summary['average_confidence']:.2f}")
        print(f"📊 High Significance Signals: {summary['high_significance_signals']}")
        print(f"📊 Trading Recommendations:")
        for symbol, recs in summary['trading_recommendations'].items():
            print(f"   {symbol}: {recs['moderate']}")
    else:
        print(f"❌ Integration test failed: {results['error']}")
