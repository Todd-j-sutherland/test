#!/usr/bin/env python3
"""
Enhanced Sentiment Integration Demo
Shows how the enhanced sentiment system improves upon legacy sentiment analysis
"""

import sys
import os
from datetime import datetime, timedelta

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

from enhanced_sentiment_integration import SentimentIntegrationManager

def create_demo_scenarios():
    """Create different sentiment scenarios to demonstrate the system"""
    
    scenarios = {
        'bullish_earnings': {
            'symbol': 'CBA.AX',
            'timestamp': datetime.now().isoformat(),
            'news_count': 12,
            'sentiment_scores': {
                'average_sentiment': 0.45,
                'positive_count': 9,
                'negative_count': 2,
                'neutral_count': 1
            },
            'reddit_sentiment': {
                'average_sentiment': 0.35,
                'posts_analyzed': 25
            },
            'significant_events': {
                'events_detected': [
                    {'type': 'earnings_report', 'sentiment_impact': 0.6},
                    {'type': 'dividend_increase', 'sentiment_impact': 0.3}
                ]
            },
            'overall_sentiment': 0.42,
            'confidence': 0.85,
            'recent_headlines': [
                'CBA beats earnings expectations by 15%',
                'Record quarterly profit announced',
                'Dividend increased by 8%',
                'Strong loan growth continues',
                'Banking sector outperforms market'
            ]
        },
        
        'bearish_crisis': {
            'symbol': 'WBC.AX',
            'timestamp': datetime.now().isoformat(),
            'news_count': 18,
            'sentiment_scores': {
                'average_sentiment': -0.65,
                'positive_count': 2,
                'negative_count': 14,
                'neutral_count': 2
            },
            'reddit_sentiment': {
                'average_sentiment': -0.55,
                'posts_analyzed': 45
            },
            'significant_events': {
                'events_detected': [
                    {'type': 'regulatory_action', 'sentiment_impact': -0.7},
                    {'type': 'executive_departure', 'sentiment_impact': -0.4}
                ]
            },
            'overall_sentiment': -0.58,
            'confidence': 0.92,
            'recent_headlines': [
                'ASIC launches investigation into lending practices',
                'CEO announces unexpected resignation',
                'Share price plummets 12% in trading',
                'Credit rating under review',
                'Institutional investors reduce holdings'
            ]
        },
        
        'neutral_mixed': {
            'symbol': 'ANZ.AX',
            'timestamp': datetime.now().isoformat(),
            'news_count': 6,
            'sentiment_scores': {
                'average_sentiment': 0.05,
                'positive_count': 3,
                'negative_count': 2,
                'neutral_count': 1
            },
            'reddit_sentiment': {
                'average_sentiment': -0.02,
                'posts_analyzed': 8
            },
            'significant_events': {
                'events_detected': []
            },
            'overall_sentiment': 0.02,
            'confidence': 0.45,
            'recent_headlines': [
                'ANZ maintains steady performance',
                'Mixed analyst ratings on banking sector',
                'Interest rate changes impact margins'
            ]
        }
    }
    
    return scenarios

def demonstrate_enhancement(scenario_name, legacy_data):
    """Demonstrate the enhancement process for a specific scenario"""
    
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name.upper().replace('_', ' ')}")
    print(f"Symbol: {legacy_data['symbol']}")
    print(f"{'='*60}")
    
    manager = SentimentIntegrationManager()
    
    # Convert legacy to enhanced
    enhanced_metrics = manager.convert_legacy_to_enhanced(legacy_data)
    
    # Show comparison
    print(f"\n📊 SENTIMENT ANALYSIS COMPARISON:")
    print(f"┌─────────────────────────┬─────────────┬─────────────┐")
    print(f"│ Metric                  │ Legacy      │ Enhanced    │")
    print(f"├─────────────────────────┼─────────────┼─────────────┤")
    print(f"│ Raw Score               │ {legacy_data['overall_sentiment']:>10.3f} │ {enhanced_metrics.raw_score:>10.1f} │")
    print(f"│ Normalized Score        │ {((legacy_data['overall_sentiment'] + 1) * 50):>10.1f} │ {enhanced_metrics.normalized_score:>10.1f} │")
    print(f"│ Confidence              │ {legacy_data['confidence']:>10.2f} │ {enhanced_metrics.confidence:>10.2f} │")
    print(f"│ Statistical Significance│ N/A         │ {enhanced_metrics.z_score:>10.2f} │")
    print(f"│ Historical Percentile   │ N/A         │ {enhanced_metrics.percentile_rank:>9.1f}% │")
    print(f"│ Strength Category       │ Basic       │ {enhanced_metrics.strength_category.name:>10} │")
    print(f"└─────────────────────────┴─────────────┴─────────────┘")
    
    # Show enhanced features
    print(f"\n🚀 ENHANCED FEATURES:")
    print(f"• Market Regime Adjustment: {enhanced_metrics.market_adjusted_score != enhanced_metrics.raw_score}")
    print(f"• Volatility Adjustment: {enhanced_metrics.volatility_adjusted_score != enhanced_metrics.raw_score}")
    print(f"• Multi-component Analysis: ✓")
    print(f"• Statistical Validation: ✓")
    
    # Generate trading signals
    signals = manager.generate_enhanced_trading_signals(legacy_data)
    
    print(f"\n📈 TRADING SIGNALS BY RISK PROFILE:")
    print(f"┌─────────────┬─────────────┬─────────────────────────────────────┐")
    print(f"│ Risk Level  │ Signal      │ Reasoning                           │")
    print(f"├─────────────┼─────────────┼─────────────────────────────────────┤")
    
    for risk_level, signal_data in signals.items():
        if risk_level != 'enhanced_analysis':
            signal = f"{signal_data['signal']} {signal_data['strength']}"
            reasoning = signal_data['reasoning'][:35] + "..." if len(signal_data['reasoning']) > 35 else signal_data['reasoning']
            print(f"│ {risk_level.capitalize():<11} │ {signal:<11} │ {reasoning:<35} │")
    
    print(f"└─────────────┴─────────────┴─────────────────────────────────────┘")
    
    # Show enhancement metrics
    enhancement = signals['enhanced_analysis']['improvement_over_legacy']
    print(f"\n💡 ENHANCEMENT SUMMARY:")
    print(f"• Score Refinement: {enhancement['score_refinement']:+.1f} points")
    print(f"• Confidence Improvement: {enhancement['confidence_improvement']:+.2f}")
    print(f"• Statistical Significance: {enhancement['statistical_significance']:.2f}")
    print(f"• Enhancement Summary: {enhancement['enhancement_summary']}")

def main():
    """Run the complete enhanced sentiment integration demonstration"""
    
    print("🎯 ENHANCED SENTIMENT INTEGRATION DEMONSTRATION")
    print("Using .venv312 Python Environment")
    print(f"Demonstration Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get demo scenarios
    scenarios = create_demo_scenarios()
    
    # Demonstrate each scenario
    for scenario_name, legacy_data in scenarios.items():
        demonstrate_enhancement(scenario_name, legacy_data)
    
    # Show integration performance
    manager = SentimentIntegrationManager()
    
    # Process all scenarios to build history
    for legacy_data in scenarios.values():
        manager.convert_legacy_to_enhanced(legacy_data)
    
    print(f"\n{'='*60}")
    print("INTEGRATION PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    report = manager.get_integration_performance_report()
    print(f"Total Conversions: {report['total_conversions']}")
    print(f"Average Score Improvement: {report['average_score_improvement']:+.2f}")
    print(f"Average Confidence Improvement: {report['average_confidence_improvement']:+.3f}")
    print(f"Enhancement Effectiveness: {report['enhancement_effectiveness']}")
    print(f"Symbols Processed: {', '.join(report['symbols_processed'])}")
    
    print(f"\n✅ Demo completed successfully!")
    print("The enhanced sentiment system provides:")
    print("• Statistical significance testing")
    print("• Market regime adjustments")
    print("• Multi-component analysis")
    print("• Risk-adjusted trading signals")
    print("• Historical percentile ranking")
    print("• Improved confidence metrics")

if __name__ == "__main__":
    main()
