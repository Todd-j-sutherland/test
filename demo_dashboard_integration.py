#!/usr/bin/env python3
"""
Enhanced Sentiment Dashboard Integration Demo
Shows how to integrate enhanced sentiment into your existing professional dashboard
"""

import sys
import os
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

try:
    from enhanced_sentiment_integration import SentimentIntegrationManager, get_enhanced_trading_signals
    ENHANCED_SENTIMENT_AVAILABLE = True
except ImportError:
    ENHANCED_SENTIMENT_AVAILABLE = False

def create_enhanced_sentiment_widget(legacy_sentiment_data):
    """Create a Streamlit widget showing enhanced sentiment analysis"""
    
    if not ENHANCED_SENTIMENT_AVAILABLE:
        st.error("Enhanced sentiment system not available")
        return None
    
    st.subheader("🚀 Enhanced Sentiment Analysis")
    
    # Get enhanced signals
    signals = get_enhanced_trading_signals(legacy_sentiment_data)
    enhanced_analysis = signals['enhanced_analysis']
    
    # Create metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Enhanced Score", 
            f"{enhanced_analysis['normalized_score']:.1f}/100",
            f"{enhanced_analysis['improvement_over_legacy']['score_refinement']:+.1f}"
        )
    
    with col2:
        st.metric(
            "Strength", 
            enhanced_analysis['strength_category'],
            f"Z-Score: {enhanced_analysis['z_score']:.2f}"
        )
    
    with col3:
        st.metric(
            "Confidence", 
            f"{enhanced_analysis['confidence']:.2f}",
            f"{enhanced_analysis['improvement_over_legacy']['confidence_improvement']:+.3f}"
        )
    
    with col4:
        st.metric(
            "Percentile", 
            f"{enhanced_analysis['percentile_rank']:.0f}%",
            "vs Historical"
        )
    
    # Trading signals section
    st.subheader("📈 Risk-Adjusted Trading Signals")
    
    signal_data = []
    for risk_level, signal_info in signals.items():
        if risk_level != 'enhanced_analysis':
            signal_data.append({
                'Risk Profile': risk_level.title(),
                'Signal': f"{signal_info['signal']} {signal_info['strength']}",
                'Reasoning': signal_info['reasoning']
            })
    
    df_signals = pd.DataFrame(signal_data)
    st.dataframe(df_signals, use_container_width=True)
    
    # Enhancement summary
    st.subheader("💡 Enhancement Summary")
    enhancement = enhanced_analysis['improvement_over_legacy']
    
    st.write(f"**Enhancements Applied:** {enhancement['enhancement_summary']}")
    
    enhancements_info = {
        'Market Adjustment': enhancement['market_adjustment_applied'],
        'Volatility Adjustment': enhancement['volatility_adjustment_applied'],
        'Statistical Significance': f"{enhancement['statistical_significance']:.2f}",
        'Score Refinement': f"{enhancement['score_refinement']:+.1f} points"
    }
    
    for label, value in enhancements_info.items():
        if isinstance(value, bool):
            st.write(f"• **{label}:** {'✅ Applied' if value else '❌ Not Applied'}")
        else:
            st.write(f"• **{label}:** {value}")
    
    return enhanced_analysis

def demo_dashboard_integration():
    """Demo showing integration with dashboard"""
    
    st.title("🎯 Enhanced Sentiment Dashboard Integration")
    st.caption("Demonstration using .venv312 environment")
    
    # Simulate legacy sentiment data (as would come from your existing system)
    st.subheader("📊 Legacy Sentiment Data (Input)")
    
    # Create sample data that user can modify
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.selectbox("Symbol", ["CBA.AX", "ANZ.AX", "WBC.AX", "NAB.AX"])
        overall_sentiment = st.slider("Overall Sentiment", -1.0, 1.0, 0.15, 0.01)
        confidence = st.slider("Confidence", 0.0, 1.0, 0.7, 0.01)
    
    with col2:
        news_count = st.number_input("News Count", 1, 50, 10)
        avg_sentiment = st.slider("News Avg Sentiment", -1.0, 1.0, 0.2, 0.01)
        social_sentiment = st.slider("Social Sentiment", -1.0, 1.0, 0.1, 0.01)
    
    # Create the legacy sentiment data structure
    legacy_sentiment_data = {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'news_count': news_count,
        'sentiment_scores': {
            'average_sentiment': avg_sentiment,
            'positive_count': int(news_count * 0.6),
            'negative_count': int(news_count * 0.2),
            'neutral_count': int(news_count * 0.2)
        },
        'reddit_sentiment': {
            'average_sentiment': social_sentiment,
            'posts_analyzed': news_count * 2
        },
        'significant_events': {
            'events_detected': [
                {'type': 'earnings_report', 'sentiment_impact': avg_sentiment * 0.8}
            ] if abs(avg_sentiment) > 0.3 else []
        },
        'overall_sentiment': overall_sentiment,
        'confidence': confidence,
        'recent_headlines': [
            f'{symbol} reports quarterly results',
            f'Banking sector analysis for {symbol}',
            f'Market update: {symbol} performance'
        ]
    }
    
    # Show the legacy data
    st.json(legacy_sentiment_data)
    
    # Process button
    if st.button("🚀 Enhance Sentiment Analysis", type="primary"):
        
        st.markdown("---")
        
        # Show enhanced sentiment widget
        enhanced_analysis = create_enhanced_sentiment_widget(legacy_sentiment_data)
        
        if enhanced_analysis:
            # Show comparison chart
            st.subheader("📈 Legacy vs Enhanced Comparison")
            
            legacy_normalized = ((overall_sentiment + 1) / 2) * 100
            enhanced_normalized = enhanced_analysis['normalized_score']
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Legacy System',
                x=['Sentiment Score'],
                y=[legacy_normalized],
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name='Enhanced System',
                x=['Sentiment Score'],
                y=[enhanced_normalized],
                marker_color='darkblue'
            ))
            
            fig.update_layout(
                title='Sentiment Score Comparison',
                yaxis_title='Score (0-100)',
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Integration code example
            st.subheader("🔧 Integration Code Example")
            
            code_example = f'''
# In your existing dashboard, replace basic sentiment with enhanced:

# OLD CODE:
# sentiment_score = legacy_sentiment_data['overall_sentiment']
# confidence = legacy_sentiment_data['confidence']

# NEW CODE (Enhanced):
from app.core.sentiment.integration import get_enhanced_trading_signals

def get_enhanced_sentiment_for_dashboard(legacy_data):
    signals = get_enhanced_trading_signals(legacy_data)
    enhanced = signals['enhanced_analysis']
    
    return {{
        'score': enhanced['normalized_score'],
        'confidence': enhanced['confidence'], 
        'strength': enhanced['strength_category'],
        'percentile': enhanced['percentile_rank'],
        'trading_signals': {{risk: signals[risk]['signal'] for risk in ['conservative', 'moderate', 'aggressive']}}
    }}

# Usage in your dashboard:
enhanced_data = get_enhanced_sentiment_for_dashboard(legacy_sentiment_data)
st.metric("Enhanced Score", f"{{enhanced_data['score']:.1f}}/100")
'''
            
            st.code(code_example, language='python')
            
            st.success("✅ Enhanced sentiment integration completed successfully!")
            st.info("This enhanced system provides more accurate, statistically validated sentiment analysis with risk-adjusted trading signals.")

if __name__ == "__main__":
    demo_dashboard_integration()
