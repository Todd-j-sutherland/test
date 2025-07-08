#!/usr/bin/env python3
"""
News Analysis Dashboard
Interactive web dashboard displaying news sentiment analysis results for Australian banks
"""

import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="📰 ASX Bank News Analysis Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .bank-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .news-item {
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        background-color: white;
    }
    .sentiment-score {
        font-weight: bold;
        font-size: 1.2rem;
    }
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    .neutral { color: #6c757d; }
    .confidence-high { background-color: #d4edda; }
    .confidence-medium { background-color: #fff3cd; }
    .confidence-low { background-color: #f8d7da; }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .event-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: bold;
        margin: 0.1rem;
    }
    .event-positive { background-color: #d4edda; color: #155724; }
    .event-negative { background-color: #f8d7da; color: #721c24; }
    .event-neutral { background-color: #e2e3e5; color: #383d41; }
</style>
""", unsafe_allow_html=True)

class NewsAnalysisDashboard:
    """Dashboard for displaying news analysis results"""
    
    def __init__(self):
        self.data_path = "data/sentiment_history"
        self.bank_symbols = ["CBA.AX", "WBC.AX", "ANZ.AX", "NAB.AX", "MQG.AX"]
        self.bank_names = {
            "CBA.AX": "Commonwealth Bank",
            "WBC.AX": "Westpac Banking Corp",
            "ANZ.AX": "ANZ Banking Group",
            "NAB.AX": "National Australia Bank",
            "MQG.AX": "Macquarie Group"
        }
        
    def load_sentiment_data(self) -> Dict[str, List[Dict]]:
        """Load sentiment history data for all banks"""
        all_data = {}
        
        for symbol in self.bank_symbols:
            file_path = os.path.join(self.data_path, f"{symbol}_history.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        all_data[symbol] = data if isinstance(data, list) else [data]
                except Exception as e:
                    logger.error(f"Error loading data for {symbol}: {e}")
                    all_data[symbol] = []
            else:
                all_data[symbol] = []
                
        return all_data
    
    def get_latest_analysis(self, data: List[Dict]) -> Dict:
        """Get the most recent analysis from the data"""
        if not data:
            return {}
        
        # Sort by timestamp and get the latest
        try:
            sorted_data = sorted(data, key=lambda x: x.get('timestamp', ''), reverse=True)
            return sorted_data[0] if sorted_data else {}
        except Exception:
            return data[-1] if data else {}
    
    def format_sentiment_score(self, score: float) -> tuple:
        """Format sentiment score with color class"""
        if score > 0.2:
            return f"+{score:.3f}", "positive"
        elif score < -0.2:
            return f"{score:.3f}", "negative"
        else:
            return f"{score:.3f}", "neutral"
    
    def get_confidence_level(self, confidence: float) -> tuple:
        """Get confidence level description and CSS class"""
        if confidence >= 0.8:
            return "HIGH", "confidence-high"
        elif confidence >= 0.6:
            return "MEDIUM", "confidence-medium"
        else:
            return "LOW", "confidence-low"
    
    def create_sentiment_overview_chart(self, all_data: Dict) -> go.Figure:
        """Create overview chart of all bank sentiments"""
        symbols = []
        scores = []
        confidences = []
        colors = []
        
        for symbol, data in all_data.items():
            latest = self.get_latest_analysis(data)
            if latest:
                symbols.append(self.bank_names.get(symbol, symbol))
                score = latest.get('overall_sentiment', 0)
                confidence = latest.get('confidence', 0)
                
                scores.append(score)
                confidences.append(confidence)
                
                # Color based on sentiment
                if score > 0.2:
                    colors.append('#28a745')  # Green
                elif score < -0.2:
                    colors.append('#dc3545')  # Red
                else:
                    colors.append('#6c757d')  # Gray
        
        fig = go.Figure()
        
        # Add sentiment bars
        fig.add_trace(go.Bar(
            x=symbols,
            y=scores,
            name='Sentiment Score',
            marker_color=colors,
            text=[f"{s:.3f}" for s in scores],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>' +
                         'Sentiment: %{y:.3f}<br>' +
                         'Confidence: %{customdata:.2f}<extra></extra>',
            customdata=confidences
        ))
        
        fig.update_layout(
            title="Bank Sentiment Overview",
            xaxis_title="Banks",
            yaxis_title="Sentiment Score",
            height=400,
            showlegend=False,
            yaxis=dict(range=[-1, 1])
        )
        
        return fig
    
    def create_confidence_distribution_chart(self, all_data: Dict) -> go.Figure:
        """Create confidence distribution chart"""
        symbols = []
        confidences = []
        
        for symbol, data in all_data.items():
            latest = self.get_latest_analysis(data)
            if latest:
                symbols.append(self.bank_names.get(symbol, symbol))
                confidences.append(latest.get('confidence', 0))
        
        fig = go.Figure(go.Bar(
            x=symbols,
            y=confidences,
            marker_color=['#28a745' if c >= 0.8 else '#ffc107' if c >= 0.6 else '#dc3545' for c in confidences],
            text=[f"{c:.2f}" for c in confidences],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Analysis Confidence Levels",
            xaxis_title="Banks",
            yaxis_title="Confidence Score",
            height=400,
            showlegend=False,
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def create_news_impact_chart(self, news_data: List[Dict], events_data: Dict) -> go.Figure:
        """Create chart showing news impact on sentiment"""
        # Analyze news sentiment impact
        impact_data = []
        
        for news in news_data[:10]:  # Top 10 recent news
            title = news.get('title', 'Unknown')[:50] + "..."
            sentiment_impact = 0
            
            # Extract sentiment if available in news analysis
            if 'sentiment_analysis' in news:
                sentiment_impact = news['sentiment_analysis'].get('composite', 0)
            
            impact_data.append({
                'title': title,
                'impact': sentiment_impact,
                'source': news.get('source', 'Unknown'),
                'relevance': news.get('relevance', 'medium')
            })
        
        if not impact_data:
            # Create empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No news impact data available",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="News Impact on Sentiment",
                height=300,
                showlegend=False
            )
            return fig
        
        df = pd.DataFrame(impact_data)
        
        fig = go.Figure(go.Bar(
            x=df['title'],
            y=df['impact'],
            marker_color=['#28a745' if i > 0 else '#dc3545' if i < 0 else '#6c757d' for i in df['impact']],
            hovertemplate='<b>%{x}</b><br>' +
                         'Impact: %{y:.3f}<br>' +
                         'Source: %{customdata}<extra></extra>',
            customdata=df['source']
        ))
        
        fig.update_layout(
            title="News Impact on Sentiment",
            xaxis_title="News Articles",
            yaxis_title="Sentiment Impact",
            height=400,
            showlegend=False,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def display_confidence_legend(self):
        """Display confidence score legend and decision criteria"""
        st.markdown("### 📊 Confidence Score Legend & Decision Criteria")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card confidence-high">
                <h4>🟢 HIGH CONFIDENCE (≥0.8)</h4>
                <p><strong>Action:</strong> Strong Buy/Sell Signal</p>
                <p><strong>Criteria:</strong> Multiple reliable sources, consistent sentiment, significant news volume</p>
                <p><strong>Decision:</strong> Execute trades with full position sizing</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card confidence-medium">
                <h4>🟡 MEDIUM CONFIDENCE (0.6-0.8)</h4>
                <p><strong>Action:</strong> Moderate Buy/Sell Signal</p>
                <p><strong>Criteria:</strong> Some reliable sources, moderate sentiment consistency</p>
                <p><strong>Decision:</strong> Execute trades with reduced position sizing</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card confidence-low">
                <h4>🔴 LOW CONFIDENCE (<0.6)</h4>
                <p><strong>Action:</strong> Hold/Monitor</p>
                <p><strong>Criteria:</strong> Limited sources, inconsistent sentiment, low news volume</p>
                <p><strong>Decision:</strong> Avoid trading, wait for better signals</p>
            </div>
            """, unsafe_allow_html=True)
    
    def display_sentiment_scale(self):
        """Display sentiment scale explanation"""
        st.markdown("### 📈 Sentiment Score Scale")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown('<div class="metric-card"><strong>Very Negative</strong><br>-1.0 to -0.5<br><span class="negative">Strong Sell</span></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card"><strong>Negative</strong><br>-0.5 to -0.2<br><span class="negative">Sell</span></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card"><strong>Neutral</strong><br>-0.2 to +0.2<br><span class="neutral">Hold</span></div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card"><strong>Positive</strong><br>+0.2 to +0.5<br><span class="positive">Buy</span></div>', unsafe_allow_html=True)
        
        with col5:
            st.markdown('<div class="metric-card"><strong>Very Positive</strong><br>+0.5 to +1.0<br><span class="positive">Strong Buy</span></div>', unsafe_allow_html=True)
    
    def display_bank_analysis(self, symbol: str, data: List[Dict]):
        """Display detailed analysis for a specific bank"""
        latest = self.get_latest_analysis(data)
        
        if not latest:
            st.warning(f"No analysis data available for {self.bank_names.get(symbol, symbol)}")
            return
        
        bank_name = self.bank_names.get(symbol, symbol)
        
        # Header
        st.markdown(f"### 🏦 {bank_name} ({symbol})")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sentiment = latest.get('overall_sentiment', 0)
            score_text, score_class = self.format_sentiment_score(sentiment)
            st.markdown(f"""
            <div class="metric-card">
                <h4>Sentiment Score</h4>
                <div class="sentiment-score {score_class}">{score_text}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            confidence = latest.get('confidence', 0)
            conf_level, conf_class = self.get_confidence_level(confidence)
            st.markdown(f"""
            <div class="metric-card">
                <h4>Confidence</h4>
                <div class="{conf_class}">{confidence:.2f} ({conf_level})</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            news_count = latest.get('news_count', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h4>News Articles</h4>
                <div>{news_count}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            timestamp = latest.get('timestamp', 'Unknown')
            if timestamp != 'Unknown':
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    time_str = timestamp[:16]
            else:
                time_str = 'Unknown'
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Last Updated</h4>
                <div>{time_str}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Sentiment components breakdown
        if 'sentiment_components' in latest:
            st.markdown("#### 📊 Sentiment Components Breakdown")
            components = latest['sentiment_components']
            
            comp_df = pd.DataFrame([
                {'Component': 'News', 'Score': components.get('news', 0), 'Weight': '35%'},
                {'Component': 'Reddit', 'Score': components.get('reddit', 0), 'Weight': '15%'},
                {'Component': 'Events', 'Score': components.get('events', 0), 'Weight': '20%'},
                {'Component': 'Volume', 'Score': components.get('volume', 0), 'Weight': '10%'},
                {'Component': 'Momentum', 'Score': components.get('momentum', 0), 'Weight': '10%'},
                {'Component': 'ML Trading', 'Score': components.get('ml_trading', 0), 'Weight': '10%'}
            ])
            
            st.dataframe(comp_df, use_container_width=True)
        
        # Recent headlines
        if 'recent_headlines' in latest:
            st.markdown("#### 📰 Recent Headlines")
            headlines = latest['recent_headlines']
            for headline in headlines[:5]:
                if headline:  # Skip empty headlines
                    st.markdown(f"- {headline}")
        
        # Significant events
        if 'significant_events' in latest:
            st.markdown("#### 🚨 Significant Events Detected")
            events = latest['significant_events']
            
            if 'events_detected' in events and events['events_detected']:
                for event in events['events_detected']:
                    event_type = event.get('type', 'unknown')
                    headline = event.get('headline', 'No headline')
                    sentiment_impact = event.get('sentiment_impact', 0)
                    
                    # Determine event badge class
                    if sentiment_impact > 0.1:
                        badge_class = "event-positive"
                    elif sentiment_impact < -0.1:
                        badge_class = "event-negative"
                    else:
                        badge_class = "event-neutral"
                    
                    st.markdown(f"""
                    <div class="news-item">
                        <span class="event-badge {badge_class}">{event_type.replace('_', ' ').title()}</span>
                        <br><strong>{headline}</strong>
                        <br><small>Sentiment Impact: {sentiment_impact:+.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No significant events detected in recent analysis")
        
        # News sentiment analysis details
        if 'news_sentiment' in latest:
            news_data = latest['news_sentiment']
            
            st.markdown("#### 📊 News Analysis Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Sentiment Distribution:**")
                if 'sentiment_distribution' in news_data:
                    dist = news_data['sentiment_distribution']
                    dist_df = pd.DataFrame([
                        {'Category': 'Very Positive', 'Count': dist.get('very_positive', 0)},
                        {'Category': 'Positive', 'Count': dist.get('positive', 0)},
                        {'Category': 'Neutral', 'Count': dist.get('neutral', 0)},
                        {'Category': 'Negative', 'Count': dist.get('negative', 0)},
                        {'Category': 'Very Negative', 'Count': dist.get('very_negative', 0)}
                    ])
                    st.dataframe(dist_df, use_container_width=True)
            
            with col2:
                st.markdown("**Analysis Method Performance:**")
                if 'method_breakdown' in news_data:
                    method = news_data['method_breakdown']
                    if 'traditional' in method and 'composite' in method:
                        method_df = pd.DataFrame([
                            {'Method': 'Traditional (TextBlob/VADER)', 'Mean Score': method['traditional'].get('mean', 0)},
                            {'Method': 'Transformer Models', 'Mean Score': method['transformer'].get('mean', 0)},
                            {'Method': 'Composite (Final)', 'Mean Score': method['composite'].get('mean', 0)}
                        ])
                        st.dataframe(method_df, use_container_width=True)
        
        # Reddit sentiment if available
        if 'reddit_sentiment' in latest:
            reddit_data = latest['reddit_sentiment']
            posts_analyzed = reddit_data.get('posts_analyzed', 0)
            
            if posts_analyzed > 0:
                st.markdown("#### 💬 Reddit Sentiment Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Posts Analyzed", posts_analyzed)
                    st.metric("Average Sentiment", f"{reddit_data.get('average_sentiment', 0):.3f}")
                
                with col2:
                    st.metric("Bullish Posts", reddit_data.get('bullish_count', 0))
                    st.metric("Bearish Posts", reddit_data.get('bearish_count', 0))
    
    def run_dashboard(self):
        """Main dashboard application"""
        # Header
        st.markdown('<div class="main-header">📰 ASX Bank News Analysis Dashboard</div>', unsafe_allow_html=True)
        
        # Load data
        with st.spinner("Loading sentiment analysis data..."):
            all_data = self.load_sentiment_data()
        
        # Sidebar
        st.sidebar.markdown("## 🎛️ Dashboard Controls")
        
        # Bank selection
        selected_banks = st.sidebar.multiselect(
            "Select Banks to Display",
            options=list(self.bank_names.keys()),
            default=list(self.bank_names.keys()),
            format_func=lambda x: self.bank_names[x]
        )
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.experimental_rerun()
        
        # Display confidence legend and sentiment scale
        self.display_confidence_legend()
        self.display_sentiment_scale()
        
        # Overview charts
        st.markdown("## 📊 Market Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_chart = self.create_sentiment_overview_chart(all_data)
            st.plotly_chart(sentiment_chart, use_container_width=True)
        
        with col2:
            confidence_chart = self.create_confidence_distribution_chart(all_data)
            st.plotly_chart(confidence_chart, use_container_width=True)
        
        # Individual bank analysis
        st.markdown("## 🏦 Individual Bank Analysis")
        
        for symbol in selected_banks:
            if symbol in all_data:
                with st.expander(f"{self.bank_names[symbol]} ({symbol})", expanded=True):
                    self.display_bank_analysis(symbol, all_data[symbol])
        
        # Summary statistics
        st.markdown("## 📈 Analysis Summary")
        
        # Calculate summary stats
        total_analyses = sum(len(data) for data in all_data.values())
        latest_analyses = [self.get_latest_analysis(data) for data in all_data.values() if data]
        
        if latest_analyses:
            avg_sentiment = sum(a.get('overall_sentiment', 0) for a in latest_analyses) / len(latest_analyses)
            avg_confidence = sum(a.get('confidence', 0) for a in latest_analyses) / len(latest_analyses)
            total_news = sum(a.get('news_count', 0) for a in latest_analyses)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Analyses", total_analyses)
            
            with col2:
                st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
            
            with col3:
                st.metric("Average Confidence", f"{avg_confidence:.2f}")
            
            with col4:
                st.metric("Total News Articles", total_news)
        
        # Footer
        st.markdown("---")
        st.markdown("**Last Updated:** " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        st.markdown("**Data Source:** ASX Bank News Analysis System")

def main():
    """Run the dashboard"""
    dashboard = NewsAnalysisDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
