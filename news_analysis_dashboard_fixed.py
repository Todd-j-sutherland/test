#!/usr/bin/env python3
"""
Modern News Analysis Dashboard with Enhanced UI/UX
Interactive web dashboard displaying news sentiment analysis and technical indicators for Australian banks
"""

import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Import technical analysis
try:
    from src.technical_analysis import TechnicalAnalyzer, get_market_data
except ImportError:
    TechnicalAnalyzer = None
    get_market_data = None

# Add ML imports
import sqlite3
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ASX Bank Analytics Hub",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS with enhanced styling
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Card Components */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
        line-height: 1;
    }
    
    .metric-label {
        color: #6c757d;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    /* Bank Analysis Cards */
    .bank-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .bank-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.12);
    }
    
    /* Sentiment Colors */
    .positive { 
        color: #10b981; 
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    }
    .negative { 
        color: #ef4444; 
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
    }
    .neutral { 
        color: #6b7280; 
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
    }
    
    /* Section Headers */
    .section-header {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 2rem 0 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2937;
        margin: 0 0 0.5rem 0;
    }
    
    .section-subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin: 0;
    }
    
    /* Charts */
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .error-message {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    .success-message {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

class ModernNewsAnalysisDashboard:
    """Modern dashboard for displaying news analysis and technical analysis results"""
    
    def __init__(self):
        self.data_path = "data/sentiment_history"
        self.bank_symbols = ["CBA.AX", "WBC.AX", "ANZ.AX", "NAB.AX", "MQG.AX", "SUN.AX", "QBE.AX"]
        self.bank_names = {
            "CBA.AX": "Commonwealth Bank",
            "WBC.AX": "Westpac Banking Corp",
            "ANZ.AX": "ANZ Banking Group",
            "NAB.AX": "National Australia Bank",
            "MQG.AX": "Macquarie Group",
            "SUN.AX": "Suncorp Group",
            "QBE.AX": "QBE Insurance Group"
        }
        # Initialize technical analyzer if available
        self.tech_analyzer = TechnicalAnalyzer() if TechnicalAnalyzer else None
        self.technical_data = {}  # Cache for technical analysis
    
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
    
    def get_latest_analysis(self, data: List[Dict]) -> Optional[Dict]:
        """Get the most recent analysis from the data"""
        if not data:
            return None
        
        # Sort by timestamp and get the latest
        try:
            sorted_data = sorted(data, key=lambda x: x.get('timestamp', ''), reverse=True)
            return sorted_data[0] if sorted_data else None
        except Exception as e:
            logger.error(f"Error getting latest analysis: {e}")
            return data[-1] if data else None
    
    def create_modern_metric_card(self, title: str, value: str, subtitle: str = "", icon: str = "📊") -> str:
        """Create a modern metric card HTML"""
        return f"""
        <div class="metric-card">
            <div class="metric-label">{icon} {title}</div>
            <div class="metric-value">{value}</div>
            <div style="font-size: 0.8rem; color: #6b7280; margin-top: 0.5rem;">{subtitle}</div>
        </div>
        """
    
    def create_sentiment_overview_chart(self, all_data: Dict) -> go.Figure:
        """Create sentiment overview chart"""
        symbols = []
        sentiments = []
        colors = []
        
        for symbol, data in all_data.items():
            latest = self.get_latest_analysis(data)
            if latest:
                symbols.append(self.bank_names.get(symbol, symbol))
                sentiment = latest.get('overall_sentiment', 0)
                sentiments.append(sentiment)
                
                # Color based on sentiment
                if sentiment > 0.1:
                    colors.append('#10b981')  # Green for positive
                elif sentiment < -0.1:
                    colors.append('#ef4444')  # Red for negative
                else:
                    colors.append('#f59e0b')  # Yellow for neutral
        
        fig = go.Figure(go.Bar(
            x=symbols,
            y=sentiments,
            marker_color=colors,
            text=[f"{s:.3f}" for s in sentiments],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Sentiment: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Overall Sentiment Scores",
            xaxis_title="Banks",
            yaxis_title="Sentiment Score",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(255, 255, 255, 0.9)',
            paper_bgcolor='rgba(255, 255, 255, 0)',
            yaxis=dict(range=[-1, 1])
        )
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_hline(y=0.2, line_dash="dot", line_color="green", annotation_text="Positive Threshold")
        fig.add_hline(y=-0.2, line_dash="dot", line_color="red", annotation_text="Negative Threshold")
        
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
            marker_color='rgba(102, 126, 234, 0.8)',
            text=[f"{c:.2f}" for c in confidences],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Analysis Confidence Levels",
            xaxis_title="Banks",
            yaxis_title="Confidence Score",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(255, 255, 255, 0.9)',
            paper_bgcolor='rgba(255, 255, 255, 0)',
            yaxis=dict(range=[0, 1])
        )
        
        # Add confidence thresholds
        fig.add_hline(y=0.8, line_dash="dot", line_color="green", annotation_text="High Confidence")
        fig.add_hline(y=0.6, line_dash="dot", line_color="orange", annotation_text="Medium Confidence")
        fig.add_hline(y=0.4, line_dash="dot", line_color="red", annotation_text="Low Confidence")
        
        return fig
    
    def display_bank_analysis(self, symbol: str, data: List[Dict]):
        """Display analysis for a specific bank"""
        latest = self.get_latest_analysis(data)
        
        if not latest:
            st.markdown(f"""
            <div class="error-message">
                ⚠️ No analysis data available for {self.bank_names.get(symbol, symbol)}
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Bank header
        st.markdown(f"""
        <div class="bank-card">
            <h3 style="color: #1f2937; margin-bottom: 1rem;">
                🏦 {self.bank_names.get(symbol, symbol)} ({symbol})
            </h3>
        """, unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sentiment = latest.get('overall_sentiment', 0)
            sentiment_color = "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"
            metric_html = self.create_modern_metric_card(
                "Sentiment Score", 
                f"<span class='{sentiment_color}'>{sentiment:.3f}</span>",
                "Overall Market Feeling",
                "📊"
            )
            st.markdown(metric_html, unsafe_allow_html=True)
        
        with col2:
            confidence = latest.get('confidence', 0)
            conf_color = "positive" if confidence > 0.8 else "negative" if confidence < 0.6 else "neutral"
            metric_html = self.create_modern_metric_card(
                "Confidence", 
                f"<span class='{conf_color}'>{confidence:.2f}</span>",
                "Analysis Reliability",
                "🎯"
            )
            st.markdown(metric_html, unsafe_allow_html=True)
        
        with col3:
            news_count = latest.get('news_count', 0)
            metric_html = self.create_modern_metric_card(
                "News Articles", 
                str(news_count),
                "Analysis Sources",
                "📰"
            )
            st.markdown(metric_html, unsafe_allow_html=True)
        
        with col4:
            timestamp = latest.get('timestamp', 'Unknown')
            if timestamp != 'Unknown':
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_ago = datetime.now() - dt.replace(tzinfo=None)
                    if time_ago.days > 0:
                        time_str = f"{time_ago.days} days ago"
                    elif time_ago.seconds > 3600:
                        time_str = f"{time_ago.seconds // 3600} hours ago"
                    else:
                        time_str = f"{time_ago.seconds // 60} minutes ago"
                except:
                    time_str = "Recently"
            else:
                time_str = "Unknown"
            
            metric_html = self.create_modern_metric_card(
                "Last Updated", 
                time_str,
                "Data Freshness",
                "⏰"
            )
            st.markdown(metric_html, unsafe_allow_html=True)
        
        # Recent headlines if available
        if 'recent_headlines' in latest and latest['recent_headlines']:
            st.markdown("#### 📰 Recent Headlines")
            headlines = latest['recent_headlines'][:5]  # Show top 5
            for i, headline in enumerate(headlines):
                if headline:
                    st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.9); padding: 0.75rem; 
                                margin: 0.5rem 0; border-radius: 8px; border-left: 3px solid #667eea;">
                        <strong>#{i+1}:</strong> {headline}
                    </div>
                    """, unsafe_allow_html=True)
        
        # ML Predictions if available
        if 'ml_prediction' in latest and latest['ml_prediction']:
            ml_pred = latest['ml_prediction']
            if ml_pred.get('prediction') != 'UNKNOWN':
                st.markdown("#### 🤖 ML Prediction")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    prediction = ml_pred.get('prediction', 'UNKNOWN')
                    pred_color = "positive" if prediction == 'BUY' else "negative" if prediction == 'SELL' else "neutral"
                    metric_html = self.create_modern_metric_card(
                        "AI Signal", 
                        f"<span class='{pred_color}'>{prediction}</span>",
                        "Model Recommendation",
                        "🤖"
                    )
                    st.markdown(metric_html, unsafe_allow_html=True)
                
                with col2:
                    probability = ml_pred.get('probability', 0)
                    prob_color = "positive" if probability > 0.7 else "negative" if probability < 0.5 else "neutral"
                    metric_html = self.create_modern_metric_card(
                        "Probability", 
                        f"<span class='{prob_color}'>{probability:.1%}</span>",
                        "Success Likelihood",
                        "📈"
                    )
                    st.markdown(metric_html, unsafe_allow_html=True)
                
                with col3:
                    ml_confidence = ml_pred.get('confidence', 0)
                    ml_conf_color = "positive" if ml_confidence > 0.8 else "negative" if ml_confidence < 0.6 else "neutral"
                    metric_html = self.create_modern_metric_card(
                        "ML Confidence", 
                        f"<span class='{ml_conf_color}'>{ml_confidence:.1%}</span>",
                        "Algorithm Certainty",
                        "🔬"
                    )
                    st.markdown(metric_html, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def run_dashboard(self):
        """Main dashboard application"""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🏦 ASX Bank Analytics Hub</h1>
            <p>Advanced sentiment analysis & market intelligence for Australian banking sector</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load data
        with st.spinner("🔄 Loading sentiment analysis data..."):
            all_data = self.load_sentiment_data()
        
        # Sidebar controls
        st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
            <h3 style="margin: 0; text-align: center;">🎛️ Dashboard Controls</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Bank selection
        selected_banks = st.sidebar.multiselect(
            "🏦 Select Banks to Display",
            options=list(self.bank_names.keys()),
            default=list(self.bank_names.keys())[:4],  # Default to first 4
            format_func=lambda x: self.bank_names[x]
        )
        
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        
        # Overview section
        if selected_banks:
            filtered_data = {k: v for k, v in all_data.items() if k in selected_banks}
            
            st.markdown("""
            <div class="section-header">
                <div class="section-title">📊 Market Overview</div>
                <div class="section-subtitle">Real-time sentiment analysis across selected banks</div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                sentiment_chart = self.create_sentiment_overview_chart(filtered_data)
                st.plotly_chart(sentiment_chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                confidence_chart = self.create_confidence_distribution_chart(filtered_data)
                st.plotly_chart(confidence_chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Individual bank analyses
            st.markdown("""
            <div class="section-header">
                <div class="section-title">🏦 Individual Bank Analysis</div>
                <div class="section-subtitle">Detailed analysis for each selected bank</div>
            </div>
            """, unsafe_allow_html=True)
            
            for symbol in selected_banks:
                if symbol in filtered_data:
                    with st.expander(f"📊 {self.bank_names[symbol]} ({symbol})", expanded=False):
                        self.display_bank_analysis(symbol, filtered_data[symbol])
            
            # Summary statistics
            st.markdown("""
            <div class="section-header">
                <div class="section-title">📈 Summary Statistics</div>
                <div class="section-subtitle">Key metrics across all selected banks</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate summary stats
            latest_analyses = [self.get_latest_analysis(data) for data in filtered_data.values()]
            latest_analyses = [a for a in latest_analyses if a is not None]
            
            if latest_analyses:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_sentiment = sum(a.get('overall_sentiment', 0) for a in latest_analyses) / len(latest_analyses)
                    sent_color = "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral"
                    metric_html = self.create_modern_metric_card(
                        "Average Sentiment", 
                        f"<span class='{sent_color}'>{avg_sentiment:.3f}</span>",
                        "Market Mood",
                        "📊"
                    )
                    st.markdown(metric_html, unsafe_allow_html=True)
                
                with col2:
                    avg_confidence = sum(a.get('confidence', 0) for a in latest_analyses) / len(latest_analyses)
                    conf_color = "positive" if avg_confidence > 0.7 else "negative" if avg_confidence < 0.5 else "neutral"
                    metric_html = self.create_modern_metric_card(
                        "Average Confidence", 
                        f"<span class='{conf_color}'>{avg_confidence:.2f}</span>",
                        "Analysis Quality",
                        "🎯"
                    )
                    st.markdown(metric_html, unsafe_allow_html=True)
                
                with col3:
                    total_news = sum(a.get('news_count', 0) for a in latest_analyses)
                    metric_html = self.create_modern_metric_card(
                        "Total News Articles", 
                        str(total_news),
                        "Analysis Sources",
                        "📰"
                    )
                    st.markdown(metric_html, unsafe_allow_html=True)
                
                with col4:
                    positive_banks = sum(1 for a in latest_analyses if a.get('overall_sentiment', 0) > 0.1)
                    negative_banks = sum(1 for a in latest_analyses if a.get('overall_sentiment', 0) < -0.1)
                    neutral_banks = len(latest_analyses) - positive_banks - negative_banks
                    
                    sentiment_dist = f"🟢{positive_banks} 🟡{neutral_banks} 🔴{negative_banks}"
                    metric_html = self.create_modern_metric_card(
                        "Sentiment Distribution", 
                        sentiment_dist,
                        "Pos/Neu/Neg",
                        "🏦"
                    )
                    st.markdown(metric_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="error-message">
                ⚠️ Please select at least one bank to display analysis
            </div>
            """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
        <div style="margin-top: 3rem; padding: 2rem; background: rgba(255, 255, 255, 0.95); 
                    border-radius: 12px; text-align: center;">
            <div style="color: #6b7280; font-size: 0.9rem;">
                <strong>Last Updated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
                <strong>System:</strong> ASX Bank Analytics Hub | 
                <strong>Status:</strong> <span style="color: #10b981;">🟢 Active</span>
            </div>
        </div>
        """.format(datetime=datetime), unsafe_allow_html=True)

def main():
    """Run the dashboard"""
    dashboard = ModernNewsAnalysisDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
