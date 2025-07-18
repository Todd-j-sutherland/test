#!/usr/bin/env python3
"""
News Analysis Dashboard with Technical Analysis Integration
Professional interactive web dashboard displaying news sentiment analysis and technical indicators for Australian banks
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
import numpy as np
import sqlite3

# Import technical analysis and config
import sys
sys.path.append('..')  # Add parent directory to path
from src.technical_analysis import TechnicalAnalyzer, get_market_data
from config.settings import Settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ASX Bank Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-color: #2c3e50;
        --secondary-color: #3498db;
        --accent-color: #e74c3c;
        --success-color: #27ae60;
        --warning-color: #f39c12;
        --info-color: #17a2b8;
        --light-gray: #f8f9fa;
        --medium-gray: #6c757d;
        --dark-gray: #343a40;
        --border-color: #dee2e6;
        --shadow: 0 2px 4px rgba(0,0,0,0.1);
        --shadow-hover: 0 4px 8px rgba(0,0,0,0.15);
        --border-radius: 8px;
        --border-radius-lg: 12px;
    }
    
    /* Global font styling */
    .main, .sidebar, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 2rem;
        border-radius: var(--border-radius-lg);
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: var(--shadow);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
        text-align: center;
        height: 100%;
    }
    
    .metric-card:hover {
        box-shadow: var(--shadow-hover);
        transform: translateY(-2px);
    }
    
    .metric-card h4 {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--medium-gray);
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    
    .metric-subtitle {
        font-size: 0.75rem;
        color: var(--medium-gray);
        margin-top: 0.25rem;
        font-weight: 500;
    }
    
    /* Status indicators */
    .status-positive { color: var(--success-color); }
    .status-negative { color: var(--accent-color); }
    .status-neutral { color: var(--medium-gray); }
    .status-warning { color: var(--warning-color); }
    
    /* Enhanced confidence indicators */
    .confidence-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .confidence-high {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .confidence-medium {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        color: #856404;
        border: 1px solid #ffeeba;
    }
    
    .confidence-low {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    /* Professional section headers */
    .section-header {
        background: white;
        padding: 1.5rem;
        border-radius: var(--border-radius) var(--border-radius) 0 0;
        border-bottom: 3px solid var(--secondary-color);
        margin-bottom: 0;
        box-shadow: var(--shadow);
    }
    
    .section-header h2 {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--primary-color);
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-header .subtitle {
        color: var(--medium-gray);
        font-size: 0.875rem;
        font-weight: 400;
        margin-top: 0.25rem;
    }
    
    /* Enhanced bank cards */
    .bank-card {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius-lg);
        padding: 0;
        margin: 1.5rem 0;
        box-shadow: var(--shadow);
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .bank-card:hover {
        box-shadow: var(--shadow-hover);
        border-color: var(--secondary-color);
    }
    
    .bank-card-header {
        background: linear-gradient(135deg, var(--light-gray) 0%, #e9ecef 100%);
        padding: 1.25rem 1.5rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .bank-card-header h3 {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--primary-color);
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .bank-card-body {
        padding: 1.5rem;
    }
    
    /* Enhanced news items */
    .news-item {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 1rem;
        margin: 0.75rem 0;
        transition: all 0.3s ease;
        border-left: 4px solid var(--border-color);
    }
    
    .news-item:hover {
        box-shadow: var(--shadow);
        border-left-color: var(--secondary-color);
    }
    
    .news-item.positive { border-left-color: var(--success-color); }
    .news-item.negative { border-left-color: var(--accent-color); }
    .news-item.neutral { border-left-color: var(--medium-gray); }
    
    /* Enhanced event badges */
    .event-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.375rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0.25rem 0.25rem 0.25rem 0;
    }
    
    .event-positive {
        background: linear-gradient(135deg, var(--success-color) 0%, #229954 100%);
        color: white;
    }
    
    .event-negative {
        background: linear-gradient(135deg, var(--accent-color) 0%, #c0392b 100%);
        color: white;
    }
    
    .event-neutral {
        background: linear-gradient(135deg, var(--medium-gray) 0%, #5a6268 100%);
        color: white;
    }
    
    /* Professional table styling */
    .stDataFrame {
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        overflow: hidden;
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--light-gray) 0%, white 100%);
    }
    
    /* Professional buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--secondary-color) 0%, #2980b9 100%);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: var(--shadow);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2980b9 0%, #1f618d 100%);
        box-shadow: var(--shadow-hover);
        transform: translateY(-1px);
    }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--light-gray);
        padding: 0.5rem;
        border-radius: var(--border-radius);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--secondary-color);
        color: white;
        border-color: var(--secondary-color);
    }
    
    /* Legend containers */
    .legend-container {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius-lg);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }
    
    .legend-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--primary-color);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Professional alerts */
    .alert {
        padding: 1rem 1.25rem;
        border-radius: var(--border-radius);
        border: 1px solid;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .alert-info {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-color: #bee5eb;
        color: #0c5460;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        border-color: #ffeeba;
        color: #856404;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-color: #c3e6cb;
        color: #155724;
    }
    
    /* Footer styling */
    .footer {
        background: var(--primary-color);
        color: white;
        padding: 1.5rem;
        border-radius: var(--border-radius);
        text-align: center;
        margin-top: 2rem;
        font-size: 0.875rem;
    }
    
    /* Loading animations */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .bank-card-header {
            padding: 1rem;
        }
        
        .bank-card-body {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

class NewsAnalysisDashboard:
    """Professional dashboard for displaying news analysis and technical analysis results"""
    
    def __init__(self):
        self.data_path = "data/sentiment_history"
        self.settings = Settings()
        self.bank_symbols = self.settings.BANK_SYMBOLS
        self.bank_names = {
            "CBA.AX": "Commonwealth Bank",
            "WBC.AX": "Westpac Banking Corp",
            "ANZ.AX": "ANZ Banking Group",
            "NAB.AX": "National Australia Bank",
            "MQG.AX": "Macquarie Group",
            "SUN.AX": "Suncorp Group",
            "QBE.AX": "QBE Insurance Group"
        }
        # Initialize technical analyzer
        self.tech_analyzer = TechnicalAnalyzer()
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
            return f"+{score:.3f}", "status-positive"
        elif score < -0.2:
            return f"{score:.3f}", "status-negative"
        else:
            return f"{score:.3f}", "status-neutral"
    
    def get_confidence_level(self, confidence: float) -> tuple:
        """Get confidence level description and CSS class"""
        if confidence >= 0.8:
            return "HIGH", "confidence-high"
        elif confidence >= 0.6:
            return "MEDIUM", "confidence-medium"
        else:
            return "LOW", "confidence-low"
    
    def create_professional_header(self):
        """Create professional dashboard header"""
        st.markdown("""
        <div class="main-header">
            <h1>📊 ASX Bank Analytics Platform</h1>
            <p>Professional sentiment analysis and technical indicators for Australian banking sector</p>
        </div>
        """, unsafe_allow_html=True)
    
    def create_section_header(self, title: str, subtitle: str = "", icon: str = ""):
        """Create professional section header"""
        st.markdown(f"""
        <div class="section-header">
            <h2>{icon} {title}</h2>
            {f'<div class="subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        """, unsafe_allow_html=True)
    
    def display_professional_metrics(self, metrics: List[Dict]):
        """Display metrics in professional card layout"""
        cols = st.columns(len(metrics))
        
        for i, metric in enumerate(metrics):
            with cols[i]:
                value = metric.get('value', 'N/A')
                delta = metric.get('delta', '')
                status = metric.get('status', 'neutral')
                subtitle = metric.get('subtitle', '')
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{metric['title']}</h4>
                    <div class="metric-value status-{status}">{value}</div>
                    {f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ''}
                    {f'<div class="metric-subtitle status-{status}">{delta}</div>' if delta else ''}
                </div>
                """, unsafe_allow_html=True)
    
    def create_sentiment_overview_chart(self, all_data: Dict) -> go.Figure:
        """Create professional overview chart of all bank sentiments"""
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
                
                # Professional color scheme
                if score > 0.2:
                    colors.append('#27ae60')  # Success green
                elif score < -0.2:
                    colors.append('#e74c3c')  # Professional red
                else:
                    colors.append('#6c757d')  # Neutral gray
        
        fig = go.Figure()
        
        # Add sentiment bars with professional styling
        fig.add_trace(go.Bar(
            x=symbols,
            y=scores,
            name='Sentiment Score',
            marker=dict(
                color=colors,
                line=dict(color='white', width=1)
            ),
            text=[f"{s:.3f}" for s in scores],
            textposition='auto',
            textfont=dict(size=12, family="Inter"),
            hovertemplate='<b>%{x}</b><br>' +
                         'Sentiment: %{y:.3f}<br>' +
                         'Confidence: %{customdata:.2f}<br>' +
                         '<extra></extra>',
            customdata=confidences
        ))
        
        fig.update_layout(
            title=dict(
                text="Bank Sentiment Overview",
                font=dict(size=18, family="Inter", weight=600, color="#2c3e50")
            ),
            xaxis=dict(
                title=dict(text="Banks", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                gridcolor="#f1f1f1",
                linecolor="#dee2e6"
            ),
            yaxis=dict(
                title=dict(text="Sentiment Score", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                range=[-1, 1],
                gridcolor="#f1f1f1",
                linecolor="#dee2e6",
                zeroline=True,
                zerolinecolor="#dee2e6",
                zerolinewidth=2
            ),
            height=400,
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter"),
            margin=dict(l=60, r=60, t=60, b=60)
        )
        
        return fig
    
    def create_confidence_distribution_chart(self, all_data: Dict) -> go.Figure:
        """Create professional confidence distribution chart"""
        symbols = []
        confidences = []
        
        for symbol, data in all_data.items():
            latest = self.get_latest_analysis(data)
            if latest:
                symbols.append(self.bank_names.get(symbol, symbol))
                confidences.append(latest.get('confidence', 0))
        
        # Professional color mapping for confidence
        colors = []
        for c in confidences:
            if c >= 0.8:
                colors.append('#27ae60')  # High confidence - green
            elif c >= 0.6:
                colors.append('#f39c12')  # Medium confidence - orange
            else:
                colors.append('#e74c3c')  # Low confidence - red
        
        fig = go.Figure(go.Bar(
            x=symbols,
            y=confidences,
            marker=dict(
                color=colors,
                line=dict(color='white', width=1)
            ),
            text=[f"{c:.2f}" for c in confidences],
            textposition='auto',
            textfont=dict(size=12, family="Inter"),
            hovertemplate='<b>%{x}</b><br>' +
                         'Confidence: %{y:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text="Analysis Confidence Levels",
                font=dict(size=18, family="Inter", weight=600, color="#2c3e50")
            ),
            xaxis=dict(
                title=dict(text="Banks", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                gridcolor="#f1f1f1",
                linecolor="#dee2e6"
            ),
            yaxis=dict(
                title=dict(text="Confidence Score", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                range=[0, 1],
                gridcolor="#f1f1f1",
                linecolor="#dee2e6"
            ),
            height=400,
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter"),
            margin=dict(l=60, r=60, t=60, b=60)
        )
        
        return fig
    
    def create_news_impact_chart(self, news_data: List[Dict], events_data: Dict) -> go.Figure:
        """Create professional chart showing news impact on sentiment"""
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
            # Create empty chart with professional styling
            fig = go.Figure()
            fig.add_annotation(
                text="No news impact data available",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, family="Inter", color="#6c757d")
            )
            fig.update_layout(
                title=dict(
                    text="News Impact on Sentiment",
                    font=dict(size=18, family="Inter", weight=600, color="#2c3e50")
                ),
                height=300,
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white"
            )
            return fig
        
        df = pd.DataFrame(impact_data)
        
        # Professional color mapping
        colors = ['#27ae60' if i > 0 else '#e74c3c' if i < 0 else '#6c757d' for i in df['impact']]
        
        fig = go.Figure(go.Bar(
            x=df['title'],
            y=df['impact'],
            marker=dict(
                color=colors,
                line=dict(color='white', width=1)
            ),
            hovertemplate='<b>%{x}</b><br>' +
                         'Impact: %{y:.3f}<br>' +
                         'Source: %{customdata}<br>' +
                         '<extra></extra>',
            customdata=df['source']
        ))
        
        fig.update_layout(
            title=dict(
                text="News Impact on Sentiment",
                font=dict(size=18, family="Inter", weight=600, color="#2c3e50")
            ),
            xaxis=dict(
                title=dict(text="News Articles", font=dict(size=14, family="Inter")),
                tickfont=dict(size=10, family="Inter"),
                tickangle=-45,
                gridcolor="#f1f1f1",
                linecolor="#dee2e6"
            ),
            yaxis=dict(
                title=dict(text="Sentiment Impact", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                gridcolor="#f1f1f1",
                linecolor="#dee2e6"
            ),
            height=400,
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter"),
            margin=dict(l=60, r=60, t=60, b=100)
        )
        
        return fig
    
    def display_confidence_legend(self):
        """Display professional confidence score legend and decision criteria"""
        st.markdown("""
        <div class="legend-container">
            <div class="legend-title">📊 Confidence Score Legend & Decision Criteria</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card confidence-high">
                <h4>🟢 HIGH CONFIDENCE (≥0.8)</h4>
                <div style="margin-top: 1rem;">
                    <p><strong>Action:</strong> Strong Buy/Sell Signal</p>
                    <p><strong>Criteria:</strong> Multiple reliable sources, consistent sentiment, significant news volume</p>
                    <p><strong>Decision:</strong> Execute trades with full position sizing</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card confidence-medium">
                <h4>🟡 MEDIUM CONFIDENCE (0.6-0.8)</h4>
                <div style="margin-top: 1rem;">
                    <p><strong>Action:</strong> Moderate Buy/Sell Signal</p>
                    <p><strong>Criteria:</strong> Some reliable sources, moderate sentiment consistency</p>
                    <p><strong>Decision:</strong> Execute trades with reduced position sizing</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card confidence-low">
                <h4>🔴 LOW CONFIDENCE (<0.6)</h4>
                <div style="margin-top: 1rem;">
                    <p><strong>Action:</strong> Hold/Monitor</p>
                    <p><strong>Criteria:</strong> Limited sources, inconsistent sentiment, low news volume</p>
                    <p><strong>Decision:</strong> Avoid trading, wait for better signals</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def display_sentiment_scale(self):
        """Display professional sentiment scale explanation"""
        st.markdown("""
        <div class="legend-container">
            <div class="legend-title">📈 Sentiment Score Scale</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown('<div class="metric-card"><h4>Very Negative</h4><div class="metric-value status-negative">-1.0 to -0.5</div><div class="metric-subtitle">Strong Sell</div></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card"><h4>Negative</h4><div class="metric-value status-negative">-0.5 to -0.2</div><div class="metric-subtitle">Sell</div></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card"><h4>Neutral</h4><div class="metric-value status-neutral">-0.2 to +0.2</div><div class="metric-subtitle">Hold</div></div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card"><h4>Positive</h4><div class="metric-value status-positive">+0.2 to +0.5</div><div class="metric-subtitle">Buy</div></div>', unsafe_allow_html=True)
        
        with col5:
            st.markdown('<div class="metric-card"><h4>Very Positive</h4><div class="metric-value status-positive">+0.5 to +1.0</div><div class="metric-subtitle">Strong Buy</div></div>', unsafe_allow_html=True)
    
    def display_bank_analysis(self, symbol: str, data: List[Dict]):
        """Display detailed analysis for a specific bank with professional styling"""
        latest = self.get_latest_analysis(data)
        
        if not latest:
            st.markdown(f"""
            <div class="alert alert-warning">
                <strong>No Data Available</strong><br>
                No analysis data available for {self.bank_names.get(symbol, symbol)}
            </div>
            """, unsafe_allow_html=True)
            return
        
        bank_name = self.bank_names.get(symbol, symbol)
        
        # Professional bank card header
        st.markdown(f"""
        <div class="bank-card">
            <div class="bank-card-header">
                <h3>🏦 {bank_name} ({symbol})</h3>
            </div>
            <div class="bank-card-body">
        """, unsafe_allow_html=True)
        
        # Key metrics with professional layout
        sentiment = latest.get('overall_sentiment', 0)
        confidence = latest.get('confidence', 0)
        news_count = latest.get('news_count', 0)
        timestamp = latest.get('timestamp', 'Unknown')
        
        # Format timestamp
        if timestamp != 'Unknown':
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime('%Y-%m-%d %H:%M')
            except:
                time_str = timestamp[:16]
        else:
            time_str = 'Unknown'
        
        # Prepare metrics data
        score_text, score_class = self.format_sentiment_score(sentiment)
        conf_level, conf_class = self.get_confidence_level(confidence)
        
        metrics = [
            {
                'title': 'Sentiment Score',
                'value': score_text,
                'status': score_class.replace('status-', ''),
                'subtitle': 'Current market sentiment'
            },
            {
                'title': 'Confidence Level',
                'value': f"{confidence:.2f}",
                'status': 'neutral',
                'subtitle': f"{conf_level} confidence"
            },
            {
                'title': 'News Articles',
                'value': str(news_count),
                'status': 'neutral',
                'subtitle': 'Articles analyzed'
            },
            {
                'title': 'Last Updated',
                'value': time_str.split(' ')[0],
                'status': 'neutral',
                'subtitle': time_str.split(' ')[1] if ' ' in time_str else ''
            }
        ]
        
        self.display_professional_metrics(metrics)
        
        # Enhanced sentiment components breakdown
        if 'sentiment_components' in latest:
            st.markdown("#### 📊 Sentiment Components Analysis")
            components = latest['sentiment_components']
            
            # Get technical analysis for additional components
            tech_analysis = self.get_technical_analysis(symbol)
            
            # Calculate technical components
            volume_score = 0
            momentum_score = 0
            ml_trading_score = 0
            
            if tech_analysis and 'momentum' in tech_analysis:
                momentum_data = tech_analysis['momentum']
                momentum_score = max(0, min(100, (momentum_data['score'] + 100) / 2))
                
                if momentum_data['volume_momentum'] == 'high':
                    volume_score = 75
                elif momentum_data['volume_momentum'] == 'normal':
                    volume_score = 50
                else:
                    volume_score = 25
                
                if 'overall_signal' in tech_analysis:
                    ml_trading_score = max(0, min(100, (tech_analysis['overall_signal'] + 100) / 2))
            
            comp_df = pd.DataFrame([
                {'Component': 'News Sentiment', 'Score': f"{components.get('news', 0):.1f}", 'Weight': '35%', 'Status': '🟢' if components.get('news', 0) > 0 else '🔴' if components.get('news', 0) < 0 else '🟡'},
                {'Component': 'Social Media', 'Score': f"{components.get('reddit', 0):.1f}", 'Weight': '15%', 'Status': '🟢' if components.get('reddit', 0) > 0 else '🔴' if components.get('reddit', 0) < 0 else '🟡'},
                {'Component': 'Market Events', 'Score': f"{components.get('events', 0):.1f}", 'Weight': '20%', 'Status': '🟢' if components.get('events', 0) > 0 else '🔴' if components.get('events', 0) < 0 else '🟡'},
                {'Component': 'Volume Analysis', 'Score': f"{volume_score:.1f}", 'Weight': '10%', 'Status': '🟢' if volume_score > 60 else '🔴' if volume_score < 40 else '🟡'},
                {'Component': 'Technical Momentum', 'Score': f"{momentum_score:.1f}", 'Weight': '10%', 'Status': '🟢' if momentum_score > 60 else '🔴' if momentum_score < 40 else '🟡'},
                {'Component': 'ML Prediction', 'Score': f"{ml_trading_score:.1f}", 'Weight': '10%', 'Status': '🟢' if ml_trading_score > 60 else '🔴' if ml_trading_score < 40 else '🟡'}
            ])
            
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        # Professional headlines section
        if 'recent_headlines' in latest:
            st.markdown("#### 📰 Recent Market Headlines")
            headlines = latest['recent_headlines']
            
            for i, headline in enumerate(headlines[:5]):
                if headline:
                    st.markdown(f"""
                    <div class="news-item neutral">
                        <strong>#{i+1}</strong> {headline}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Enhanced significant events section
        if 'significant_events' in latest:
            st.markdown("#### 🚨 Significant Market Events")
            events = latest['significant_events']
            
            if 'events_detected' in events and events['events_detected']:
                for event in events['events_detected']:
                    event_type = event.get('type', 'unknown')
                    headline = event.get('headline', 'No headline')
                    sentiment_impact = event.get('sentiment_impact', 0)
                    
                    # Determine event styling
                    if sentiment_impact > 0.1:
                        event_class = "positive"
                        badge_class = "event-positive"
                        impact_icon = "📈"
                    elif sentiment_impact < -0.1:
                        event_class = "negative" 
                        badge_class = "event-negative"
                        impact_icon = "📉"
                    else:
                        event_class = "neutral"
                        badge_class = "event-neutral"
                        impact_icon = "➡️"
                    
                    st.markdown(f"""
                    <div class="news-item {event_class}">
                        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
                            <span class="event-badge {badge_class}">{event_type.replace('_', ' ').title()}</span>
                            <span style="font-size: 1.2rem;">{impact_icon}</span>
                        </div>
                        <div style="font-weight: 600; margin-bottom: 0.5rem;">{headline}</div>
                        <div style="color: #6c757d; font-size: 0.875rem;">
                            Sentiment Impact: <span class="status-{'positive' if sentiment_impact > 0 else 'negative' if sentiment_impact < 0 else 'neutral'}">{sentiment_impact:+.3f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert alert-info">
                    <strong>Market Status:</strong> No significant events detected in recent analysis
                </div>
                """, unsafe_allow_html=True)
        
        # Professional news sentiment analysis details
        if 'news_sentiment' in latest:
            news_data = latest['news_sentiment']
            
            st.markdown("#### 📊 Detailed News Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Sentiment Distribution:**")
                if 'sentiment_distribution' in news_data:
                    dist = news_data['sentiment_distribution']
                    dist_df = pd.DataFrame([
                        {'Category': '🟢 Very Positive', 'Count': dist.get('very_positive', 0), 'Percentage': f"{(dist.get('very_positive', 0) / max(1, sum(dist.values())) * 100):.1f}%"},
                        {'Category': '🟢 Positive', 'Count': dist.get('positive', 0), 'Percentage': f"{(dist.get('positive', 0) / max(1, sum(dist.values())) * 100):.1f}%"},
                        {'Category': '🟡 Neutral', 'Count': dist.get('neutral', 0), 'Percentage': f"{(dist.get('neutral', 0) / max(1, sum(dist.values())) * 100):.1f}%"},
                        {'Category': '🔴 Negative', 'Count': dist.get('negative', 0), 'Percentage': f"{(dist.get('negative', 0) / max(1, sum(dist.values())) * 100):.1f}%"},
                        {'Category': '🔴 Very Negative', 'Count': dist.get('very_negative', 0), 'Percentage': f"{(dist.get('very_negative', 0) / max(1, sum(dist.values())) * 100):.1f}%"}
                    ])
                    st.dataframe(dist_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**Analysis Method Performance:**")
                if 'method_breakdown' in news_data:
                    method = news_data['method_breakdown']
                    if 'traditional' in method and 'composite' in method:
                        method_df = pd.DataFrame([
                            {'Method': 'Traditional NLP', 'Score': f"{method['traditional'].get('mean', 0):.3f}", 'Accuracy': 'Good'},
                            {'Method': 'Transformer Models', 'Score': f"{method['transformer'].get('mean', 0):.3f}", 'Accuracy': 'High'},
                            {'Method': 'Composite Final', 'Score': f"{method['composite'].get('mean', 0):.3f}", 'Accuracy': 'Optimal'}
                        ])
                        st.dataframe(method_df, use_container_width=True, hide_index=True)
        
        # Professional Reddit sentiment section
        if 'reddit_sentiment' in latest:
            reddit_data = latest['reddit_sentiment']
            st.markdown("#### 🔴 Social Media Sentiment")
            
            reddit_metrics = [
                {
                    'title': 'Reddit Score',
                    'value': f"{reddit_data.get('overall_sentiment', 0):.3f}",
                    'status': 'positive' if reddit_data.get('overall_sentiment', 0) > 0 else 'negative' if reddit_data.get('overall_sentiment', 0) < 0 else 'neutral',
                    'subtitle': 'Social sentiment'
                },
                {
                    'title': 'Posts Analyzed',
                    'value': str(reddit_data.get('post_count', 0)),
                    'status': 'neutral',
                    'subtitle': 'Discussion volume'
                }
            ]
            
            self.display_professional_metrics(reddit_metrics)
        
        # Enhanced Historical Trends Charts
        st.markdown("#### 📈 Historical Performance Analysis")
        
        chart_tabs = st.tabs(["📊 Trend Analysis", "🔗 Correlation Matrix", "📋 Performance Summary"])
        
        with chart_tabs[0]:
            historical_chart = self.create_historical_trends_chart(symbol, {symbol: data})
            st.plotly_chart(historical_chart, use_container_width=True)
            
            st.markdown("""
            <div class="alert alert-info">
                <strong>Chart Analysis:</strong> Blue line shows sentiment progression, orange indicates confidence levels, 
                green represents normalized stock price movement, and colored bars display daily momentum shifts.
            </div>
            """, unsafe_allow_html=True)
        
        with chart_tabs[1]:
            correlation_chart = self.create_correlation_chart(symbol, {symbol: data})
            st.plotly_chart(correlation_chart, use_container_width=True)
            
            st.markdown("""
            <div class="alert alert-info">
                <strong>Correlation Insights:</strong> Each point represents analysis correlation between sentiment and price movement. 
                Point size indicates confidence level, with trend line showing overall relationship strength.
            </div>
            """, unsafe_allow_html=True)
        
        with chart_tabs[2]:
            # Performance summary metrics
            performance_metrics = self.calculate_performance_metrics(symbol, data)
            if performance_metrics:
                st.markdown("**Performance Summary (Last 30 Days)**")
                self.display_professional_metrics(performance_metrics)

        # Enhanced Technical Analysis Section
        st.markdown("#### 📈 Technical Analysis & Market Indicators")
        
        tech_analysis = self.get_technical_analysis(symbol)
        if tech_analysis and tech_analysis.get('current_price', 0) > 0:
            # Technical metrics with professional styling
            current_price = tech_analysis.get('current_price', 0)
            momentum_score = tech_analysis.get('momentum', {}).get('score', 0)
            recommendation = tech_analysis.get('recommendation', 'HOLD')
            rsi = tech_analysis.get('indicators', {}).get('rsi', 50)
            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Normal"
            
            technical_metrics = [
                {
                    'title': 'Current Price',
                    'value': f"${current_price:.2f}",
                    'status': 'neutral',
                    'subtitle': 'Latest market price'
                },
                {
                    'title': 'Momentum Score',
                    'value': f"{momentum_score:.1f}",
                    'status': 'positive' if momentum_score > 5 else 'negative' if momentum_score < -5 else 'neutral',
                    'subtitle': 'Technical momentum'
                },
                {
                    'title': 'Technical Signal',
                    'value': recommendation,
                    'status': 'positive' if 'BUY' in recommendation else 'negative' if 'SELL' in recommendation else 'neutral',
                    'subtitle': 'Trading recommendation'
                },
                {
                    'title': 'RSI Indicator',
                    'value': f"{rsi:.1f}",
                    'status': 'warning' if rsi > 70 or rsi < 30 else 'neutral',
                    'subtitle': rsi_status
                }
            ]
            
            self.display_professional_metrics(technical_metrics)
            
            # Detailed technical indicators with professional tables
            st.markdown("**📊 Technical Indicators Overview:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                indicators = tech_analysis.get('indicators', {})
                macd = indicators.get('macd', {})
                
                indicator_data = []
                indicator_data.append({
                    "Indicator": "RSI (14)", 
                    "Value": f"{indicators.get('rsi', 0):.2f}", 
                    "Signal": "🔴 Overbought" if indicators.get('rsi', 50) > 70 else "🟢 Oversold" if indicators.get('rsi', 50) < 30 else "🟡 Neutral",
                    "Strength": "Strong" if abs(indicators.get('rsi', 50) - 50) > 20 else "Moderate"
                })
                
                macd_signal = "🟢 Bullish" if macd.get('histogram', 0) > 0 else "🔴 Bearish"
                indicator_data.append({
                    "Indicator": "MACD", 
                    "Value": f"{macd.get('line', 0):.4f}", 
                    "Signal": macd_signal,
                    "Strength": "Strong" if abs(macd.get('histogram', 0)) > 0.1 else "Weak"
                })
                
                if 'sma' in indicators:
                    sma_20 = indicators['sma'].get('sma_20', 0)
                    sma_50 = indicators['sma'].get('sma_50', 0)
                    price_vs_sma20 = "🟢 Above" if current_price > sma_20 else "🔴 Below"
                    price_vs_sma50 = "🟢 Above" if current_price > sma_50 else "🔴 Below"
                    
                    indicator_data.append({
                        "Indicator": "SMA 20", 
                        "Value": f"${sma_20:.2f}", 
                        "Signal": f"Price {price_vs_sma20}",
                        "Strength": "Strong" if abs(current_price - sma_20) / sma_20 > 0.02 else "Weak"
                    })
                    
                    indicator_data.append({
                        "Indicator": "SMA 50", 
                        "Value": f"${sma_50:.2f}", 
                        "Signal": f"Price {price_vs_sma50}",
                        "Strength": "Strong" if abs(current_price - sma_50) / sma_50 > 0.05 else "Weak"
                    })
                
                st.dataframe(pd.DataFrame(indicator_data), use_container_width=True, hide_index=True)
            
            with col2:
                momentum = tech_analysis.get('momentum', {})
                trend = tech_analysis.get('trend', {})
                
                momentum_data = []
                momentum_data.append({
                    "Timeframe": "1-Day Change", 
                    "Value": f"{momentum.get('price_change_1d', 0):+.2f}%",
                    "Status": "🟢" if momentum.get('price_change_1d', 0) > 0 else "🔴",
                    "Significance": "High" if abs(momentum.get('price_change_1d', 0)) > 2 else "Low"
                })
                
                momentum_data.append({
                    "Timeframe": "5-Day Change", 
                    "Value": f"{momentum.get('price_change_5d', 0):+.2f}%",
                    "Status": "🟢" if momentum.get('price_change_5d', 0) > 0 else "🔴",
                    "Significance": "High" if abs(momentum.get('price_change_5d', 0)) > 5 else "Low"
                })
                
                momentum_data.append({
                    "Timeframe": "20-Day Change", 
                    "Value": f"{momentum.get('price_change_20d', 0):+.2f}%",
                    "Status": "🟢" if momentum.get('price_change_20d', 0) > 0 else "🔴",
                    "Significance": "High" if abs(momentum.get('price_change_20d', 0)) > 10 else "Low"
                })
                
                momentum_data.append({
                    "Timeframe": "Trend Direction", 
                    "Value": trend.get('direction', 'Unknown').replace('_', ' ').title(),
                    "Status": "🟢" if 'up' in trend.get('direction', '') else "🔴" if 'down' in trend.get('direction', '') else "🟡",
                    "Significance": trend.get('strength', 'Moderate').title()
                })
                
                momentum_data.append({
                    "Timeframe": "Volume Pattern", 
                    "Value": momentum.get('volume_momentum', 'Normal').title(),
                    "Status": "🟢" if momentum.get('volume_momentum') == 'high' else "🔴" if momentum.get('volume_momentum') == 'low' else "🟡",
                    "Significance": "Market Interest"
                })
                
                st.dataframe(pd.DataFrame(momentum_data), use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div class="alert alert-warning">
                <strong>Technical Data Unavailable</strong><br>
                Technical analysis data is currently not available for this symbol. Please try refreshing or check back later.
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced ML Prediction Section
        st.markdown("#### 🤖 Machine Learning Predictions")
        
        ml_prediction = latest.get('ml_prediction', {})
        
        if ml_prediction and ml_prediction.get('prediction') != 'UNKNOWN':
            prediction = ml_prediction['prediction']
            probability = ml_prediction.get('probability', 0)
            ml_confidence = ml_prediction.get('confidence', 0)
            threshold = ml_prediction.get('threshold', 0.5)
            
            ml_metrics = [
                {
                    'title': 'ML Signal',
                    'value': f"{'📈 PROFITABLE' if prediction == 'PROFITABLE' else '📉 UNPROFITABLE'}",
                    'status': 'positive' if prediction == 'PROFITABLE' else 'negative',
                    'subtitle': 'Model prediction'
                },
                {
                    'title': 'Probability',
                    'value': f"{probability:.1%}",
                    'status': 'positive' if probability > 0.7 else 'warning' if probability > 0.6 else 'negative',
                    'subtitle': 'Prediction confidence'
                },
                {
                    'title': 'Model Confidence',
                    'value': f"{ml_confidence:.1%}",
                    'status': 'positive' if ml_confidence > 0.8 else 'warning' if ml_confidence > 0.6 else 'negative',
                    'subtitle': 'Algorithm certainty'
                },
                {
                    'title': 'Decision Threshold',
                    'value': f"{threshold:.2f}",
                    'status': 'neutral',
                    'subtitle': 'Classification cutoff'
                }
            ]
            
            self.display_professional_metrics(ml_metrics)
            
            # Show ML model performance if available
            if st.button("📊 View Model Performance Details", key=f"ml_perf_{symbol}"):
                self.display_ml_performance(symbol)
        else:
            st.markdown("""
            <div class="alert alert-info">
                <strong>ML Model Status</strong><br>
                Machine learning predictions are not currently available. Train models using: 
                <code>python scripts/retrain_ml_models.py</code>
            </div>
            """, unsafe_allow_html=True)
        
        # Close bank card
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    def calculate_performance_metrics(self, symbol: str, data: List[Dict]) -> List[Dict]:
        """Calculate performance metrics for the last 30 days"""
        if len(data) < 2:
            return []
        
        recent_data = data[-30:]  # Last 30 entries
        
        # Calculate metrics
        sentiments = [entry.get('overall_sentiment', 0) for entry in recent_data]
        confidences = [entry.get('confidence', 0) for entry in recent_data]
        
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        sentiment_volatility = np.std(sentiments) if len(sentiments) > 1 else 0
        
        # Trend calculation
        if len(sentiments) >= 2:
            recent_trend = sentiments[-1] - sentiments[-7] if len(sentiments) >= 7 else sentiments[-1] - sentiments[0]
        else:
            recent_trend = 0
        
        return [
            {
                'title': 'Avg Sentiment',
                'value': f"{avg_sentiment:.3f}",
                'status': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral',
                'subtitle': '30-day average'
            },
            {
                'title': 'Avg Confidence',
                'value': f"{avg_confidence:.2f}",
                'status': 'positive' if avg_confidence > 0.7 else 'warning' if avg_confidence > 0.5 else 'negative',
                'subtitle': 'Analysis quality'
            },
            {
                'title': 'Volatility',
                'value': f"{sentiment_volatility:.3f}",
                'status': 'negative' if sentiment_volatility > 0.2 else 'warning' if sentiment_volatility > 0.1 else 'positive',
                'subtitle': 'Sentiment stability'
            },
            {
                'title': 'Recent Trend',
                'value': f"{recent_trend:+.3f}",
                'status': 'positive' if recent_trend > 0.05 else 'negative' if recent_trend < -0.05 else 'neutral',
                'subtitle': 'Week over week'
            }
        ]
    
    def get_technical_analysis(self, symbol: str, force_refresh: bool = False) -> Dict:
        """Get technical analysis for a symbol with caching"""
        if symbol not in self.technical_data or force_refresh:
            try:
                market_data = get_market_data(symbol, period='3mo')
                if not market_data.empty:
                    self.technical_data[symbol] = self.tech_analyzer.analyze(symbol, market_data)
                else:
                    self.technical_data[symbol] = self.tech_analyzer._empty_analysis(symbol)
            except Exception as e:
                logger.error(f"Error getting technical analysis for {symbol}: {e}")
                self.technical_data[symbol] = self.tech_analyzer._empty_analysis(symbol)
        
        return self.technical_data[symbol]
    
    def display_ml_performance(self, symbol: str):
        """Display ML model performance metrics with professional styling"""
        try:
            from src.ml_training_pipeline import MLTrainingPipeline
            ml_pipeline = MLTrainingPipeline()
            
            conn = sqlite3.connect(ml_pipeline.db_path)
            
            # Get model performance
            query = '''
                SELECT * FROM model_performance 
                ORDER BY training_date DESC 
                LIMIT 5
            '''
            
            performance_df = pd.read_sql_query(query, conn)
            
            if not performance_df.empty:
                st.markdown("##### 📊 Recent Model Performance")
                
                # Display metrics in professional cards
                for idx, row in performance_df.iterrows():
                    with st.expander(f"🤖 Model: {row['model_type']} - {row['model_version']}", expanded=(idx==0)):
                        performance_metrics = [
                            {
                                'title': 'Validation Score',
                                'value': f"{row['validation_score']:.3f}",
                                'status': 'positive' if row['validation_score'] > 0.7 else 'warning' if row['validation_score'] > 0.6 else 'negative',
                                'subtitle': 'Model accuracy'
                            },
                            {
                                'title': 'Precision',
                                'value': f"{row['precision_score']:.3f}",
                                'status': 'positive' if row['precision_score'] > 0.7 else 'warning' if row['precision_score'] > 0.6 else 'negative',
                                'subtitle': 'True positive rate'
                            },
                            {
                                'title': 'Recall',
                                'value': f"{row['recall_score']:.3f}",
                                'status': 'positive' if row['recall_score'] > 0.7 else 'warning' if row['recall_score'] > 0.6 else 'negative',
                                'subtitle': 'Sensitivity score'
                            }
                        ]
                        
                        self.display_professional_metrics(performance_metrics)
                        
                        # Show feature importance if available
                        if row['feature_importance']:
                            try:
                                importance = json.loads(row['feature_importance'])
                                importance_df = pd.DataFrame(list(importance.items()), 
                                                           columns=['Feature', 'Importance'])
                                importance_df = importance_df.sort_values('Importance', ascending=False)
                                importance_df['Importance'] = importance_df['Importance'].apply(lambda x: f"{x:.4f}")
                                
                                st.markdown("**🔍 Feature Importance Analysis:**")
                                st.dataframe(importance_df, use_container_width=True, hide_index=True)
                            except:
                                st.markdown("""
                                <div class="alert alert-info">
                                    <strong>Feature Analysis:</strong> Feature importance data format not compatible for display
                                </div>
                                """, unsafe_allow_html=True)
            
            # Get recent predictions accuracy
            query = '''
                SELECT 
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN outcome_label = 1 THEN 1 ELSE 0 END) as profitable_trades,
                    AVG(return_pct) as avg_return
                FROM trading_outcomes
                WHERE symbol = ?
                AND exit_timestamp > datetime('now', '-30 days')
            '''
            
            accuracy_df = pd.read_sql_query(query, conn, params=[symbol])
            
            if not accuracy_df.empty and accuracy_df['total_predictions'].iloc[0] > 0:
                st.markdown("##### 📈 Recent Trading Performance (30 days)")
                
                total = accuracy_df['total_predictions'].iloc[0]
                profitable = accuracy_df['profitable_trades'].iloc[0]
                win_rate = profitable / total if total > 0 else 0
                avg_return = accuracy_df['avg_return'].iloc[0]
                
                trading_metrics = [
                    {
                        'title': 'Total Trades',
                        'value': str(total),
                        'status': 'neutral',
                        'subtitle': 'Executed positions'
                    },
                    {
                        'title': 'Win Rate',
                        'value': f"{win_rate:.1%}",
                        'status': 'positive' if win_rate > 0.6 else 'warning' if win_rate > 0.4 else 'negative',
                        'subtitle': 'Success percentage'
                    },
                    {
                        'title': 'Average Return',
                        'value': f"{avg_return:+.2f}%",
                        'status': 'positive' if avg_return > 0 else 'negative',
                        'subtitle': 'Per trade return'
                    }
                ]
                
                self.display_professional_metrics(trading_metrics)
            
            conn.close()
            
        except Exception as e:
            st.markdown(f"""
            <div class="alert alert-warning">
                <strong>Performance Data Unavailable</strong><br>
                Error loading ML performance metrics: {str(e)}
            </div>
            """, unsafe_allow_html=True)
    
    def run_dashboard(self):
        """Main professional dashboard application"""
        # Professional header
        self.create_professional_header()
        
        # Load data with professional loading indicator
        with st.spinner("🔄 Loading sentiment analysis data..."):
            all_data = self.load_sentiment_data()
        
        # Professional sidebar
        st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
            <h2 style="margin: 0; font-size: 1.5rem; font-weight: 600;">🎛️ Dashboard Controls</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;">Configure your analysis view</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced bank selection
        st.sidebar.markdown("**📊 Bank Selection**")
        selected_banks = st.sidebar.multiselect(
            "Choose banks to analyze:",
            options=list(self.bank_names.keys()),
            default=list(self.bank_names.keys()),
            format_func=lambda x: f"🏦 {self.bank_names[x]}"
        )
        
        st.sidebar.markdown("---")
        
        # Professional action buttons
        st.sidebar.markdown("**⚡ Quick Actions**")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                self.technical_data = {}
                st.rerun()
        
        with col2:
            if st.button("📈 Tech Update", use_container_width=True):
                with st.spinner("Updating technical analysis..."):
                    for symbol in self.bank_symbols:
                        self.get_technical_analysis(symbol, force_refresh=True)
                st.success("✅ Technical analysis updated!")
        
        st.sidebar.markdown("---")
        
        # Professional info section
        st.sidebar.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #3498db;">
            <h4 style="margin: 0 0 0.5rem 0; color: #2c3e50;">📋 Analysis Info</h4>
            <p style="margin: 0; font-size: 0.85rem; color: #6c757d;">
                Real-time sentiment analysis combining news, social media, and technical indicators for ASX banking sector.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display professional legends
        self.display_confidence_legend()
        self.display_sentiment_scale()
        
        # Market Overview Section
        self.create_section_header(
            "Market Overview", 
            "Real-time sentiment and confidence analysis across all banks",
            "📊"
        )
        
        overview_col1, overview_col2 = st.columns(2)
        
        with overview_col1:
            sentiment_chart = self.create_sentiment_overview_chart(all_data)
            st.plotly_chart(sentiment_chart, use_container_width=True)
        
        with overview_col2:
            confidence_chart = self.create_confidence_distribution_chart(all_data)
            st.plotly_chart(confidence_chart, use_container_width=True)
        
        # Technical Analysis Section
        self.create_section_header(
            "Technical Analysis & Momentum", 
            "Advanced technical indicators and momentum analysis",
            "📈"
        )
        
        # Load technical analysis data with progress
        with st.spinner("📊 Loading technical analysis..."):
            for symbol in self.bank_symbols:
                self.get_technical_analysis(symbol)
        
        tech_tab1, tech_tab2, tech_tab3 = st.tabs([
            "🎯 Momentum Analysis", 
            "📊 Technical Signals", 
            "🔄 Combined Strategy"
        ])
        
        with tech_tab1:
            momentum_chart = self.create_momentum_chart(all_data)
            st.plotly_chart(momentum_chart, use_container_width=True)
            
            st.markdown("### 📋 Momentum Performance Summary")
            momentum_data = []
            for symbol in self.bank_symbols:
                tech_analysis = self.get_technical_analysis(symbol)
                if tech_analysis and 'momentum' in tech_analysis:
                    momentum = tech_analysis['momentum']
                    
                    # Determine momentum status
                    score = momentum['score']
                    if score > 20:
                        status = "🚀 Very Strong"
                    elif score > 5:
                        status = "💪 Strong"
                    elif score > -5:
                        status = "➡️ Neutral"
                    elif score > -20:
                        status = "📉 Weak"
                    else:
                        status = "⚠️ Very Weak"
                    
                    momentum_data.append({
                        'Bank': self.bank_names.get(symbol, symbol),
                        'Symbol': symbol,
                        'Momentum Score': f"{momentum['score']:.1f}",
                        'Status': status,
                        'Strength': momentum['strength'].replace('_', ' ').title(),
                        '1D Change': f"{momentum['price_change_1d']:+.2f}%",
                        '5D Change': f"{momentum['price_change_5d']:+.2f}%",
                        'Volume': momentum['volume_momentum'].title()
                    })
            
            if momentum_data:
                st.dataframe(pd.DataFrame(momentum_data), use_container_width=True, hide_index=True)
        
        with tech_tab2:
            signals_chart = self.create_technical_signals_chart(all_data)
            st.plotly_chart(signals_chart, use_container_width=True)
            
            st.markdown("### 📊 Technical Indicators Dashboard")
            tech_data = []
            for symbol in self.bank_symbols:
                tech_analysis = self.get_technical_analysis(symbol)
                if tech_analysis:
                    indicators = tech_analysis.get('indicators', {})
                    rsi = indicators.get('rsi', 50)
                    
                    # RSI status with emojis
                    if rsi > 70:
                        rsi_status = "🔴 Overbought"
                    elif rsi < 30:
                        rsi_status = "🟢 Oversold"
                    else:
                        rsi_status = "🟡 Normal"
                    
                    # MACD status
                    macd_hist = indicators.get('macd', {}).get('histogram', 0)
                    macd_status = "🟢 Bullish" if macd_hist > 0 else "🔴 Bearish"
                    
                    # Recommendation with styling
                    rec = tech_analysis.get('recommendation', 'HOLD')
                    if rec in ['STRONG_BUY', 'BUY']:
                        rec_display = f"🟢 {rec}"
                    elif rec in ['STRONG_SELL', 'SELL']:
                        rec_display = f"🔴 {rec}"
                    else:
                        rec_display = f"🟡 {rec}"
                    
                    tech_data.append({
                        'Bank': self.bank_names.get(symbol, symbol),
                        'Symbol': symbol,
                        'Current Price': f"${tech_analysis.get('current_price', 0):.2f}",
                        'RSI': f"{rsi:.1f}",
                        'RSI Status': rsi_status,
                        'MACD Signal': macd_status,
                        'Trend': tech_analysis.get('trend', {}).get('direction', 'Unknown').replace('_', ' ').title(),
                        'Recommendation': rec_display
                    })
            
            if tech_data:
                st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)
        
        with tech_tab3:
            combined_chart = self.create_combined_analysis_chart(all_data)
            st.plotly_chart(combined_chart, use_container_width=True)
            
            st.markdown("### 🎯 Integrated Trading Recommendations")
            
            st.markdown("""
            <div class="alert alert-info">
                <strong>Strategy Overview:</strong> Our integrated approach combines sentiment analysis, 
                technical momentum, and machine learning predictions to provide comprehensive trading signals.
            </div>
            """, unsafe_allow_html=True)
            
            recommendations = []
            for symbol in self.bank_symbols:
                sentiment_data = self.get_latest_analysis(all_data.get(symbol, []))
                tech_analysis = self.get_technical_analysis(symbol)
                
                if sentiment_data or tech_analysis:
                    sentiment_score = sentiment_data.get('overall_sentiment', 0) if sentiment_data else 0
                    tech_recommendation = tech_analysis.get('recommendation', 'HOLD') if tech_analysis else 'HOLD'
                    momentum_score = tech_analysis.get('momentum', {}).get('score', 0) if tech_analysis else 0
                    confidence = sentiment_data.get('confidence', 0) if sentiment_data else 0
                    
                    # Enhanced combined recommendation logic
                    if sentiment_score > 0.2 and tech_recommendation in ['BUY', 'STRONG_BUY'] and momentum_score > 10 and confidence > 0.7:
                        combined_rec = '🟢 STRONG BUY'
                        rec_strength = 'High'
                    elif sentiment_score > 0.1 and tech_recommendation in ['BUY'] and momentum_score > 0 and confidence > 0.5:
                        combined_rec = '🟢 BUY'
                        rec_strength = 'Medium'
                    elif sentiment_score < -0.2 and tech_recommendation in ['SELL', 'STRONG_SELL'] and momentum_score < -10 and confidence > 0.7:
                        combined_rec = '🔴 STRONG SELL'
                        rec_strength = 'High'
                    elif sentiment_score < -0.1 and tech_recommendation in ['SELL'] and momentum_score < 0 and confidence > 0.5:
                        combined_rec = '🔴 SELL'
                        rec_strength = 'Medium'
                    else:
                        combined_rec = '🟡 HOLD'
                        rec_strength = 'Low'
                    
                    recommendations.append({
                        'Bank': self.bank_names.get(symbol, symbol),
                        'Symbol': symbol,
                        'News Sentiment': f"{sentiment_score:+.3f}",
                        'Technical Signal': tech_recommendation,
                        'Momentum': f"{momentum_score:+.1f}",
                        'Confidence': f"{confidence:.2f}",
                        'Combined Recommendation': combined_rec,
                        'Signal Strength': rec_strength
                    })
            
            if recommendations:
                st.dataframe(pd.DataFrame(recommendations), use_container_width=True, hide_index=True)
        
        # Historical Trends Section
        self.create_section_header(
            "Historical Trends Analysis", 
            "Comprehensive historical performance and correlation analysis",
            "📈"
        )
        
        trend_tabs = st.tabs([
            "📊 Multi-Bank Comparison", 
            "🎯 Performance Leaders", 
            "📋 Trend Intelligence"
        ])
        
        with trend_tabs[0]:
            st.markdown("### 📊 Recent Performance Trends")
            
            # Enhanced multi-bank comparison chart
            multi_bank_fig = go.Figure()
            
            for symbol in selected_banks:
                if symbol in all_data and all_data[symbol]:
                    recent_data = all_data[symbol][-10:]
                    dates = []
                    sentiments = []
                    confidences = []
                    
                    for entry in recent_data:
                        try:
                            timestamp = entry.get('timestamp', '')
                            if timestamp:
                                if 'T' in timestamp:
                                    date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                else:
                                    date = datetime.fromisoformat(timestamp)
                                dates.append(date)
                                sentiments.append(entry.get('overall_sentiment', 0))
                                confidences.append(entry.get('confidence', 0))
                        except Exception:
                            continue
                    
                    if dates and sentiments:
                        multi_bank_fig.add_trace(go.Scatter(
                            x=dates,
                            y=sentiments,
                            mode='lines+markers',
                            name=self.bank_names.get(symbol, symbol),
                            line=dict(width=3),
                            marker=dict(size=8),
                            hovertemplate='<b>%{fullData.name}</b><br>' +
                                         'Date: %{x}<br>' +
                                         'Sentiment: %{y:.3f}<br>' +
                                         '<extra></extra>'
                        ))
            
            multi_bank_fig.update_layout(
                title=dict(
                    text="Recent Sentiment Trends - Selected Banks",
                    font=dict(size=18, family="Inter", weight=600, color="#2c3e50")
                ),
                xaxis=dict(
                    title=dict(text="Date", font=dict(size=14, family="Inter")),
                    tickfont=dict(size=12, family="Inter"),
                    gridcolor="#f1f1f1",
                    linecolor="#dee2e6"
                ),
                yaxis=dict(
                    title=dict(text="Sentiment Score", font=dict(size=14, family="Inter")),
                    tickfont=dict(size=12, family="Inter"),
                    range=[-0.5, 0.5],
                    gridcolor="#f1f1f1",
                    linecolor="#dee2e6",
                    zeroline=True,
                    zerolinecolor="#dee2e6",
                    zerolinewidth=2
                ),
                height=500,
                legend=dict(x=0, y=1),
                hovermode='x unified',
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Inter"),
                margin=dict(l=60, r=60, t=60, b=60)
            )
            
            st.plotly_chart(multi_bank_fig, use_container_width=True)
        
        with trend_tabs[1]:
            st.markdown("### 🎯 Top Performance Movers (7-Day Analysis)")
            
            # Calculate and display top movers with enhanced analysis
            movers_data = []
            for symbol in selected_banks:
                if symbol in all_data and len(all_data[symbol]) >= 2:
                    recent_data = all_data[symbol][-7:]
                    if len(recent_data) >= 2:
                        latest_sentiment = recent_data[-1].get('overall_sentiment', 0)
                        week_ago_sentiment = recent_data[0].get('overall_sentiment', 0)
                        change = latest_sentiment - week_ago_sentiment
                        
                        # Get price data for comparison
                        try:
                            price_data = get_market_data(symbol, period='1mo', interval='1d')
                            if not price_data.empty:
                                current_price = price_data['Close'].iloc[-1]
                                week_ago_price = price_data['Close'].iloc[-7] if len(price_data) >= 7 else price_data['Close'].iloc[0]
                                price_change = ((current_price - week_ago_price) / week_ago_price) * 100
                            else:
                                price_change = 0
                        except Exception:
                            price_change = 0
                        
                        # Determine trend classification
                        if abs(change) > 0.15:
                            trend = "🔥 Explosive"
                        elif abs(change) > 0.1:
                            trend = "🚀 High Momentum"
                        elif change > 0.05:
                            trend = "📈 Rising"
                        elif change < -0.05:
                            trend = "📉 Declining"
                        else:
                            trend = "➡️ Stable"
                        
                        movers_data.append({
                            'Bank': self.bank_names.get(symbol, symbol),
                            'Symbol': symbol,
                            'Sentiment Change': change,
                            'Price Change (%)': price_change,
                            'Current Sentiment': latest_sentiment,
                            'Trend Classification': trend,
                            'Volatility': 'High' if abs(change) > 0.1 else 'Medium' if abs(change) > 0.05 else 'Low'
                        })
            
            if movers_data:
                # Sort by absolute sentiment change
                movers_df = pd.DataFrame(movers_data)
                movers_df['abs_change'] = movers_df['Sentiment Change'].abs()
                movers_df = movers_df.sort_values('abs_change', ascending=False)
                
                # Display top movers in professional expandable cards
                for idx, row in movers_df.iterrows():
                    with st.expander(f"{row['Trend Classification']} {row['Bank']} ({row['Symbol']})", expanded=(idx < 3)):
                        
                        mover_metrics = [
                            {
                                'title': 'Sentiment Shift',
                                'value': f"{row['Sentiment Change']:+.3f}",
                                'status': 'positive' if row['Sentiment Change'] > 0 else 'negative' if row['Sentiment Change'] < 0 else 'neutral',
                                'subtitle': '7-day change'
                            },
                            {
                                'title': 'Price Movement',
                                'value': f"{row['Price Change (%)']:+.2f}%",
                                'status': 'positive' if row['Price Change (%)'] > 0 else 'negative' if row['Price Change (%)'] < 0 else 'neutral',
                                'subtitle': 'Market performance'
                            },
                            {
                                'title': 'Current Level',
                                'value': f"{row['Current Sentiment']:.3f}",
                                'status': 'positive' if row['Current Sentiment'] > 0.1 else 'negative' if row['Current Sentiment'] < -0.1 else 'neutral',
                                'subtitle': 'Present sentiment'
                            },
                            {
                                'title': 'Volatility',
                                'value': row['Volatility'],
                                'status': 'warning' if row['Volatility'] == 'High' else 'neutral',
                                'subtitle': 'Movement intensity'
                            }
                        ]
                        
                        self.display_professional_metrics(mover_metrics)
        
        with trend_tabs[2]:
            st.markdown("### 📋 Trend Intelligence Summary")
            
            # Enhanced trend summary with market insights
            if all_data:
                # Calculate market-wide statistics
                all_latest = [self.get_latest_analysis(data) for data in all_data.values() if data]
                
                if all_latest:
                    market_sentiment = sum(a.get('overall_sentiment', 0) for a in all_latest) / len(all_latest)
                    market_confidence = sum(a.get('confidence', 0) for a in all_latest) / len(all_latest)
                    total_news = sum(a.get('news_count', 0) for a in all_latest)
                    
                    # Market overview metrics
                    market_metrics = [
                        {
                            'title': 'Market Sentiment',
                            'value': f"{market_sentiment:.3f}",
                            'status': 'positive' if market_sentiment > 0.1 else 'negative' if market_sentiment < -0.1 else 'neutral',
                            'subtitle': 'Overall market mood'
                        },
                        {
                            'title': 'Avg Confidence',
                            'value': f"{market_confidence:.2f}",
                            'status': 'positive' if market_confidence > 0.7 else 'warning' if market_confidence > 0.5 else 'negative',
                            'subtitle': 'Analysis quality'
                        },
                        {
                            'title': 'News Volume',
                            'value': str(total_news),
                            'status': 'positive' if total_news > 50 else 'warning' if total_news > 20 else 'negative',
                            'subtitle': 'Market coverage'
                        },
                        {
                            'title': 'Active Banks',
                            'value': str(len([a for a in all_latest if a.get('news_count', 0) > 0])),
                            'status': 'neutral',
                            'subtitle': 'With recent news'
                        }
                    ]
                    
                    st.markdown("**🌍 Market Overview:**")
                    self.display_professional_metrics(market_metrics)
                    
                    # Individual bank trend summary
                    st.markdown("**🏦 Individual Bank Trends:**")
                    trend_summary = []
                    
                    for symbol in selected_banks:
                        if symbol in all_data and all_data[symbol]:
                            latest = self.get_latest_analysis(all_data[symbol])
                            
                            # Calculate trend direction with more sophisticated logic
                            if len(all_data[symbol]) >= 5:
                                recent_sentiments = [entry.get('overall_sentiment', 0) for entry in all_data[symbol][-5:]]
                                
                                # Calculate trend using linear regression
                                if len(recent_sentiments) == 5:
                                    x = np.array(range(5))
                                    y = np.array(recent_sentiments)
                                    slope = np.polyfit(x, y, 1)[0]
                                    
                                    if slope > 0.02:
                                        trend_direction = "📈 Strongly Rising"
                                        trend_strength = "Strong"
                                    elif slope > 0.01:
                                        trend_direction = "📈 Rising"
                                        trend_strength = "Moderate"
                                    elif slope < -0.02:
                                        trend_direction = "📉 Strongly Falling"
                                        trend_strength = "Strong"
                                    elif slope < -0.01:
                                        trend_direction = "📉 Falling"
                                        trend_strength = "Moderate"
                                    else:
                                        trend_direction = "➡️ Stable"
                                        trend_strength = "Stable"
                                else:
                                    trend_direction = "❓ Insufficient Data"
                                    trend_strength = "Unknown"
                            else:
                                trend_direction = "❓ Insufficient Data"
                                trend_strength = "Unknown"
                            
                            # Risk assessment
                            volatility = np.std([entry.get('overall_sentiment', 0) for entry in all_data[symbol][-10:]]) if len(all_data[symbol]) >= 10 else 0
                            risk_level = "High" if volatility > 0.2 else "Medium" if volatility > 0.1 else "Low"
                            
                            trend_summary.append({
                                'Bank': self.bank_names.get(symbol, symbol),
                                'Current Sentiment': f"{latest.get('overall_sentiment', 0):.3f}",
                                'Confidence': f"{latest.get('confidence', 0):.2f}",
                                'Trend Direction': trend_direction,
                                'Trend Strength': trend_strength,
                                'Risk Level': risk_level,
                                'News Activity': latest.get('news_count', 0),
                                'Last Update': latest.get('timestamp', 'Unknown')[:10] if latest.get('timestamp') else 'Unknown'
                            })
                    
                    if trend_summary:
                        st.dataframe(pd.DataFrame(trend_summary), use_container_width=True, hide_index=True)

        # Individual Bank Analysis Section
        self.create_section_header(
            "Individual Bank Analysis", 
            "Detailed sentiment, technical, and ML analysis for each bank",
            "🏦"
        )
        
        for symbol in selected_banks:
            if symbol in all_data:
                with st.expander(f"🏦 {self.bank_names[symbol]} ({symbol}) - Comprehensive Analysis", expanded=True):
                    self.display_bank_analysis(symbol, all_data[symbol])
        
        # Enhanced Summary Section
        self.create_section_header(
            "Executive Summary", 
            "Key performance indicators and market insights",
            "📈"
        )
        
        # Calculate comprehensive summary stats
        total_analyses = sum(len(data) for data in all_data.values())
        latest_analyses = [self.get_latest_analysis(data) for data in all_data.values() if data]
        
        if latest_analyses:
            avg_sentiment = sum(a.get('overall_sentiment', 0) for a in latest_analyses) / len(latest_analyses)
            avg_confidence = sum(a.get('confidence', 0) for a in latest_analyses) / len(latest_analyses)
            total_news = sum(a.get('news_count', 0) for a in latest_analyses)
            high_confidence_count = len([a for a in latest_analyses if a.get('confidence', 0) >= 0.8])
            
            summary_metrics = [
                {
                    'title': 'Total Analyses',
                    'value': str(total_analyses),
                    'status': 'neutral',
                    'subtitle': 'Historical data points'
                },
                {
                    'title': 'Market Sentiment',
                    'value': f"{avg_sentiment:.3f}",
                    'status': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral',
                    'subtitle': 'Weighted average'
                },
                {
                    'title': 'Avg Confidence',
                    'value': f"{avg_confidence:.2f}",
                    'status': 'positive' if avg_confidence > 0.7 else 'warning' if avg_confidence > 0.5 else 'negative',
                    'subtitle': 'Analysis quality'
                },
                {
                    'title': 'News Coverage',
                    'value': str(total_news),
                    'status': 'positive' if total_news > 100 else 'warning' if total_news > 50 else 'negative',
                    'subtitle': 'Articles analyzed'
                },
                {
                    'title': 'High Confidence',
                    'value': f"{high_confidence_count}/{len(latest_analyses)}",
                    'status': 'positive' if high_confidence_count > len(latest_analyses)/2 else 'warning',
                    'subtitle': 'Reliable signals'
                }
            ]
            
            self.display_professional_metrics(summary_metrics)
        
        # Professional footer
        st.markdown("""
        <div class="footer">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <strong>ASX Bank Analytics Platform</strong><br>
                    <small>Real-time sentiment analysis & technical indicators</small>
                </div>
                <div style="text-align: right;">
                    <strong>Last Updated:</strong> {}<br>
                    <small>Data Source: Multi-source aggregation</small>
                </div>
            </div>
        </div>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
    
    def create_momentum_chart(self, all_data: Dict) -> go.Figure:
        """Create professional momentum analysis chart"""
        symbols = []
        momentum_scores = []
        colors = []
        hover_text = []
        
        for symbol, data in all_data.items():
            tech_analysis = self.get_technical_analysis(symbol)
            if tech_analysis and 'momentum' in tech_analysis:
                symbols.append(self.bank_names.get(symbol, symbol))
                momentum_score = tech_analysis['momentum']['score']
                momentum_scores.append(momentum_score)
                
                # Professional color gradient based on momentum
                if momentum_score > 20:
                    colors.append('#27ae60')  # Strong positive
                elif momentum_score > 5:
                    colors.append('#2ecc71')  # Moderate positive
                elif momentum_score > -5:
                    colors.append('#f39c12')  # Neutral
                elif momentum_score > -20:
                    colors.append('#e67e22')  # Moderate negative
                else:
                    colors.append('#e74c3c')  # Strong negative
                
                # Enhanced hover information
                strength = tech_analysis['momentum']['strength']
                volume = tech_analysis['momentum']['volume_momentum']
                hover_text.append(f"Strength: {strength.title()}<br>Volume: {volume.title()}")
        
        fig = go.Figure(go.Bar(
            x=symbols,
            y=momentum_scores,
            marker=dict(
                color=colors,
                line=dict(color='white', width=1)
            ),
            text=[f"{s:.1f}" for s in momentum_scores],
            textposition='auto',
            textfont=dict(size=12, family="Inter", weight=600),
            hovertemplate='<b>%{x}</b><br>' +
                         'Momentum Score: %{y:.1f}<br>' +
                         '%{customdata}<br>' +
                         '<extra></extra>',
            customdata=hover_text
        ))
        
        fig.update_layout(
            title=dict(
                text="Technical Momentum Analysis",
                font=dict(size=18, family="Inter", weight=600, color="#2c3e50")
            ),
            xaxis=dict(
                title=dict(text="Banks", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                gridcolor="#f1f1f1",
                linecolor="#dee2e6"
            ),
            yaxis=dict(
                title=dict(text="Momentum Score", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                range=[-100, 100],
                gridcolor="#f1f1f1",
                linecolor="#dee2e6",
                zeroline=True,
                zerolinecolor="#dee2e6",
                zerolinewidth=2
            ),
            height=400,
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter"),
            margin=dict(l=60, r=60, t=60, b=60)
        )
        
        # Add reference lines for momentum zones
        fig.add_hline(y=20, line_dash="dash", line_color="#27ae60", annotation_text="Strong Bullish", annotation_position="top right")
        fig.add_hline(y=-20, line_dash="dash", line_color="#e74c3c", annotation_text="Strong Bearish", annotation_position="bottom right")
        
        return fig
    
    def create_technical_signals_chart(self, all_data: Dict) -> go.Figure:
        """Create professional technical analysis signals chart"""
        symbols = []
        signals = []
        colors = []
        recommendations = []
        hover_data = []
        
        for symbol, data in all_data.items():
            tech_analysis = self.get_technical_analysis(symbol)
            if tech_analysis:
                symbols.append(self.bank_names.get(symbol, symbol))
                signal_score = tech_analysis.get('overall_signal', 0)
                signals.append(signal_score)
                recommendation = tech_analysis.get('recommendation', 'HOLD')
                recommendations.append(recommendation)
                
                # Professional color scheme for recommendations
                if recommendation in ['STRONG_BUY']:
                    colors.append('#27ae60')
                elif recommendation in ['BUY']:
                    colors.append('#2ecc71')
                elif recommendation in ['STRONG_SELL']:
                    colors.append('#e74c3c')
                elif recommendation in ['SELL']:
                    colors.append('#e67e22')
                else:
                    colors.append('#95a5a6')
                
                # Enhanced hover data
                rsi = tech_analysis.get('indicators', {}).get('rsi', 50)
                trend = tech_analysis.get('trend', {}).get('direction', 'sideways')
                hover_data.append(f"RSI: {rsi:.1f}<br>Trend: {trend.title()}")
        
        fig = go.Figure(go.Bar(
            x=symbols,
            y=signals,
            marker=dict(
                color=colors,
                line=dict(color='white', width=1)
            ),
            text=recommendations,
            textposition='auto',
            textfont=dict(size=10, family="Inter", weight=600),
            hovertemplate='<b>%{x}</b><br>' +
                         'Signal Score: %{y:.1f}<br>' +
                         'Recommendation: %{text}<br>' +
                         '%{customdata}<br>' +
                         '<extra></extra>',
            customdata=hover_data
        ))
        
        fig.update_layout(
            title=dict(
                text="Technical Analysis Signals",
                font=dict(size=18, family="Inter", weight=600, color="#2c3e50")
            ),
            xaxis=dict(
                title=dict(text="Banks", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                gridcolor="#f1f1f1",
                linecolor="#dee2e6"
            ),
            yaxis=dict(
                title=dict(text="Signal Score", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                range=[-100, 100],
                gridcolor="#f1f1f1",
                linecolor="#dee2e6",
                zeroline=True,
                zerolinecolor="#dee2e6",
                zerolinewidth=2
            ),
            height=400,
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter"),
            margin=dict(l=60, r=60, t=60, b=60)
        )
        
        # Add signal threshold lines
        fig.add_hline(y=30, line_dash="dash", line_color="#27ae60", annotation_text="Strong Buy Zone")
        fig.add_hline(y=-30, line_dash="dash", line_color="#e74c3c", annotation_text="Strong Sell Zone")
        
        return fig
    
    def create_combined_analysis_chart(self, all_data: Dict) -> go.Figure:
        """Create professional combined sentiment and technical analysis chart"""
        symbols = []
        sentiment_scores = []
        technical_scores = []
        confidence_scores = []
        
        for symbol, data in all_data.items():
            latest_sentiment = self.get_latest_analysis(data)
            tech_analysis = self.get_technical_analysis(symbol)
            
            if latest_sentiment or tech_analysis:
                symbols.append(self.bank_names.get(symbol, symbol))
                
                # Normalize sentiment score to -100 to 100 scale
                sent_score = latest_sentiment.get('overall_sentiment', 0) * 100 if latest_sentiment else 0
                sentiment_scores.append(sent_score)
                
                tech_score = tech_analysis.get('overall_signal', 0) if tech_analysis else 0
                technical_scores.append(tech_score)
                
                confidence = latest_sentiment.get('confidence', 0) if latest_sentiment else 0
                confidence_scores.append(confidence)
        
        fig = go.Figure()
        
        # Add sentiment bars with professional styling
        fig.add_trace(go.Bar(
            name='News Sentiment',
            x=symbols,
            y=sentiment_scores,
            marker=dict(
                color='rgba(52, 152, 219, 0.8)',
                line=dict(color='white', width=1)
            ),
            yaxis='y',
            hovertemplate='<b>%{x}</b><br>' +
                         'News Sentiment: %{y:.1f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add technical bars with professional styling
        fig.add_trace(go.Bar(
            name='Technical Analysis',
            x=symbols,
            y=technical_scores,
            marker=dict(
                color='rgba(231, 76, 60, 0.8)',
                line=dict(color='white', width=1)
            ),
            yaxis='y',
            hovertemplate='<b>%{x}</b><br>' +
                         'Technical Signal: %{y:.1f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add confidence line
        if confidence_scores:
            fig.add_trace(go.Scatter(
                name='Confidence',
                x=symbols,
                y=[c * 100 for c in confidence_scores],  # Scale to match other data
                mode='lines+markers',
                line=dict(color='#f39c12', width=3),
                marker=dict(size=10, color='#f39c12'),
                yaxis='y',
                hovertemplate='<b>%{x}</b><br>' +
                             'Confidence: %{customdata:.2f}<br>' +
                             '<extra></extra>',
                customdata=confidence_scores
            ))
        
        fig.update_layout(
            title=dict(
                text="Combined Analysis: News Sentiment vs Technical Signals",
                font=dict(size=18, family="Inter", weight=600, color="#2c3e50")
            ),
            xaxis=dict(
                title=dict(text="Banks", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                gridcolor="#f1f1f1",
                linecolor="#dee2e6"
            ),
            yaxis=dict(
                title=dict(text="Signal Strength", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                range=[-100, 100],
                gridcolor="#f1f1f1",
                linecolor="#dee2e6",
                zeroline=True,
                zerolinecolor="#dee2e6",
                zerolinewidth=2
            ),
            height=500,
            barmode='group',
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter"),
            margin=dict(l=60, r=60, t=60, b=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_historical_trends_chart(self, symbol: str, all_data: Dict) -> go.Figure:
        """Create professional historical trends chart"""
        sentiment_data = all_data.get(symbol, [])
        
        # Get historical price data
        try:
            price_data = get_market_data(symbol, period='3mo', interval='1d')
            if price_data.empty:
                logger.warning(f"No price data available for {symbol}")
            else:
                logger.info(f"Retrieved {len(price_data)} price data points for {symbol}")
        except Exception as e:
            logger.error(f"Error getting price data for {symbol}: {e}")
            price_data = pd.DataFrame()
        
        # Prepare data for plotting
        dates = []
        sentiment_scores = []
        confidence_scores = []
        prices = []
        momentum_scores = []
        
        # Process sentiment history
        for entry in sentiment_data[-30:]:  # Last 30 entries
            try:
                timestamp = entry.get('timestamp', '')
                if timestamp:
                    if 'T' in timestamp:
                        date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        date = datetime.fromisoformat(timestamp)
                    
                    dates.append(date)
                    sentiment_scores.append(entry.get('overall_sentiment', 0))
                    confidence_scores.append(entry.get('confidence', 0))
                    
                    # Get corresponding price data
                    price_date = date.date()
                    price_found = False
                    
                    if not price_data.empty:
                        exact_matches = price_data.index.date == price_date
                        if exact_matches.any():
                            price = price_data.loc[exact_matches, 'Close'].iloc[0]
                            prices.append(price)
                            price_found = True
                        else:
                            price_dates = price_data.index.date
                            date_diffs = [abs((pd_date - price_date).days) for pd_date in price_dates]
                            if date_diffs and min(date_diffs) <= 3:
                                closest_idx = date_diffs.index(min(date_diffs))
                                price = price_data.iloc[closest_idx]['Close']
                                prices.append(price)
                                price_found = True
                    
                    if not price_found:
                        if prices:
                            prices.append(prices[-1])
                        else:
                            fallback_price = self.get_current_price(symbol)
                            prices.append(fallback_price)
                    
                    # Calculate momentum
                    if len(prices) > 1:
                        momentum = ((prices[-1] - prices[-2]) / prices[-2]) * 100
                    else:
                        momentum = 0
                    momentum_scores.append(momentum)
            except Exception as e:
                logger.warning(f"Error processing entry: {e}")
                continue
        
        # Create professional figure
        fig = go.Figure()
        
        if dates:
            # Normalize price data for better visualization
            if prices and max(prices) != min(prices):
                price_norm = [(p - min(prices)) / (max(prices) - min(prices)) * 2 - 1 for p in prices]
            else:
                price_norm = [0] * len(dates)
            
            # Add sentiment trace
            fig.add_trace(go.Scatter(
                x=dates,
                y=sentiment_scores,
                mode='lines+markers',
                name='Sentiment Score',
                line=dict(color='#3498db', width=3),
                marker=dict(size=6, color='#3498db'),
                hovertemplate='<b>Sentiment</b><br>' +
                             'Date: %{x}<br>' +
                             'Score: %{y:.3f}<extra></extra>'
            ))
            
            # Add confidence trace
            fig.add_trace(go.Scatter(
                x=dates,
                y=confidence_scores,
                mode='lines+markers',
                name='Confidence',
                line=dict(color='#f39c12', width=2),
                marker=dict(size=5, color='#f39c12'),
                hovertemplate='<b>Confidence</b><br>' +
                             'Date: %{x}<br>' +
                             'Level: %{y:.3f}<extra></extra>'
            ))
            
            # Add normalized price trace
            fig.add_trace(go.Scatter(
                x=dates,
                y=price_norm,
                mode='lines',
                name='Price (Normalized)',
                line=dict(color='#27ae60', width=2),
                hovertemplate='<b>Price</b><br>' +
                             'Date: %{x}<br>' +
                             'Actual: $%{customdata:.2f}<br>' +
                             'Normalized: %{y:.3f}<extra></extra>',
                customdata=prices
            ))
            
            # Add momentum bars
            if momentum_scores:
                momentum_colors = ['#e74c3c' if m < 0 else '#27ae60' for m in momentum_scores]
                fig.add_trace(go.Bar(
                    x=dates,
                    y=[m/100 for m in momentum_scores],  # Scale momentum
                    name='Daily Momentum (%)',
                    marker=dict(color=momentum_colors, opacity=0.4),
                    hovertemplate='<b>Momentum</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Change: %{customdata:.2f}%<extra></extra>',
                    customdata=momentum_scores
                ))
        else:
            fig.add_annotation(
                text="No historical data available",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, family="Inter", color="#6c757d")
            )
        
        fig.update_layout(
            title=dict(
                text=f"Historical Trends - {self.bank_names.get(symbol, symbol)}",
                font=dict(size=18, family="Inter", weight=600, color="#2c3e50")
            ),
            xaxis=dict(
                title=dict(text="Date", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                gridcolor="#f1f1f1",
                linecolor="#dee2e6"
            ),
            yaxis=dict(
                title=dict(text="Score / Normalized Value", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                range=[-1.2, 1.2],
                gridcolor="#f1f1f1",
                linecolor="#dee2e6",
                zeroline=True,
                zerolinecolor="#dee2e6",
                zerolinewidth=2
            ),
            height=500,
            legend=dict(x=0, y=1),
            hovermode='x unified',
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter"),
            margin=dict(l=60, r=60, t=60, b=60)
        )
        
        return fig
    
    def create_correlation_chart(self, symbol: str, all_data: Dict) -> go.Figure:
        """Create professional correlation chart between sentiment and price movement"""
        sentiment_data = all_data.get(symbol, [])
        
        try:
            price_data = get_market_data(symbol, period='3mo', interval='1d')
        except Exception:
            price_data = pd.DataFrame()
        
        sentiment_values = []
        price_changes = []
        confidence_values = []
        dates = []
        
        # Process data for correlation analysis
        for entry in sentiment_data[-20:]:  # Last 20 entries
            try:
                timestamp = entry.get('timestamp', '')
                if timestamp:
                    if 'T' in timestamp:
                        date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        date = datetime.fromisoformat(timestamp)
                    
                    sentiment = entry.get('overall_sentiment', 0)
                    confidence = entry.get('confidence', 0)
                    
                    # Get price change for this date
                    price_date = date.date()
                    if not price_data.empty and price_date in price_data.index.date:
                        price_row = price_data.loc[price_data.index.date == price_date]
                        if not price_row.empty:
                            close_price = price_row['Close'].iloc[0]
                            open_price = price_row['Open'].iloc[0]
                            price_change = ((close_price - open_price) / open_price) * 100
                            
                            sentiment_values.append(sentiment)
                            price_changes.append(price_change)
                            confidence_values.append(confidence)
                            dates.append(date)
            except Exception as e:
                logger.warning(f"Error processing correlation data: {e}")
                continue
        
        # Create professional scatter plot
        fig = go.Figure()
        
        if sentiment_values and price_changes:
            # Color points by confidence with professional color scale
            fig.add_trace(go.Scatter(
                x=sentiment_values,
                y=price_changes,
                mode='markers',
                marker=dict(
                    size=[max(8, c*25) for c in confidence_values],  # Size by confidence
                    color=confidence_values,
                    colorscale='RdYlGn',
                    colorbar=dict(
                        title=dict(text="Confidence Level", font=dict(size=14, family="Inter")),
                        tickfont=dict(size=10, family="Inter")
                    ),
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ),
                text=[f"Date: {d.strftime('%Y-%m-%d')}<br>Confidence: {c:.2f}" 
                      for d, c in zip(dates, confidence_values)],
                hovertemplate='<b>Correlation Analysis</b><br>' +
                             'Sentiment: %{x:.3f}<br>' +
                             'Price Change: %{y:.2f}%<br>' +
                             '%{text}<extra></extra>',
                name='Data Points'
            ))
            
            # Add trend line with professional styling
            if len(sentiment_values) > 3:
                z = np.polyfit(sentiment_values, price_changes, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(sentiment_values), max(sentiment_values), 100)
                y_trend = p(x_trend)
                
                # Calculate R-squared for trend line
                correlation = np.corrcoef(sentiment_values, price_changes)[0, 1]
                r_squared = correlation ** 2
                
                fig.add_trace(go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode='lines',
                    name=f'Trend Line (R² = {r_squared:.3f})',
                    line=dict(color='#e74c3c', dash='dash', width=2),
                    hoverinfo='skip'
                ))
        else:
            fig.add_annotation(
                text="Insufficient data for correlation analysis",
                x=0, y=0,
                showarrow=False,
                font=dict(size=16, family="Inter", color="#6c757d")
            )
        
        fig.update_layout(
            title=dict(
                text=f"Sentiment vs Price Movement Correlation - {self.bank_names.get(symbol, symbol)}",
                font=dict(size=16, family="Inter", weight=600, color="#2c3e50")
            ),
            xaxis=dict(
                title=dict(text="Sentiment Score", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                gridcolor="#f1f1f1",
                linecolor="#dee2e6",
                zeroline=True,
                zerolinecolor="#dee2e6"
            ),
            yaxis=dict(
                title=dict(text="Price Change (%)", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                gridcolor="#f1f1f1",
                linecolor="#dee2e6",
                zeroline=True,
                zerolinecolor="#dee2e6"
            ),
            height=400,
            showlegend=True,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter"),
            margin=dict(l=60, r=60, t=60, b=60)
        )
        
        return fig

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol with fallback options"""
        try:
            current_data = get_market_data(symbol, period='5d', interval='1d')
            if not current_data.empty:
                return float(current_data['Close'].iloc[-1])
        except Exception as e:
            logger.warning(f"Could not get current price for {symbol}: {e}")
        
        # Fallback to cached technical analysis data
        if symbol in self.technical_data:
            current_price = self.technical_data[symbol].get('current_price', 0)
            if current_price > 0:
                return float(current_price)
        
        # Try to get fresh technical analysis
        try:
            tech_analysis = self.get_technical_analysis(symbol, force_refresh=True)
            current_price = tech_analysis.get('current_price', 0)
            if current_price > 0:
                return float(current_price)
        except Exception as e:
            logger.warning(f"Could not get technical analysis price for {symbol}: {e}")
        
        # Known approximate prices for ASX banks (as last resort)
        fallback_prices = {
            'CBA.AX': 179.0,
            'WBC.AX': 34.0,
            'ANZ.AX': 30.0,
            'NAB.AX': 40.0,
            'MQG.AX': 221.0,
            'SUN.AX': 13.0,
            'QBE.AX': 17.0
        }
        
        return fallback_prices.get(symbol, 100.0)

def main():
    """Run the professional dashboard"""
    dashboard = NewsAnalysisDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
