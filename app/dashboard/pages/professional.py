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
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import technical analysis and config
from app.core.analysis.technical import TechnicalAnalyzer, get_market_data
from app.config.settings import Settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ML Progression Tracker
try:
    from app.core.ml_progression_tracker import MLProgressionTracker
    ML_TRACKER_AVAILABLE = True
    logger.info("ML Progression Tracker loaded successfully")
except ImportError:
    ML_TRACKER_AVAILABLE = False
    logger.warning("ML Progression Tracker not available")

# Import Position Risk Assessor
try:
    from app.core.trading.risk_management import PositionRiskAssessor
    POSITION_RISK_AVAILABLE = True
    logger.info("Position Risk Assessor loaded successfully")
except ImportError:
    POSITION_RISK_AVAILABLE = False
    logger.warning("Position Risk Assessor not available - ensure src/position_risk_assessor.py exists")

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
    
    /* Position Risk Assessment Enhancements */
    .form-section {
        background: linear-gradient(135deg, var(--light-gray) 0%, #f1f3f4 100%);
        padding: 1.5rem;
        border-radius: var(--border-radius);
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
    }
    
    .form-section h4 {
        color: var(--primary-color);
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }
    
    .form-section p {
        color: var(--medium-gray);
        margin: 0;
        font-size: 0.9rem;
    }
    
    .position-preview {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }
    
    .position-preview h5 {
        color: var(--primary-color);
        margin: 0 0 0.75rem 0;
        font-weight: 600;
    }
    
    .preview-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        font-size: 0.9rem;
    }
    
    .preview-grid div {
        padding: 0.25rem 0;
    }
    
    .profit {
        color: var(--success-color);
        font-weight: 600;
    }
    
    .loss {
        color: var(--accent-color);
        font-weight: 600;
    }
    
    .low-risk {
        color: var(--success-color);
        font-weight: 600;
    }
    
    .medium-risk {
        color: var(--warning-color);
        font-weight: 600;
    }
    
    .high-risk {
        color: var(--accent-color);
        font-weight: 600;
    }
    
    /* Enhanced Risk Assessment Results */
    .risk-results-container {
        background: white;
        border-radius: var(--border-radius-lg);
        box-shadow: var(--shadow);
        margin: 1.5rem 0;
        overflow: hidden;
    }
    
    .risk-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 1.5rem;
        text-align: center;
    }
    
    .risk-header h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .risk-summary {
        display: flex;
        justify-content: space-around;
        margin-top: 1rem;
    }
    
    .risk-summary-item {
        text-align: center;
    }
    
    .risk-summary-value {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .risk-summary-label {
        font-size: 0.85rem;
        opacity: 0.9;
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
        
        # Initialize ML progression tracker
        if ML_TRACKER_AVAILABLE:
            self.ml_tracker = MLProgressionTracker()
            logger.info("ML Progression Tracker initialized")
        else:
            self.ml_tracker = None
            logger.warning("ML Progression Tracker not available")
    
    def load_sentiment_data(self) -> Dict[str, List[Dict]]:
        """Load sentiment history data for all banks"""
        all_data = {}
        
        logger.info(f"Loading sentiment data from: {self.data_path}")
        
        for symbol in self.bank_symbols:
            file_path = os.path.join(self.data_path, f"{symbol}_history.json")
            logger.info(f"Checking file: {file_path}")
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        all_data[symbol] = data if isinstance(data, list) else [data]
                        logger.info(f"Loaded {len(all_data[symbol])} records for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading data for {symbol}: {e}")
                    all_data[symbol] = []
            else:
                logger.warning(f"File not found: {file_path}")
                all_data[symbol] = []
        
        total_records = sum(len(data) for data in all_data.values())
        logger.info(f"Total records loaded: {total_records}")
        
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
                }
            ]
            
            self.display_professional_metrics(ml_metrics)
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
            from app.core.ml.training.pipeline import MLTrainingPipeline
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
    
    def create_historical_trends_chart(self, symbol: str, all_data: Dict) -> go.Figure:
        """Create professional historical trends chart for sentiment and price"""
        sentiment_data = all_data.get(symbol, [])
        
        # Prepare data for chart
        dates = []
        sentiments = []
        prices = []
        confidences = []
        
        for entry in sentiment_data:
            try:
                timestamp = entry.get('timestamp', '')
                if timestamp:
                    try:
                        if 'T' in timestamp:
                            date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        else:
                            date = datetime.fromisoformat(timestamp)
                        
                        # Get corresponding price data
                        price_data = get_market_data(symbol, period='1mo', interval='1d')
                        if not price_data.empty:
                            price_dates = price_data.index.date
                            entry_date = date.date()
                            
                            # Find closest price date
                            date_diffs = [abs((pd_date - entry_date).days) for pd_date in price_dates]
                            if date_diffs and min(date_diffs) <= 3:
                                closest_idx = date_diffs.index(min(date_diffs))
                                
                                # Append data for chart
                                dates.append(date)
                                sentiments.append(entry.get('overall_sentiment', 0))
                                prices.append(price_data.iloc[closest_idx]['Close'])
                                confidences.append(entry.get('confidence', 0))
                    except Exception:
                        continue
                        
            except Exception:
                continue
        
        if not dates:
            # Return empty chart if no data
            fig = go.Figure()
            fig.update_layout(title="No Historical Data Available")
            return fig
        
        # Create professional chart
        fig = go.Figure()
        
        # Add sentiment trace
        fig.add_trace(go.Scatter(
            x=dates,
            y=sentiments,
            mode='lines+markers',
            name='Sentiment Score',
            line=dict(color='#3498db', width=2),
            marker=dict(size=8, color='#3498db'),
            text=[f"Sentiment: {s:.3f}<br>Confidence: {c:.2f}" for s, c in zip(sentiments, confidences)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add price trace
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name='Price',
            line=dict(color='#2ecc71', width=2, dash='dash'),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f"Historical Sentiment and Price Trends - {self.bank_names.get(symbol, symbol)}",
                font=dict(size=18, family="Inter", weight=600, color="#2c3e50")
            ),
            xaxis=dict(
                title=dict(text="Date", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                gridcolor="#f1f1f1",
                linecolor="#dee2e6"
            ),
            yaxis=dict(
                title=dict(text="Value", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                gridcolor="#f1f1f1",
                linecolor="#dee2e6"
            ),
            height=500,
            showlegend=True,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter"),
            margin=dict(l=60, r=60, t=60, b=60)
        )
        
        return fig
    
    def create_correlation_chart(self, symbol: str, all_data: Dict) -> go.Figure:
        """Create professional correlation analysis chart"""
        sentiment_data = all_data.get(symbol, [])
        
        # Prepare correlation data
        correlation_points = []
        
        for entry in sentiment_data[-20:]:  # Last 20 entries
            try:
                sentiment = entry.get('overall_sentiment', 0)
                confidence = entry.get('confidence', 0)
                
                # Try to get corresponding price data
                timestamp = entry.get('timestamp', '')
                if timestamp:
                    try:
                        if 'T' in timestamp:
                            date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        else:
                            date = datetime.fromisoformat(timestamp)
                        
                        # Get price change for correlation analysis
                        price_data = get_market_data(symbol, period='1mo', interval='1d')
                        if not price_data.empty:
                            price_dates = price_data.index.date
                            entry_date = date.date()
                            
                            # Find closest price date
                            date_diffs = [abs((pd_date - entry_date).days) for pd_date in price_dates]
                            if date_diffs and min(date_diffs) <= 3:
                                closest_idx = date_diffs.index(min(date_diffs))
                                
                                # Calculate price change from previous day
                                if closest_idx > 0:
                                    current_price = price_data.iloc[closest_idx]['Close']
                                    prev_price = price_data.iloc[closest_idx-1]['Close']
                                    price_change = ((current_price - prev_price) / prev_price) * 100
                                    
                                    correlation_points.append({
                                        'sentiment': sentiment,
                                        'price_change': price_change,
                                        'confidence': confidence,
                                        'date': date
                                    })
                    except Exception:
                        continue
                        
            except Exception:
                continue
        
        fig = go.Figure()
        
        if correlation_points:
            sentiments = [p['sentiment'] for p in correlation_points]
            price_changes = [p['price_change'] for p in correlation_points]
            confidences = [p['confidence'] for p in correlation_points]
            dates = [p['date'].strftime('%Y-%m-%d') for p in correlation_points]
            
            # Create scatter plot with confidence as size
            fig.add_trace(go.Scatter(
                x=sentiments,
                y=price_changes,
                mode='markers',
                marker=dict(
                    size=[c * 30 + 5 for c in confidences],  # Scale confidence to marker size
                    color=confidences,
                    colorscale='Viridis',
                    colorbar=dict(title="Confidence Level"),
                    line=dict(width=1, color='white')
                ),
                text=dates,
                hovertemplate='<b>Correlation Point</b><br>' +
                             'Sentiment: %{x:.3f}<br>' +
                             'Price Change: %{y:.2f}%<br>' +
                             'Confidence: %{marker.color:.2f}<br>' +
                             'Date: %{text}<extra></extra>'
            ))
            
            # Add trend line if enough data points
            if len(correlation_points) >= 3:
                try:
                    # Calculate correlation coefficient
                    correlation_coef = np.corrcoef(sentiments, price_changes)[0, 1]
                    
                    # Add trend line
                    z = np.polyfit(sentiments, price_changes, 1)
                    p = np.poly1d(z)
                    trend_x = np.linspace(min(sentiments), max(sentiments), 100)
                    trend_y = p(trend_x)
                    
                    fig.add_trace(go.Scatter(
                        x=trend_x,
                        y=trend_y,
                        mode='lines',
                        name=f'Trend (r={correlation_coef:.3f})',
                        line=dict(color='red', width=2, dash='dash'),
                        hovertemplate='<b>Trend Line</b><br>' +
                                     'Correlation: %{name}<extra></extra>'
                    ))
                except Exception as e:
                    logger.warning(f"Error calculating correlation trend: {e}")
        else:
            fig.add_annotation(
                text="Insufficient data for correlation analysis",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, family="Inter", color="#6c757d")
            )
        
        fig.update_layout(
            title=dict(
                text=f"Sentiment vs Price Movement Correlation - {self.bank_names.get(symbol, symbol)}",
                font=dict(size=18, family="Inter", weight=600, color="#2c3e50")
            ),
            xaxis=dict(
                title=dict(text="Sentiment Score", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                gridcolor="#f1f1f1",
                linecolor="#dee2e6"
            ),
            yaxis=dict(
                title=dict(text="Price Change (%)", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                gridcolor="#f1f1f1",
                linecolor="#dee2e6",
                zeroline=True,
                zerolinecolor="#dee2e6",
                zerolinewidth=2
            ),
            height=500,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter"),
            margin=dict(l=60, r=60, t=60, b=60)
        )
        
        return fig
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            market_data = get_market_data(symbol, period='1d', interval='1m')
            if not market_data.empty:
                return float(market_data['Close'].iloc[-1])
            else:
                # Fallback to daily data
                daily_data = get_market_data(symbol, period='5d', interval='1d')
                if not daily_data.empty:
                    return float(daily_data['Close'].iloc[-1])
                return 0.0
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return 0.0
    
    def display_position_risk_section(self):
        """Display the Position Risk Assessor section"""
        if not POSITION_RISK_AVAILABLE:
            st.markdown("""
            <div class="alert alert-warning">
                <strong>⚠️ Position Risk Assessor Unavailable</strong><br>
                The Position Risk Assessor module is not available. Please ensure 
                <code>src/position_risk_assessor.py</code> is properly installed.
            </div>
            """, unsafe_allow_html=True)
            return
        
        self.create_section_header(
            "Position Risk Assessment", 
            "ML-powered analysis for existing positions and recovery predictions",
            "🎯"
        )
        
        # Position input form with enhanced UI
        with st.form("position_risk_form"):
            st.markdown("""
            <div class="form-section">
                <h4>📊 Position Details</h4>
                <p>Enter your position information for comprehensive ML-powered risk analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced form layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                symbol = st.selectbox(
                    "📈 Select Bank:",
                    options=list(self.bank_symbols),
                    format_func=lambda x: f"🏦 {self.bank_names.get(x, x)} ({x})",
                    help="Choose the bank stock you want to analyze"
                )
            
            with col2:
                entry_price = st.number_input(
                    "💰 Entry Price ($):",
                    min_value=0.01,
                    value=100.00,
                    step=0.01,
                    format="%.2f",
                    help="The price at which you entered the position"
                )
            
            with col3:
                current_price = st.number_input(
                    "📊 Current Price ($):",
                    min_value=0.01,
                    value=self.get_current_price(symbol) or 99.00,
                    step=0.01,
                    format="%.2f",
                    help="Current market price of the stock"
                )
            
            col4, col5, col6 = st.columns(3)
            
            with col4:
                position_type = st.selectbox(
                    "📍 Position Type:",
                    options=['long', 'short'],
                    format_func=lambda x: f"📈 Long Position (Buy)" if x == 'long' else f"📉 Short Position (Sell)",
                    help="Type of position you have taken"
                )
            
            with col5:
                entry_date = st.date_input(
                    "📅 Entry Date:",
                    value=datetime.now().date() - timedelta(days=7),
                    max_value=datetime.now().date(),
                    help="When you entered this position"
                )
            
            with col6:
                position_size = st.number_input(
                    "📦 Position Size ($):",
                    min_value=100.0,
                    value=10000.0,
                    step=100.0,
                    format="%.0f",
                    help="Total dollar value of your position"
                )
            
            # Advanced options expander
            with st.expander("⚙️ Advanced Risk Parameters", expanded=False):
                adv_col1, adv_col2 = st.columns(2)
                
                with adv_col1:
                    max_loss_tolerance = st.slider(
                        "🛡️ Maximum Loss Tolerance (%):",
                        min_value=1.0,
                        max_value=50.0,
                        value=10.0,
                        step=1.0,
                        help="Maximum loss you're willing to accept"
                    )
                    
                    time_horizon = st.selectbox(
                        "⏰ Investment Time Horizon:",
                        options=['short_term', 'medium_term', 'long_term'],
                        format_func=lambda x: {
                            'short_term': '📅 Short-term (< 1 month)',
                            'medium_term': '📆 Medium-term (1-6 months)', 
                            'long_term': '📋 Long-term (> 6 months)'
                        }.get(x, x),
                        help="Your planned holding period"
                    )
                
                with adv_col2:
                    risk_appetite = st.selectbox(
                        "🎯 Risk Appetite:",
                        options=['conservative', 'moderate', 'aggressive'],
                        format_func=lambda x: {
                            'conservative': '🛡️ Conservative',
                            'moderate': '⚖️ Moderate',
                            'aggressive': '🚀 Aggressive'
                        }.get(x, x),
                        help="Your general risk tolerance"
                    )
                    
                    include_sentiment = st.checkbox(
                        "📰 Include Sentiment Analysis",
                        value=True,
                        help="Factor in news sentiment for risk assessment"
                    )
            
            # Calculate current P&L for preview
            current_return = ((current_price - entry_price) / entry_price * 100) if position_type == 'long' else ((entry_price - current_price) / entry_price * 100)
            current_pnl = position_size * (current_return / 100)
            
            # Position preview
            st.markdown(f"""
            <div class="position-preview">
                <h5>📊 Position Preview</h5>
                <div class="preview-grid">
                    <div>Current Return: <span class="{'profit' if current_return > 0 else 'loss'}">{current_return:+.2f}%</span></div>
                    <div>P&L: <span class="{'profit' if current_pnl > 0 else 'loss'}">${current_pnl:+,.2f}</span></div>
                    <div>Days Held: {(datetime.now().date() - entry_date).days} days</div>
                    <div>Risk Level: <span class="{'low-risk' if abs(current_return) < 5 else 'medium-risk' if abs(current_return) < 15 else 'high-risk'}">{('Low' if abs(current_return) < 5 else 'Medium' if abs(current_return) < 15 else 'High')}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced submit button
            submitted = st.form_submit_button(
                "🎯 Perform Advanced Risk Assessment", 
                use_container_width=True,
                type="primary"
            )
        
        if submitted:
            with st.spinner("🔄 Analyzing position risk..."):
                try:
                    # Initialize Position Risk Assessor
                    assessor = PositionRiskAssessor()
                    
                    # Perform risk assessment
                    result = assessor.assess_position_risk(
                        symbol=symbol,
                        entry_price=entry_price,
                        current_price=current_price,
                        position_type=position_type,
                        entry_date=datetime.combine(entry_date, datetime.min.time())
                    )
                    
                    if 'error' in result:
                        st.error(f"❌ Error in risk assessment: {result['error']}")
                    else:
                        self.display_risk_assessment_results(result, symbol, entry_price, current_price, position_type)
                        
                except Exception as e:
                    st.error(f"❌ Error initializing Position Risk Assessor: {str(e)}")
                    # Fallback to heuristic assessment
                    st.warning("🔄 Falling back to heuristic risk assessment...")
                    self.display_fallback_risk_assessment(symbol, entry_price, current_price, position_type)
    
    def display_risk_assessment_results(self, result: Dict, symbol: str, entry_price: float, current_price: float, position_type: str):
        """Display comprehensive risk assessment results with enhanced UI"""
        
        # Header
        bank_name = self.bank_names.get(symbol, symbol)
        current_return = result.get('current_return_pct', 0)
        position_status = result.get('position_status', 'unknown')
        
        # Status styling
        status_icon = "🟢" if position_status == 'profitable' else "🔴"
        status_text = "Profitable" if position_status == 'profitable' else "Underwater"
        status_color = "#27ae60" if position_status == 'profitable' else "#e74c3c"
        
        # Get risk metrics
        risk_metrics = result.get('risk_metrics', {})
        overall_risk_score = risk_metrics.get('overall_risk_score', 5)
        
        # Risk level determination
        if overall_risk_score <= 3:
            risk_level = "Low"
            risk_color = "#27ae60"
            risk_icon = "🟢"
        elif overall_risk_score <= 6:
            risk_level = "Medium"
            risk_color = "#f39c12"
            risk_icon = "🟡"
        else:
            risk_level = "High"
            risk_color = "#e74c3c"
            risk_icon = "🔴"
        
        # Enhanced header with summary
        st.markdown(f"""
        <div class="risk-results-container">
            <div class="risk-header">
                <h3>{status_icon} Position Risk Assessment: {bank_name} ({symbol})</h3>
                <div class="risk-summary">
                    <div class="risk-summary-item">
                        <div class="risk-summary-value" style="color: {status_color};">{current_return:+.2f}%</div>
                        <div class="risk-summary-label">Current Return</div>
                    </div>
                    <div class="risk-summary-item">
                        <div class="risk-summary-value">{risk_icon} {risk_level}</div>
                        <div class="risk-summary-label">Risk Level</div>
                    </div>
                    <div class="risk-summary-item">
                        <div class="risk-summary-value">${entry_price:.2f} → ${current_price:.2f}</div>
                        <div class="risk-summary-label">Price Movement</div>
                    </div>
                    <div class="risk-summary-item">
                        <div class="risk-summary-value">{overall_risk_score:.1f}/10</div>
                        <div class="risk-summary-label">Risk Score</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Recovery Predictions
        recovery_predictions = result.get('recovery_predictions', {})
        if recovery_predictions:
            st.markdown("### 🔮 Recovery Predictions")
            self.create_recovery_probability_chart(recovery_predictions)
        
        # Risk Metrics
        risk_metrics = result.get('risk_metrics', {})
        if risk_metrics:
            st.markdown("### ⚠️ Risk Analysis")
            self.display_risk_breakdown(risk_metrics)
        
        # Recommendations
        recommendations = result.get('recommendations', {})
        if recommendations:
            st.markdown("### 💡 AI Recommendations")
            self.display_position_recommendations(recommendations)
        
        # Market Context
        market_context = result.get('market_context', {})
        if market_context:
            st.markdown("### 🌍 Market Context")
            self.display_market_context(market_context, symbol)
        
        # Action Plan
        if recommendations and risk_metrics:
            st.markdown("### 🎯 Action Plan")
            self.display_action_plan(recommendations, current_return, risk_metrics.get('overall_risk_score', 5))


    def create_recovery_probability_chart(self, recovery_predictions: Dict):
        """Create recovery probability visualization chart"""
        timeframes = list(recovery_predictions.keys())
        probabilities = [recovery_predictions[tf].get('probability', 0) * 100 for tf in timeframes]
        
        # Professional color scheme based on probability
        colors = []
        for prob in probabilities:
            if prob >= 80:
                colors.append('#27ae60')  # High probability - green
            elif prob >= 60:
                colors.append('#f39c12')  # Medium probability - orange
            else:
                colors.append('#e74c3c')  # Low probability - red
        
        fig = go.Figure(go.Bar(
            x=[tf.replace('_', ' ').title() for tf in timeframes],
            y=probabilities,
            marker=dict(
                color=colors,
                line=dict(color='white', width=1)
            ),
            text=[f"{p:.1f}%" for p in probabilities],
            textposition='auto',
            textfont=dict(size=12, family="Inter", weight=600),
            hovertemplate='<b>%{x}</b><br>' +
                         'Recovery Probability: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text="Recovery Probability by Timeframe",
                font=dict(size=18, family="Inter", weight=600, color="#2c3e50")
            ),
            xaxis=dict(
                title=dict(text="Timeframe", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                gridcolor="#f1f1f1",
                linecolor="#dee2e6"
            ),
            yaxis=dict(
                title=dict(text="Probability (%)", font=dict(size=14, family="Inter")),
                tickfont=dict(size=12, family="Inter"),
                range=[0, 100],
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
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recovery details table
        recovery_data = []
        for tf, pred in recovery_predictions.items():
            recovery_data.append({
                'Timeframe': tf.replace('_', ' ').title(),
                'Probability': f"{pred.get('probability', 0) * 100:.1f}%",
                'Confidence': f"{pred.get('confidence', 0):.2f}",
                'Expected Return': f"{pred.get('expected_return', 0):+.2f}%",
                'Risk Level': pred.get('risk_level', 'Unknown').title()
            })
        
        st.dataframe(pd.DataFrame(recovery_data), use_container_width=True, hide_index=True)
    
    def display_position_recommendations(self, recommendations: Dict):
        """Display AI-generated position recommendations"""
        primary_action = recommendations.get('primary_action', 'MONITOR')
        confidence = recommendations.get('confidence', 0)
        reasoning = recommendations.get('reasoning', 'No reasoning provided')
        risk_level = recommendations.get('risk_level', 'Medium')
        
        # Action styling
        if primary_action in ['EXIT_RECOMMENDED', 'STRONG_SELL']:
            action_color = "🔴"
            action_class = "negative"
        elif primary_action in ['REDUCE_POSITION', 'PARTIAL_EXIT']:
            action_color = "🟡"
            action_class = "warning"
        elif primary_action in ['HOLD', 'MONITOR']:
            action_color = "🟢"
            action_class = "positive"
        else:
            action_color = "🔵"
            action_class = "neutral"
        
        st.markdown(f"""
        <div class="bank-card">
            <div class="bank-card-header">
                <h3>{action_color} Primary Recommendation: {primary_action.replace('_', ' ').title()}</h3>
            </div>
            <div class="bank-card-body">
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    <strong>Reasoning:</strong> {reasoning}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendation metrics
        rec_metrics = [
            {
                'title': 'Action Confidence',
                'value': f"{confidence:.1%}",
                'status': 'positive' if confidence > 0.8 else 'warning' if confidence > 0.6 else 'negative',
                'subtitle': 'AI certainty level'
            },
            {
                'title': 'Risk Assessment',
                'value': risk_level,
                'status': 'negative' if risk_level == 'High' else 'warning' if risk_level == 'Medium' else 'positive',
                'subtitle': 'Position risk level'
            }
        ]
        
        # Add specific recommendations if available
        if 'stop_loss_recommendation' in recommendations:
            rec_metrics.append({
                'title': 'Stop Loss',
                'value': f"${recommendations['stop_loss_recommendation']:.2f}",
                'status': 'warning',
                'subtitle': 'Suggested stop level'
            })
        
        if 'target_price' in recommendations:
            rec_metrics.append({
                'title': 'Target Price',
                'value': f"${recommendations['target_price']:.2f}",
                'status': 'positive',
                'subtitle': 'Recovery target'
            })
        
        self.display_professional_metrics(rec_metrics)
        
        # Additional recommendations
        if 'additional_actions' in recommendations and recommendations['additional_actions']:
            st.markdown("#### 📋 Additional Recommendations:")
            for action in recommendations['additional_actions']:
                st.markdown(f"• {action}")
    
    def display_risk_breakdown(self, risk_metrics: Dict):
        """Display detailed risk metrics breakdown"""
        overall_risk = risk_metrics.get('overall_risk_score', 5)
        max_adverse_excursion = risk_metrics.get('max_adverse_excursion_prediction', 0)
        recovery_timeframe = risk_metrics.get('estimated_recovery_days', 0)
        volatility_risk = risk_metrics.get('volatility_risk', 'Medium')
        
        # Risk color coding
        if overall_risk >= 8:
            risk_color = "🚨"
            risk_status = "critical"
        elif overall_risk >= 6:
            risk_color = "🔴"
            risk_status = "negative"
        elif overall_risk >= 4:
            risk_color = "🟡"
            risk_status = "warning"
        else:
            risk_color = "🟢"
            risk_status = "positive"
        
        st.markdown(f"""
        <div class="bank-card">
            <div class="bank-card-header">
                <h3>{risk_color} Risk Analysis Breakdown</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        risk_breakdown_metrics = [
            {
                'title': 'Overall Risk Score',
                'value': f"{overall_risk:.1f}/10",
                'status': risk_status,
                'subtitle': 'Composite risk level'
            },
            {
                'title': 'Max Adverse Excursion',
                'value': f"{max_adverse_excursion:+.2f}%",
                'status': 'negative' if max_adverse_excursion < -5 else 'warning' if max_adverse_excursion < -2 else 'positive',
                'subtitle': 'Worst-case scenario'
            },
            {
                'title': 'Recovery Timeframe',
                'value': f"{recovery_timeframe:.0f} days",
                'status': 'positive' if recovery_timeframe < 10 else 'warning' if recovery_timeframe < 30 else 'negative',
                'subtitle': 'Estimated recovery time'
            },
            {
                'title': 'Volatility Risk',
                'value': volatility_risk,
                'status': 'negative' if volatility_risk == 'High' else 'warning' if volatility_risk == 'Medium' else 'positive',
                'subtitle': 'Price volatility assessment'
            }
        ]
        
        self.display_professional_metrics(risk_breakdown_metrics)
        
        # Risk factors
        if 'risk_factors' in risk_metrics and risk_metrics['risk_factors']:
            st.markdown("#### ⚠️ Key Risk Factors:")
            for factor in risk_metrics['risk_factors']:
                st.markdown(f"• {factor}")
    
    def display_market_context(self, market_context: Dict, symbol: str):
        """Display market context and environmental factors"""
        volatility_5d = market_context.get('volatility_5d', 0)
        volatility_20d = market_context.get('volatility_20d', 0)
        volume_ratio = market_context.get('volume_ratio', 1.0)
        support_distance = market_context.get('support_distance', 0)
        resistance_distance = market_context.get('resistance_distance', 0)
        
        context_metrics = [
            {
                'title': '5-Day Volatility',
                'value': f"{volatility_5d:.1%}",
                'status': 'negative' if volatility_5d > 0.3 else 'warning' if volatility_5d > 0.2 else 'positive',
                'subtitle': 'Recent price volatility'
            },
            {
                'title': '20-Day Volatility',
                'value': f"{volatility_20d:.1%}",
                'status': 'negative' if volatility_20d > 0.25 else 'warning' if volatility_20d > 0.15 else 'positive',
                'subtitle': 'Medium-term volatility'
            },
            {
                'title': 'Volume Activity',
                'value': f"{volume_ratio:.1f}x",
                'status': 'positive' if volume_ratio > 1.5 else 'warning' if volume_ratio > 0.8 else 'negative',
                'subtitle': 'vs 20-day average'
            },
            {
                'title': 'Support Distance',
                'value': f"{support_distance:.1%}",
                'status': 'positive' if support_distance < 0.02 else 'warning' if support_distance < 0.05 else 'negative',
                'subtitle': 'Distance to support'
            }
        ]
                
        self.display_professional_metrics(context_metrics)
        
        # Technical context from existing analysis
        if symbol in self.technical_data:
            tech_data = self.technical_data[symbol]
            if tech_data and 'trend' in tech_data:
                trend_info = tech_data['trend']
                st.markdown(f"""
                **📈 Technical Context:**
                - **Trend Direction:** {trend_info.get('direction', 'Unknown').replace('_', ' ').title()}
                - **Trend Strength:** {trend_info.get('strength', 'Unknown').title()}
                - **Current Recommendation:** {tech_data.get('recommendation', 'HOLD')}
                """)
    
    def display_action_plan(self, recommendations: Dict, current_return: float, risk_score: float):
        """Display comprehensive action plan based on analysis"""
        
        st.markdown("##### 🎯 Comprehensive Action Plan")
        
        # Immediate actions
        st.markdown("**🚨 Immediate Actions (Next 24 Hours):**")
        
        if current_return < -5:
            st.error("""
            🚨 **Critical Action Required:**
            1. Review position size immediately
            2. Consider setting tight stop-loss
            3. Monitor market news closely
            4. Prepare exit strategy
            """)
        elif current_return < -2:
            st.warning("""
            ⚠️ **Monitor Closely:**
            1. Set price alerts for key levels
            2. Review stop-loss placement
            3. Watch for sentiment changes
            4. Consider position reduction
            """)
        else:
            st.info("""
            ℹ️ **Standard Monitoring:**
            1. Continue regular monitoring
            2. Watch for trend changes
            3. Maintain risk management
            4. Look for exit opportunities if profitable
            """)
        
        # Short-term plan (1-2 weeks)
        st.markdown("**📅 Short-term Plan (1-2 Weeks):**")
        
        primary_action = recommendations.get('primary_action', 'MONITOR')
        
        if primary_action in ['CONSIDER_EXIT', 'EXIT_RECOMMENDED']:
            st.markdown("""
            🔴 **Exit Strategy:**
            1. Plan staged exit over 2-3 trading sessions
            2. Monitor market conditions for optimal timing
            3. Consider partial exits to reduce risk
            4. Document lessons learned for future trades
            """)
        elif primary_action == 'REDUCE_POSITION':
            st.markdown("""
            🟡 **Position Reduction:**
            1. Reduce position size by 25-50%
            2. Maintain core position for potential recovery
            3. Reassess weekly based on performance
            4. Set stricter stop-loss on remaining position
            """)
        else:
            st.markdown("""
            🟢 **Hold Strategy:**
            1. Maintain current position size
            2. Monitor technical and sentiment indicators
            3. Set profit-taking targets if position recovers
            4. Review strategy if new information emerges
            """)
        
        # Risk management checklist
        st.markdown("**✅ Risk Management Checklist:**")
        checklist_items = [
            "Position size appropriate for risk tolerance",
            "Stop-loss levels clearly defined",
            "Recovery targets set and monitored",
            "Market context regularly updated",
            "Alternative scenarios planned",
            "Exit strategy documented and ready"
        ]
        
        for item in checklist_items:
            st.markdown(f"- [ ] {item}")
    
    def display_fallback_risk_assessment(self, symbol: str, entry_price: float, current_price: float, position_type: str):
        """Display a simplified heuristic risk assessment when full system unavailable"""
        
        bank_name = self.bank_names.get(symbol, symbol)
        
        # Calculate basic metrics
        if position_type == 'long':
            current_return = ((current_price - entry_price) / entry_price) * 100
        else:
            current_return = ((entry_price - current_price) / entry_price) * 100
        
        # Simple heuristic risk scoring
        loss_magnitude = abs(min(0, current_return))
        
        if loss_magnitude > 10:
            risk_score = 9
            risk_status = "🚨 Critical"
            risk_color = "#c0392b"
            action = "IMMEDIATE EXIT RECOMMENDED"
        elif loss_magnitude > 5:
            risk_score = 7
            risk_status = "🔴 High Risk"
            risk_color = "#e74c3c"
            action = "CONSIDER EXIT"
        elif loss_magnitude > 2:
            risk_score = 5
            risk_status = "🟡 Moderate Risk"
            risk_color = "#f39c12"
            action = "REDUCE POSITION"
        elif loss_magnitude > 0:
            risk_score = 3
            risk_status = "🟢 Low Risk"
            risk_color = "#27ae60"
            action = "MONITOR CLOSELY"
        else:
            risk_score = 1
            risk_status = "🟢 Profitable"
            risk_color = "#27ae60"
            action = "HOLD"
        
        # Display simplified assessment with enhanced styling
        st.markdown(f"""
        <div class="risk-results-container">
            <div class="risk-header">
                <h3>🔧 Basic Risk Assessment: {bank_name} ({symbol})</h3>
                <p style="opacity: 0.9; margin: 0.5rem 0 0 0;">Simplified analysis - Full ML assessment unavailable</p>
                <div class="risk-summary">
                    <div class="risk-summary-item">
                        <div class="risk-summary-value" style="color: {'#27ae60' if current_return > 0 else '#e74c3c'};">{current_return:+.2f}%</div>
                        <div class="risk-summary-label">Current Return</div>
                    </div>
                    <div class="risk-summary-item">
                        <div class="risk-summary-value" style="color: {risk_color};">{risk_status}</div>
                        <div class="risk-summary-label">Risk Level</div>
                    </div>
                    <div class="risk-summary-item">
                        <div class="risk-summary-value">{risk_score}/10</div>
                        <div class="risk-summary-label">Risk Score</div>
                    </div>
                    <div class="risk-summary-item">
                        <div class="risk-summary-value">{action}</div>
                        <div class="risk-summary-label">Recommended Action</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Basic recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💡 Basic Recommendations")
            recommendations = []
            
            if current_return < -5:
                recommendations.append("🛑 Consider stopping losses to prevent further decline")
                recommendations.append("📉 Position is showing significant adverse movement")
            elif current_return < 0:
                recommendations.append("⚠️ Monitor position closely for recovery signs")
                recommendations.append("📊 Consider reducing position size if losses continue")
            else:
                recommendations.append("✅ Position is currently profitable")
                recommendations.append("📈 Consider taking partial profits if desired")
            
            recommendations.append(f"🎯 Current risk level: {risk_status}")
            recommendations.append("🔄 Enable full ML assessment for detailed analysis")
            
            for rec in recommendations:
                st.markdown(f"• {rec}")
        
        with col2:
            st.markdown("#### 📊 Position Metrics")
            st.metric("Entry Price", f"${entry_price:.2f}")
            st.metric("Current Price", f"${current_price:.2f}")
            st.metric("Return", f"{current_return:+.2f}%", delta=f"{current_return:.2f}%")
            
            # Simple sentiment context
            st.markdown("#### 📰 Context")
            st.info("💡 For detailed sentiment analysis and ML predictions, please ensure the Position Risk Assessor is properly configured.")
        
        # Upgrade notice
        st.markdown("""
        ---
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px; margin: 1rem 0;">
            <h5 style="color: #495057; margin: 0 0 0.5rem 0;">🚀 Upgrade to Full ML Assessment</h5>
            <p style="color: #6c757d; margin: 0; font-size: 0.9rem;">
                Get advanced recovery predictions, sentiment analysis, and personalized recommendations
                by configuring the complete Position Risk Assessor system.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="bank-card">
            <div class="bank-card-header">
                <h3>🎯 Heuristic Risk Assessment: {self.bank_names.get(symbol, symbol)}</h3>
            </div>
            <div class="bank-card-body">
                <div style="background: {'#f8d7da' if loss_magnitude > 2 else '#fff3cd' if loss_magnitude > 0 else '#d4edda'}; 
                           padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    <strong>Current Return:</strong> {current_return:+.2f}% | 
                    <strong>Risk Level:</strong> {risk_status} | 
                    <strong>Recommended Action:</strong> {action}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Basic metrics
        heuristic_metrics = [
            {
                'title': 'Risk Score',
                'value': f"{risk_score}/10",
                'status': 'negative' if risk_score > 6 else 'warning' if risk_score > 3 else 'positive',
                'subtitle': 'Heuristic assessment'
            },
            {
                'title': 'Loss Magnitude',
                'value': f"{loss_magnitude:.1f}%",
                'status': 'negative' if loss_magnitude > 5 else 'warning' if loss_magnitude > 2 else 'positive',
                'subtitle': 'Current drawdown'
            },
            {
                'title': 'Position Status',
                'value': "Underwater" if current_return < 0 else "Profitable",
                'status': 'negative' if current_return < 0 else 'positive',
                'subtitle': 'Current state'
            },
            {
                'title': 'Action Required',
                'value': action.split()[0],
                'status': 'negative' if 'EXIT' in action else 'warning' if 'REDUCE' in action else 'positive',
                'subtitle': 'Immediate step'
            }
        ]
        
        self.display_professional_metrics(heuristic_metrics)
        
        # Basic recommendations
        st.markdown("#### 💡 Basic Recommendations")
        
        if loss_magnitude > 5:
            st.error("🚨 **High Risk Position**: Consider immediate action to limit further losses")
        elif loss_magnitude > 2:
            st.warning("⚠️ **Moderate Risk**: Monitor closely and consider position reduction")
        elif current_return < 0:
            st.info("ℹ️ **Minor Loss**: Normal market fluctuation, continue monitoring")
        else:
            st.success("✅ **Profitable Position**: Maintain current strategy")

# Main Dashboard Execution
def main():
    """Main function to run the professional dashboard"""
    try:
        # Initialize dashboard
        dashboard = NewsAnalysisDashboard()
        
        # Debug: Check if data is loading
        st.sidebar.markdown("### 🔍 Debug Info")
        all_data = dashboard.load_sentiment_data()
        data_count = sum(len(data) for data in all_data.values())
        st.sidebar.info(f"📊 Total data points: {data_count}")
        st.sidebar.info(f"📁 Available banks: {len(all_data)}")
        
        # Create professional header
        dashboard.create_professional_header()
        
        # Professional sidebar navigation
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); 
                       border-radius: 10px; margin-bottom: 1.5rem;">
                <h3 style="color: white; margin: 0;">🏦 ASX Bank Analytics</h3>
                <p style="color: #ecf0f1; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Professional Trading Dashboard</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Navigation options
            selected_view = st.selectbox(
                "📊 Select Analysis View:",
                options=[
                    "📈 Market Overview",
                    "🏦 Individual Bank Analysis", 
                    "📊 Technical Analysis",
                    "🎯 Position Risk Assessor",
                    "📰 News & Sentiment",
                    "⚙️ System Status"
                ],
                index=0
            )
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.cache_data.clear()
                dashboard.technical_data.clear()
                st.success("✅ Data refreshed!")
                st.rerun()
            
            if st.button("📊 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("✅ Cache cleared!")
                st.rerun()
            
            # System information
            st.markdown("### ℹ️ System Info")
            st.info(f"""
            **🔧 Status:** Operational
            **🕐 Last Update:** {datetime.now().strftime('%H:%M:%S')}
            **📊 Banks Monitored:** {len(dashboard.bank_symbols)}
            **🎯 Risk Assessor:** {'✅ Available' if POSITION_RISK_AVAILABLE else '❌ Unavailable'}
            **📈 Technical Analysis:** Available
            **Dashboard Version:** 2.0.0
            """)
        
        # Main content area
        if selected_view == "📈 Market Overview":
            display_market_overview(dashboard)
        elif selected_view == "🏦 Individual Bank Analysis":
            display_individual_bank_analysis(dashboard)
        elif selected_view == "📊 Technical Analysis":
            display_technical_analysis(dashboard)
        elif selected_view == "🎯 Position Risk Assessor":
            dashboard.display_position_risk_section()
        elif selected_view == "📰 News & Sentiment":
            display_news_sentiment_analysis(dashboard)
        elif selected_view == "⚙️ System Status":
            display_system_status(dashboard)
            
    except Exception as e:
        st.error(f"❌ Dashboard Error: {str(e)}")
        logger.error(f"Dashboard error: {e}")
        
        # Fallback simple interface
        st.markdown("## 🚨 Fallback Mode")
        st.warning("The main dashboard encountered an error. Using simplified interface.")
        
        # Basic data display
        try:
            dashboard = NewsAnalysisDashboard()
            data = dashboard.load_sentiment_data()
            
            if data:
                st.markdown("### Available Data:")
                for symbol, symbol_data in data.items():
                    if symbol_data:
                        latest = dashboard.get_latest_analysis(symbol_data)
                        if latest:
                            st.write(f"**{dashboard.bank_names.get(symbol, symbol)}:** {latest.get('overall_sentiment', 0):.3f}")
            else:
                st.warning("No sentiment data available")
                
        except Exception as fallback_error:
            st.error(f"❌ Fallback mode also failed: {str(fallback_error)}")

def display_market_overview(dashboard):
    """Display market overview with professional charts"""
    dashboard.create_section_header(
        "Market Overview", 
        "Real-time sentiment analysis and market indicators for all ASX banks",
        "📈"
    )
    
    # Load data
    with st.spinner("📊 Loading market data..."):
        all_data = dashboard.load_sentiment_data()
    
    # Debug info
    st.info(f"🔍 Debug: Found {len(all_data)} banks with {sum(len(data) for data in all_data.values())} total data points")
    
    if not any(all_data.values()):
        st.warning("⚠️ No sentiment data available. Please run data collection first.")
        st.markdown("""
        ### 🚀 Getting Started
        
        To see data in this dashboard, you need to run the data collection process:
        
        1. **Run the news collector:**
           ```bash
           python core/news_trading_analyzer.py
           ```
        
        2. **Or run the main analysis:**
           ```bash
           python enhanced_main.py
           ```
        
        3. **Check data directory:**
           - Data should appear in `data/sentiment_history/`
           - Each bank should have a `[SYMBOL]_history.json` file
        """)
        return
    
    # Market summary metrics
    total_banks = len(dashboard.bank_symbols)
    analyzed_banks = sum(1 for data in all_data.values() if data)
    
    latest_sentiments = []
    latest_confidences = []
    
    for data in all_data.values():
        if data:
            latest = dashboard.get_latest_analysis(data)
            if latest:
                latest_sentiments.append(latest.get('overall_sentiment', 0))
                latest_confidences.append(latest.get('confidence', 0))
    
    avg_sentiment = sum(latest_sentiments) / len(latest_sentiments) if latest_sentiments else 0
    avg_confidence = sum(latest_confidences) / len(latest_confidences) if latest_confidences else 0
    
    positive_count = sum(1 for s in latest_sentiments if s > 0.1)
    negative_count = sum(1 for s in latest_sentiments if s < -0.1)
    neutral_count = len(latest_sentiments) - positive_count - negative_count
    
    # Display market metrics
    market_metrics = [
        {
            'title': 'Banks Analyzed',
            'value': f"{analyzed_banks}/{total_banks}",
            'status': 'positive' if analyzed_banks == total_banks else 'warning',
            'subtitle': 'Data coverage'
        },
        {
            'title': 'Market Sentiment',
            'value': f"{avg_sentiment:+.3f}",
            'status': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral',
            'subtitle': 'Average sentiment'
        },
        {
            'title': 'Confidence Level',
            'value': f"{avg_confidence:.2f}",
            'status': 'positive' if avg_confidence > 0.7 else 'warning' if avg_confidence > 0.5 else 'negative',
            'subtitle': 'Analysis quality'
        },
        {
            'title': 'Positive Banks',
            'value': f"{positive_count}",
            'status': 'positive',
            'subtitle': f'vs {negative_count} negative'
        }
    ]
    
    dashboard.display_professional_metrics(market_metrics)
    
    # ML Progression Summary Section
    if dashboard.ml_tracker:
        st.markdown("---")
        dashboard.create_section_header(
            "🤖 Machine Learning Performance", 
            "Historical feedback on model improvement as more data is analyzed",
            "🧠"
        )
        
        try:
            # Get ML summary for last 30 days
            ml_summary = dashboard.ml_tracker.get_overall_ml_summary(30)
            
            # Display ML metrics
            ml_metrics = [
                {
                    'title': 'Models Tracked',
                    'value': f"{len(ml_summary['models_tracked'])}",
                    'status': 'positive',
                    'subtitle': 'Active ML models'
                },
                {
                    'title': 'Accuracy Trend',
                    'value': ml_summary['prediction_accuracy_trend'].title(),
                    'status': 'positive' if ml_summary['prediction_accuracy_trend'] == 'improving' else 'warning' if ml_summary['prediction_accuracy_trend'] == 'stable' else 'negative',
                    'subtitle': 'Model performance'
                },
                {
                    'title': 'Best Performer',
                    'value': ml_summary['best_performing_model'].replace('_', ' ').title() if ml_summary['best_performing_model'] else 'N/A',
                    'status': 'positive',
                    'subtitle': 'Highest accuracy'
                },
                {
                    'title': 'Most Improved',
                    'value': ml_summary['most_improved_model'].replace('_', ' ').title() if ml_summary['most_improved_model'] else 'N/A',
                    'status': 'positive',
                    'subtitle': 'Fastest progress'
                }
            ]
            
            dashboard.display_professional_metrics(ml_metrics)
            
            # ML Progression Chart
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 📊 Model Performance Progression")
                try:
                    ml_chart = dashboard.ml_tracker.create_progression_chart(days=30)
                    st.plotly_chart(ml_chart, use_container_width=True)
                except Exception as e:
                    st.warning(f"Unable to generate ML progression chart: {str(e)}")
            
            with col2:
                st.markdown("#### 💡 ML Recommendations")
                recommendations = ml_summary.get('recommendations', [])
                if recommendations:
                    for rec in recommendations[:5]:  # Show top 5 recommendations
                        st.markdown(f"• {rec}")
                else:
                    st.info("🔄 Gathering more data to generate recommendations...")
                
                # Additional ML insights
                st.markdown("#### 📈 Data Growth")
                data_growth = ml_summary.get('data_volume_growth', 0)
                if data_growth > 1000:
                    st.success(f"📚 Strong data growth: +{data_growth:,} records")
                elif data_growth > 100:
                    st.info(f"📊 Moderate growth: +{data_growth:,} records")
                else:
                    st.warning("⚡ Low data growth - consider increasing collection frequency")
        
        except Exception as e:
            st.warning(f"⚠️ ML progression tracking temporarily unavailable: {str(e)}")
    else:
        st.info("🤖 ML progression tracking not available - install ML tracker module for enhanced insights")
    
    st.markdown("---")
    
    # Professional charts in tabs
    chart_tabs = st.tabs(["📊 Sentiment Overview", "🎯 Confidence Analysis", "📈 Market Trends"])
    
    with chart_tabs[0]:
        sentiment_chart = dashboard.create_sentiment_overview_chart(all_data)
        st.plotly_chart(sentiment_chart, use_container_width=True)
    
    with chart_tabs[1]:
        confidence_chart = dashboard.create_confidence_distribution_chart(all_data)
        st.plotly_chart(confidence_chart, use_container_width=True)
    
    with chart_tabs[2]:
        st.info("📈 Market trends chart will be displayed here based on historical data")
    
    # Professional legends
    dashboard.display_confidence_legend()
    dashboard.display_sentiment_scale()

def display_individual_bank_analysis(dashboard):
    """Display detailed analysis for individual banks"""
    dashboard.create_section_header(
        "Individual Bank Analysis", 
        "Comprehensive sentiment, technical, and risk analysis for specific banks",
        "🏦"
    )
    
    # Bank selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_bank = st.selectbox(
            "🏦 Select Bank for Detailed Analysis:",
            options=dashboard.bank_symbols,
            format_func=lambda x: f"🏦 {dashboard.bank_names.get(x, x)} ({x})"
        )
    
    with col2:
        if st.button("🔄 Refresh Analysis", use_container_width=True):
            dashboard.technical_data.pop(selected_bank, None)
            st.cache_data.clear()
            st.rerun()
    
    # Load and display analysis
    all_data = dashboard.load_sentiment_data()
    bank_data = all_data.get(selected_bank, [])
    
    if bank_data:
        dashboard.display_bank_analysis(selected_bank, bank_data)
    else:
        st.warning(f"⚠️ No analysis data available for {dashboard.bank_names.get(selected_bank, selected_bank)}")

def display_technical_analysis(dashboard):
    """Display technical analysis view"""
    dashboard.create_section_header(
        "Technical Analysis", 
        "Advanced technical indicators and market analysis",
        "📊"
    )
    
    # Technical analysis for all banks
    st.markdown("### 📈 Technical Analysis Summary")
    
    # Create technical summary
    technical_summary = []
    
    for symbol in dashboard.bank_symbols:
        try:
            tech_analysis = dashboard.get_technical_analysis(symbol)
            if tech_analysis and 'current_price' in tech_analysis:
                
                current_price = tech_analysis.get('current_price', 0)
                recommendation = tech_analysis.get('recommendation', 'HOLD')
                momentum_score = tech_analysis.get('momentum', {}).get('score', 0)
                rsi = tech_analysis.get('indicators', {}).get('rsi', 50)
                
                technical_summary.append({
                    'Bank': dashboard.bank_names.get(symbol, symbol),
                    'Symbol': symbol,
                    'Price': f"${current_price:.2f}",
                    'Recommendation': recommendation,
                    'Momentum': f"{momentum_score:.1f}",
                    'RSI': f"{rsi:.1f}",
                    'Status': '🟢' if 'BUY' in recommendation else '🔴' if 'SELL' in recommendation else '🟡'
                })
        except Exception as e:
            logger.warning(f"Error getting technical analysis for {symbol}: {e}")
    
    if technical_summary:
        tech_df = pd.DataFrame(technical_summary)
        st.dataframe(tech_df, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Technical analysis data is currently unavailable")

def display_news_sentiment_analysis(dashboard):
    """Display news and sentiment analysis"""
    dashboard.create_section_header(
        "News & Sentiment Analysis", 
        "Latest news headlines and sentiment impact analysis",
        "📰"
    )
    
    # Load sentiment data
    all_data = dashboard.load_sentiment_data()
    
    # Aggregate recent news from all banks
    recent_news = []
    
    for symbol, data in all_data.items():
        if data:
            latest = dashboard.get_latest_analysis(data)
            if latest and 'recent_headlines' in latest:
                headlines = latest['recent_headlines']
                for headline in headlines[:3]:  # Top 3 from each bank
                    if headline:
                        recent_news.append({
                            'Bank': dashboard.bank_names.get(symbol, symbol),
                            'Headline': headline,
                            'Symbol': symbol,
                            'Sentiment': latest.get('overall_sentiment', 0)
                        })
    
    if recent_news:
        st.markdown("### 📰 Latest Market Headlines")
        
        for news in recent_news[:10]:  # Show top 10
            sentiment = news['Sentiment']
            sentiment_color = "🟢" if sentiment > 0.1 else "🔴" if sentiment < -0.1 else "🟡"
            
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; 
                       border-left: 4px solid {'#27ae60' if sentiment > 0.1 else '#e74c3c' if sentiment < -0.1 else '#f39c12'};">
                <strong>{sentiment_color} {news['Bank']} ({news['Symbol']})</strong><br>
                {news['Headline']}<br>
                <small>Sentiment: {sentiment:+.3f}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No recent news data available")

def display_system_status(dashboard):
    """Display system status and diagnostics"""
    dashboard.create_section_header(
        "System Status", 
        "Technical system diagnostics and data quality metrics",
        "⚙️"
    )
    
    # System metrics
    st.markdown("### 🔧 System Diagnostics")
    
    # Check data availability
    all_data = dashboard.load_sentiment_data()
    data_quality = []
    
    for symbol in dashboard.bank_symbols:
        symbol_data = all_data.get(symbol, [])
        
        data_quality.append({
            'Bank': dashboard.bank_names.get(symbol, symbol),
            'Symbol': symbol,
            'Records': len(symbol_data),
            'Latest': symbol_data[-1].get('timestamp', 'N/A')[:10] if symbol_data else 'N/A',
            'Status': '✅ Good' if len(symbol_data) > 5 else '⚠️ Limited' if len(symbol_data) > 0 else '❌ No Data'
        })
    
    st.dataframe(pd.DataFrame(data_quality), use_container_width=True, hide_index=True)
    
    # System information
    st.markdown("### ℹ️ System Information")
    
    system_info = {
        'Python Version': sys.version.split()[0],
        'Streamlit Version': st.__version__,
        'Total Banks Monitored': len(dashboard.bank_symbols),
        'Position Risk Assessor': 'Available' if POSITION_RISK_AVAILABLE else 'Not Available',
        'Technical Analysis': 'Available',
        'Dashboard Version': '2.0.0'
    }
    
    for key, value in system_info.items():
        st.write(f"**{key}:** {value}")
    
    # Show data loading status at bottom
    with st.expander("📊 Data Loading Status", expanded=False):
        all_data_check = dashboard.load_sentiment_data()
        for symbol, data in all_data_check.items():
            bank_name = dashboard.bank_names.get(symbol, symbol)
            status = f"✅ {len(data)} records" if data else "❌ No data"
            st.write(f"**{bank_name}:** {status}")


            # Run the dashboard
if __name__ == "__main__":
    main()
else:
    # For Streamlit cloud or when imported
    main()