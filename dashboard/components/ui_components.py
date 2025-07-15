"""
UI components for the dashboard
Handles rendering of headers, metrics, legends, and other UI elements
"""

import streamlit as st
from typing import Dict, List, Optional, Any

from ..utils.logging_config import setup_dashboard_logger, log_error_with_context
from ..utils.helpers import format_sentiment_score, get_confidence_level, format_timestamp

logger = setup_dashboard_logger(__name__)

class UIComponents:
    """Handles UI component rendering for the dashboard"""
    
    def __init__(self):
        self.css_loaded = False
        logger.info("UIComponents initialized")
    
    def load_professional_css(self):
        """Load professional CSS styling for the dashboard"""
        if self.css_loaded:
            return
        
        try:
            css = """
            <style>
            /* Professional Dashboard Styling */
            .main-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .main-header h1 {
                margin: 0;
                font-size: 2.5rem;
                font-weight: 600;
                font-family: 'Inter', sans-serif;
            }
            
            .main-header p {
                margin: 0.5rem 0 0 0;
                font-size: 1.1rem;
                opacity: 0.9;
            }
            
            .section-header {
                border-left: 4px solid #3498db;
                padding-left: 1rem;
                margin: 2rem 0 1rem 0;
            }
            
            .section-header h2 {
                color: #2c3e50;
                margin: 0;
                font-size: 1.8rem;
                font-weight: 600;
            }
            
            .subtitle {
                color: #6c757d;
                font-size: 1rem;
                margin-top: 0.25rem;
            }
            
            .metric-card {
                background: white;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 1.5rem;
                margin: 0.5rem 0;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                transition: transform 0.2s, box-shadow 0.2s;
            }
            
            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            
            .metric-card h4 {
                color: #495057;
                margin: 0 0 0.5rem 0;
                font-size: 0.9rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .metric-value {
                font-size: 1.8rem;
                font-weight: 700;
                margin: 0.5rem 0;
                line-height: 1.2;
            }
            
            .metric-subtitle {
                font-size: 0.8rem;
                color: #6c757d;
                margin-top: 0.5rem;
            }
            
            .status-positive { color: #27ae60; }
            .status-negative { color: #e74c3c; }
            .status-neutral { color: #6c757d; }
            .status-warning { color: #f39c12; }
            
            .confidence-high { 
                border-left: 4px solid #27ae60;
                background: linear-gradient(90deg, rgba(39, 174, 96, 0.05) 0%, transparent 100%);
            }
            .confidence-medium { 
                border-left: 4px solid #f39c12;
                background: linear-gradient(90deg, rgba(243, 156, 18, 0.05) 0%, transparent 100%);
            }
            .confidence-low { 
                border-left: 4px solid #e74c3c;
                background: linear-gradient(90deg, rgba(231, 76, 60, 0.05) 0%, transparent 100%);
            }
            
            .bank-card {
                background: white;
                border: 1px solid #dee2e6;
                border-radius: 12px;
                margin: 1rem 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
                overflow: hidden;
            }
            
            .bank-card-header {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 1.5rem;
                border-bottom: 1px solid #dee2e6;
            }
            
            .bank-card-header h3 {
                margin: 0;
                color: #2c3e50;
                font-size: 1.4rem;
                font-weight: 600;
            }
            
            .bank-card-body {
                padding: 2rem;
            }
            
            .news-item {
                background: white;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
                border-left: 4px solid #6c757d;
            }
            
            .news-item.positive {
                border-left-color: #27ae60;
                background: linear-gradient(90deg, rgba(39, 174, 96, 0.03) 0%, transparent 100%);
            }
            
            .news-item.negative {
                border-left-color: #e74c3c;
                background: linear-gradient(90deg, rgba(231, 76, 60, 0.03) 0%, transparent 100%);
            }
            
            .news-item.neutral {
                border-left-color: #6c757d;
            }
            
            .event-badge {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .event-positive { background: #d4edda; color: #155724; }
            .event-negative { background: #f8d7da; color: #721c24; }
            .event-neutral { background: #e2e3e5; color: #383d41; }
            
            .alert {
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
                border: 1px solid transparent;
            }
            
            .alert-info {
                background: #d1ecf1;
                border-color: #bee5eb;
                color: #0c5460;
            }
            
            .alert-warning {
                background: #fff3cd;
                border-color: #ffeaa7;
                color: #856404;
            }
            
            .alert-success {
                background: #d4edda;
                border-color: #c3e6cb;
                color: #155724;
            }
            
            .legend-container {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 1.5rem;
                margin: 1rem 0;
                border: 1px solid #dee2e6;
            }
            
            .legend-title {
                font-size: 1.2rem;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 1rem;
            }
            
            /* Responsive adjustments */
            @media (max-width: 768px) {
                .main-header h1 { font-size: 2rem; }
                .metric-value { font-size: 1.5rem; }
                .bank-card-body { padding: 1rem; }
            }
            </style>
            """
            
            st.markdown(css, unsafe_allow_html=True)
            self.css_loaded = True
            logger.debug("Professional CSS loaded successfully")
            
        except Exception as e:
            log_error_with_context(logger, e, "loading professional CSS")
    
    def create_professional_header(self):
        """Create professional dashboard header"""
        try:
            st.markdown("""
            <div class="main-header">
                <h1>📊 ASX Bank Analytics Platform</h1>
                <p>Professional sentiment analysis and technical indicators for Australian banking sector</p>
            </div>
            """, unsafe_allow_html=True)
            
            logger.debug("Professional header created")
            
        except Exception as e:
            log_error_with_context(logger, e, "creating professional header")
            st.title("📊 ASX Bank Analytics Platform")
    
    def create_section_header(self, title: str, subtitle: str = "", icon: str = ""):
        """Create professional section header"""
        try:
            icon_text = f"{icon} " if icon else ""
            subtitle_html = f'<div class="subtitle">{subtitle}</div>' if subtitle else ''
            
            st.markdown(f"""
            <div class="section-header">
                <h2>{icon_text}{title}</h2>
                {subtitle_html}
            </div>
            """, unsafe_allow_html=True)
            
            logger.debug(f"Section header created: {title}")
            
        except Exception as e:
            log_error_with_context(logger, e, f"creating section header: {title}")
            st.subheader(f"{icon} {title}")
    
    def display_professional_metrics(self, metrics: List[Dict]):
        """Display metrics in professional card layout"""
        try:
            if not metrics:
                logger.warning("No metrics provided for display")
                return
            
            cols = st.columns(len(metrics))
            
            for i, metric in enumerate(metrics):
                with cols[i]:
                    try:
                        value = metric.get('value', 'N/A')
                        delta = metric.get('delta', '')
                        status = metric.get('status', 'neutral')
                        subtitle = metric.get('subtitle', '')
                        title = metric.get('title', f'Metric {i+1}')
                        
                        delta_html = f'<div class="metric-subtitle status-{status}">{delta}</div>' if delta else ''
                        subtitle_html = f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ''
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{title}</h4>
                            <div class="metric-value status-{status}">{value}</div>
                            {subtitle_html}
                            {delta_html}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        log_error_with_context(logger, e, f"displaying metric {i}", metric=metric)
                        st.error(f"Error displaying metric: {metric.get('title', 'Unknown')}")
            
            logger.debug(f"Displayed {len(metrics)} professional metrics")
            
        except Exception as e:
            log_error_with_context(logger, e, "displaying professional metrics", metrics_count=len(metrics))
    
    def display_confidence_legend(self):
        """Display professional confidence score legend and decision criteria"""
        try:
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
            
            logger.debug("Confidence legend displayed")
            
        except Exception as e:
            log_error_with_context(logger, e, "displaying confidence legend")
    
    def display_sentiment_scale(self):
        """Display professional sentiment scale explanation"""
        try:
            st.markdown("""
            <div class="legend-container">
                <div class="legend-title">📈 Sentiment Score Scale</div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            scale_items = [
                ("Very Negative", "-1.0 to -0.5", "Strong Sell", "negative"),
                ("Negative", "-0.5 to -0.2", "Sell", "negative"),
                ("Neutral", "-0.2 to +0.2", "Hold", "neutral"),
                ("Positive", "+0.2 to +0.5", "Buy", "positive"),
                ("Very Positive", "+0.5 to +1.0", "Strong Buy", "positive")
            ]
            
            cols = [col1, col2, col3, col4, col5]
            
            for i, (label, range_text, action, status) in enumerate(scale_items):
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{label}</h4>
                        <div class="metric-value status-{status}">{range_text}</div>
                        <div class="metric-subtitle">{action}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            logger.debug("Sentiment scale displayed")
            
        except Exception as e:
            log_error_with_context(logger, e, "displaying sentiment scale")
    
    def display_alert(self, message: str, alert_type: str = "info", title: str = ""):
        """Display professional alert message"""
        try:
            title_html = f"<strong>{title}</strong><br>" if title else ""
            
            st.markdown(f"""
            <div class="alert alert-{alert_type}">
                {title_html}{message}
            </div>
            """, unsafe_allow_html=True)
            
            logger.debug(f"Alert displayed: {alert_type} - {title}")
            
        except Exception as e:
            log_error_with_context(logger, e, f"displaying alert: {alert_type}")
            st.error(message)
    
    def display_news_item(self, headline: str, sentiment_impact: float = 0, 
                         event_type: str = "", source: str = ""):
        """Display a news item with professional styling"""
        try:
            # Determine styling based on sentiment impact
            if sentiment_impact > 0.1:
                item_class = "positive"
                impact_icon = "📈"
            elif sentiment_impact < -0.1:
                item_class = "negative"
                impact_icon = "📉"
            else:
                item_class = "neutral"
                impact_icon = "➡️"
            
            # Format event type badge if provided
            event_badge = ""
            if event_type:
                badge_class = f"event-{item_class}"
                event_badge = f'<span class="event-badge {badge_class}">{event_type.replace("_", " ").title()}</span>'
            
            # Format source if provided
            source_text = f" - {source}" if source else ""
            
            st.markdown(f"""
            <div class="news-item {item_class}">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
                    {event_badge}
                    <span style="font-size: 1.2rem;">{impact_icon}</span>
                </div>
                <div style="font-weight: 600; margin-bottom: 0.5rem;">{headline}</div>
                <div style="color: #6c757d; font-size: 0.875rem;">
                    Sentiment Impact: <span class="status-{item_class}">{sentiment_impact:+.3f}</span>{source_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            logger.debug(f"News item displayed: {headline[:50]}...")
            
        except Exception as e:
            log_error_with_context(logger, e, "displaying news item", headline=headline[:50])
    
    def create_bank_card_header(self, symbol: str, bank_name: str):
        """Create professional bank card header"""
        try:
            st.markdown(f"""
            <div class="bank-card">
                <div class="bank-card-header">
                    <h3>🏦 {bank_name} ({symbol})</h3>
                </div>
                <div class="bank-card-body">
            """, unsafe_allow_html=True)
            
            logger.debug(f"Bank card header created: {bank_name}")
            
        except Exception as e:
            log_error_with_context(logger, e, f"creating bank card header: {bank_name}")
    
    def close_bank_card(self):
        """Close bank card HTML"""
        try:
            st.markdown("</div></div>", unsafe_allow_html=True)
            logger.debug("Bank card closed")
        except Exception as e:
            log_error_with_context(logger, e, "closing bank card")
