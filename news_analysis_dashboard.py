#!/usr/bin/env python3
"""
News Analysis Dashboard with Technical Analysis Integration
Interactive web dashboard displaying news sentiment analysis and technical indicators for Australian banks
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

# Import technical analysis
from src.technical_analysis import TechnicalAnalyzer, get_market_data

# Add ML imports
import sqlite3
import joblib
import numpy as np

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
    """Dashboard for displaying news analysis and technical analysis results"""
    
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
            
            # Get technical analysis for additional components
            tech_analysis = self.get_technical_analysis(symbol)
            
            # Calculate technical components
            volume_score = 0
            momentum_score = 0
            ml_trading_score = 0
            
            if tech_analysis and 'momentum' in tech_analysis:
                momentum_data = tech_analysis['momentum']
                
                # Convert momentum score to 0-100 scale
                momentum_score = max(0, min(100, (momentum_data['score'] + 100) / 2))
                
                # Volume score based on volume momentum
                if momentum_data['volume_momentum'] == 'high':
                    volume_score = 75
                elif momentum_data['volume_momentum'] == 'normal':
                    volume_score = 50
                else:
                    volume_score = 25
                
                # ML Trading score based on overall signal
                if 'overall_signal' in tech_analysis:
                    ml_trading_score = max(0, min(100, (tech_analysis['overall_signal'] + 100) / 2))
            
            comp_df = pd.DataFrame([
                {'Component': 'News', 'Score': components.get('news', 0), 'Weight': '35%'},
                {'Component': 'Reddit', 'Score': components.get('reddit', 0), 'Weight': '15%'},
                {'Component': 'Events', 'Score': components.get('events', 0), 'Weight': '20%'},
                {'Component': 'Volume', 'Score': round(volume_score, 1), 'Weight': '10%'},
                {'Component': 'Momentum', 'Score': round(momentum_score, 1), 'Weight': '10%'},
                {'Component': 'ML Trading', 'Score': round(ml_trading_score, 1), 'Weight': '10%'}
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
            st.markdown("#### 🔴 Reddit Sentiment Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Reddit Score", f"{reddit_data.get('overall_sentiment', 0):.3f}")
            with col2:
                st.metric("Reddit Posts", reddit_data.get('post_count', 0))
        
        # Add Historical Trends Charts
        st.markdown("#### 📈 Historical Trends")
        
        chart_tabs = st.tabs(["📊 Historical Data", "🔗 Correlation Analysis"])
        
        with chart_tabs[0]:
            historical_chart = self.create_historical_trends_chart(symbol, {symbol: data})
            st.plotly_chart(historical_chart, use_container_width=True)
            
            st.markdown("""
            **Chart Legend:**
            - **Blue Line**: Sentiment Score (-1 to +1)
            - **Orange Line**: Confidence Level (0 to 1) 
            - **Green Line**: Stock Price (Normalized)
            - **Red/Green Bars**: Daily Momentum (% price change)
            """)
        
        with chart_tabs[1]:
            correlation_chart = self.create_correlation_chart(symbol, {symbol: data})
            st.plotly_chart(correlation_chart, use_container_width=True)
            
            st.markdown("""
            **Correlation Analysis:**
            - Each point represents one analysis day
            - Point size indicates confidence level
            - Color scale shows confidence (red=low, green=high)
            - Trend line shows overall sentiment-price relationship
            """)

        # Technical Analysis Section
        st.markdown("#### 📈 Technical Analysis & Momentum")
        
        tech_analysis = self.get_technical_analysis(symbol)
        if tech_analysis and tech_analysis.get('current_price', 0) > 0:
            # Technical metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = tech_analysis.get('current_price', 0)
                st.metric("Current Price", f"${current_price:.2f}")
            
            with col2:
                momentum_score = tech_analysis.get('momentum', {}).get('score', 0)
                delta_color = "normal" if -5 <= momentum_score <= 5 else "inverse" if momentum_score < 0 else "normal"
                st.metric("Momentum Score", f"{momentum_score:.1f}", delta=momentum_score, delta_color=delta_color)
            
            with col3:
                recommendation = tech_analysis.get('recommendation', 'HOLD')
                st.metric("Technical Signal", recommendation)
            
            with col4:
                rsi = tech_analysis.get('indicators', {}).get('rsi', 50)
                rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Normal"
                st.metric("RSI", f"{rsi:.1f}", rsi_status)
            
            # Detailed technical indicators
            st.markdown("**📊 Technical Indicators:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                indicators = tech_analysis.get('indicators', {})
                macd = indicators.get('macd', {})
                
                indicator_data = []
                indicator_data.append({"Indicator": "RSI (14)", "Value": f"{indicators.get('rsi', 0):.2f}", "Signal": "Overbought" if indicators.get('rsi', 50) > 70 else "Oversold" if indicators.get('rsi', 50) < 30 else "Neutral"})
                indicator_data.append({"Indicator": "MACD", "Value": f"{macd.get('line', 0):.4f}", "Signal": "Bullish" if macd.get('histogram', 0) > 0 else "Bearish"})
                
                if 'sma' in indicators:
                    sma_20 = indicators['sma'].get('sma_20', 0)
                    sma_50 = indicators['sma'].get('sma_50', 0)
                    price_vs_sma20 = "Above" if current_price > sma_20 else "Below"
                    price_vs_sma50 = "Above" if current_price > sma_50 else "Below"
                    
                    indicator_data.append({"Indicator": "SMA 20", "Value": f"${sma_20:.2f}", "Signal": f"Price {price_vs_sma20}"})
                    indicator_data.append({"Indicator": "SMA 50", "Value": f"${sma_50:.2f}", "Signal": f"Price {price_vs_sma50}"})
                
                st.dataframe(pd.DataFrame(indicator_data), use_container_width=True)
            
            with col2:
                momentum = tech_analysis.get('momentum', {})
                trend = tech_analysis.get('trend', {})
                
                momentum_data = []
                momentum_data.append({"Metric": "1-Day Change", "Value": f"{momentum.get('price_change_1d', 0):.2f}%"})
                momentum_data.append({"Metric": "5-Day Change", "Value": f"{momentum.get('price_change_5d', 0):.2f}%"})
                momentum_data.append({"Metric": "20-Day Change", "Value": f"{momentum.get('price_change_20d', 0):.2f}%"})
                momentum_data.append({"Metric": "Trend Direction", "Value": trend.get('direction', 'Unknown').replace('_', ' ').title()})
                momentum_data.append({"Metric": "Volume Momentum", "Value": momentum.get('volume_momentum', 'Normal').title()})
                
                st.dataframe(pd.DataFrame(momentum_data), use_container_width=True)
        else:
            st.info("Technical analysis data not available")
        
        # Add ML Prediction Section
        st.markdown("#### 🤖 Machine Learning Prediction")
        
        ml_prediction = latest.get('ml_prediction', {})
        
        if ml_prediction and ml_prediction.get('prediction') != 'UNKNOWN':
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                prediction = ml_prediction['prediction']
                if prediction == 'PROFITABLE':
                    st.metric("ML Signal", "📈 PROFITABLE", delta="BUY")
                else:
                    st.metric("ML Signal", "📉 UNPROFITABLE", delta="AVOID")
            
            with col2:
                probability = ml_prediction.get('probability', 0)
                st.metric("Probability", f"{probability:.1%}")
            
            with col3:
                ml_confidence = ml_prediction.get('confidence', 0)
                st.metric("ML Confidence", f"{ml_confidence:.1%}")
            
            with col4:
                threshold = ml_prediction.get('threshold', 0.5)
                st.metric("Decision Threshold", f"{threshold:.2f}")
            
            # Show ML model performance if available
            if st.button("📊 Show ML Model Performance", key=f"ml_perf_{symbol}"):
                self.display_ml_performance(symbol)
        else:
            st.info("ML predictions not available. Train models using: `python scripts/retrain_ml_models.py`")
        
        # Continue with existing sections...
    
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
        """Display ML model performance metrics"""
        try:
            # Load ML pipeline to access database
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
                st.markdown("##### Recent Model Performance")
                
                # Display metrics
                for idx, row in performance_df.iterrows():
                    with st.expander(f"Model: {row['model_type']} - {row['model_version']}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Validation Score", f"{row['validation_score']:.3f}")
                        with col2:
                            st.metric("Precision", f"{row['precision_score']:.3f}")
                        with col3:
                            st.metric("Recall", f"{row['recall_score']:.3f}")
                        
                        # Show feature importance if available
                        if row['feature_importance']:
                            try:
                                importance = json.loads(row['feature_importance'])
                                importance_df = pd.DataFrame(list(importance.items()), 
                                                           columns=['Feature', 'Importance'])
                                importance_df = importance_df.sort_values('Importance', ascending=False)
                                
                                st.markdown("**Feature Importance:**")
                                st.dataframe(importance_df, use_container_width=True)
                            except:
                                st.info("Feature importance data not available")
            
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
                st.markdown("##### Recent Trading Performance (30 days)")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total = accuracy_df['total_predictions'].iloc[0]
                    st.metric("Total Trades", total)
                
                with col2:
                    profitable = accuracy_df['profitable_trades'].iloc[0]
                    win_rate = profitable / total if total > 0 else 0
                    st.metric("Win Rate", f"{win_rate:.1%}")
                
                with col3:
                    avg_return = accuracy_df['avg_return'].iloc[0]
                    st.metric("Avg Return", f"{avg_return:.2f}%")
            
            conn.close()
            
        except Exception as e:
            st.error(f"Error loading ML performance: {e}")
    
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
            # Clear technical analysis cache to get fresh data
            self.technical_data = {}
            st.experimental_rerun()
        
        # Technical analysis refresh
        if st.sidebar.button("📈 Refresh Technical Analysis"):
            with st.spinner("Updating technical analysis..."):
                for symbol in self.bank_symbols:
                    self.get_technical_analysis(symbol, force_refresh=True)
            st.success("Technical analysis updated!")
        
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
        
        # Technical Analysis Section
        st.markdown("## 📈 Technical Analysis & Momentum")
        
        # Load technical analysis data
        with st.spinner("Loading technical analysis..."):
            for symbol in self.bank_symbols:
                self.get_technical_analysis(symbol)
        
        tab1, tab2, tab3 = st.tabs(["🎯 Momentum Analysis", "📊 Technical Signals", "🔄 Combined Analysis"])
        
        with tab1:
            momentum_chart = self.create_momentum_chart(all_data)
            st.plotly_chart(momentum_chart, use_container_width=True)
            
            st.markdown("### 📋 Momentum Summary")
            momentum_data = []
            for symbol in self.bank_symbols:
                tech_analysis = self.get_technical_analysis(symbol)
                if tech_analysis and 'momentum' in tech_analysis:
                    momentum = tech_analysis['momentum']
                    momentum_data.append({
                        'Bank': self.bank_names.get(symbol, symbol),
                        'Symbol': symbol,
                        'Momentum Score': f"{momentum['score']:.1f}",
                        'Strength': momentum['strength'].replace('_', ' ').title(),
                        '1D Change %': f"{momentum['price_change_1d']:.2f}%",
                        '5D Change %': f"{momentum['price_change_5d']:.2f}%",
                        'Volume Momentum': momentum['volume_momentum'].title()
                    })
            
            if momentum_data:
                st.dataframe(pd.DataFrame(momentum_data), use_container_width=True)
        
        with tab2:
            signals_chart = self.create_technical_signals_chart(all_data)
            st.plotly_chart(signals_chart, use_container_width=True)
            
            st.markdown("### 📊 Technical Indicators Summary")
            tech_data = []
            for symbol in self.bank_symbols:
                tech_analysis = self.get_technical_analysis(symbol)
                if tech_analysis:
                    indicators = tech_analysis.get('indicators', {})
                    tech_data.append({
                        'Bank': self.bank_names.get(symbol, symbol),
                        'Symbol': symbol,
                        'Price': f"${tech_analysis.get('current_price', 0):.2f}",
                        'RSI': f"{indicators.get('rsi', 0):.1f}",
                        'MACD Signal': 'Bullish' if indicators.get('macd', {}).get('histogram', 0) > 0 else 'Bearish',
                        'Trend': tech_analysis.get('trend', {}).get('direction', 'Unknown').replace('_', ' ').title(),
                        'Recommendation': tech_analysis.get('recommendation', 'HOLD')
                    })
            
            if tech_data:
                st.dataframe(pd.DataFrame(tech_data), use_container_width=True)
        
        with tab3:
            combined_chart = self.create_combined_analysis_chart(all_data)
            st.plotly_chart(combined_chart, use_container_width=True)
            
            st.markdown("### 🎯 Trading Recommendations")
            st.info("This combines news sentiment analysis with technical momentum to provide comprehensive trading signals.")
            
            recommendations = []
            for symbol in self.bank_symbols:
                sentiment_data = self.get_latest_analysis(all_data.get(symbol, []))
                tech_analysis = self.get_technical_analysis(symbol)
                
                if sentiment_data or tech_analysis:
                    # Use 'overall_sentiment' instead of 'sentiment_score' to match the data structure
                    sentiment_score = sentiment_data.get('overall_sentiment', 0) if sentiment_data else 0
                    tech_recommendation = tech_analysis.get('recommendation', 'HOLD') if tech_analysis else 'HOLD'
                    momentum_score = tech_analysis.get('momentum', {}).get('score', 0) if tech_analysis else 0
                    
                    # Simple combined recommendation logic
                    if sentiment_score > 0.2 and tech_recommendation in ['BUY', 'STRONG_BUY'] and momentum_score > 10:
                        combined_rec = '🟢 STRONG BUY'
                    elif sentiment_score > 0.1 and tech_recommendation in ['BUY'] and momentum_score > 0:
                        combined_rec = '🟢 BUY'
                    elif sentiment_score < -0.2 and tech_recommendation in ['SELL', 'STRONG_SELL'] and momentum_score < -10:
                        combined_rec = '🔴 STRONG SELL'
                    elif sentiment_score < -0.1 and tech_recommendation in ['SELL'] and momentum_score < 0:
                        combined_rec = '🔴 SELL'
                    else:
                        combined_rec = '🟡 HOLD'
                    
                    recommendations.append({
                        'Bank': self.bank_names.get(symbol, symbol),
                        'Symbol': symbol,
                        'News Sentiment': f"{sentiment_score:.3f}",
                        'Technical Rec': tech_recommendation,
                        'Momentum': f"{momentum_score:.1f}",
                        'Combined Recommendation': combined_rec
                    })
            
            if recommendations:
                st.dataframe(pd.DataFrame(recommendations), use_container_width=True)
        
        # Historical Trends Overview
        st.markdown("## 📈 Historical Trends Overview")
        
        # Create tabs for different trend views
        trend_tabs = st.tabs(["📊 Multi-Bank Trends", "🎯 Top Movers", "📋 Trend Summary"])
        
        with trend_tabs[0]:
            st.markdown("### 📊 Recent Performance Trends")
            
            # Multi-bank comparison chart
            multi_bank_fig = go.Figure()
            
            for symbol in selected_banks:
                if symbol in all_data and all_data[symbol]:
                    # Get last 10 data points for trending
                    recent_data = all_data[symbol][-10:]
                    dates = []
                    sentiments = []
                    
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
                        except Exception:
                            continue
                    
                    if dates and sentiments:
                        multi_bank_fig.add_trace(go.Scatter(
                            x=dates,
                            y=sentiments,
                            mode='lines+markers',
                            name=self.bank_names.get(symbol, symbol),
                            hovertemplate='<b>%{fullData.name}</b><br>' +
                                         'Date: %{x}<br>' +
                                         'Sentiment: %{y:.3f}<extra></extra>'
                        ))
            
            multi_bank_fig.update_layout(
                title="Recent Sentiment Trends - All Selected Banks",
                xaxis_title="Date",
                yaxis_title="Sentiment Score",
                height=500,
                legend=dict(x=0, y=1),
                hovermode='x unified',
                yaxis=dict(range=[-0.5, 0.5])
            )
            
            st.plotly_chart(multi_bank_fig, use_container_width=True)
        
        with trend_tabs[1]:
            st.markdown("### 🎯 Top Movers (Last 7 Days)")
            
            # Calculate recent performance changes
            movers_data = []
            for symbol in selected_banks:
                if symbol in all_data and len(all_data[symbol]) >= 2:
                    recent_data = all_data[symbol][-7:]  # Last 7 entries
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
                        
                        movers_data.append({
                            'Bank': self.bank_names.get(symbol, symbol),
                            'Symbol': symbol,
                            'Sentiment Change': change,
                            'Price Change (%)': price_change,
                            'Current Sentiment': latest_sentiment,
                            'Trend': '🔥 Hot' if abs(change) > 0.1 else '📈 Rising' if change > 0 else '📉 Falling' if change < 0 else '➡️ Stable'
                        })
            
            if movers_data:
                # Sort by absolute sentiment change
                movers_df = pd.DataFrame(movers_data)
                movers_df['abs_change'] = movers_df['Sentiment Change'].abs()
                movers_df = movers_df.sort_values('abs_change', ascending=False)
                
                # Display top movers
                for idx, row in movers_df.iterrows():
                    with st.expander(f"{row['Trend']} {row['Bank']} ({row['Symbol']})"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Sentiment Change", f"{row['Sentiment Change']:+.3f}")
                        
                        with col2:
                            st.metric("Price Change", f"{row['Price Change (%)']:+.2f}%")
                        
                        with col3:
                            st.metric("Current Sentiment", f"{row['Current Sentiment']:.3f}")
        
        with trend_tabs[2]:
            st.markdown("### 📋 Weekly Trend Summary")
            
            # Calculate latest analyses for this section
            latest_analyses = [self.get_latest_analysis(data) for data in all_data.values() if data]
            
            # Create summary statistics
            if latest_analyses:
                trend_summary = []
                
                for symbol in selected_banks:
                    if symbol in all_data and all_data[symbol]:
                        latest = self.get_latest_analysis(all_data[symbol])
                        
                        # Calculate trend direction
                        if len(all_data[symbol]) >= 3:
                            recent_sentiments = [entry.get('overall_sentiment', 0) for entry in all_data[symbol][-3:]]
                            if len(recent_sentiments) == 3:
                                if recent_sentiments[-1] > recent_sentiments[-2] > recent_sentiments[-3]:
                                    trend_direction = "📈 Strongly Rising"
                                elif recent_sentiments[-1] > recent_sentiments[-3]:
                                    trend_direction = "📈 Rising"
                                elif recent_sentiments[-1] < recent_sentiments[-2] < recent_sentiments[-3]:
                                    trend_direction = "📉 Strongly Falling"
                                elif recent_sentiments[-1] < recent_sentiments[-3]:
                                    trend_direction = "📉 Falling"
                                else:
                                    trend_direction = "➡️ Stable"
                            else:
                                trend_direction = "❓ Insufficient Data"
                        else:
                            trend_direction = "❓ Insufficient Data"
                        
                        trend_summary.append({
                            'Bank': self.bank_names.get(symbol, symbol),
                            'Current Sentiment': f"{latest.get('overall_sentiment', 0):.3f}",
                            'Confidence': f"{latest.get('confidence', 0):.2f}",
                            'Trend Direction': trend_direction,
                            'News Volume': latest.get('news_count', 0),
                            'Last Updated': latest.get('timestamp', 'Unknown')[:10] if latest.get('timestamp') else 'Unknown'
                        })
                
                if trend_summary:
                    st.dataframe(pd.DataFrame(trend_summary), use_container_width=True)

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
    
    def create_momentum_chart(self, all_data: Dict) -> go.Figure:
        """Create momentum analysis chart showing technical momentum for all banks"""
        symbols = []
        momentum_scores = []
        colors = []
        
        for symbol, data in all_data.items():
            tech_analysis = self.get_technical_analysis(symbol)
            if tech_analysis and 'momentum' in tech_analysis:
                symbols.append(self.bank_names.get(symbol, symbol))
                momentum_score = tech_analysis['momentum']['score']
                momentum_scores.append(momentum_score)
                
                # Color based on momentum strength
                if momentum_score > 20:
                    colors.append('#00ff00')  # Strong green
                elif momentum_score > 5:
                    colors.append('#90EE90')  # Light green
                elif momentum_score > -5:
                    colors.append('#FFD700')  # Yellow (neutral)
                elif momentum_score > -20:
                    colors.append('#FFA07A')  # Light red
                else:
                    colors.append('#ff0000')  # Strong red
        
        fig = go.Figure(go.Bar(
            x=symbols,
            y=momentum_scores,
            marker_color=colors,
            text=[f"{s:.1f}" for s in momentum_scores],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>' +
                         'Momentum Score: %{y:.1f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title="Technical Momentum Analysis",
            xaxis_title="Banks",
            yaxis_title="Momentum Score",
            height=400,
            showlegend=False,
            yaxis=dict(range=[-100, 100])
        )
        
        # Add horizontal line at zero
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        return fig
    
    def create_technical_signals_chart(self, all_data: Dict) -> go.Figure:
        """Create technical analysis signals chart"""
        symbols = []
        signals = []
        colors = []
        recommendations = []
        
        for symbol, data in all_data.items():
            tech_analysis = self.get_technical_analysis(symbol)
            if tech_analysis:
                symbols.append(self.bank_names.get(symbol, symbol))
                signal_score = tech_analysis.get('overall_signal', 0)
                signals.append(signal_score)
                recommendation = tech_analysis.get('recommendation', 'HOLD')
                recommendations.append(recommendation)
                
                # Color based on recommendation
                if recommendation in ['STRONG_BUY', 'BUY']:
                    colors.append('#28a745')
                elif recommendation in ['STRONG_SELL', 'SELL']:
                    colors.append('#dc3545')
                else:
                    colors.append('#6c757d')
        
        fig = go.Figure(go.Bar(
            x=symbols,
            y=signals,
            marker_color=colors,
            text=recommendations,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>' +
                         'Signal Score: %{y:.1f}<br>' +
                         'Recommendation: %{text}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title="Technical Analysis Signals",
            xaxis_title="Banks",
            yaxis_title="Signal Score",
            height=400,
            showlegend=False,
            yaxis=dict(range=[-100, 100])
        )
        
        # Add horizontal lines for signal thresholds
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Strong Buy")
        fig.add_hline(y=-30, line_dash="dash", line_color="red", annotation_text="Strong Sell")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        return fig
    
    def create_combined_analysis_chart(self, all_data: Dict) -> go.Figure:
        """Create combined sentiment and technical analysis chart"""
        symbols = []
        sentiment_scores = []
        technical_scores = []
        
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
        
        fig = go.Figure()
        
        # Add sentiment bars
        fig.add_trace(go.Bar(
            name='News Sentiment',
            x=symbols,
            y=sentiment_scores,
            marker_color='rgba(55, 128, 191, 0.7)',
            yaxis='y'
        ))
        
        # Add technical bars
        fig.add_trace(go.Bar(
            name='Technical Analysis',
            x=symbols,
            y=technical_scores,
            marker_color='rgba(219, 64, 82, 0.7)',
            yaxis='y'
        ))
        
        fig.update_layout(
            title="Combined Analysis: News Sentiment vs Technical Signals",
            xaxis_title="Banks",
            yaxis_title="Signal Strength",
            height=500,
            barmode='group',
            yaxis=dict(range=[-100, 100])
        )
        
        # Add horizontal line at zero
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        return fig
    
    def create_historical_trends_chart(self, symbol: str, all_data: Dict) -> go.Figure:
        """Create historical trends line chart showing price, sentiment, momentum, and confidence"""
        # Get historical sentiment data
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
                    # Parse timestamp
                    if 'T' in timestamp:
                        date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        date = datetime.fromisoformat(timestamp)
                    
                    dates.append(date)
                    sentiment_scores.append(entry.get('overall_sentiment', 0))
                    confidence_scores.append(entry.get('confidence', 0))
                    
                    # Get corresponding price data with improved matching
                    price_date = date.date()
                    price_found = False
                    
                    if not price_data.empty:
                        # Try exact date match first
                        exact_matches = price_data.index.date == price_date
                        if exact_matches.any():
                            price = price_data.loc[exact_matches, 'Close'].iloc[0]
                            prices.append(price)
                            price_found = True
                        else:
                            # Try to find closest date within 3 days
                            price_dates = price_data.index.date
                            date_diffs = [abs((pd_date - price_date).days) for pd_date in price_dates]
                            if date_diffs and min(date_diffs) <= 3:
                                closest_idx = date_diffs.index(min(date_diffs))
                                price = price_data.iloc[closest_idx]['Close']
                                prices.append(price)
                                price_found = True
                    
                    if not price_found:
                        # Use last known price or fetch current price as fallback
                        if prices:
                            prices.append(prices[-1])
                        else:
                            # Get realistic current price as fallback
                            fallback_price = self.get_current_price(symbol)
                            prices.append(fallback_price)
                        
                    # Calculate simple momentum (price change %)
                    if len(prices) > 1:
                        momentum = ((prices[-1] - prices[-2]) / prices[-2]) * 100
                    else:
                        momentum = 0
                    momentum_scores.append(momentum)
            except Exception as e:
                logger.warning(f"Error processing entry: {e}")
                continue
        
        # If no sentiment data, use price data only
        if not dates and not price_data.empty:
            price_data_recent = price_data.tail(30)
            dates = price_data_recent.index.tolist()
            prices = price_data_recent['Close'].tolist()
            
            # Calculate momentum from price data
            momentum_scores = []
            for i in range(len(prices)):
                if i == 0:
                    momentum_scores.append(0)
                else:
                    momentum = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
                    momentum_scores.append(momentum)
            
            # Default values for sentiment/confidence when no data
            sentiment_scores = [0] * len(dates)
            confidence_scores = [0.5] * len(dates)
        
        # Create figure with secondary y-axes
        fig = go.Figure()
        
        if dates:
            # Normalize data for better visualization
            if prices:
                price_norm = [(p - min(prices)) / (max(prices) - min(prices)) if max(prices) != min(prices) else 0.5 for p in prices]
            else:
                price_norm = [0.5] * len(dates)
            
            # Add traces
            fig.add_trace(go.Scatter(
                x=dates,
                y=sentiment_scores,
                mode='lines+markers',
                name='Sentiment Score',
                line=dict(color='#1f77b4', width=3),
                hovertemplate='<b>Sentiment Score</b><br>' +
                             'Date: %{x}<br>' +
                             'Score: %{y:.3f}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=confidence_scores,
                mode='lines+markers',
                name='Confidence',
                line=dict(color='#ff7f0e', width=2),
                hovertemplate='<b>Confidence</b><br>' +
                             'Date: %{x}<br>' +
                             'Confidence: %{y:.3f}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=price_norm,
                mode='lines',
                name='Price (Normalized)',
                line=dict(color='#2ca02c', width=2),
                hovertemplate='<b>Price (Normalized)</b><br>' +
                             'Date: %{x}<br>' +
                             'Normalized: %{y:.3f}<br>' +
                             'Actual: $%{customdata:.2f}<extra></extra>',
                customdata=prices
            ))
            
            # Momentum as bar chart overlay
            if momentum_scores:
                momentum_colors = ['red' if m < 0 else 'green' for m in momentum_scores]
                fig.add_trace(go.Bar(
                    x=dates,
                    y=[m/100 for m in momentum_scores],  # Scale momentum to fit
                    name='Momentum (%)',
                    marker_color=momentum_colors,
                    opacity=0.3,
                    hovertemplate='<b>Momentum</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Change: %{customdata:.2f}%<extra></extra>',
                    customdata=momentum_scores
                ))
        
        else:
            # No data available
            fig.add_annotation(
                text="No historical data available",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
        
        fig.update_layout(
            title=f"Historical Trends - {self.bank_names.get(symbol, symbol)}",
            xaxis_title="Date",
            yaxis_title="Score / Normalized Value",
            height=500,
            legend=dict(x=0, y=1),
            hovermode='x unified',
            yaxis=dict(range=[-1.2, 1.2])
        )
        
        return fig

    def create_correlation_chart(self, symbol: str, all_data: Dict) -> go.Figure:
        """Create correlation chart between sentiment and price movement"""
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
        
        # Create scatter plot
        fig = go.Figure()
        
        if sentiment_values and price_changes:
            # Color points by confidence
            fig.add_trace(go.Scatter(
                x=sentiment_values,
                y=price_changes,
                mode='markers',
                marker=dict(
                    size=[c*20 for c in confidence_values],  # Size by confidence
                    color=confidence_values,
                    colorscale='RdYlGn',
                    colorbar=dict(title="Confidence"),
                    line=dict(width=1, color='black')
                ),
                text=[f"Date: {d.strftime('%Y-%m-%d')}<br>Confidence: {c:.2f}" 
                      for d, c in zip(dates, confidence_values)],
                hovertemplate='<b>Sentiment vs Price Movement</b><br>' +
                             'Sentiment: %{x:.3f}<br>' +
                             'Price Change: %{y:.2f}%<br>' +
                             '%{text}<extra></extra>',
                name='Data Points'
            ))
            
            # Add trend line if enough data points
            if len(sentiment_values) > 3:
                z = np.polyfit(sentiment_values, price_changes, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(sentiment_values), max(sentiment_values), 100)
                y_trend = p(x_trend)
                
                fig.add_trace(go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', dash='dash'),
                    hoverinfo='skip'
                ))
        else:
            fig.add_annotation(
                text="Insufficient data for correlation analysis",
                x=0, y=0,
                showarrow=False,
                font=dict(size=16)
            )
        
        fig.update_layout(
            title=f"Sentiment vs Price Movement Correlation - {self.bank_names.get(symbol, symbol)}",
            xaxis_title="Sentiment Score",
            yaxis_title="Price Change (%)",
            height=400,
            showlegend=True
        )
        
        return fig

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol with fallback options"""
        try:
            # Try to get recent price data
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

    # ...existing code...
def main():
    """Run the dashboard"""
    dashboard = NewsAnalysisDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
