# trading_dashboard.py
"""
Web-based dashboard for the ASX Bank Trading System
Real-time monitoring and control interface
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.trading_orchestrator import TradingOrchestrator
from src.async_main import ASXBankTradingSystemAsync
from config.settings import Settings

# Page config
st.set_page_config(
    page_title="ASX Bank Trading System Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .status-running {
        color: #28a745;
        font-weight: bold;
    }
    .status-stopped {
        color: #dc3545;
        font-weight: bold;
    }
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_portfolio_chart(portfolio_data: Dict) -> go.Figure:
    """Create portfolio performance chart"""
    daily_pnl = portfolio_data.get('daily_pnl', [])
    
    if not daily_pnl:
        # Create empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No portfolio data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Portfolio Performance",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=400
        )
        return fig
    
    # Extract data for chart
    dates = [pd.to_datetime(d['date']) for d in daily_pnl]
    values = [d['total_value'] for d in daily_pnl]
    
    fig = go.Figure()
    
    # Portfolio value line
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines+markers',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Add baseline (starting value)
    fig.add_hline(y=100000, line_dash="dash", line_color="gray", 
                  annotation_text="Starting Value ($100k)")
    
    fig.update_layout(
        title="Portfolio Performance Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_positions_chart(positions: List[Dict]) -> go.Figure:
    """Create positions overview chart"""
    if not positions:
        fig = go.Figure()
        fig.add_annotation(
            text="No open positions",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(title="Open Positions", height=300)
        return fig
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Position P&L', 'Position Allocation'),
        specs=[[{"type": "bar"}, {"type": "pie"}]]
    )
    
    symbols = [pos['symbol'] for pos in positions]
    pnls = [pos['pnl'] for pos in positions]
    values = [pos['quantity'] * pos['current_price'] for pos in positions]
    
    # P&L bar chart
    colors = ['green' if pnl >= 0 else 'red' for pnl in pnls]
    fig.add_trace(
        go.Bar(x=symbols, y=pnls, name='P&L', marker_color=colors),
        row=1, col=1
    )
    
    # Allocation pie chart
    fig.add_trace(
        go.Pie(labels=symbols, values=values, name='Allocation'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def create_signals_chart(recent_analysis: Dict) -> go.Figure:
    """Create trading signals visualization"""
    if not recent_analysis:
        fig = go.Figure()
        fig.add_annotation(
            text="No recent analysis available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(title="Trading Signals", height=300)
        return fig
    
    symbols = []
    confidences = []
    directions = []
    colors = []
    
    for symbol, analysis in recent_analysis.items():
        if 'prediction' in analysis:
            pred = analysis['prediction']
            symbols.append(symbol)
            confidences.append(pred.get('confidence', 0))
            direction = pred.get('direction', 'neutral')
            directions.append(direction)
            
            # Color based on direction
            if direction == 'bullish':
                colors.append('green')
            elif direction == 'bearish':
                colors.append('red')
            else:
                colors.append('gray')
    
    fig = go.Figure(data=[
        go.Bar(
            x=symbols,
            y=confidences,
            marker_color=colors,
            text=[f"{d.title()}" for d in directions],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Current Trading Signals",
        xaxis_title="Symbol",
        yaxis_title="Confidence",
        height=300,
        yaxis=dict(range=[0, 1])
    )
    
    return fig

async def get_latest_analysis():
    """Get the latest analysis results"""
    try:
        async with ASXBankTradingSystemAsync() as system:
            results = await system.analyze_all_banks_async()
            return results
    except Exception as e:
        st.error(f"Error fetching analysis: {e}")
        return {}

# Main Dashboard
def main():
    # Auto-refresh every 30 seconds
    st_autorefresh(interval=30000, key="dashboard_refresh")
    
    # Header
    st.markdown('<div class="main-header">🏛️ ASX Bank Trading System Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar - Control Panel
    st.sidebar.header("🎛️ Control Panel")
    
    # Trading Mode Selection
    paper_trading = st.sidebar.checkbox("Paper Trading Mode", value=True)
    
    # System Status
    st.sidebar.subheader("System Status")
    if st.session_state.is_running:
        st.sidebar.markdown('<div class="status-running">🟢 RUNNING</div>', 
                           unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="status-stopped">🔴 STOPPED</div>', 
                           unsafe_allow_html=True)
    
    # Control buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("▶️ Start", disabled=st.session_state.is_running):
            if not st.session_state.orchestrator:
                st.session_state.orchestrator = TradingOrchestrator(paper_trading=paper_trading)
            
            # Start trading in background (this is simplified for demo)
            st.session_state.is_running = True
            st.success("Trading system started!")
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop", disabled=not st.session_state.is_running):
            st.session_state.is_running = False
            st.success("Trading system stopped!")
            st.rerun()
    
    # Settings
    st.sidebar.subheader("Settings")
    min_confidence = st.sidebar.slider("Min Confidence", 0.1, 0.9, 0.3, 0.1)
    max_position_size = st.sidebar.slider("Max Position Size", 0.05, 0.25, 0.10, 0.05)
    
    # Main content area
    
    # Get portfolio data (simulated for demo)
    if st.session_state.orchestrator:
        portfolio_data = st.session_state.orchestrator.get_portfolio_summary()
    else:
        portfolio_data = {
            'account_balance': 100000.0,
            'total_value': 100000.0,
            'cash_available': 100000.0,
            'total_return': 0.0,
            'num_positions': 0,
            'unrealized_pnl': 0.0,
            'positions': [],
            'recent_trades': [],
            'daily_pnl': []
        }
    
    # Key Metrics Row
    st.subheader("📊 Portfolio Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Portfolio Value",
            value=f"${portfolio_data['total_value']:,.2f}",
            delta=f"${portfolio_data['total_value'] - 100000:.2f}"
        )
    
    with col2:
        st.metric(
            label="Cash Available",
            value=f"${portfolio_data['cash_available']:,.2f}",
            delta=f"{portfolio_data['cash_available'] / portfolio_data['total_value'] * 100:.1f}%"
        )
    
    with col3:
        st.metric(
            label="Total Return",
            value=f"{portfolio_data['total_return'] * 100:.2f}%",
            delta=f"${portfolio_data['unrealized_pnl']:.2f}"
        )
    
    with col4:
        st.metric(
            label="Open Positions",
            value=portfolio_data['num_positions'],
            delta=f"Unrealized: ${portfolio_data['unrealized_pnl']:.2f}"
        )
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        # Portfolio performance chart
        portfolio_fig = create_portfolio_chart(portfolio_data)
        st.plotly_chart(portfolio_fig, use_container_width=True)
    
    with col2:
        # Positions chart
        positions_fig = create_positions_chart(portfolio_data['positions'])
        st.plotly_chart(positions_fig, use_container_width=True)
    
    # Current Analysis Section
    st.subheader("🔍 Current Market Analysis")
    
    # Get latest analysis (this would be real-time in production)
    if st.button("🔄 Refresh Analysis"):
        with st.spinner("Fetching latest analysis..."):
            # This would normally be: latest_analysis = await get_latest_analysis()
            # For demo, we'll simulate some data
            latest_analysis = {
                'CBA.AX': {
                    'current_price': 178.00,
                    'prediction': {'direction': 'bullish', 'confidence': 0.35}
                },
                'WBC.AX': {
                    'current_price': 33.63,
                    'prediction': {'direction': 'neutral', 'confidence': 0.15}
                },
                'ANZ.AX': {
                    'current_price': 30.32,
                    'prediction': {'direction': 'bearish', 'confidence': 0.42}
                },
                'NAB.AX': {
                    'current_price': 39.15,
                    'prediction': {'direction': 'neutral', 'confidence': 0.08}
                }
            }
            
            # Display signals chart
            signals_fig = create_signals_chart(latest_analysis)
            st.plotly_chart(signals_fig, use_container_width=True)
    
    # Positions Table
    if portfolio_data['positions']:
        st.subheader("📋 Open Positions")
        positions_df = pd.DataFrame(portfolio_data['positions'])
        
        # Format the dataframe
        positions_df['pnl_pct'] = positions_df['pnl_pct'].apply(lambda x: f"{x:.2f}%")
        positions_df['pnl'] = positions_df['pnl'].apply(lambda x: f"${x:.2f}")
        positions_df['entry_price'] = positions_df['entry_price'].apply(lambda x: f"${x:.2f}")
        positions_df['current_price'] = positions_df['current_price'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(positions_df, use_container_width=True)
    
    # Recent Trades
    if portfolio_data['recent_trades']:
        st.subheader("📈 Recent Trades")
        trades_df = pd.DataFrame(portfolio_data['recent_trades'])
        
        # Format the dataframe
        if 'price' in trades_df.columns:
            trades_df['price'] = trades_df['price'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(trades_df, use_container_width=True)
    
    # System Logs
    st.subheader("📝 System Logs")
    log_placeholder = st.empty()
    
    # Simulate some logs
    sample_logs = [
        "2025-01-07 10:30:00 - INFO - Analysis cycle completed",
        "2025-01-07 10:30:05 - INFO - CBA.AX: Bullish signal detected (35% confidence)",
        "2025-01-07 10:30:10 - INFO - Position opened: BUY 100 CBA.AX @ $178.00",
        "2025-01-07 10:30:15 - INFO - Portfolio updated: Total value $100,500"
    ]
    
    with log_placeholder.container():
        for log in sample_logs[-10:]:  # Show last 10 logs
            st.text(log)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: gray;'>"
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Dashboard v1.0 | ASX Bank Trading System"
        f"</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
