"""
Trading Performance Component for Enhanced Dashboard
Displays detailed performance metrics and historical tracking
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
from pathlib import Path

def display_trading_performance_log(ml_tracker=None, days_back: int = 30):
    """Display detailed trading performance log similar to professional dashboard"""
    st.header("📋 Detailed Performance Log")
    
    if not ml_tracker:
        st.warning("⚠️ ML tracker not available - performance data cannot be loaded")
        return
    
    try:
        # Get recent predictions from tracker
        recent_predictions = [
            p for p in ml_tracker.prediction_history 
            if datetime.fromisoformat(p['timestamp']) >= datetime.now() - timedelta(days=days_back)
        ]
        
        if not recent_predictions:
            st.info("📊 No recent trading predictions found")
            return
        
        # Create performance table data
        table_data = []
        for pred in recent_predictions[-50:]:  # Show last 50 predictions
            success_indicator = "⏳"  # Pending
            outcome_text = "Pending"
            
            if pred['status'] == 'completed':
                if ml_tracker._is_successful_prediction(pred):
                    success_indicator = "✅"
                else:
                    success_indicator = "❌"
                
                # Get outcome percentage
                outcome = pred.get('actual_outcome', {})
                price_change = outcome.get('price_change_percent', 0)
                outcome_text = f"{price_change:+.2f}%"
            
            table_data.append({
                'Date': pred['timestamp'][:10],
                'Time': pred['timestamp'][11:16],
                'Symbol': pred['symbol'],
                'Signal': pred['prediction'].get('signal', 'N/A'),
                'Confidence': f"{pred['prediction'].get('confidence', 0):.1%}",
                'Sentiment': f"{pred['prediction'].get('sentiment_score', 0):+.3f}",
                'Outcome': outcome_text,
                'Success': success_indicator,
                'Status': pred['status']
            })
        
        # Display the table
        df_performance = pd.DataFrame(table_data)
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol_filter = st.selectbox(
                "Filter by Symbol", 
                ["All"] + list(df_performance['Symbol'].unique())
            )
        
        with col2:
            signal_filter = st.selectbox(
                "Filter by Signal",
                ["All"] + list(df_performance['Signal'].unique())
            )
        
        with col3:
            status_filter = st.selectbox(
                "Filter by Status",
                ["All", "completed", "pending"]
            )
        
        # Apply filters
        filtered_df = df_performance.copy()
        
        if symbol_filter != "All":
            filtered_df = filtered_df[filtered_df['Symbol'] == symbol_filter]
        
        if signal_filter != "All":
            filtered_df = filtered_df[filtered_df['Signal'] == signal_filter]
        
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df['Status'] == status_filter]
        
        # Display filtered table
        st.dataframe(
            filtered_df.drop('Status', axis=1),  # Hide status column for cleaner view
            use_container_width=True,
            height=400
        )
        
        # Performance summary metrics
        completed_predictions = [p for p in recent_predictions if p['status'] == 'completed']
        
        if completed_predictions:
            st.subheader("📊 Performance Summary")
            
            # Calculate success rate
            successful = sum(1 for p in completed_predictions if ml_tracker._is_successful_prediction(p))
            total = len(completed_predictions)
            success_rate = (successful / total) * 100 if total > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Success Rate", f"{success_rate:.1f}%", f"{successful}/{total}")
            
            with col2:
                avg_confidence = sum(p['prediction'].get('confidence', 0) for p in completed_predictions) / len(completed_predictions)
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            with col3:
                buy_signals = sum(1 for p in completed_predictions if p['prediction'].get('signal') == 'BUY')
                st.metric("Buy Signals", buy_signals)
            
            with col4:
                sell_signals = sum(1 for p in completed_predictions if p['prediction'].get('signal') == 'SELL')
                st.metric("Sell Signals", sell_signals)
            
            # Performance by symbol chart
            symbol_performance = {}
            for pred in completed_predictions:
                symbol = pred['symbol']
                if symbol not in symbol_performance:
                    symbol_performance[symbol] = {'successful': 0, 'total': 0}
                
                symbol_performance[symbol]['total'] += 1
                if ml_tracker._is_successful_prediction(pred):
                    symbol_performance[symbol]['successful'] += 1
            
            # Create performance chart
            symbols = list(symbol_performance.keys())
            success_rates = [
                (symbol_performance[symbol]['successful'] / symbol_performance[symbol]['total']) * 100
                for symbol in symbols
            ]
            
            fig_performance = px.bar(
                x=symbols,
                y=success_rates,
                title="Success Rate by Symbol",
                labels={'x': 'Symbol', 'y': 'Success Rate (%)'},
                color=success_rates,
                color_continuous_scale='RdYlGn'
            )
            
            fig_performance.update_layout(height=400)
            st.plotly_chart(fig_performance, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error displaying performance log: {e}")


def display_ml_learning_metrics(ml_tracker=None):
    """Display ML learning and improvement metrics"""
    st.header("🧠 ML Learning Metrics")
    
    if not ml_tracker:
        st.warning("⚠️ ML tracker not available")
        return
    
    try:
        # Get performance history
        performance_data = ml_tracker.performance_history
        
        if not performance_data:
            st.info("📊 No performance history available")
            return
        
        # Create time series of performance
        dates = [entry['date'] for entry in performance_data]
        success_rates = [
            (entry['successful_trades'] / entry['total_trades']) * 100 
            if entry['total_trades'] > 0 else 0
            for entry in performance_data
        ]
        model_confidences = [entry.get('model_confidence', 0) * 100 for entry in performance_data]
        
        # Performance trend chart
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=dates,
            y=success_rates,
            mode='lines+markers',
            name='Success Rate (%)',
            line=dict(color='green', width=3),
            marker=dict(size=6)
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=dates,
            y=model_confidences,
            mode='lines+markers',
            name='Model Confidence (%)',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
            yaxis='y2'
        ))
        
        fig_trend.update_layout(
            title='ML Performance Over Time',
            xaxis_title='Date',
            yaxis_title='Success Rate (%)',
            yaxis2=dict(
                title='Model Confidence (%)',
                overlaying='y',
                side='right'
            ),
            height=400
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Recent performance metrics
        if performance_data:
            latest = performance_data[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                success_rate = (latest['successful_trades'] / latest['total_trades']) * 100 if latest['total_trades'] > 0 else 0
                st.metric("Latest Success Rate", f"{success_rate:.1f}%")
            
            with col2:
                st.metric("Model Confidence", f"{latest.get('model_confidence', 0):.1%}")
            
            with col3:
                st.metric("Predictions Made", latest.get('predictions_made', 0))
            
            with col4:
                accuracy = latest.get('accuracy_metrics', {}).get('accuracy', 0)
                st.metric("Accuracy", f"{accuracy:.1%}")
        
    except Exception as e:
        st.error(f"❌ Error displaying learning metrics: {e}")


def display_trading_signals_vs_outcomes():
    """Compare trading signals with actual outcomes"""
    st.header("🎯 Signals vs Outcomes Analysis")
    
    # This would analyze how well BUY/SELL signals performed
    # vs actual price movements
    
    st.info("📊 Signal effectiveness analysis would go here")
    # Implementation would load actual trading data and compare
    # signal predictions with real market outcomes


def enhanced_dashboard_performance_section(ml_tracker=None):
    """Complete performance section for enhanced dashboard"""
    
    # Create tabs for different performance views
    tab1, tab2, tab3 = st.tabs(["📋 Performance Log", "🧠 Learning Metrics", "🎯 Signal Analysis"])
    
    with tab1:
        display_trading_performance_log(ml_tracker)
    
    with tab2:
        display_ml_learning_metrics(ml_tracker)
    
    with tab3:
        display_trading_signals_vs_outcomes()
