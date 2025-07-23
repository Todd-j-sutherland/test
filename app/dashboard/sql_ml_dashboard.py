#!/usr/bin/env python3
"""
SQL-Powered ML Trading Dashboard - Uses SQL data manager for real-time data
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add app root to path
app_root = Path(__file__).parent.parent
sys.path.append(str(app_root))

from app.dashboard.utils.sql_data_manager import DashboardDataManagerSQL, SQLDashboardManager

class SQLMLTradingDashboard:
    def __init__(self):
        st.set_page_config(
            page_title="SQL ML Trading Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize SQL data manager
        try:
            self.sql_manager = SQLDashboardManager()
            self.data_manager = DashboardDataManagerSQL()
            self.connection_status = "✅ Connected to SQL Database"
        except Exception as e:
            self.connection_status = f"❌ Database Error: {e}"
            self.sql_manager = None
            self.data_manager = None
        
        st.title("📊 SQL-Powered ML Trading Dashboard")
        st.markdown(f"**Status:** {self.connection_status}")
        st.markdown("---")
    
    def show_confidence_analysis(self):
        """Show confidence analysis using SQL data"""
        st.subheader("🎯 Confidence Analysis")
        
        if not self.sql_manager:
            st.error("Database connection required for confidence analysis")
            return
        
        # Get confidence distribution
        conf_analysis = self.sql_manager.get_confidence_distribution(days_back=7)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Unique Confidence Values", 
                conf_analysis.get("unique_confidence_values", 0),
                help="More unique values = better model diversity"
            )
        
        with col2:
            quality = conf_analysis.get("quality_score", "UNKNOWN")
            st.metric("Data Quality", quality)
        
        with col3:
            total = conf_analysis.get("total_records", 0)
            st.metric("Total Records", total)
        
        # Confidence distribution chart
        if "distribution" in conf_analysis:
            distribution = conf_analysis["distribution"]
            
            # Create DataFrame for plotting
            df = pd.DataFrame([
                {"Confidence": f"{conf:.1f}%", "Count": count, "Confidence_Val": conf}
                for conf, count in distribution.items()
            ]).sort_values("Confidence_Val")
            
            fig = px.bar(
                df, 
                x="Confidence", 
                y="Count",
                title="Confidence Distribution",
                color="Count",
                color_continuous_scale="viridis"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def show_recent_predictions(self):
        """Show recent predictions from SQL database"""
        st.subheader("📈 Recent Predictions")
        
        if not self.sql_manager:
            st.error("Database connection required")
            return
        
        # Get recent predictions
        predictions = self.sql_manager.get_today_predictions()
        
        if not predictions:
            st.warning("No predictions found for today")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(predictions)
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Today's Predictions", len(predictions))
        
        with col2:
            buy_signals = len([p for p in predictions if p["signal"] == "BUY"])
            st.metric("BUY Signals", buy_signals)
        
        with col3:
            sell_signals = len([p for p in predictions if p["signal"] == "SELL"])
            st.metric("SELL Signals", sell_signals)
        
        with col4:
            avg_confidence = df["confidence"].str.rstrip("%").astype(float).mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        # Display recent predictions table
        st.subheader("Latest Predictions")
        
        # Format the dataframe for display
        display_df = df[["time", "symbol", "signal", "confidence", "sentiment", "news_count"]].copy()
        display_df.columns = ["Time", "Symbol", "Signal", "Confidence", "Sentiment", "News Count"]
        
        st.dataframe(display_df, use_container_width=True)
    
    def show_database_stats(self):
        """Show database statistics"""
        st.subheader("🗄️ Database Statistics")
        
        if not self.sql_manager:
            st.error("Database connection required")
            return
        
        stats = self.sql_manager.get_database_stats()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", stats.get("total_records", 0))
        
        with col2:
            st.metric("Today's Records", stats.get("today_records", 0))
        
        with col3:
            st.metric("Recent Activity (24h)", stats.get("recent_activity_24h", 0))
        
        # Show symbol distribution
        symbol_dist = stats.get("symbol_distribution", {})
        if symbol_dist:
            st.subheader("Symbol Distribution")
            
            df_symbols = pd.DataFrame([
                {"Symbol": symbol, "Count": count}
                for symbol, count in symbol_dist.items()
            ])
            
            fig = px.pie(
                df_symbols, 
                values="Count", 
                names="Symbol",
                title="Records by Symbol"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def show_timeline(self):
        """Show prediction timeline"""
        st.subheader("📅 Prediction Timeline")
        
        if not self.sql_manager:
            st.error("Database connection required")
            return
        
        # Get timeline data
        timeline = self.sql_manager.get_prediction_timeline(hours_back=24)
        
        if not timeline:
            st.warning("No timeline data available")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(timeline)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Create timeline chart
        fig = go.Figure()
        
        # Add confidence timeline
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["confidence"],
            mode="lines+markers",
            name="Confidence",
            line=dict(color="blue"),
            hovertemplate="<b>%{x}</b><br>Confidence: %{y:.1%}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Confidence Over Time (Last 24 Hours)",
            xaxis_title="Time",
            yaxis_title="Confidence",
            yaxis=dict(tickformat=".0%")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Main dashboard runner"""
        
        # Sidebar
        st.sidebar.header("Dashboard Controls")
        
        page = st.sidebar.selectbox(
            "Select View",
            ["Overview", "Confidence Analysis", "Recent Predictions", "Database Stats", "Timeline"]
        )
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.experimental_rerun()
        
        # Main content based on selection
        if page == "Overview":
            st.header("📊 Dashboard Overview")
            
            # Show key metrics
            if self.sql_manager:
                stats = self.sql_manager.get_database_stats()
                conf_analysis = self.sql_manager.get_confidence_distribution()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Records", stats.get("total_records", 0))
                
                with col2:
                    st.metric("Today Records", stats.get("today_records", 0))
                
                with col3:
                    st.metric("Unique Confidences", conf_analysis.get("unique_confidence_values", 0))
                
                with col4:
                    quality = conf_analysis.get("quality_score", "UNKNOWN")
                    st.metric("Data Quality", quality)
            
            # Show recent predictions preview
            self.show_recent_predictions()
            
        elif page == "Confidence Analysis":
            self.show_confidence_analysis()
            
        elif page == "Recent Predictions":
            self.show_recent_predictions()
            
        elif page == "Database Stats":
            self.show_database_stats()
            
        elif page == "Timeline":
            self.show_timeline()


if __name__ == "__main__":
    dashboard = SQLMLTradingDashboard()
    dashboard.run()
