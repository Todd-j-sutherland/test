"""
Chart generation utilities for the trading analysis dashboard
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Any, Optional
import numpy as np


class ChartGenerator:
    """Generate various charts for the trading analysis dashboard"""
    
    def __init__(self):
        """Initialize the chart generator"""
        self.color_scheme = {
            'primary': '#1e3d59',
            'secondary': '#17a2b8',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'info': '#17a2b8'
        }
    
    def create_sentiment_overview_chart(self, sentiment_data: Dict[str, Any]) -> go.Figure:
        """
        Create a sentiment overview chart
        
        Args:
            sentiment_data: Dictionary containing sentiment analysis data
            
        Returns:
            Plotly figure object
        """
        try:
            if not sentiment_data:
                return self._create_empty_chart("No sentiment data available")
            
            # Extract sentiment values
            symbols = list(sentiment_data.keys())
            sentiments = [sentiment_data[symbol].get('sentiment', 0) for symbol in symbols]
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=symbols,
                    y=sentiments,
                    marker_color=[
                        self.color_scheme['success'] if s > 0.2 else 
                        self.color_scheme['danger'] if s < -0.2 else 
                        self.color_scheme['warning'] 
                        for s in sentiments
                    ],
                    text=[f"{s:.3f}" for s in sentiments],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Sentiment Overview by Symbol",
                xaxis_title="Symbols",
                yaxis_title="Sentiment Score",
                template="plotly_white",
                height=400
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_chart(f"Error creating sentiment chart: {str(e)}")
    
    def create_confidence_distribution_chart(self, confidence_data: Dict[str, Any]) -> go.Figure:
        """
        Create a confidence distribution chart
        
        Args:
            confidence_data: Dictionary containing confidence data
            
        Returns:
            Plotly figure object
        """
        try:
            if not confidence_data:
                return self._create_empty_chart("No confidence data available")
            
            # Extract confidence values
            symbols = list(confidence_data.keys())
            confidences = [confidence_data[symbol].get('confidence', 0) for symbol in symbols]
            
            # Create histogram
            fig = go.Figure(data=[
                go.Histogram(
                    x=confidences,
                    nbinsx=10,
                    marker_color=self.color_scheme['info'],
                    opacity=0.7
                )
            ])
            
            fig.update_layout(
                title="Confidence Score Distribution",
                xaxis_title="Confidence Score",
                yaxis_title="Count",
                template="plotly_white",
                height=400
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_chart(f"Error creating confidence chart: {str(e)}")
    
    def create_historical_trends_chart(self, historical_data: Dict[str, Any]) -> go.Figure:
        """
        Create a historical trends chart
        
        Args:
            historical_data: Dictionary containing historical trend data
            
        Returns:
            Plotly figure object
        """
        try:
            if not historical_data:
                return self._create_empty_chart("No historical data available")
            
            fig = go.Figure()
            
            for symbol, data in historical_data.items():
                if isinstance(data, dict) and 'dates' in data and 'values' in data:
                    fig.add_trace(go.Scatter(
                        x=data['dates'],
                        y=data['values'],
                        mode='lines',
                        name=symbol,
                        line=dict(width=2)
                    ))
            
            fig.update_layout(
                title="Historical Sentiment Trends",
                xaxis_title="Date",
                yaxis_title="Sentiment Score",
                template="plotly_white",
                height=400,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_chart(f"Error creating historical chart: {str(e)}")
    
    def create_correlation_chart(self, correlation_data: Dict[str, Any]) -> go.Figure:
        """
        Create a correlation heatmap chart
        
        Args:
            correlation_data: Dictionary containing correlation data
            
        Returns:
            Plotly figure object
        """
        try:
            if not correlation_data:
                return self._create_empty_chart("No correlation data available")
            
            # Convert correlation data to matrix format
            symbols = list(correlation_data.keys())
            
            # Create correlation matrix
            correlation_matrix = []
            for symbol1 in symbols:
                row = []
                for symbol2 in symbols:
                    if symbol1 == symbol2:
                        row.append(1.0)
                    else:
                        # Use dummy correlation or extract from data
                        corr_value = correlation_data.get(symbol1, {}).get(symbol2, 
                                   np.random.uniform(-0.5, 0.5))  # Dummy data for demo
                        row.append(corr_value)
                correlation_matrix.append(row)
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=symbols,
                y=symbols,
                colorscale='RdBu',
                zmid=0,
                text=[[f"{val:.2f}" for val in row] for row in correlation_matrix],
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title="Symbol Correlation Matrix",
                template="plotly_white",
                height=400,
                width=500
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_chart(f"Error creating correlation chart: {str(e)}")
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """
        Create an empty chart with a message
        
        Args:
            message: Message to display
            
        Returns:
            Empty Plotly figure with message
        """
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            template="plotly_white",
            height=400,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
        )
        return fig
