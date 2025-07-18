#!/usr/bin/env python3
"""
Position Risk Dashboard Integration Example
Shows how to integrate the Position Risk Assessor with the professional dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append('src')

def create_position_risk_section():
    """Create position risk assessment section for the dashboard"""
    
    st.markdown("---")
    st.header("🎯 Position Risk Assessment")
    st.markdown("**ML-powered recovery prediction for your positions**")
    
    # Input section
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        symbol = st.selectbox(
            "Select Symbol",
            ['CBA.AX', 'WBC.AX', 'ANZ.AX', 'NAB.AX', 'WES.AX', 'CSL.AX', 'MQG.AX'],
            index=0
        )
    
    with col2:
        entry_price = st.number_input(
            "Entry Price ($)",
            min_value=0.01,
            value=100.00,
            step=0.01,
            format="%.2f"
        )
    
    with col3:
        current_price = st.number_input(
            "Current Price ($)",
            min_value=0.01,
            value=99.00,
            step=0.01,
            format="%.2f"
        )
    
    with col4:
        position_type = st.selectbox(
            "Position Type",
            ['long', 'short'],
            index=0
        )
    
    # Assessment button
    if st.button("🔍 Assess Position Risk", type="primary"):
        
        # Import and run assessment
        try:
            from position_risk_assessor import PositionRiskAssessor
            
            assessor = PositionRiskAssessor()
            result = assessor.assess_position_risk(
                symbol=symbol,
                entry_price=entry_price,
                current_price=current_price,
                position_type=position_type
            )
            
            if 'error' in result:
                st.error(f"Assessment error: {result['error']}")
                return
            
            # Display results
            display_risk_assessment(result)
            
        except ImportError:
            st.error("Position Risk Assessor module not found. Please ensure it's properly installed.")
        except Exception as e:
            st.error(f"Error running assessment: {str(e)}")

def display_risk_assessment(result):
    """Display the risk assessment results"""
    
    # Header metrics
    col1, col2, col3, col4 = st.columns(4)
    
    current_return = result.get('current_return_pct', 0)
    risk_metrics = result.get('risk_metrics', {})
    
    with col1:
        st.metric(
            "Current Return",
            f"{current_return}%",
            delta=None,
            delta_color="inverse"
        )
    
    with col2:
        risk_score = risk_metrics.get('risk_score', 0)
        st.metric(
            "Risk Score",
            f"{risk_score}/10",
            delta=None,
            delta_color="inverse" if risk_score > 5 else "normal"
        )
    
    with col3:
        recovery_prob = risk_metrics.get('breakeven_probability_30d', 0)
        st.metric(
            "Recovery Probability",
            f"{recovery_prob:.1%}",
            delta=None,
            delta_color="normal" if recovery_prob > 0.5 else "inverse"
        )
    
    with col4:
        recovery_days = risk_metrics.get('estimated_recovery_days', 0)
        st.metric(
            "Est. Recovery Time",
            f"{recovery_days} days",
            delta=None
        )
    
    # Main results sections
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Recovery probability chart
        create_recovery_chart(result.get('recovery_predictions', {}))
    
    with col2:
        # Recommendations
        display_recommendations(result.get('recommendations', {}))
    
    # Detailed analysis
    st.subheader("📊 Detailed Analysis")
    
    # Recovery scenarios table
    recovery_data = format_recovery_data(result.get('recovery_predictions', {}))
    if recovery_data:
        st.dataframe(recovery_data, use_container_width=True)
    
    # Risk metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚠️ Risk Metrics")
        risk_metrics = result.get('risk_metrics', {})
        
        st.write(f"**Current Loss:** {risk_metrics.get('current_loss_pct', 0)}%")
        st.write(f"**Expected Recovery:** {risk_metrics.get('expected_recovery_probability', 0):.1%}")
        st.write(f"**Recommended Reduction:** {risk_metrics.get('recommended_position_reduction', 0)*100:.0f}%")
        st.write(f"**Max Acceptable Loss:** {risk_metrics.get('max_acceptable_loss', 0)}%")
    
    with col2:
        st.subheader("📈 Market Context")
        market_context = result.get('market_context', {})
        
        st.write(f"**Volatility (20d):** {market_context.get('volatility_20d', 0):.1%}")
        st.write(f"**RSI:** {market_context.get('rsi', 50):.1f}")
        st.write(f"**Support Distance:** {market_context.get('support_distance', 0)*100:.1f}%")
        st.write(f"**Volume Ratio:** {market_context.get('volume_ratio', 1):.2f}x")

def create_recovery_chart(recovery_predictions):
    """Create recovery probability visualization"""
    
    st.subheader("🔮 Recovery Probability Timeline")
    
    if not recovery_predictions:
        st.warning("No recovery predictions available")
        return
    
    # Extract data for chart
    timeframes = []
    probabilities = []
    scenarios = []
    
    for scenario_name, scenario_data in recovery_predictions.items():
        if scenario_name == 'max_adverse_excursion':
            continue
            
        for timeframe, prediction in scenario_data.items():
            timeframes.append(prediction.get('days', 0))
            probabilities.append(prediction.get('probability', 0) * 100)
            scenarios.append(f"{scenario_name.replace('_', ' ').title()}")
    
    if not timeframes:
        st.warning("No recovery data to display")
        return
    
    # Create chart
    fig = go.Figure()
    
    # Group by scenario
    scenario_colors = {
        'Quick Recovery': '#ff6b6b',
        'Full Recovery': '#4ecdc4', 
        'Profit Recovery': '#45b7d1'
    }
    
    for scenario in set(scenarios):
        scenario_timeframes = [t for i, t in enumerate(timeframes) if scenarios[i] == scenario]
        scenario_probs = [p for i, p in enumerate(probabilities) if scenarios[i] == scenario]
        
        fig.add_trace(go.Scatter(
            x=scenario_timeframes,
            y=scenario_probs,
            mode='lines+markers',
            name=scenario,
            line=dict(color=scenario_colors.get(scenario, '#96ceb4'), width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title=dict(text="Recovery Probability vs Time", font=dict(size=16)),
        xaxis=dict(title="Days", gridcolor="#f1f1f1"),
        yaxis=dict(title="Probability (%)", gridcolor="#f1f1f1", range=[0, 100]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_recommendations(recommendations):
    """Display actionable recommendations"""
    
    st.subheader("💡 Recommendations")
    
    if not recommendations:
        st.warning("No recommendations available")
        return
    
    # Primary action
    primary_action = recommendations.get('primary_action', 'MONITOR')
    confidence = recommendations.get('confidence_level', 'medium')
    
    # Color code the action
    action_colors = {
        'HOLD': '🟢',
        'MONITOR': '🟡',
        'REDUCE_POSITION': '🟠',
        'CONSIDER_EXIT': '🔴',
        'EXIT_RECOMMENDED': '🚨'
    }
    
    icon = action_colors.get(primary_action, '⚪')
    
    st.markdown(f"""
    **Primary Action:** {icon} **{primary_action.replace('_', ' ')}**
    
    **Confidence:** {confidence.upper()}
    """)
    
    # Position management
    position_mgmt = recommendations.get('position_management', [])
    if position_mgmt:
        st.markdown("**Position Management:**")
        for rec in position_mgmt:
            st.markdown(f"• {rec}")
    
    # Exit strategy
    exit_strategy = recommendations.get('exit_strategy', [])
    if exit_strategy:
        st.markdown("**Exit Strategy:**")
        for rec in exit_strategy:
            st.markdown(f"• {rec}")
    
    # Monitoring
    monitoring = recommendations.get('monitoring', [])
    if monitoring:
        st.markdown("**Monitoring:**")
        for rec in monitoring:
            st.markdown(f"• {rec}")

def format_recovery_data(recovery_predictions):
    """Format recovery predictions for table display"""
    
    if not recovery_predictions:
        return None
    
    data = []
    
    for scenario_name, scenario_data in recovery_predictions.items():
        if scenario_name == 'max_adverse_excursion':
            continue
            
        for timeframe, prediction in scenario_data.items():
            data.append({
                'Scenario': scenario_name.replace('_', ' ').title(),
                'Timeframe': f"{prediction.get('days', 0)} days",
                'Probability': f"{prediction.get('probability', 0):.1%}",
                'Confidence': prediction.get('confidence', 'medium').title()
            })
    
    return data

# Example usage
if __name__ == "__main__":
    st.set_page_config(
        page_title="Position Risk Assessment",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Position Risk Assessment Demo")
    st.markdown("**ML-powered position recovery prediction**")
    
    create_position_risk_section()
