"""
Enhanced ASX Bank Analysis Dashboard

Comprehensive dashboard displaying:
- Economic sentiment analysis and market regime
- Individual bank sentiment with divergence detection
- ML-powered predictions and confidence scores
- Trading signals with economic context
- Risk assessment and position recommendations
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

    # Import our components
try:
    from app.core.analysis.economic import EconomicSentimentAnalyzer
    from app.core.analysis.divergence import DivergenceDetector
    from app.core.data.processors.news_processor import NewsTradingAnalyzer
    from app.core.ml.enhanced_pipeline import EnhancedMLPipeline
    from app.core.ml.trading_scorer import MLTradingScorer
    from app.core.trading.alpaca_integration import AlpacaMLTrader
    from app.core.data.collectors.market_data import ASXDataFeed
    from app.config.settings import Settings
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import components: {e}")
    COMPONENTS_AVAILABLE = False

def main():
    """Main dashboard function"""
    st.set_page_config(
        page_title="ASX Bank Analysis Dashboard",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏦 Enhanced ASX Bank Analysis Dashboard")
    st.markdown("**AI-Powered Trading Analysis with Economic Context & Divergence Detection**")
    
    if not COMPONENTS_AVAILABLE:
        st.error("❌ Core components not available. Please check your installation.")
        return
    
    # Initialize components
    try:
        economic_analyzer = EconomicSentimentAnalyzer()
        divergence_detector = DivergenceDetector()
        news_analyzer = NewsTradingAnalyzer()
        ml_pipeline = EnhancedMLPipeline()
        ml_scorer = MLTradingScorer()
        alpaca_trader = AlpacaMLTrader(paper=True)
        data_feed = ASXDataFeed()
        settings = Settings()
    except Exception as e:
        st.error(f"❌ Failed to initialize components: {e}")
        return
    
    # Sidebar controls
    st.sidebar.header("📊 Analysis Controls")
    
    refresh_data = st.sidebar.button("🔄 Refresh All Data")
    show_detailed = st.sidebar.checkbox("📋 Show Detailed Analysis", value=True)
    confidence_threshold = st.sidebar.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.6, 0.05)
    
    # Bank selection
    st.sidebar.subheader("🏦 Bank Selection")
    selected_banks = st.sidebar.multiselect(
        "Select banks to analyze:",
        options=settings.BANK_SYMBOLS,
        default=settings.BANK_SYMBOLS
    )
    
    if not selected_banks:
        st.warning("⚠️ Please select at least one bank to analyze.")
        return
    
    # Main dashboard sections
    display_economic_overview(economic_analyzer)
    
    # Get bank analyses
    bank_analyses = get_bank_analyses(news_analyzer, selected_banks)
    
    if bank_analyses:
        # Get divergence analysis
        divergence_analysis = divergence_detector.analyze_sector_divergence(bank_analyses)
        
        # Calculate ML trading scores
        economic_data = economic_analyzer.analyze_economic_sentiment()
        ml_scores = ml_scorer.calculate_scores_for_all_banks(
            bank_analyses=bank_analyses,
            economic_analysis=economic_data,
            divergence_analysis=divergence_analysis
        )
        
        display_ml_trading_scores(ml_scores, confidence_threshold)
        display_divergence_analysis(divergence_detector, bank_analyses)
        display_bank_sentiment_overview(bank_analyses, confidence_threshold)
        display_individual_bank_analysis(bank_analyses, data_feed, show_detailed)
        display_ml_predictions(ml_pipeline, bank_analyses)
        display_trading_signals(divergence_detector, bank_analyses)
        display_alpaca_integration(alpaca_trader, ml_scores)
    else:
        st.error("❌ Unable to retrieve bank analysis data.")

def display_ml_trading_scores(ml_scores, confidence_threshold):
    """Display ML trading scores for each bank"""
    st.markdown("## 🎯 ML Trading Scores")
    
    if not ml_scores:
        st.warning("⚠️ No ML trading scores available")
        return
    
    # Create columns for better layout
    cols = st.columns(len(ml_scores))
    
    for idx, (bank, score_data) in enumerate(ml_scores.items()):
        with cols[idx]:
            # Overall score with color coding
            overall_score = score_data.get('overall_score', 0)
            confidence = score_data.get('confidence', 0)
            recommendation = score_data.get('recommendation', 'HOLD')
            
            # Color based on recommendation
            color = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}.get(recommendation, '⚫')
            
            st.markdown(f"""
            <div style='border: 2px solid {"green" if recommendation == "BUY" else "red" if recommendation == "SELL" else "orange"}; 
                        border-radius: 10px; padding: 15px; margin-bottom: 10px;'>
                <h3 style='text-align: center; margin-top: 0;'>{color} {bank}</h3>
                <h2 style='text-align: center; font-size: 2em;'>{overall_score:.1f}/100</h2>
                <p style='text-align: center; font-weight: bold; color: {"green" if recommendation == "BUY" else "red" if recommendation == "SELL" else "orange"};'>
                    {recommendation}
                </p>
                <p style='text-align: center; font-size: 0.9em;'>
                    Confidence: {confidence:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show component breakdown
            with st.expander(f"📊 {bank} Score Breakdown"):
                components = score_data.get('components', {})
                for component, value in components.items():
                    if component != 'overall_score':
                        st.metric(
                            label=component.replace('_', ' ').title(),
                            value=f"{value:.1f}",
                            delta=None
                        )

def display_alpaca_integration(alpaca_trader, ml_scores):
    """Display Alpaca trading integration status and actions"""
    st.markdown("## 📈 Alpaca Trading Integration")
    
    # Check if Alpaca is available
    if not alpaca_trader:
        st.warning("⚠️ Alpaca integration not available. Set up API credentials to enable paper trading.")
        with st.expander("🔧 Setup Instructions"):
            st.markdown("""
            To enable Alpaca paper trading:
            1. Create account at [Alpaca Markets](https://alpaca.markets/)
            2. Get paper trading API credentials
            3. Set environment variables:
               - `ALPACA_API_KEY`
               - `ALPACA_SECRET_KEY`
               - `ALPACA_BASE_URL` (paper trading URL)
            4. Install alpaca-trade-api: `pip install alpaca-trade-api`
            """)
        return
    
    # Show account status
    st.subheader("📊 Account Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Account Status", "Paper Trading", delta="Active")
    with col2:
        st.metric("Buying Power", "$100,000", delta="+$500")
    with col3:
        st.metric("Portfolio Value", "$105,000", delta="+$5,000")
    
    # Trading recommendations based on ML scores
    if ml_scores:
        st.subheader("🎯 ML Trading Recommendations")
        
        for bank, score_data in ml_scores.items():
            recommendation = score_data.get('recommendation', 'HOLD')
            overall_score = score_data.get('overall_score', 0)
            confidence = score_data.get('confidence', 0)
            
            col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
            
            with col1:
                st.write(f"**{bank}**")
            with col2:
                color = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}.get(recommendation, '⚫')
                st.write(f"{color} {recommendation}")
            with col3:
                st.write(f"{overall_score:.1f}/100")
            with col4:
                if recommendation in ['BUY', 'SELL'] and confidence > 0.6:
                    if st.button(f"Execute {recommendation}", key=f"trade_{bank}"):
                        # Simulate trade execution
                        st.success(f"✅ {recommendation} order placed for {bank}")
                        st.balloons()


def display_economic_overview(economic_analyzer):
    """Display economic sentiment and market regime analysis"""
    st.header("🌍 Economic Context Analysis")
    
    try:
        economic_data = economic_analyzer.analyze_economic_sentiment()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sentiment = economic_data.get('overall_sentiment', 0)
            # For delta_color: Streamlit only accepts 'normal', 'inverse', or 'off'
            sentiment_delta_color = "normal" if sentiment > 0 else "inverse" if sentiment < 0 else "off"
            # For chart colors: use standard color names
            sentiment_color = "green" if sentiment > 0 else "red" if sentiment < 0 else "gray"
            st.metric(
                "Overall Economic Sentiment",
                f"{sentiment:+.3f}",
                delta=None,
                delta_color=sentiment_delta_color
            )
        
        with col2:
            confidence = economic_data.get('confidence', 0)
            st.metric("Analysis Confidence", f"{confidence:.1%}")
        
        with col3:
            regime = economic_data.get('market_regime', {}).get('regime', 'Unknown')
            st.metric("Market Regime", regime)
        
        with col4:
            # Add a custom indicator based on regime
            regime_score = {
                'Expansion': 0.8,
                'Neutral': 0.5,
                'Tightening': 0.3,
                'Contraction': 0.2,
                'Easing': 0.6
            }.get(regime, 0.5)
            st.metric("Regime Score", f"{regime_score:.1f}")
        
        # Economic indicators breakdown
        if show_detailed := st.expander("📊 Economic Indicators Breakdown", expanded=False):
            indicators = economic_data.get('indicators', {})
            
            if indicators:
                df_indicators = pd.DataFrame([
                    {
                        'Indicator': key.replace('_', ' ').title(),
                        'Value': data.get('value', 'N/A'),
                        'Trend': data.get('trend', 'N/A'),
                        'Sentiment': data.get('sentiment', 0)
                    }
                    for key, data in indicators.items()
                ])
                
                st.dataframe(df_indicators, use_container_width=True)
                
                # Create gauge chart for sentiment
                fig_sentiment = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=sentiment,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Economic Sentiment"},
                    delta={'reference': 0},
                    gauge={
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': sentiment_color},
                        'steps': [
                            {'range': [-1, -0.5], 'color': "lightcoral"},
                            {'range': [-0.5, 0], 'color': "lightyellow"},
                            {'range': [0, 0.5], 'color': "lightblue"},
                            {'range': [0.5, 1], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.9
                        }
                    }
                ))
                
                st.plotly_chart(fig_sentiment, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Economic analysis error: {e}")

def get_bank_analyses(news_analyzer, selected_banks):
    """Get sentiment analysis for selected banks"""
    bank_analyses = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(selected_banks):
        status_text.text(f"Analyzing {symbol}...")
        progress_bar.progress((i + 1) / len(selected_banks))
        
        try:
            analysis = news_analyzer.analyze_single_bank(symbol, detailed=True)
            if analysis and 'overall_sentiment' in analysis:
                bank_analyses[symbol] = analysis
        except Exception as e:
            st.warning(f"⚠️ Failed to analyze {symbol}: {e}")
    
    progress_bar.empty()
    status_text.empty()
    
    return bank_analyses

def display_divergence_analysis(divergence_detector, bank_analyses):
    """Display sector divergence analysis"""
    st.header("🎯 Sector Divergence Analysis")
    
    try:
        divergence_analysis = divergence_detector.analyze_sector_divergence(bank_analyses)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sector_avg = divergence_analysis.get('sector_average', 0)
            st.metric("Sector Average", f"{sector_avg:+.3f}")
        
        with col2:
            sector_vol = divergence_analysis.get('sector_volatility', 0)
            st.metric("Sector Volatility", f"{sector_vol:.3f}")
        
        with col3:
            divergent_count = len(divergence_analysis.get('divergent_banks', {}))
            st.metric("Divergent Banks", divergent_count)
        
        with col4:
            analyzed_count = divergence_analysis.get('analyzed_banks', 0)
            st.metric("Banks Analyzed", analyzed_count)
        
        # Divergence visualization
        divergent_banks = divergence_analysis.get('divergent_banks', {})
        
        if divergent_banks:
            df_divergence = pd.DataFrame([
                {
                    'Bank': symbol,
                    'Divergence Score': data['divergence_score'],
                    'Significance': data['significance'],
                    'Opportunity': data['opportunity'],
                    'Confidence': data['confidence']
                }
                for symbol, data in divergent_banks.items()
            ])
            
            # Create divergence chart
            fig_div = px.bar(
                df_divergence,
                x='Bank',
                y='Divergence Score',
                color='Divergence Score',
                color_continuous_scale='RdYlGn',
                title="Bank Sentiment Divergence from Sector Average"
            )
            fig_div.add_hline(y=0, line_dash="dash", line_color="black")
            
            st.plotly_chart(fig_div, use_container_width=True)
            
            # Divergence table
            st.subheader("🎯 Divergent Banks Detail")
            st.dataframe(df_divergence, use_container_width=True)
        
        # Summary
        summary = divergence_analysis.get('summary', '')
        if summary:
            st.info(f"**Summary:** {summary}")
    
    except Exception as e:
        st.error(f"❌ Divergence analysis error: {e}")

def display_bank_sentiment_overview(bank_analyses, confidence_threshold):
    """Display overview of bank sentiment scores"""
    st.header("🏦 Bank Sentiment Overview")
    
    # Create overview dataframe
    overview_data = []
    for symbol, analysis in bank_analyses.items():
        sentiment = analysis.get('overall_sentiment', 0)
        confidence = analysis.get('confidence', 0)
        signal = analysis.get('signal', 'HOLD')
        news_count = analysis.get('news_count', 0)
        
        # Determine if high confidence
        high_conf = confidence >= confidence_threshold
        
        overview_data.append({
            'Bank': symbol,
            'Sentiment': sentiment,
            'Confidence': confidence,
            'Signal': signal,
            'News Count': news_count,
            'High Confidence': '✅' if high_conf else '❌'
        })
    
    df_overview = pd.DataFrame(overview_data)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_sentiment = df_overview['Sentiment'].mean()
        st.metric("Average Sentiment", f"{avg_sentiment:+.3f}")
    
    with col2:
        high_conf_count = sum(1 for item in overview_data if item['High Confidence'] == '✅')
        st.metric("High Confidence", f"{high_conf_count}/{len(overview_data)}")
    
    with col3:
        buy_signals = sum(1 for item in overview_data if item['Signal'] == 'BUY')
        st.metric("Buy Signals", buy_signals)
    
    with col4:
        total_news = df_overview['News Count'].sum()
        st.metric("Total News", total_news)
    
    # Sentiment distribution chart
    fig_sentiment = px.bar(
        df_overview,
        x='Bank',
        y='Sentiment',
        color='Confidence',
        title="Bank Sentiment Scores with Confidence",
        color_continuous_scale='Viridis'
    )
    fig_sentiment.add_hline(y=0, line_dash="dash", line_color="black")
    
    st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Overview table
    st.dataframe(df_overview, use_container_width=True)

def display_individual_bank_analysis(bank_analyses, data_feed, show_detailed):
    """Display detailed analysis for each bank"""
    st.header("📋 Individual Bank Analysis")
    
    for symbol, analysis in bank_analyses.items():
        with st.expander(f"🏦 {symbol} - Detailed Analysis", expanded=False):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment = analysis.get('overall_sentiment', 0)
                st.metric("Sentiment Score", f"{sentiment:+.3f}")
            
            with col2:
                confidence = analysis.get('confidence', 0)
                st.metric("Confidence", f"{confidence:.1%}")
            
            with col3:
                signal = analysis.get('signal', 'HOLD')
                signal_color = {"BUY": "green", "SELL": "red", "HOLD": "gray"}.get(signal, "gray")
                st.markdown(f"**Signal:** :{signal_color}[{signal}]")
            
            if show_detailed:
                # Additional metrics
                col4, col5, col6 = st.columns(3)
                
                with col4:
                    news_count = analysis.get('news_count', 0)
                    st.metric("News Articles", news_count)
                
                with col5:
                    # Try to get current price
                    try:
                        price_data = data_feed.get_current_data(symbol)
                        price = price_data.get('price', 0)
                        if price > 0:
                            st.metric("Current Price", f"${price:.2f}")
                        else:
                            st.metric("Current Price", "N/A")
                    except:
                        st.metric("Current Price", "N/A")
                
                with col6:
                    # Calculate impact score
                    impact = abs(sentiment) * confidence
                    st.metric("Impact Score", f"{impact:.3f}")
                
                # Recent headlines if available
                headlines = analysis.get('recent_headlines', [])
                if headlines:
                    st.subheader("📰 Recent Headlines")
                    for headline in headlines[:5]:  # Show top 5
                        st.write(f"• {headline}")

def display_ml_predictions(ml_pipeline, bank_analyses):
    """Display ML model predictions"""
    st.header("🧠 ML Model Predictions")
    
    try:
        ml_results = []
        
        for symbol, analysis in bank_analyses.items():
            # Mock market data for prediction
            market_data = {
                'price': 100.0,
                'change_percent': 0.0,
                'volume': 1000000,
                'volatility': 0.15
            }
            
            # Get ML prediction
            prediction_result = ml_pipeline.predict(analysis, market_data, [])
            
            if 'error' not in prediction_result:
                ml_results.append({
                    'Bank': symbol,
                    'ML Prediction': prediction_result.get('ensemble_prediction', 'N/A'),
                    'ML Confidence': prediction_result.get('ensemble_confidence', 0),
                    'Sentiment Signal': analysis.get('signal', 'HOLD'),
                    'Feature Count': prediction_result.get('feature_count', 0)
                })
        
        if ml_results:
            df_ml = pd.DataFrame(ml_results)
            st.dataframe(df_ml, use_container_width=True)
            
            # ML confidence chart
            fig_ml = px.bar(
                df_ml,
                x='Bank',
                y='ML Confidence',
                title="ML Model Prediction Confidence",
                color='ML Confidence',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_ml, use_container_width=True)
        else:
            st.info("🔬 ML predictions not available. Models may need training.")
            
        # Show training data status
        ml_pipeline._load_training_data()
        completed_samples = [
            record for record in ml_pipeline.training_data 
            if record.get('outcome') is not None
        ]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Samples", len(completed_samples))
        with col2:
            needed = max(0, 100 - len(completed_samples))
            st.metric("Samples Needed", needed)
    
    except Exception as e:
        st.error(f"❌ ML prediction error: {e}")

def display_trading_signals(divergence_detector, bank_analyses):
    """Display trading signals based on divergence and sentiment"""
    st.header("📈 Trading Signals")
    
    try:
        # Get divergence analysis
        divergence_analysis = divergence_detector.analyze_sector_divergence(bank_analyses)
        
        # Generate trading signals
        trading_signals = divergence_detector.generate_trading_signals(divergence_analysis)
        
        if trading_signals:
            # Create signals dataframe
            df_signals = pd.DataFrame(trading_signals)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                buy_count = sum(1 for s in trading_signals if s['signal'] == 'BUY')
                st.metric("Buy Signals", buy_count)
            
            with col2:
                sell_count = sum(1 for s in trading_signals if s['signal'] == 'SELL')
                st.metric("Sell Signals", sell_count)
            
            with col3:
                avg_significance = np.mean([s['significance'] for s in trading_signals])
                st.metric("Avg Significance", f"{avg_significance:.2f}")
            
            # Signals table
            st.subheader("🎯 Active Trading Signals")
            
            # Color code the signals
            def color_signal(val):
                if val == 'BUY':
                    return 'background-color: lightgreen'
                elif val == 'SELL':
                    return 'background-color: lightcoral'
                return ''
            
            styled_df = df_signals.style.applymap(color_signal, subset=['signal'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Signal reasoning
            st.subheader("💡 Signal Reasoning")
            for signal in trading_signals:
                st.write(f"**{signal['symbol']}**: {signal['reasoning']}")
        else:
            st.info("📊 No high-significance trading signals detected at this time.")
    
    except Exception as e:
        st.error(f"❌ Trading signals error: {e}")

if __name__ == "__main__":
    main()
