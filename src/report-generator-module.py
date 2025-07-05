# src/report_generator.py
"""
Report generation module for creating comprehensive analysis reports
Supports HTML, PDF, and markdown formats
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import logging
from jinja2 import Template
import plotly.graph_objects as go
import plotly.offline as pyo
import pandas as pd

# Import settings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Settings

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates comprehensive trading reports"""
    
    def __init__(self):
        self.settings = Settings()
        self.report_dir = "reports"
        os.makedirs(self.report_dir, exist_ok=True)
        
        # HTML template for reports
        self.html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                .header {
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                }
                .section {
                    background-color: white;
                    margin: 20px 0;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .bank-card {
                    background-color: #f9f9f9;
                    border: 1px solid #ddd;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                }
                .metric {
                    display: inline-block;
                    margin: 10px 20px 10px 0;
                }
                .metric-label {
                    font-weight: bold;
                    color: #666;
                }
                .metric-value {
                    font-size: 1.2em;
                    color: #2c3e50;
                }
                .positive { color: #27ae60; }
                .negative { color: #e74c3c; }
                .neutral { color: #95a5a6; }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th, td {
                    padding: 10px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #2c3e50;
                    color: white;
                }
                tr:hover { background-color: #f5f5f5; }
                .chart-container {
                    margin: 20px 0;
                }
                .recommendation {
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                    font-weight: bold;
                }
                .buy { background-color: #d4edda; color: #155724; }
                .sell { background-color: #f8d7da; color: #721c24; }
                .hold { background-color: #fff3cd; color: #856404; }
                .footer {
                    text-align: center;
                    color: #666;
                    margin-top: 50px;
                    padding: 20px;
                }
            </style>
        </head>
        <body>
            {{ content }}
        </body>
        </html>
        """
    
    def generate_daily_report(self, analysis_results: Dict, market_data: Dict) -> str:
        """Generate comprehensive daily report"""
        
        timestamp = datetime.now()
        filename = f"daily_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(self.report_dir, filename)
        
        content = self._generate_daily_content(analysis_results, market_data, timestamp)
        
        # Create HTML report
        template = Template(self.html_template)
        html_content = template.render(
            title=f"ASX Bank Analysis - {timestamp.strftime('%Y-%m-%d')}",
            content=content
        )
        
        # Save report
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Weekly report generated: {filepath}")
        return filepath
    
    def _assess_indicator_impact(self, indicator: str, value: float) -> Dict:
        """Assess the impact of an economic indicator"""
        
        impacts = {
            'cash_rate': {
                'high': (value > 4.0, 'negative', 'High rates pressure margins'),
                'low': (value < 2.0, 'positive', 'Low rates support lending')
            },
            'inflation': {
                'high': (value > 3.0, 'negative', 'High inflation concern'),
                'low': (value < 2.0, 'neutral', 'Low inflation')
            },
            'gdp_growth': {
                'high': (value > 3.0, 'positive', 'Strong economic growth'),
                'low': (value < 1.0, 'negative', 'Weak growth')
            },
            'unemployment': {
                'high': (value > 5.0, 'negative', 'High unemployment risk'),
                'low': (value < 4.0, 'positive', 'Low unemployment')
            }
        }
        
        if indicator in impacts:
            for condition, (check, impact_class, impact_text) in impacts[indicator].items():
                if (condition == 'high' and check) or (condition == 'low' and not check):
                    return {'class': impact_class, 'text': impact_text}
        
        return {'class': 'neutral', 'text': 'Neutral impact'}
    
    def _get_sentiment_class(self, sentiment: float) -> str:
        """Get CSS class for sentiment value"""
        if sentiment > 0.3:
            return 'positive'
        elif sentiment < -0.3:
            return 'negative'
        else:
            return 'neutral'
    
    def _get_sentiment_text(self, sentiment: float) -> str:
        """Convert sentiment score to text"""
        if sentiment > 0.5:
            return 'Very Positive'
        elif sentiment > 0.2:
            return 'Positive'
        elif sentiment < -0.5:
            return 'Very Negative'
        elif sentiment < -0.2:
            return 'Negative'
        else:
            return 'Neutral'
    
    def _format_overnight_news(self, news: List[Dict]) -> str:
        """Format overnight news for morning brief"""
        
        if not news:
            return "<p>No significant overnight news.</p>"
        
        html = "<ul>"
        for item in news[:5]:  # Top 5 news items
            html += f"""
            <li>
                <strong>{item.get('source', 'Unknown')}:</strong> {item.get('title', 'No title')}
                <br><small>{item.get('published', '')}</small>
            </li>
            """
        html += "</ul>"
        
        return html
    
    def _format_us_impact(self, us_impact: Dict) -> str:
        """Format US market impact analysis"""
        
        html = f"""
        <p><strong>US Market Trend:</strong> {us_impact.get('us_trend', 'Unknown').upper()}</p>
        <p><strong>Impact on ASX Banks:</strong> {us_impact.get('impact', 'Neutral').upper()}</p>
        <p><strong>Key Factors:</strong></p>
        <ul>
        """
        
        for factor in us_impact.get('key_factors', []):
            html += f"<li>{factor}</li>"
        
        html += "</ul>"
        
        return html
    
    def _generate_watchlist(self) -> str:
        """Generate watchlist for the day"""
        
        watchlist = [
            "RBA announcements or speeches",
            "Banking sector earnings releases",
            "Housing market data releases",
            "Major economic indicators",
            "Regulatory news from APRA/ASIC"
        ]
        
        html = "<ul>"
        for item in watchlist:
            html += f"<li>{item}</li>"
        html += "</ul>"
        
        return html
    
    def _generate_performance_summary(self, results: Dict) -> str:
        """Generate performance summary for EOD report"""
        
        html = "<table>"
        html += "<tr><th>Bank</th><th>Open</th><th>Close</th><th>Change</th><th>Volume</th></tr>"
        
        for symbol, analysis in results.items():
            current = analysis.get('current_price', 0)
            # These would come from actual data in production
            open_price = current * 0.99  # Placeholder
            change = ((current - open_price) / open_price) * 100
            volume = 1000000  # Placeholder
            
            change_class = 'positive' if change > 0 else 'negative'
            
            html += f"""
            <tr>
                <td>{symbol}</td>
                <td>${open_price:.2f}</td>
                <td>${current:.2f}</td>
                <td class="{change_class}">{change:+.2f}%</td>
                <td>{volume:,}</td>
            </tr>
            """
        
        html += "</table>"
        
        return html
    
    def _generate_volume_analysis(self, results: Dict) -> str:
        """Generate volume analysis section"""
        
        html = "<p>Volume analysis helps identify unusual trading activity that may signal upcoming moves.</p>"
        html += "<ul>"
        
        # Placeholder analysis
        html += "<li>CBA showed above-average volume with positive price action</li>"
        html += "<li>WBC volume remained subdued despite price movement</li>"
        html += "<li>No unusual volume spikes detected in other banks</li>"
        
        html += "</ul>"
        
        return html
    
    def _generate_tomorrow_outlook(self, results: Dict) -> str:
        """Generate tomorrow's outlook"""
        
        html = "<h3>Key Levels to Watch</h3>"
        html += "<table>"
        html += "<tr><th>Bank</th><th>Support</th><th>Resistance</th><th>Bias</th></tr>"
        
        for symbol, analysis in results.items():
            technical = analysis.get('technical_analysis', {})
            sr = technical.get('support_resistance', {})
            prediction = analysis.get('prediction', {})
            
            support = sr.get('support', [0])[0] if sr.get('support') else 0
            resistance = sr.get('resistance', [0])[-1] if sr.get('resistance') else 0
            bias = prediction.get('direction', 'neutral')
            
            html += f"""
            <tr>
                <td>{symbol}</td>
                <td>${support:.2f}</td>
                <td>${resistance:.2f}</td>
                <td class="{bias}">{bias.upper()}</td>
            </tr>
            """
        
        html += "</table>"
        
        return html
    
    def _generate_weekly_performance(self, weekly_data: Dict) -> str:
        """Generate weekly performance summary"""
        
        html = "<table>"
        html += "<tr><th>Bank</th><th>Week Open</th><th>Week Close</th><th>Weekly Change</th><th>Trend</th></tr>"
        
        for symbol, data in weekly_data.items():
            if isinstance(data, pd.DataFrame) and not data.empty:
                week_open = data['Open'].iloc[0]
                week_close = data['Close'].iloc[-1]
                weekly_change = ((week_close - week_open) / week_open) * 100
                
                # Simple trend determination
                if data['Close'].iloc[-1] > data['Close'].iloc[-5]:
                    trend = "Uptrend"
                    trend_class = "positive"
                else:
                    trend = "Downtrend"
                    trend_class = "negative"
                
                change_class = 'positive' if weekly_change > 0 else 'negative'
                
                html += f"""
                <tr>
                    <td>{symbol}</td>
                    <td>${week_open:.2f}</td>
                    <td>${week_close:.2f}</td>
                    <td class="{change_class}">{weekly_change:+.2f}%</td>
                    <td class="{trend_class}">{trend}</td>
                </tr>
                """
        
        html += "</table>"
        
        return html
    
    def _generate_weekly_technicals(self, weekly_data: Dict) -> str:
        """Generate weekly technical analysis summary"""
        
        html = "<p>Technical indicators provide insights into momentum and potential reversal points.</p>"
        html += "<table>"
        html += "<tr><th>Bank</th><th>50-Day MA</th><th>200-Day MA</th><th>RSI</th><th>Signal</th></tr>"
        
        # Placeholder technical data
        for symbol in weekly_data.keys():
            # These would be calculated from actual data
            ma50 = 100.0
            ma200 = 95.0
            rsi = 55
            signal = "Buy" if ma50 > ma200 else "Sell"
            
            html += f"""
            <tr>
                <td>{symbol}</td>
                <td>${ma50:.2f}</td>
                <td>${ma200:.2f}</td>
                <td>{rsi}</td>
                <td class="{'buy' if signal == 'Buy' else 'sell'}">{signal}</td>
            </tr>
            """
        
        html += "</table>"
        
        return html
    
    def _generate_next_week_outlook(self) -> str:
        """Generate next week outlook"""
        
        html = """
        <h3>Economic Calendar</h3>
        <ul>
            <li>Tuesday: RBA Interest Rate Decision</li>
            <li>Wednesday: Employment Data Release</li>
            <li>Thursday: Housing Finance Data</li>
            <li>Friday: Consumer Confidence Index</li>
        </ul>
        
        <h3>Trading Strategy</h3>
        <p>Based on current analysis:</p>
        <ul>
            <li>Watch for RBA decision impact on banking margins</li>
            <li>Monitor support levels for potential entry points</li>
            <li>Consider reducing exposure ahead of major data releases</li>
            <li>Focus on banks with strongest technical setups</li>
        </ul>
        """
        
        return html
    
    def export_to_csv(self, analysis_results: Dict, filename: Optional[str] = None) -> str:
        """Export analysis results to CSV"""
        
        if not filename:
            timestamp = datetime.now()
            filename = f"analysis_export_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = os.path.join(self.report_dir, filename)
        
        # Prepare data for CSV
        data = []
        for symbol, analysis in analysis_results.items():
            row = {
                'Symbol': symbol,
                'Bank': self.settings.get_bank_name(symbol),
                'Current Price': analysis.get('current_price', 0),
                'Recommendation': analysis.get('recommendation', {}).get('action', ''),
                'Confidence': analysis.get('recommendation', {}).get('confidence', ''),
                'Risk Score': analysis.get('risk_reward', {}).get('risk_score', 0),
                'Technical Signal': analysis.get('technical_analysis', {}).get('overall_signal', 0),
                'P/E Ratio': analysis.get('fundamental_analysis', {}).get('metrics', {}).get('pe_ratio', 0),
                'Dividend Yield': analysis.get('fundamental_analysis', {}).get('metrics', {}).get('dividend_yield', 0) * 100,
                'Sentiment': analysis.get('sentiment_analysis', {}).get('overall_sentiment', 0),
                'Prediction Score': analysis.get('prediction', {}).get('score', 0)
            }
            data.append(row)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Analysis exported to CSV: {filepath}")
        return filepath
    
    def generate_custom_report(self, report_type: str, data: Dict, **kwargs) -> str:
        """Generate custom reports based on type"""
        
        timestamp = datetime.now()
        filename = f"{report_type}_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(self.report_dir, filename)
        
        if report_type == 'risk_analysis':
            content = self._generate_risk_report(data)
        elif report_type == 'technical_analysis':
            content = self._generate_technical_report(data)
        elif report_type == 'fundamental_analysis':
            content = self._generate_fundamental_report(data)
        else:
            content = "<p>Report type not supported</p>"
        
        template = Template(self.html_template)
        html_content = template.render(
            title=f"{report_type.replace('_', ' ').title()} Report",
            content=content
        )
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Custom report generated: {filepath}")
        return filepath("Daily report generated: {filepath}")
        return filepath
    
    def _generate_daily_content(self, analysis_results: Dict, market_data: Dict, timestamp: datetime) -> str:
        """Generate content for daily report"""
        
        content = f"""
        <div class="header">
            <h1>ASX Bank Trading Analysis</h1>
            <h2>{timestamp.strftime('%A, %B %d, %Y')}</h2>
            <p>Generated at {timestamp.strftime('%I:%M %p AEST')}</p>
        </div>
        
        <!-- Market Overview -->
        <div class="section">
            <h2>Market Overview</h2>
            {self._generate_market_overview(market_data)}
        </div>
        
        <!-- Executive Summary -->
        <div class="section">
            <h2>Executive Summary</h2>
            {self._generate_executive_summary(analysis_results)}
        </div>
        
        <!-- Individual Bank Analysis -->
        <div class="section">
            <h2>Bank Analysis</h2>
            {self._generate_bank_analysis(analysis_results)}
        </div>
        
        <!-- Risk Assessment -->
        <div class="section">
            <h2>Risk Assessment</h2>
            {self._generate_risk_assessment(analysis_results)}
        </div>
        
        <!-- Trading Recommendations -->
        <div class="section">
            <h2>Trading Recommendations</h2>
            {self._generate_recommendations(analysis_results)}
        </div>
        
        <!-- Charts -->
        <div class="section">
            <h2>Visual Analysis</h2>
            {self._generate_charts(analysis_results)}
        </div>
        
        <div class="footer">
            <p>This report is for informational purposes only. Always do your own research before making investment decisions.</p>
            <p>Generated by ASX Bank Trading Analysis System</p>
        </div>
        """
        
        return content
    
    def _generate_market_overview(self, market_data: Dict) -> str:
        """Generate market overview section"""
        
        asx200 = market_data.get('ASX200', {})
        trend = market_data.get('trend', 'neutral')
        
        trend_class = 'positive' if trend == 'bullish' else 'negative' if trend == 'bearish' else 'neutral'
        
        html = f"""
        <div class="metric">
            <span class="metric-label">ASX 200:</span>
            <span class="metric-value {trend_class}">
                {asx200.get('value', 0):.2f} ({asx200.get('change_percent', 0):+.2f}%)
            </span>
        </div>
        <div class="metric">
            <span class="metric-label">Market Trend:</span>
            <span class="metric-value {trend_class}">{trend.upper()}</span>
        </div>
        """
        
        # Sector performance
        sector = market_data.get('sector_performance', {})
        if sector:
            html += f"""
            <div class="metric">
                <span class="metric-label">Banking Sector:</span>
                <span class="metric-value">{sector.get('outlook', 'neutral').upper()}</span>
            </div>
            """
        
        # Economic indicators
        indicators = market_data.get('economic_indicators', {})
        if indicators:
            html += "<h3>Key Economic Indicators</h3>"
            html += "<table>"
            html += "<tr><th>Indicator</th><th>Value</th><th>Impact</th></tr>"
            
            for key, value in indicators.items():
                impact = self._assess_indicator_impact(key, value)
                html += f"""
                <tr>
                    <td>{key.replace('_', ' ').title()}</td>
                    <td>{value}</td>
                    <td class="{impact['class']}">{impact['text']}</td>
                </tr>
                """
            
            html += "</table>"
        
        return html
    
    def _generate_executive_summary(self, analysis_results: Dict) -> str:
        """Generate executive summary"""
        
        # Count recommendations
        buy_count = sum(1 for r in analysis_results.values() 
                       if 'BUY' in r.get('recommendation', {}).get('action', ''))
        sell_count = sum(1 for r in analysis_results.values() 
                        if 'SELL' in r.get('recommendation', {}).get('action', ''))
        hold_count = len(analysis_results) - buy_count - sell_count
        
        # Average risk score
        risk_scores = [r.get('risk_reward', {}).get('risk_score', 50) 
                      for r in analysis_results.values()]
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 50
        
        html = f"""
        <p><strong>Analysis Summary:</strong></p>
        <ul>
            <li>Banks Analyzed: {len(analysis_results)}</li>
            <li>Buy Recommendations: {buy_count}</li>
            <li>Sell Recommendations: {sell_count}</li>
            <li>Hold Recommendations: {hold_count}</li>
            <li>Average Risk Score: {avg_risk:.1f}/100</li>
        </ul>
        """
        
        # Key findings
        html += "<p><strong>Key Findings:</strong></p><ul>"
        
        # Find best opportunity
        best_opportunity = max(analysis_results.items(), 
                             key=lambda x: x[1].get('prediction', {}).get('score', 0))
        if best_opportunity[1].get('prediction', {}).get('score', 0) > 50:
            html += f"<li>Best Opportunity: {best_opportunity[0]} - {best_opportunity[1].get('recommendation', {}).get('action', 'HOLD')}</li>"
        
        # Highest risk
        highest_risk = max(analysis_results.items(), 
                          key=lambda x: x[1].get('risk_reward', {}).get('risk_score', 0))
        if highest_risk[1].get('risk_reward', {}).get('risk_score', 0) > 70:
            html += f"<li>Highest Risk: {highest_risk[0]} - Risk Score: {highest_risk[1].get('risk_reward', {}).get('risk_score', 0)}/100</li>"
        
        html += "</ul>"
        
        return html
    
    def _generate_bank_analysis(self, analysis_results: Dict) -> str:
        """Generate individual bank analysis cards"""
        
        html = ""
        
        for symbol, analysis in analysis_results.items():
            bank_name = self.settings.get_bank_name(symbol)
            recommendation = analysis.get('recommendation', {})
            technical = analysis.get('technical_analysis', {})
            fundamental = analysis.get('fundamental_analysis', {})
            sentiment = analysis.get('sentiment_analysis', {})
            risk_reward = analysis.get('risk_reward', {})
            prediction = analysis.get('prediction', {})
            
            # Determine recommendation class
            action = recommendation.get('action', 'HOLD')
            rec_class = 'buy' if 'BUY' in action else 'sell' if 'SELL' in action else 'hold'
            
            html += f"""
            <div class="bank-card">
                <h3>{bank_name} ({symbol})</h3>
                
                <div class="recommendation {rec_class}">
                    {action} - {recommendation.get('confidence', 'MEDIUM')} Confidence
                </div>
                
                <div style="display: flex; flex-wrap: wrap;">
                    <div class="metric">
                        <span class="metric-label">Current Price:</span>
                        <span class="metric-value">${analysis.get('current_price', 0):.2f}</span>
                    </div>
                    
                    <div class="metric">
                        <span class="metric-label">Technical Signal:</span>
                        <span class="metric-value">{technical.get('overall_signal', 0):.1f}</span>
                    </div>
                    
                    <div class="metric">
                        <span class="metric-label">P/E Ratio:</span>
                        <span class="metric-value">{fundamental.get('metrics', {}).get('pe_ratio', 0):.1f}</span>
                    </div>
                    
                    <div class="metric">
                        <span class="metric-label">Dividend Yield:</span>
                        <span class="metric-value">{fundamental.get('metrics', {}).get('dividend_yield', 0)*100:.2f}%</span>
                    </div>
                    
                    <div class="metric">
                        <span class="metric-label">Sentiment:</span>
                        <span class="metric-value {self._get_sentiment_class(sentiment.get('overall_sentiment', 0))}">
                            {self._get_sentiment_text(sentiment.get('overall_sentiment', 0))}
                        </span>
                    </div>
                    
                    <div class="metric">
                        <span class="metric-label">Risk Score:</span>
                        <span class="metric-value">{risk_reward.get('risk_score', 0):.0f}/100</span>
                    </div>
                </div>
                
                <p><strong>Key Factors:</strong></p>
                <ul>
            """
            
            # Add key factors
            for factor in prediction.get('key_factors', [])[:3]:
                html += f"<li>{factor}</li>"
            
            html += """
                </ul>
            </div>
            """
        
        return html
    
    def _generate_risk_assessment(self, analysis_results: Dict) -> str:
        """Generate risk assessment section"""
        
        html = "<table>"
        html += "<tr><th>Bank</th><th>Risk Score</th><th>Risk/Reward</th><th>Stop Loss</th><th>Target</th></tr>"
        
        for symbol, analysis in analysis_results.items():
            risk_reward = analysis.get('risk_reward', {})
            risk_score = risk_reward.get('risk_score', 0)
            rr_ratio = risk_reward.get('risk_metrics', {}).get('risk_reward_ratio', 0)
            stop_loss = risk_reward.get('stop_loss', {}).get('recommended', 0)
            target = risk_reward.get('take_profit', {}).get('target_1', 0)
            
            risk_class = 'negative' if risk_score > 70 else 'positive' if risk_score < 40 else 'neutral'
            
            html += f"""
            <tr>
                <td>{symbol}</td>
                <td class="{risk_class}">{risk_score:.0f}</td>
                <td>{rr_ratio:.2f}:1</td>
                <td>${stop_loss:.2f}</td>
                <td>${target:.2f}</td>
            </tr>
            """
        
        html += "</table>"
        
        return html
    
    def _generate_recommendations(self, analysis_results: Dict) -> str:
        """Generate trading recommendations table"""
        
        html = "<table>"
        html += "<tr><th>Bank</th><th>Action</th><th>Entry</th><th>Position Size</th><th>Timeframe</th></tr>"
        
        for symbol, analysis in analysis_results.items():
            recommendation = analysis.get('recommendation', {})
            risk_reward = analysis.get('risk_reward', {})
            prediction = analysis.get('prediction', {})
            
            action = recommendation.get('action', 'HOLD')
            if action != 'HOLD':
                entry = analysis.get('current_price', 0)
                position = risk_reward.get('position_size', {}).get('recommended_shares', 0)
                timeframe = prediction.get('timeframes', {}).get('short_term', {}).get('prediction', 'unknown')
                
                html += f"""
                <tr>
                    <td>{symbol}</td>
                    <td class="{'buy' if 'BUY' in action else 'sell'}">{action}</td>
                    <td>${entry:.2f}</td>
                    <td>{position} shares</td>
                    <td>{timeframe}</td>
                </tr>
                """
        
        html += "</table>"
        
        return html
    
    def _generate_charts(self, analysis_results: Dict) -> str:
        """Generate interactive charts"""
        
        # Create comparison chart
        symbols = list(analysis_results.keys())
        scores = [a.get('prediction', {}).get('score', 0) for a in analysis_results.values()]
        risk_scores = [a.get('risk_reward', {}).get('risk_score', 0) for a in analysis_results.values()]
        
        # Prediction scores chart
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=symbols,
            y=scores,
            name='Prediction Score',
            marker_color=['green' if s > 0 else 'red' for s in scores]
        ))
        fig1.update_layout(
            title='Prediction Scores by Bank',
            xaxis_title='Bank',
            yaxis_title='Score',
            height=400
        )
        
        # Risk scores chart
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=symbols,
            y=risk_scores,
            name='Risk Score',
            marker_color=['red' if s > 70 else 'yellow' if s > 40 else 'green' for s in risk_scores]
        ))
        fig2.update_layout(
            title='Risk Scores by Bank',
            xaxis_title='Bank',
            yaxis_title='Risk Score',
            height=400
        )
        
        # Convert to HTML
        chart1_html = pyo.plot(fig1, output_type='div', include_plotlyjs='cdn')
        chart2_html = pyo.plot(fig2, output_type='div', include_plotlyjs='cdn')
        
        return f"""
        <div class="chart-container">
            {chart1_html}
        </div>
        <div class="chart-container">
            {chart2_html}
        </div>
        """
    
    def generate_morning_brief(self, overnight_news: List[Dict], us_impact: Dict) -> str:
        """Generate morning brief report"""
        
        timestamp = datetime.now()
        filename = f"morning_brief_{timestamp.strftime('%Y%m%d')}.html"
        filepath = os.path.join(self.report_dir, filename)
        
        content = f"""
        <div class="header">
            <h1>Morning Brief - ASX Banks</h1>
            <h2>{timestamp.strftime('%A, %B %d, %Y')}</h2>
        </div>
        
        <div class="section">
            <h2>Overnight Developments</h2>
            {self._format_overnight_news(overnight_news)}
        </div>
        
        <div class="section">
            <h2>US Market Impact</h2>
            {self._format_us_impact(us_impact)}
        </div>
        
        <div class="section">
            <h2>Key Things to Watch Today</h2>
            {self._generate_watchlist()}
        </div>
        """
        
        template = Template(self.html_template)
        html_content = template.render(
            title="Morning Brief - ASX Banks",
            content=content
        )
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Morning brief generated: {filepath}")
        return filepath
    
    def generate_eod_report(self, results: Dict, market: Dict) -> str:
        """Generate end of day report"""
        
        timestamp = datetime.now()
        filename = f"eod_report_{timestamp.strftime('%Y%m%d')}.html"
        filepath = os.path.join(self.report_dir, filename)
        
        content = f"""
        <div class="header">
            <h1>End of Day Report - ASX Banks</h1>
            <h2>{timestamp.strftime('%A, %B %d, %Y')}</h2>
        </div>
        
        <div class="section">
            <h2>Today's Performance</h2>
            {self._generate_performance_summary(results)}
        </div>
        
        <div class="section">
            <h2>Trading Activity</h2>
            {self._generate_volume_analysis(results)}
        </div>
        
        <div class="section">
            <h2>Tomorrow's Outlook</h2>
            {self._generate_tomorrow_outlook(results)}
        </div>
        """
        
        template = Template(self.html_template)
        html_content = template.render(
            title="End of Day Report",
            content=content
        )
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"EOD report generated: {filepath}")
        return filepath
    
    def generate_weekly_report(self, weekly_data: Dict) -> str:
        """Generate weekly summary report"""
        
        timestamp = datetime.now()
        filename = f"weekly_report_{timestamp.strftime('%Y%m%d')}.html"
        filepath = os.path.join(self.report_dir, filename)
        
        content = f"""
        <div class="header">
            <h1>Weekly Report - ASX Banks</h1>
            <h2>Week Ending {timestamp.strftime('%B %d, %Y')}</h2>
        </div>
        
        <div class="section">
            <h2>Weekly Performance</h2>
            {self._generate_weekly_performance(weekly_data)}
        </div>
        
        <div class="section">
            <h2>Technical Analysis</h2>
            {self._generate_weekly_technicals(weekly_data)}
        </div>
        
        <div class="section">
            <h2>Next Week Outlook</h2>
            {self._generate_next_week_outlook()}
        </div>
        """
        
        template = Template(self.html_template)
        html_content = template.render(
            title="Weekly Report",
            content=content
        )
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(