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
        return filepath
    
    def _generate_risk_report(self, data: Dict) -> str:
        """Generate risk analysis report content"""
        
        content = f"""
        <div class="section">
            <h2>Risk Analysis Report</h2>
            <p>Comprehensive risk assessment for trading positions.</p>
            
            <h3>Portfolio Risk Metrics</h3>
            <table>
                <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
                <tr><td>Total Exposure</td><td>${data.get('total_exposure', 0):,.2f}</td><td>-</td></tr>
                <tr><td>Risk Percentage</td><td>{data.get('risk_percentage', 0):.2f}%</td><td>-</td></tr>
            </table>
        </div>
        """
        
        return content
    
    def _generate_technical_report(self, data: Dict) -> str:
        """Generate technical analysis report content"""
        
        content = f"""
        <div class="section">
            <h2>Technical Analysis Report</h2>
            <p>Detailed technical indicators and signals.</p>
            
            <h3>Key Indicators</h3>
            <ul>
                <li>RSI: {data.get('rsi', 0):.2f}</li>
                <li>MACD: {data.get('macd', 0):.2f}</li>
                <li>Overall Signal: {data.get('overall_signal', 0):.2f}</li>
            </ul>
        </div>
        """
        
        return content
    
    def _generate_fundamental_report(self, data: Dict) -> str:
        """Generate fundamental analysis report content"""
        
        content = f"""
        <div class="section">
            <h2>Fundamental Analysis Report</h2>
            <p>Company valuation and financial metrics.</p>
            
            <h3>Key Metrics</h3>
            <ul>
                <li>P/E Ratio: {data.get('pe_ratio', 0):.2f}</li>
                <li>ROE: {data.get('roe', 0):.2%}</li>
                <li>Dividend Yield: {data.get('dividend_yield', 0):.2%}</li>
            </ul>
        </div>
        """
        
        return content
    
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
        
        logger.info(f"Weekly report generated: {filepath}")
        return filepath