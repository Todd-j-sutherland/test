#!/usr/bin/env python3
"""
Simple HTML News Analysis Dashboard
Generates a static HTML dashboard from sentiment analysis data
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
import webbrowser
import tempfile

class SimpleHTMLDashboard:
    """Generate a simple HTML dashboard for news analysis"""
    
    def __init__(self):
        self.data_path = "data/sentiment_history"
        self.bank_symbols = ["CBA.AX", "WBC.AX", "ANZ.AX", "NAB.AX", "MQG.AX"]
        self.bank_names = {
            "CBA.AX": "Commonwealth Bank",
            "WBC.AX": "Westpac Banking Corp",
            "ANZ.AX": "ANZ Banking Group",
            "NAB.AX": "National Australia Bank",
            "MQG.AX": "Macquarie Group"
        }
        
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
                    print(f"Error loading data for {symbol}: {e}")
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
    
    def get_trading_signal(self, sentiment: float, confidence: float) -> str:
        """Get trading signal based on sentiment and confidence"""
        if confidence < 0.6:
            return "HOLD (Low Confidence)"
        elif sentiment > 0.5:
            return "STRONG BUY"
        elif sentiment > 0.2:
            return "BUY"
        elif sentiment < -0.5:
            return "STRONG SELL"
        elif sentiment < -0.2:
            return "SELL"
        else:
            return "HOLD"
    
    def generate_bank_card_html(self, symbol: str, data: List[Dict]) -> str:
        """Generate HTML for a single bank card"""
        latest = self.get_latest_analysis(data)
        
        if not latest:
            return f"""
            <div class="bank-card">
                <h3>{self.bank_names.get(symbol, symbol)} ({symbol})</h3>
                <p class="no-data">No analysis data available</p>
            </div>
            """
        
        bank_name = self.bank_names.get(symbol, symbol)
        sentiment = latest.get('overall_sentiment', 0)
        confidence = latest.get('confidence', 0)
        news_count = latest.get('news_count', 0)
        
        score_text, score_class = self.format_sentiment_score(sentiment)
        conf_level, conf_class = self.get_confidence_level(confidence)
        trading_signal = self.get_trading_signal(sentiment, confidence)
        
        # Timestamp formatting
        timestamp = latest.get('timestamp', 'Unknown')
        if timestamp != 'Unknown':
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime('%Y-%m-%d %H:%M')
            except:
                time_str = timestamp[:16]
        else:
            time_str = 'Unknown'
        
        # Recent headlines
        headlines_html = ""
        if 'recent_headlines' in latest:
            headlines = [h for h in latest['recent_headlines'][:5] if h]
            if headlines:
                headlines_html = "<h4>📰 Recent Headlines</h4><ul>"
                for headline in headlines:
                    headlines_html += f"<li>{headline}</li>"
                headlines_html += "</ul>"
        
        # Sentiment components
        components_html = ""
        if 'sentiment_components' in latest:
            components = latest['sentiment_components']
            components_html = """
            <h4>📊 Sentiment Breakdown</h4>
            <div class="components-grid">
            """
            for component, score in components.items():
                components_html += f"""
                <div class="component-item">
                    <span class="component-label">{component.title()}:</span>
                    <span class="component-score {self.format_sentiment_score(score)[1]}">{score:.3f}</span>
                </div>
                """
            components_html += "</div>"
        
        # Significant events
        events_html = ""
        if 'significant_events' in latest and 'events_detected' in latest['significant_events']:
            events = latest['significant_events']['events_detected']
            if events:
                events_html = "<h4>🚨 Significant Events</h4>"
                for event in events[:3]:  # Show top 3 events
                    event_type = event.get('type', 'unknown').replace('_', ' ').title()
                    headline = event.get('headline', 'No headline')
                    sentiment_impact = event.get('sentiment_impact', 0)
                    
                    impact_class = "positive" if sentiment_impact > 0 else "negative" if sentiment_impact < 0 else "neutral"
                    
                    events_html += f"""
                    <div class="event-item">
                        <span class="event-type">{event_type}</span>
                        <div class="event-headline">{headline}</div>
                        <div class="event-impact {impact_class}">Impact: {sentiment_impact:+.2f}</div>
                    </div>
                    """
        
        return f"""
        <div class="bank-card">
            <div class="bank-header">
                <h2>🏦 {bank_name} ({symbol})</h2>
                <div class="trading-signal {score_class}">{trading_signal}</div>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-label">Sentiment Score</div>
                    <div class="metric-value {score_class}">{score_text}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value {conf_class}">{confidence:.2f} ({conf_level})</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">News Articles</div>
                    <div class="metric-value">{news_count}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Last Updated</div>
                    <div class="metric-value">{time_str}</div>
                </div>
            </div>
            
            {components_html}
            {headlines_html}
            {events_html}
        </div>
        """
    
    def generate_overview_html(self, all_data: Dict) -> str:
        """Generate overview statistics HTML"""
        # Calculate summary stats
        latest_analyses = [self.get_latest_analysis(data) for data in all_data.values() if data]
        
        if not latest_analyses:
            return "<div class='overview'>No analysis data available</div>"
        
        avg_sentiment = sum(a.get('overall_sentiment', 0) for a in latest_analyses) / len(latest_analyses)
        avg_confidence = sum(a.get('confidence', 0) for a in latest_analyses) / len(latest_analyses)
        total_news = sum(a.get('news_count', 0) for a in latest_analyses)
        
        high_confidence_count = sum(1 for a in latest_analyses if a.get('confidence', 0) >= 0.8)
        bullish_count = sum(1 for a in latest_analyses if a.get('overall_sentiment', 0) > 0.2)
        bearish_count = sum(1 for a in latest_analyses if a.get('overall_sentiment', 0) < -0.2)
        
        score_text, score_class = self.format_sentiment_score(avg_sentiment)
        
        return f"""
        <div class="overview">
            <h2>📊 Market Overview</h2>
            <div class="overview-grid">
                <div class="overview-item">
                    <div class="overview-value {score_class}">{score_text}</div>
                    <div class="overview-label">Average Sentiment</div>
                </div>
                <div class="overview-item">
                    <div class="overview-value">{avg_confidence:.2f}</div>
                    <div class="overview-label">Average Confidence</div>
                </div>
                <div class="overview-item">
                    <div class="overview-value">{total_news}</div>
                    <div class="overview-label">Total News Articles</div>
                </div>
                <div class="overview-item">
                    <div class="overview-value">{high_confidence_count}/{len(latest_analyses)}</div>
                    <div class="overview-label">High Confidence Signals</div>
                </div>
                <div class="overview-item">
                    <div class="overview-value positive">{bullish_count}</div>
                    <div class="overview-label">Bullish Banks</div>
                </div>
                <div class="overview-item">
                    <div class="overview-value negative">{bearish_count}</div>
                    <div class="overview-label">Bearish Banks</div>
                </div>
            </div>
        </div>
        """
    
    def generate_legend_html(self) -> str:
        """Generate confidence legend HTML"""
        return """
        <div class="legend">
            <h2>📊 Confidence Score Legend & Decision Criteria</h2>
            <div class="legend-grid">
                <div class="legend-item confidence-high">
                    <h3>🟢 HIGH CONFIDENCE (≥0.8)</h3>
                    <p><strong>Action:</strong> Strong Buy/Sell Signal</p>
                    <p><strong>Criteria:</strong> Multiple reliable sources, consistent sentiment, significant news volume</p>
                    <p><strong>Decision:</strong> Execute trades with full position sizing</p>
                </div>
                <div class="legend-item confidence-medium">
                    <h3>🟡 MEDIUM CONFIDENCE (0.6-0.8)</h3>
                    <p><strong>Action:</strong> Moderate Buy/Sell Signal</p>
                    <p><strong>Criteria:</strong> Some reliable sources, moderate sentiment consistency</p>
                    <p><strong>Decision:</strong> Execute trades with reduced position sizing</p>
                </div>
                <div class="legend-item confidence-low">
                    <h3>🔴 LOW CONFIDENCE (<0.6)</h3>
                    <p><strong>Action:</strong> Hold/Monitor</p>
                    <p><strong>Criteria:</strong> Limited sources, inconsistent sentiment, low news volume</p>
                    <p><strong>Decision:</strong> Avoid trading, wait for better signals</p>
                </div>
            </div>
            
            <h2>📈 Sentiment Score Scale</h2>
            <div class="sentiment-scale">
                <div class="scale-item negative">
                    <div class="scale-value">-1.0 to -0.5</div>
                    <div class="scale-label">Very Negative</div>
                    <div class="scale-action">Strong Sell</div>
                </div>
                <div class="scale-item negative">
                    <div class="scale-value">-0.5 to -0.2</div>
                    <div class="scale-label">Negative</div>
                    <div class="scale-action">Sell</div>
                </div>
                <div class="scale-item neutral">
                    <div class="scale-value">-0.2 to +0.2</div>
                    <div class="scale-label">Neutral</div>
                    <div class="scale-action">Hold</div>
                </div>
                <div class="scale-item positive">
                    <div class="scale-value">+0.2 to +0.5</div>
                    <div class="scale-label">Positive</div>
                    <div class="scale-action">Buy</div>
                </div>
                <div class="scale-item positive">
                    <div class="scale-value">+0.5 to +1.0</div>
                    <div class="scale-label">Very Positive</div>
                    <div class="scale-action">Strong Buy</div>
                </div>
            </div>
        </div>
        """
    
    def generate_html_dashboard(self) -> str:
        """Generate complete HTML dashboard"""
        all_data = self.load_sentiment_data()
        
        # Generate sections
        overview_html = self.generate_overview_html(all_data)
        legend_html = self.generate_legend_html()
        
        banks_html = ""
        for symbol in self.bank_symbols:
            banks_html += self.generate_bank_card_html(symbol, all_data.get(symbol, []))
        
        # Complete HTML with CSS
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>📰 ASX Bank News Analysis Dashboard</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background-color: #f5f6fa;
                    color: #2c3e50;
                    line-height: 1.6;
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                
                .header {{
                    text-align: center;
                    margin-bottom: 40px;
                    background: white;
                    padding: 30px;
                    border-radius: 15px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                
                .header h1 {{
                    font-size: 2.5rem;
                    color: #1f77b4;
                    margin-bottom: 10px;
                }}
                
                .header p {{
                    font-size: 1.1rem;
                    color: #7f8c8d;
                }}
                
                .overview {{
                    background: white;
                    padding: 30px;
                    border-radius: 15px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                }}
                
                .overview h2 {{
                    margin-bottom: 20px;
                    color: #2c3e50;
                }}
                
                .overview-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                    gap: 20px;
                }}
                
                .overview-item {{
                    text-align: center;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 10px;
                }}
                
                .overview-value {{
                    font-size: 2rem;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                
                .overview-label {{
                    font-size: 0.9rem;
                    color: #6c757d;
                }}
                
                .legend {{
                    background: white;
                    padding: 30px;
                    border-radius: 15px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                }}
                
                .legend h2 {{
                    margin-bottom: 20px;
                    color: #2c3e50;
                }}
                
                .legend-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                
                .legend-item {{
                    padding: 20px;
                    border-radius: 10px;
                    border: 2px solid #e9ecef;
                }}
                
                .legend-item h3 {{
                    margin-bottom: 15px;
                }}
                
                .legend-item p {{
                    margin-bottom: 8px;
                }}
                
                .sentiment-scale {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                    gap: 15px;
                }}
                
                .scale-item {{
                    text-align: center;
                    padding: 15px;
                    border-radius: 8px;
                    border: 2px solid #e9ecef;
                }}
                
                .scale-value {{
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                
                .scale-label {{
                    font-size: 0.9rem;
                    margin-bottom: 5px;
                }}
                
                .scale-action {{
                    font-size: 0.8rem;
                    font-weight: bold;
                }}
                
                .bank-card {{
                    background: white;
                    margin-bottom: 30px;
                    border-radius: 15px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                
                .bank-header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 25px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                
                .bank-header h2 {{
                    margin: 0;
                }}
                
                .trading-signal {{
                    background: rgba(255,255,255,0.2);
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-weight: bold;
                    font-size: 0.9rem;
                }}
                
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    padding: 25px;
                    background: #f8f9fa;
                }}
                
                .metric-item {{
                    text-align: center;
                }}
                
                .metric-label {{
                    font-size: 0.9rem;
                    color: #6c757d;
                    margin-bottom: 5px;
                }}
                
                .metric-value {{
                    font-size: 1.3rem;
                    font-weight: bold;
                }}
                
                .components-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 15px;
                    padding: 20px;
                }}
                
                .component-item {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 10px;
                    background: #f8f9fa;
                    border-radius: 8px;
                }}
                
                .component-label {{
                    font-size: 0.9rem;
                    color: #6c757d;
                }}
                
                .component-score {{
                    font-weight: bold;
                }}
                
                .event-item {{
                    margin: 10px 0;
                    padding: 15px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    border-left: 4px solid #6c757d;
                }}
                
                .event-type {{
                    font-weight: bold;
                    color: #495057;
                    font-size: 0.9rem;
                }}
                
                .event-headline {{
                    margin: 8px 0;
                    font-weight: 500;
                }}
                
                .event-impact {{
                    font-size: 0.8rem;
                    font-weight: bold;
                }}
                
                .no-data {{
                    text-align: center;
                    color: #6c757d;
                    font-style: italic;
                    padding: 40px;
                }}
                
                /* Color classes */
                .positive {{ color: #28a745; }}
                .negative {{ color: #dc3545; }}
                .neutral {{ color: #6c757d; }}
                
                .confidence-high {{ 
                    background-color: #d4edda; 
                    border-color: #c3e6cb;
                }}
                
                .confidence-medium {{ 
                    background-color: #fff3cd; 
                    border-color: #ffeaa7;
                }}
                
                .confidence-low {{ 
                    background-color: #f8d7da; 
                    border-color: #f5c6cb;
                }}
                
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding: 20px;
                    color: #6c757d;
                    border-top: 1px solid #e9ecef;
                }}
                
                /* Responsive design */
                @media (max-width: 768px) {{
                    .container {{
                        padding: 10px;
                    }}
                    
                    .header h1 {{
                        font-size: 2rem;
                    }}
                    
                    .bank-header {{
                        flex-direction: column;
                        text-align: center;
                    }}
                    
                    .bank-header h2 {{
                        margin-bottom: 10px;
                    }}
                    
                    .metrics-grid,
                    .overview-grid,
                    .legend-grid {{
                        grid-template-columns: 1fr;
                    }}
                }}
                
                /* Print styles */
                @media print {{
                    .container {{
                        max-width: none;
                        padding: 0;
                    }}
                    
                    .bank-card {{
                        page-break-inside: avoid;
                        margin-bottom: 20px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📰 ASX Bank News Analysis Dashboard</h1>
                    <p>Real-time sentiment analysis and trading signals for Australian major banks</p>
                    <p><small>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
                </div>
                
                {overview_html}
                {legend_html}
                
                <div class="banks-section">
                    <h2 style="margin-bottom: 20px;">🏦 Individual Bank Analysis</h2>
                    {banks_html}
                </div>
                
                <div class="footer">
                    <p><strong>ASX Bank News Analysis System</strong></p>
                    <p>This dashboard displays sentiment analysis results from news articles, Reddit posts, and significant events.</p>
                    <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def save_and_open_dashboard(self):
        """Generate HTML dashboard and open in browser"""
        html_content = self.generate_html_dashboard()
        
        # Save to file
        output_file = "news_analysis_dashboard.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Dashboard generated: {output_file}")
        
        # Open in browser
        file_path = os.path.abspath(output_file)
        webbrowser.open(f'file://{file_path}')
        
        return output_file

def main():
    """Generate and display dashboard"""
    print("📰 Generating News Analysis Dashboard...")
    
    dashboard = SimpleHTMLDashboard()
    output_file = dashboard.save_and_open_dashboard()
    
    print(f"🌐 Dashboard is available at: file://{os.path.abspath(output_file)}")
    print("🔄 Run this script again to refresh the dashboard with latest data")

if __name__ == "__main__":
    main()
