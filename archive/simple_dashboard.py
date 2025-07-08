# simple_dashboard.py
"""
Simple Flask-based dashboard for the ASX Bank Trading System
Lightweight alternative to Streamlit for real-time monitoring
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.trading_orchestrator import TradingOrchestrator
from src.async_main import ASXBankTradingSystemAsync

app = Flask(__name__)

# Global state
orchestrator = None
is_running = False
last_analysis = {}

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get system status"""
    global orchestrator, is_running
    
    if orchestrator:
        portfolio = orchestrator.get_portfolio_summary()
        return jsonify({
            'status': 'running' if is_running else 'stopped',
            'portfolio': portfolio,
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({
            'status': 'not_initialized',
            'portfolio': {
                'account_balance': 100000.0,
                'total_value': 100000.0,
                'cash_available': 100000.0,
                'total_return': 0.0,
                'num_positions': 0,
                'unrealized_pnl': 0.0,
                'positions': [],
                'recent_trades': [],
                'daily_pnl': []
            },
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/analysis')
def get_analysis():
    """Get latest analysis results"""
    global last_analysis
    
    if not last_analysis:
        # Run a quick analysis
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def run_analysis():
                async with ASXBankTradingSystemAsync() as system:
                    return await system.analyze_all_banks_async()
            
            last_analysis = loop.run_until_complete(run_analysis())
            loop.close()
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify(last_analysis)

@app.route('/api/charts/portfolio')
def get_portfolio_chart():
    """Get portfolio performance chart data"""
    global orchestrator
    
    if not orchestrator:
        return jsonify({'error': 'Trading orchestrator not initialized'})
    
    portfolio = orchestrator.get_portfolio_summary()
    daily_pnl = portfolio.get('daily_pnl', [])
    
    if not daily_pnl:
        return jsonify({'error': 'No portfolio data available'})
    
    # Create portfolio chart
    dates = [pd.to_datetime(d['date']) for d in daily_pnl]
    values = [d['total_value'] for d in daily_pnl]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines+markers',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=3)
    ))
    
    fig.add_hline(y=100000, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Value ($)"
    )
    
    return json.dumps(fig, cls=PlotlyJSONEncoder)

@app.route('/api/charts/signals')
def get_signals_chart():
    """Get current trading signals chart"""
    global last_analysis
    
    if not last_analysis:
        return jsonify({'error': 'No analysis data available'})
    
    symbols = []
    confidences = []
    directions = []
    colors = []
    
    for symbol, analysis in last_analysis.items():
        if 'prediction' in analysis:
            pred = analysis['prediction']
            symbols.append(symbol)
            confidences.append(pred.get('confidence', 0))
            direction = pred.get('direction', 'neutral')
            directions.append(direction)
            
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
            text=directions,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Current Trading Signals",
        xaxis_title="Symbol",
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1])
    )
    
    return json.dumps(fig, cls=PlotlyJSONEncoder)

@app.route('/api/control/start', methods=['POST'])
def start_trading():
    """Start the trading system"""
    global orchestrator, is_running
    
    try:
        if not orchestrator:
            orchestrator = TradingOrchestrator(paper_trading=True)
        
        is_running = True
        return jsonify({'status': 'started', 'message': 'Trading system started'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/control/stop', methods=['POST'])
def stop_trading():
    """Stop the trading system"""
    global is_running
    
    try:
        is_running = False
        return jsonify({'status': 'stopped', 'message': 'Trading system stopped'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/refresh', methods=['POST'])
def refresh_analysis():
    """Refresh the analysis data"""
    global last_analysis
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_analysis():
            async with ASXBankTradingSystemAsync() as system:
                return await system.analyze_all_banks_async()
        
        last_analysis = loop.run_until_complete(run_analysis())
        loop.close()
        
        return jsonify({'status': 'success', 'message': 'Analysis refreshed'})
    except Exception as e:
        return jsonify({'error': str(e)})

# Create template directory and HTML file
def create_template():
    """Create the HTML template for the dashboard"""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASX Bank Trading System Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .header { text-align: center; color: #1f77b4; margin-bottom: 30px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; color: #1f77b4; }
        .metric-label { color: #666; font-size: 0.9em; }
        .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }
        .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .controls { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .btn { padding: 10px 20px; margin: 5px; border: none; border-radius: 4px; cursor: pointer; }
        .btn-primary { background-color: #007bff; color: white; }
        .btn-success { background-color: #28a745; color: white; }
        .btn-danger { background-color: #dc3545; color: white; }
        .btn-info { background-color: #17a2b8; color: white; }
        .status { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .status-running { background-color: #d4edda; color: #155724; }
        .status-stopped { background-color: #f8d7da; color: #721c24; }
        .table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .table th, .table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        .table th { background-color: #f8f9fa; }
        .footer { text-align: center; color: #666; margin-top: 30px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏛️ ASX Bank Trading System Dashboard</h1>
            <p>Real-time monitoring and control interface</p>
        </div>
        
        <div class="controls">
            <h3>Control Panel</h3>
            <button class="btn btn-success" onclick="startTrading()">▶️ Start Trading</button>
            <button class="btn btn-danger" onclick="stopTrading()">⏹️ Stop Trading</button>
            <button class="btn btn-info" onclick="refreshAnalysis()">🔄 Refresh Analysis</button>
            <button class="btn btn-primary" onclick="refreshDashboard()">📊 Refresh Dashboard</button>
            <div id="status" class="status"></div>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value" id="portfolio-value">$100,000</div>
                <div class="metric-label">Portfolio Value</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="cash-available">$100,000</div>
                <div class="metric-label">Cash Available</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="total-return">0.00%</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="positions">0</div>
                <div class="metric-label">Open Positions</div>
            </div>
        </div>
        
        <div class="charts">
            <div class="chart-container">
                <div id="portfolio-chart"></div>
            </div>
            <div class="chart-container">
                <div id="signals-chart"></div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Current Analysis</h3>
            <table class="table" id="analysis-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Price</th>
                        <th>Direction</th>
                        <th>Confidence</th>
                        <th>Risk Level</th>
                    </tr>
                </thead>
                <tbody id="analysis-tbody">
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Last updated: <span id="last-updated"></span> | ASX Bank Trading System v2.0</p>
        </div>
    </div>

    <script>
        // Update dashboard data
        function updateDashboard() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    const portfolio = data.portfolio;
                    
                    // Update metrics
                    document.getElementById('portfolio-value').textContent = '$' + portfolio.total_value.toLocaleString();
                    document.getElementById('cash-available').textContent = '$' + portfolio.cash_available.toLocaleString();
                    document.getElementById('total-return').textContent = (portfolio.total_return * 100).toFixed(2) + '%';
                    document.getElementById('positions').textContent = portfolio.num_positions;
                    
                    // Update status
                    const statusDiv = document.getElementById('status');
                    statusDiv.textContent = 'Status: ' + data.status.toUpperCase();
                    statusDiv.className = 'status ' + (data.status === 'running' ? 'status-running' : 'status-stopped');
                    
                    // Update timestamp
                    document.getElementById('last-updated').textContent = new Date().toLocaleString();
                });
            
            // Update analysis table
            fetch('/api/analysis')
                .then(response => response.json())
                .then(data => {
                    const tbody = document.getElementById('analysis-tbody');
                    tbody.innerHTML = '';
                    
                    for (const [symbol, analysis] of Object.entries(data)) {
                        if (analysis.prediction) {
                            const row = tbody.insertRow();
                            row.insertCell(0).textContent = symbol;
                            row.insertCell(1).textContent = '$' + (analysis.current_price || 0).toFixed(2);
                            row.insertCell(2).textContent = analysis.prediction.direction || 'neutral';
                            row.insertCell(3).textContent = ((analysis.prediction.confidence || 0) * 100).toFixed(1) + '%';
                            row.insertCell(4).textContent = analysis.risk_level || 'unknown';
                        }
                    }
                });
        }
        
        // Control functions
        function startTrading() {
            fetch('/api/control/start', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    alert(data.message || 'Trading started');
                    updateDashboard();
                });
        }
        
        function stopTrading() {
            fetch('/api/control/stop', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    alert(data.message || 'Trading stopped');
                    updateDashboard();
                });
        }
        
        function refreshAnalysis() {
            fetch('/api/refresh', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    alert(data.message || 'Analysis refreshed');
                    updateDashboard();
                });
        }
        
        function refreshDashboard() {
            updateDashboard();
        }
        
        // Initialize dashboard
        updateDashboard();
        
        // Auto-refresh every 30 seconds
        setInterval(updateDashboard, 30000);
    </script>
</body>
</html>
    """
    
    with open(os.path.join(template_dir, 'dashboard.html'), 'w') as f:
        f.write(html_content)

if __name__ == '__main__':
    create_template()
    print("🌐 Starting ASX Trading Dashboard...")
    print("📊 Access at: http://localhost:8501")
    app.run(host='0.0.0.0', port=8501, debug=True)
