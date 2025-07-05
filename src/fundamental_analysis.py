# src/fundamental_analysis.py
"""
Fundamental analysis module for evaluating company financials
Focuses on bank-specific metrics and valuation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import requests
from bs4 import BeautifulSoup

# Import settings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Settings
from utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class FundamentalAnalyzer:
    """Analyzes fundamental metrics for banks"""
    
    def __init__(self):
        self.settings = Settings()
        self.cache = CacheManager()
        self.session = requests.Session()
        
        # Bank-specific metric thresholds
        self.bank_benchmarks = {
            'pe_ratio': {'excellent': 12, 'good': 15, 'fair': 18, 'poor': 20},
            'roe': {'excellent': 0.15, 'good': 0.12, 'fair': 0.10, 'poor': 0.08},
            'tier1_ratio': {'excellent': 0.14, 'good': 0.12, 'fair': 0.10, 'poor': 0.08},
            'npm': {'excellent': 0.30, 'good': 0.25, 'fair': 0.20, 'poor': 0.15},
            'dividend_yield': {'excellent': 0.06, 'good': 0.05, 'fair': 0.04, 'poor': 0.03}
        }
    
    def analyze(self, symbol: str) -> Dict:
        """Perform comprehensive fundamental analysis"""
        
        cache_key = f"fundamental_{symbol}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            # Fetch company data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get financial statements
            financials = self._get_financial_statements(ticker)
            
            # Calculate key metrics
            metrics = self._calculate_metrics(info, financials)
            
            # Peer comparison
            peer_analysis = self._analyze_peers(symbol, metrics)
            
            # Calculate valuation score
            valuation = self._calculate_valuation(metrics, peer_analysis)
            
            # Bank-specific analysis
            bank_metrics = self._analyze_bank_specifics(info, financials)
            
            # Growth analysis
            growth = self._analyze_growth(ticker, financials)
            
            # Calculate overall score
            score = self._calculate_fundamental_score(metrics, valuation, bank_metrics, growth)
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'valuation': valuation,
                'bank_specific': bank_metrics,
                'growth': growth,
                'peer_comparison': peer_analysis,
                'score': score,
                'rating': self._generate_rating(score),
                'strengths': self._identify_strengths(metrics, bank_metrics),
                'weaknesses': self._identify_weaknesses(metrics, bank_metrics)
            }
            
            # Cache for 4 hours
            self.cache.set(cache_key, result, expiry_minutes=240)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis for {symbol}: {str(e)}")
            return self._default_analysis()
    
    def _get_financial_statements(self, ticker) -> Dict:
        """Fetch and process financial statements"""
        
        try:
            # Get financial data
            income_stmt = ticker.quarterly_income_stmt
            balance_sheet = ticker.quarterly_balance_sheet
            cash_flow = ticker.quarterly_cashflow
            
            # Process into usable format
            financials = {
                'income_statement': income_stmt.to_dict() if not income_stmt.empty else {},
                'balance_sheet': balance_sheet.to_dict() if not balance_sheet.empty else {},
                'cash_flow': cash_flow.to_dict() if not cash_flow.empty else {},
                'available': not (income_stmt.empty and balance_sheet.empty)
            }
            
            return financials
            
        except Exception as e:
            logger.warning(f"Error fetching financial statements: {str(e)}")
            return {'available': False}
    
    def _calculate_metrics(self, info: Dict, financials: Dict) -> Dict:
        """Calculate key fundamental metrics"""
        
        metrics = {}
        
        # Valuation metrics
        metrics['pe_ratio'] = info.get('trailingPE', 0)
        metrics['forward_pe'] = info.get('forwardPE', 0)
        metrics['peg_ratio'] = info.get('pegRatio', 0)
        metrics['price_to_book'] = info.get('priceToBook', 0)
        metrics['enterprise_value'] = info.get('enterpriseValue', 0)
        
        # Profitability metrics
        metrics['profit_margin'] = info.get('profitMargins', 0)
        metrics['operating_margin'] = info.get('operatingMargins', 0)
        metrics['roe'] = info.get('returnOnEquity', 0)
        metrics['roa'] = info.get('returnOnAssets', 0)
        
        # Financial health
        metrics['current_ratio'] = info.get('currentRatio', 0)
        metrics['debt_to_equity'] = info.get('debtToEquity', 0)
        metrics['quick_ratio'] = info.get('quickRatio', 0)
        
        # Dividend metrics
        metrics['dividend_yield'] = info.get('dividendYield', 0)
        metrics['dividend_rate'] = info.get('dividendRate', 0)
        metrics['payout_ratio'] = info.get('payoutRatio', 0)
        
        # Per share metrics
        metrics['book_value_per_share'] = info.get('bookValue', 0)
        metrics['earnings_per_share'] = info.get('trailingEps', 0)
        
        # Market metrics
        metrics['market_cap'] = info.get('marketCap', 0)
        metrics['beta'] = info.get('beta', 1.0)
        
        return metrics
    
    def _analyze_bank_specifics(self, info: Dict, financials: Dict) -> Dict:
        """Analyze bank-specific metrics"""
        
        bank_metrics = {}
        
        # Net Interest Margin (NIM) - Critical for banks
        # This would require more detailed financial data
        bank_metrics['net_interest_margin'] = self._estimate_nim(info)
        
        # Asset Quality
        bank_metrics['npl_ratio'] = 0.02  # Placeholder - would need detailed data
        
        # Capital Adequacy (Tier 1 ratio)
        bank_metrics['tier1_ratio'] = 0.12  # Placeholder - regulatory filing data needed
        
        # Efficiency Ratio
        if info.get('operatingMargins') and info.get('profitMargins'):
            bank_metrics['efficiency_ratio'] = 1 - info['operatingMargins']
        else:
            bank_metrics['efficiency_ratio'] = 0.5
        
        # Loan to Deposit Ratio
        bank_metrics['loan_to_deposit'] = 0.85  # Placeholder
        
        # Book Value Growth
        bank_metrics['book_value_growth'] = info.get('bookValue', 0) / info.get('priceToBook', 1) if info.get('priceToBook', 0) > 0 else 0
        
        return bank_metrics
    
    def _estimate_nim(self, info: Dict) -> float:
        """Estimate Net Interest Margin"""
        
        # Simplified estimation based on available data
        # Australian banks typically have NIM between 1.5% - 2.5%
        
        # Use profit margin as proxy
        profit_margin = info.get('profitMargins', 0.25)
        
        # Rough estimation
        if profit_margin > 0.30:
            return 0.022  # 2.2%
        elif profit_margin > 0.25:
            return 0.020  # 2.0%
        elif profit_margin > 0.20:
            return 0.018  # 1.8%
        else:
            return 0.016  # 1.6%
    
    def _analyze_growth(self, ticker, financials: Dict) -> Dict:
        """Analyze growth metrics"""
        
        growth = {}
        
        try:
            # Get historical data for growth calculation
            hist = ticker.history(period="2y")
            
            if not hist.empty:
                # Price growth
                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                growth['price_growth_2y'] = ((end_price - start_price) / start_price) * 100
                
                # Dividend growth (if available)
                dividends = ticker.dividends
                if not dividends.empty:
                    recent_div = dividends.tail(4).sum()  # Last year
                    previous_div = dividends.iloc[-8:-4].sum() if len(dividends) >= 8 else 0
                    
                    if previous_div > 0:
                        growth['dividend_growth'] = ((recent_div - previous_div) / previous_div) * 100
                    else:
                        growth['dividend_growth'] = 0
                else:
                    growth['dividend_growth'] = 0
            
            # Analyst growth estimates
            info = ticker.info
            growth['earnings_growth'] = info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0
            growth['revenue_growth'] = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
            
        except Exception as e:
            logger.warning(f"Error calculating growth metrics: {str(e)}")
            growth = {
                'price_growth_2y': 0,
                'dividend_growth': 0,
                'earnings_growth': 0,
                'revenue_growth': 0
            }
        
        return growth
    
    def _analyze_peers(self, symbol: str, metrics: Dict) -> Dict:
        """Compare with peer banks"""
        
        peer_comparison = {
            'pe_vs_peers': 'inline',
            'roe_vs_peers': 'inline',
            'dividend_vs_peers': 'inline',
            'relative_value': 0
        }
        
        try:
            # Get peer metrics
            peer_metrics = []
            
            for peer_symbol in self.settings.BANK_SYMBOLS:
                if peer_symbol != symbol:
                    try:
                        peer_ticker = yf.Ticker(peer_symbol)
                        peer_info = peer_ticker.info
                        
                        peer_metrics.append({
                            'symbol': peer_symbol,
                            'pe': peer_info.get('trailingPE', 0),
                            'roe': peer_info.get('returnOnEquity', 0),
                            'dividend_yield': peer_info.get('dividendYield', 0)
                        })
                    except:
                        continue
            
            if peer_metrics:
                # Calculate averages
                avg_pe = sum(p['pe'] for p in peer_metrics if p['pe'] > 0) / len([p for p in peer_metrics if p['pe'] > 0])
                avg_roe = sum(p['roe'] for p in peer_metrics if p['roe'] > 0) / len([p for p in peer_metrics if p['roe'] > 0])
                avg_div = sum(p['dividend_yield'] for p in peer_metrics if p['dividend_yield'] > 0) / len([p for p in peer_metrics if p['dividend_yield'] > 0])
                
                # Compare
                if metrics['pe_ratio'] > 0 and avg_pe > 0:
                    if metrics['pe_ratio'] < avg_pe * 0.9:
                        peer_comparison['pe_vs_peers'] = 'undervalued'
                        peer_comparison['relative_value'] += 20
                    elif metrics['pe_ratio'] > avg_pe * 1.1:
                        peer_comparison['pe_vs_peers'] = 'overvalued'
                        peer_comparison['relative_value'] -= 20
                
                if metrics['roe'] > avg_roe * 1.1:
                    peer_comparison['roe_vs_peers'] = 'outperforming'
                    peer_comparison['relative_value'] += 15
                elif metrics['roe'] < avg_roe * 0.9:
                    peer_comparison['roe_vs_peers'] = 'underperforming'
                    peer_comparison['relative_value'] -= 15
                
                if metrics['dividend_yield'] > avg_div * 1.1:
                    peer_comparison['dividend_vs_peers'] = 'above_average'
                    peer_comparison['relative_value'] += 10
                elif metrics['dividend_yield'] < avg_div * 0.9:
                    peer_comparison['dividend_vs_peers'] = 'below_average'
                    peer_comparison['relative_value'] -= 10
                
        except Exception as e:
            logger.warning(f"Error in peer comparison: {str(e)}")
        
        return peer_comparison
    
    def _calculate_valuation(self, metrics: Dict, peer_analysis: Dict) -> Dict:
        """Calculate comprehensive valuation score"""
        
        valuation = {
            'score': 0,
            'rating': 'fair',
            'factors': []
        }
        
        # P/E valuation
        pe = metrics.get('pe_ratio', 15)
        if 0 < pe < 12:
            valuation['score'] += 25
            valuation['factors'].append('Attractive P/E ratio')
        elif pe > 18:
            valuation['score'] -= 15
            valuation['factors'].append('High P/E ratio')
        
        # P/B valuation
        pb = metrics.get('price_to_book', 1.5)
        if 0 < pb < 1.0:
            valuation['score'] += 20
            valuation['factors'].append('Trading below book value')
        elif pb > 2.0:
            valuation['score'] -= 10
            valuation['factors'].append('High price to book')
        
        # Dividend yield
        div_yield = metrics.get('dividend_yield', 0.05)
        if div_yield > 0.06:
            valuation['score'] += 15
            valuation['factors'].append('High dividend yield')
        elif div_yield < 0.04:
            valuation['score'] -= 10
            valuation['factors'].append('Low dividend yield')
        
        # Peer relative value
        valuation['score'] += peer_analysis.get('relative_value', 0)
        
        # Determine rating
        if valuation['score'] > 40:
            valuation['rating'] = 'undervalued'
        elif valuation['score'] > 20:
            valuation['rating'] = 'slightly_undervalued'
        elif valuation['score'] < -20:
            valuation['rating'] = 'overvalued'
        elif valuation['score'] < -10:
            valuation['rating'] = 'slightly_overvalued'
        else:
            valuation['rating'] = 'fair_value'
        
        return valuation
    
    def _calculate_fundamental_score(self, metrics: Dict, valuation: Dict, 
                                    bank_metrics: Dict, growth: Dict) -> float:
        """Calculate overall fundamental score (0-100)"""
        
        score = 50  # Start neutral
        
        # Valuation component (30%)
        score += valuation['score'] * 0.3
        
        # Profitability component (25%)
        roe = metrics.get('roe', 0.12)
        if roe > 0.15:
            score += 10
        elif roe > 0.12:
            score += 5
        elif roe < 0.08:
            score -= 10
        
        # Growth component (20%)
        earnings_growth = growth.get('earnings_growth', 0)
        if earnings_growth > 10:
            score += 10
        elif earnings_growth > 5:
            score += 5
        elif earnings_growth < -5:
            score -= 10
        
        # Financial health (15%)
        debt_to_equity = metrics.get('debt_to_equity', 1)
        if debt_to_equity < 0.5:
            score += 7.5
        elif debt_to_equity > 2:
            score -= 7.5
        
        # Bank specific (10%)
        nim = bank_metrics.get('net_interest_margin', 0.02)
        if nim > 0.022:
            score += 5
        elif nim < 0.018:
            score -= 5
        
        return max(0, min(100, score))
    
    def _generate_rating(self, score: float) -> str:
        """Generate fundamental rating"""
        
        if score >= 80:
            return 'STRONG BUY'
        elif score >= 65:
            return 'BUY'
        elif score >= 35:
            return 'HOLD'
        elif score >= 20:
            return 'SELL'
        else:
            return 'STRONG SELL'
    
    def _identify_strengths(self, metrics: Dict, bank_metrics: Dict) -> List[str]:
        """Identify fundamental strengths"""
        
        strengths = []
        
        if metrics.get('roe', 0) > 0.15:
            strengths.append('Excellent return on equity')
        
        if metrics.get('dividend_yield', 0) > 0.06:
            strengths.append('Attractive dividend yield')
        
        if metrics.get('pe_ratio', 15) < 12 and metrics.get('pe_ratio', 0) > 0:
            strengths.append('Low valuation multiple')
        
        if bank_metrics.get('net_interest_margin', 0) > 0.022:
            strengths.append('Strong net interest margin')
        
        if metrics.get('debt_to_equity', 1) < 0.5:
            strengths.append('Conservative capital structure')
        
        return strengths
    
    def _identify_weaknesses(self, metrics: Dict, bank_metrics: Dict) -> List[str]:
        """Identify fundamental weaknesses"""
        
        weaknesses = []
        
        if metrics.get('roe', 0.12) < 0.08:
            weaknesses.append('Low return on equity')
        
        if metrics.get('pe_ratio', 0) > 20:
            weaknesses.append('High valuation')
        
        if bank_metrics.get('efficiency_ratio', 0.5) > 0.6:
            weaknesses.append('High cost structure')
        
        if metrics.get('payout_ratio', 0.5) > 0.8:
            weaknesses.append('High dividend payout ratio')
        
        return weaknesses
    
    def get_economic_indicators(self) -> Dict:
        """Get key economic indicators affecting banks"""
        
        # In production, these would be fetched from RBA, ABS, etc.
        return {
            'cash_rate': 4.35,  # RBA cash rate
            'inflation': 3.8,   # CPI
            'unemployment': 3.9,  # Unemployment rate
            'gdp_growth': 2.1,  # GDP growth
            'housing_growth': 5.2,  # Property price growth
            'business_confidence': 5,  # NAB business confidence
            'consumer_confidence': 85.2  # Consumer confidence index
        }
    
    def _default_analysis(self) -> Dict:
        """Return default analysis when data unavailable"""
        
        return {
            'symbol': '',
            'metrics': {},
            'valuation': {'score': 0, 'rating': 'unknown'},
            'bank_specific': {},
            'growth': {},
            'peer_comparison': {},
            'score': 50,
            'rating': 'HOLD',
            'strengths': [],
            'weaknesses': ['Unable to fetch fundamental data']
        }