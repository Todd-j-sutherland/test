#!/usr/bin/env python3
"""
Trading System Performance Analysis Script
==========================================

This script analyzes your ML trading system's performance patterns and generates
a comprehensive report. Run this periodically to track improvements over time.

Usage:
    python analyze_trading_patterns.py

The script will:
1. Analyze sentiment vs return correlations
2. Calculate success rates by various factors
3. Identify temporal patterns
4. Generate risk-adjusted metrics
5. Create a timestamped report for comparison

Author: ML Trading System
Date: July 2025
"""

import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

class TradingPatternAnalyzer:
    def __init__(self, db_path='data/ml_models/training_data.db'):
        self.db_path = db_path
        self.analysis_date = datetime.now()
        self.report_data = {}
        
    def load_data(self):
        """Load sentiment and trading outcome data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load sentiment features
            self.sentiment_df = pd.read_sql_query(
                'SELECT * FROM sentiment_features ORDER BY timestamp', conn
            )
            
            # Load trading outcomes
            self.outcomes_df = pd.read_sql_query(
                'SELECT * FROM trading_outcomes ORDER BY signal_timestamp', conn
            )
            
            # Merge datasets
            self.merged_df = pd.merge(
                self.outcomes_df, self.sentiment_df, 
                left_on='feature_id', right_on='id', 
                how='inner', suffixes=('_outcome', '_sentiment')
            )
            
            conn.close()
            
            print(f"✅ Loaded {len(self.sentiment_df)} sentiment records")
            print(f"✅ Loaded {len(self.outcomes_df)} trading outcomes")
            print(f"✅ Merged dataset: {len(self.merged_df)} records")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def analyze_basic_metrics(self):
        """Calculate basic performance metrics"""
        if self.merged_df.empty:
            return
        
        # Basic statistics
        total_trades = len(self.merged_df)
        overall_success_rate = self.merged_df['outcome_label'].mean()
        avg_return = self.merged_df['return_pct'].mean()
        return_std = self.merged_df['return_pct'].std()
        
        # Sentiment-return correlation
        sentiment_return_corr = self.merged_df['sentiment_score'].corr(
            self.merged_df['return_pct']
        )
        
        # Win rate correlation
        sentiment_success_corr = self.merged_df['sentiment_score'].corr(
            self.merged_df['outcome_label']
        )
        
        self.report_data['basic_metrics'] = {
            'analysis_date': self.analysis_date.isoformat(),
            'total_trades': int(total_trades),
            'overall_success_rate': float(overall_success_rate),
            'average_return_pct': float(avg_return),
            'return_volatility': float(return_std),
            'sentiment_return_correlation': float(sentiment_return_corr),
            'sentiment_success_correlation': float(sentiment_success_corr),
            'sharpe_ratio': float(avg_return / return_std) if return_std > 0 else 0
        }
    
    def analyze_sentiment_patterns(self):
        """Analyze performance by sentiment levels"""
        if self.merged_df.empty:
            return
        
        # Sentiment quintile analysis
        self.merged_df['sentiment_quintile'] = pd.qcut(
            self.merged_df['sentiment_score'], 
            q=5, 
            labels=['Q1_Lowest', 'Q2_Low', 'Q3_Medium', 'Q4_High', 'Q5_Highest']
        )
        
        quintile_stats = self.merged_df.groupby('sentiment_quintile').agg({
            'outcome_label': ['count', 'mean'],
            'return_pct': ['mean', 'std']
        }).round(4)
        
        # Convert to dictionary format
        sentiment_analysis = {}
        for quintile in quintile_stats.index:
            sentiment_analysis[quintile] = {
                'trade_count': int(quintile_stats.loc[quintile, ('outcome_label', 'count')]),
                'success_rate': float(quintile_stats.loc[quintile, ('outcome_label', 'mean')]),
                'avg_return': float(quintile_stats.loc[quintile, ('return_pct', 'mean')]),
                'return_std': float(quintile_stats.loc[quintile, ('return_pct', 'std')])
            }
        
        # Positive vs negative sentiment
        positive_sentiment = self.merged_df[self.merged_df['sentiment_score'] > 0.1]
        negative_sentiment = self.merged_df[self.merged_df['sentiment_score'] < -0.1]
        neutral_sentiment = self.merged_df[
            (self.merged_df['sentiment_score'] >= -0.1) & 
            (self.merged_df['sentiment_score'] <= 0.1)
        ]
        
        self.report_data['sentiment_patterns'] = {
            'quintile_analysis': sentiment_analysis,
            'category_analysis': {
                'positive': {
                    'count': len(positive_sentiment),
                    'success_rate': float(positive_sentiment['outcome_label'].mean()) if len(positive_sentiment) > 0 else 0,
                    'avg_return': float(positive_sentiment['return_pct'].mean()) if len(positive_sentiment) > 0 else 0
                },
                'negative': {
                    'count': len(negative_sentiment),
                    'success_rate': float(negative_sentiment['outcome_label'].mean()) if len(negative_sentiment) > 0 else 0,
                    'avg_return': float(negative_sentiment['return_pct'].mean()) if len(negative_sentiment) > 0 else 0
                },
                'neutral': {
                    'count': len(neutral_sentiment),
                    'success_rate': float(neutral_sentiment['outcome_label'].mean()) if len(neutral_sentiment) > 0 else 0,
                    'avg_return': float(neutral_sentiment['return_pct'].mean()) if len(neutral_sentiment) > 0 else 0
                }
            }
        }
    
    def analyze_confidence_patterns(self):
        """Analyze performance by confidence levels"""
        if self.merged_df.empty:
            return
        
        confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        confidence_analysis = {}
        
        for threshold in confidence_thresholds:
            high_conf_data = self.merged_df[self.merged_df['confidence'] >= threshold]
            if len(high_conf_data) > 0:
                confidence_analysis[f'threshold_{threshold}'] = {
                    'trade_count': len(high_conf_data),
                    'success_rate': float(high_conf_data['outcome_label'].mean()),
                    'avg_return': float(high_conf_data['return_pct'].mean()),
                    'return_std': float(high_conf_data['return_pct'].std())
                }
        
        self.report_data['confidence_patterns'] = confidence_analysis
    
    def analyze_bank_performance(self):
        """Analyze performance by individual banks"""
        if self.merged_df.empty:
            return
        
        bank_analysis = {}
        for symbol in self.merged_df['symbol_outcome'].unique():
            symbol_data = self.merged_df[self.merged_df['symbol_outcome'] == symbol]
            
            if len(symbol_data) >= 3:  # Only analyze banks with 3+ trades
                avg_return = symbol_data['return_pct'].mean()
                std_return = symbol_data['return_pct'].std()
                
                bank_analysis[symbol] = {
                    'trade_count': len(symbol_data),
                    'success_rate': float(symbol_data['outcome_label'].mean()),
                    'avg_return': float(avg_return),
                    'return_std': float(std_return),
                    'sharpe_ratio': float(avg_return / std_return) if std_return > 0 else 0,
                    'max_gain': float(symbol_data['return_pct'].max()),
                    'max_loss': float(symbol_data['return_pct'].min()),
                    'avg_sentiment': float(symbol_data['sentiment_score'].mean()),
                    'avg_confidence': float(symbol_data['confidence'].mean())
                }
        
        self.report_data['bank_performance'] = bank_analysis
    
    def analyze_temporal_patterns(self):
        """Analyze time-based trading patterns"""
        if self.merged_df.empty:
            return
        
        # Add time features
        self.merged_df['signal_datetime'] = pd.to_datetime(self.merged_df['signal_timestamp'])
        self.merged_df['hour'] = self.merged_df['signal_datetime'].dt.hour
        self.merged_df['day_of_week'] = self.merged_df['signal_datetime'].dt.day_name()
        
        # Hour analysis
        hourly_stats = self.merged_df.groupby('hour').agg({
            'outcome_label': ['count', 'mean'],
            'return_pct': 'mean'
        }).round(4)
        
        hour_analysis = {}
        for hour in hourly_stats.index:
            if hourly_stats.loc[hour, ('outcome_label', 'count')] >= 3:  # 3+ trades
                hour_analysis[f'hour_{hour}'] = {
                    'trade_count': int(hourly_stats.loc[hour, ('outcome_label', 'count')]),
                    'success_rate': float(hourly_stats.loc[hour, ('outcome_label', 'mean')]),
                    'avg_return': float(hourly_stats.loc[hour, ('return_pct', 'mean')])
                }
        
        # Day of week analysis
        daily_stats = self.merged_df.groupby('day_of_week').agg({
            'outcome_label': ['count', 'mean'],
            'return_pct': 'mean'
        }).round(4)
        
        day_analysis = {}
        for day in daily_stats.index:
            day_analysis[day] = {
                'trade_count': int(daily_stats.loc[day, ('outcome_label', 'count')]),
                'success_rate': float(daily_stats.loc[day, ('outcome_label', 'mean')]),
                'avg_return': float(daily_stats.loc[day, ('return_pct', 'mean')])
            }
        
        self.report_data['temporal_patterns'] = {
            'hourly_analysis': hour_analysis,
            'daily_analysis': day_analysis
        }
    
    def analyze_recent_trends(self):
        """Analyze recent performance trends"""
        if self.merged_df.empty:
            return
        
        # Sort by timestamp
        sorted_df = self.merged_df.sort_values('signal_timestamp')
        
        # Recent performance (last 30 trades)
        recent_30 = sorted_df.tail(30)
        recent_10 = sorted_df.tail(10)
        
        # Calculate moving averages
        sorted_df['success_ma_10'] = sorted_df['outcome_label'].rolling(window=10).mean()
        sorted_df['return_ma_10'] = sorted_df['return_pct'].rolling(window=10).mean()
        
        self.report_data['recent_trends'] = {
            'last_30_trades': {
                'success_rate': float(recent_30['outcome_label'].mean()),
                'avg_return': float(recent_30['return_pct'].mean()),
                'avg_sentiment': float(recent_30['sentiment_score'].mean())
            },
            'last_10_trades': {
                'success_rate': float(recent_10['outcome_label'].mean()),
                'avg_return': float(recent_10['return_pct'].mean()),
                'avg_sentiment': float(recent_10['sentiment_score'].mean())
            },
            'trend_analysis': {
                'current_success_ma': float(sorted_df['success_ma_10'].iloc[-1]) if len(sorted_df) >= 10 else None,
                'current_return_ma': float(sorted_df['return_ma_10'].iloc[-1]) if len(sorted_df) >= 10 else None
            }
        }
    
    def generate_recommendations(self):
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Sentiment recommendations
        if 'sentiment_patterns' in self.report_data:
            sentiment_data = self.report_data['sentiment_patterns']['category_analysis']
            
            if sentiment_data['positive']['success_rate'] > 0.5:
                recommendations.append("✅ Continue focusing on positive sentiment signals")
            
            if sentiment_data['negative']['success_rate'] < 0.3:
                recommendations.append("⚠️ Avoid negative sentiment trades")
        
        # Confidence recommendations
        if 'confidence_patterns' in self.report_data:
            conf_data = self.report_data['confidence_patterns']
            
            best_threshold = None
            best_success = 0
            for threshold, data in conf_data.items():
                if data['success_rate'] > best_success and data['trade_count'] >= 10:
                    best_success = data['success_rate']
                    best_threshold = threshold
            
            if best_threshold:
                recommendations.append(f"🎯 Optimal confidence threshold: {best_threshold}")
        
        # Bank recommendations
        if 'bank_performance' in self.report_data:
            bank_data = self.report_data['bank_performance']
            
            best_banks = []
            for bank, data in bank_data.items():
                if data['success_rate'] > 0.4 and data['trade_count'] >= 5:
                    best_banks.append(bank)
            
            if best_banks:
                recommendations.append(f"🏦 Focus on banks: {', '.join(best_banks)}")
        
        self.report_data['recommendations'] = recommendations
    
    def save_report(self):
        """Save the analysis report to file"""
        timestamp = self.analysis_date.strftime("%Y%m%d_%H%M%S")
        filename = f"reports/pattern_analysis_{timestamp}.json"
        
        # Ensure reports directory exists
        os.makedirs('reports', exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.report_data, f, indent=2, default=str)
        
        return filename
    
    def print_summary(self):
        """Print a human-readable summary"""
        print("\n" + "="*60)
        print("🎯 TRADING SYSTEM PATTERN ANALYSIS SUMMARY")
        print("="*60)
        
        # Basic metrics
        if 'basic_metrics' in self.report_data:
            metrics = self.report_data['basic_metrics']
            print(f"\n📊 BASIC METRICS:")
            print(f"   Total Trades: {metrics['total_trades']}")
            print(f"   Success Rate: {metrics['overall_success_rate']:.1%}")
            print(f"   Average Return: {metrics['average_return_pct']:.3f}%")
            print(f"   Sentiment-Return Correlation: {metrics['sentiment_return_correlation']:.3f}")
            print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        
        # Best performing sentiment category
        if 'sentiment_patterns' in self.report_data:
            sentiment_cat = self.report_data['sentiment_patterns']['category_analysis']
            print(f"\n🎯 SENTIMENT PERFORMANCE:")
            print(f"   Positive: {sentiment_cat['positive']['success_rate']:.1%} success, {sentiment_cat['positive']['avg_return']:.3f}% return")
            print(f"   Neutral:  {sentiment_cat['neutral']['success_rate']:.1%} success, {sentiment_cat['neutral']['avg_return']:.3f}% return")
            print(f"   Negative: {sentiment_cat['negative']['success_rate']:.1%} success, {sentiment_cat['negative']['avg_return']:.3f}% return")
        
        # Top performing banks
        if 'bank_performance' in self.report_data:
            banks = self.report_data['bank_performance']
            print(f"\n🏦 BANK PERFORMANCE:")
            for bank, data in sorted(banks.items(), key=lambda x: x[1]['success_rate'], reverse=True):
                print(f"   {bank}: {data['success_rate']:.1%} success, {data['avg_return']:.3f}% return ({data['trade_count']} trades)")
        
        # Recommendations
        if 'recommendations' in self.report_data:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in self.report_data['recommendations']:
                print(f"   {rec}")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("🔍 Starting Trading Pattern Analysis...")
        
        if not self.load_data():
            return None
        
        print("📊 Analyzing basic metrics...")
        self.analyze_basic_metrics()
        
        print("🎯 Analyzing sentiment patterns...")
        self.analyze_sentiment_patterns()
        
        print("🔍 Analyzing confidence patterns...")
        self.analyze_confidence_patterns()
        
        print("🏦 Analyzing bank performance...")
        self.analyze_bank_performance()
        
        print("📅 Analyzing temporal patterns...")
        self.analyze_temporal_patterns()
        
        print("📈 Analyzing recent trends...")
        self.analyze_recent_trends()
        
        print("💡 Generating recommendations...")
        self.generate_recommendations()
        
        print("💾 Saving report...")
        report_file = self.save_report()
        
        self.print_summary()
        
        print(f"\n✅ Analysis complete! Report saved to: {report_file}")
        print(f"📅 Analysis date: {self.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return report_file

def compare_reports(current_file, previous_file):
    """Compare current analysis with a previous report"""
    try:
        with open(current_file, 'r') as f:
            current = json.load(f)
        with open(previous_file, 'r') as f:
            previous = json.load(f)
        
        print("\n" + "="*60)
        print("📈 PERFORMANCE COMPARISON")
        print("="*60)
        
        # Compare basic metrics
        curr_metrics = current['basic_metrics']
        prev_metrics = previous['basic_metrics']
        
        print(f"\n📊 METRICS COMPARISON:")
        print(f"   Total Trades: {prev_metrics['total_trades']} → {curr_metrics['total_trades']} ({curr_metrics['total_trades'] - prev_metrics['total_trades']:+d})")
        print(f"   Success Rate: {prev_metrics['overall_success_rate']:.1%} → {curr_metrics['overall_success_rate']:.1%} ({curr_metrics['overall_success_rate'] - prev_metrics['overall_success_rate']:+.1%})")
        print(f"   Avg Return: {prev_metrics['average_return_pct']:.3f}% → {curr_metrics['average_return_pct']:.3f}% ({curr_metrics['average_return_pct'] - prev_metrics['average_return_pct']:+.3f}%)")
        print(f"   Correlation: {prev_metrics['sentiment_return_correlation']:.3f} → {curr_metrics['sentiment_return_correlation']:.3f} ({curr_metrics['sentiment_return_correlation'] - prev_metrics['sentiment_return_correlation']:+.3f})")
        
    except Exception as e:
        print(f"❌ Error comparing reports: {e}")

if __name__ == "__main__":
    analyzer = TradingPatternAnalyzer()
    report_file = analyzer.run_full_analysis()
    
    # Ask if user wants to compare with previous report
    print(f"\n🔄 To compare with a previous analysis, run:")
    print(f"python -c \"from analyze_trading_patterns import compare_reports; compare_reports('{report_file}', 'reports/pattern_analysis_PREVIOUS.json')\"")
