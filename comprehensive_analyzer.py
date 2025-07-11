#!/usr/bin/env python3
"""
Comprehensive Analyzer - Complete data quality and readiness assessment
Provides comprehensive analysis of ML data quality, feature importance, and system readiness
"""
import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from src.ml_training_pipeline import MLTrainingPipeline

class ComprehensiveAnalyzer:
    def __init__(self):
        self.ml_pipeline = MLTrainingPipeline()
        self.analysis_results = {}
        
    def analyze_data_quality(self) -> Dict:
        """Comprehensive data quality analysis"""
        print("🔍 Analyzing data quality...")
        
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'database_status': {},
            'data_distribution': {},
            'feature_quality': {},
            'sample_adequacy': {},
            'recommendations': []
        }
        
        try:
            # Database connectivity
            conn = sqlite3.connect(self.ml_pipeline.db_path)
            cursor = conn.cursor()
            
            # Check table existence and row counts
            tables = ['sentiment_features', 'trading_outcomes', 'model_performance']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                quality_report['database_status'][table] = count
            
            # Get sample data
            X, y = self.ml_pipeline.prepare_training_dataset(min_samples=1)
            
            if X is not None and len(X) > 0:
                quality_report['sample_adequacy'] = {
                    'total_samples': len(X),
                    'feature_count': len(X.columns),
                    'class_distribution': y.value_counts().to_dict(),
                    'missing_values': X.isnull().sum().to_dict(),
                    'duplicate_rows': X.duplicated().sum()
                }
                
                # Data distribution analysis
                quality_report['data_distribution'] = {
                    'sentiment_score_range': [float(X['sentiment_score'].min()), float(X['sentiment_score'].max())],
                    'confidence_range': [float(X['confidence'].min()), float(X['confidence'].max())],
                    'news_count_range': [int(X['news_count'].min()), int(X['news_count'].max())],
                    'class_balance_ratio': self.calculate_class_balance_ratio(y)
                }
                
                # Feature quality analysis
                quality_report['feature_quality'] = self.analyze_feature_quality(X, y)
                
            else:
                quality_report['sample_adequacy'] = {'status': 'No training data available'}
            
            conn.close()
            
        except Exception as e:
            quality_report['error'] = str(e)
        
        # Generate recommendations
        quality_report['recommendations'] = self.generate_data_quality_recommendations(quality_report)
        
        return quality_report
    
    def calculate_class_balance_ratio(self, y: pd.Series) -> float:
        """Calculate class balance ratio"""
        class_counts = y.value_counts()
        if len(class_counts) >= 2:
            return min(class_counts) / max(class_counts)
        return 1.0
    
    def analyze_feature_quality(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Analyze feature quality and importance"""
        feature_quality = {}
        
        for column in X.columns:
            if X[column].dtype in ['int64', 'float64']:
                feature_quality[column] = {
                    'type': 'numeric',
                    'mean': float(X[column].mean()),
                    'std': float(X[column].std()),
                    'missing_pct': float(X[column].isnull().sum() / len(X) * 100),
                    'unique_values': int(X[column].nunique()),
                    'correlation_with_target': float(X[column].corr(y)) if not X[column].isnull().all() else 0
                }
            else:
                feature_quality[column] = {
                    'type': 'categorical',
                    'unique_values': int(X[column].nunique()),
                    'missing_pct': float(X[column].isnull().sum() / len(X) * 100),
                    'most_frequent': X[column].mode().iloc[0] if not X[column].empty else None
                }
        
        return feature_quality
    
    def analyze_model_readiness(self) -> Dict:
        """Analyze ML model readiness"""
        print("🤖 Analyzing model readiness...")
        
        readiness_report = {
            'timestamp': datetime.now().isoformat(),
            'training_readiness': {},
            'model_performance': {},
            'deployment_readiness': {},
            'recommendations': []
        }
        
        try:
            # Check training data adequacy
            X, y = self.ml_pipeline.prepare_training_dataset(min_samples=1)
            
            if X is not None:
                sample_count = len(X)
                
                readiness_report['training_readiness'] = {
                    'sample_count': sample_count,
                    'min_samples_met': sample_count >= 100,
                    'class_balance_acceptable': self.calculate_class_balance_ratio(y) >= 0.3,
                    'feature_count': len(X.columns),
                    'data_completeness': (1 - X.isnull().sum().sum() / (len(X) * len(X.columns))) * 100
                }
                
                # Check for existing models
                model_version = self.ml_pipeline.get_latest_model_version()
                if model_version:
                    readiness_report['model_performance'] = self.get_model_performance_metrics(model_version)
                    readiness_report['deployment_readiness']['model_available'] = True
                else:
                    readiness_report['deployment_readiness']['model_available'] = False
                
                # Overall readiness score
                readiness_report['deployment_readiness']['overall_score'] = self.calculate_readiness_score(readiness_report)
                
            else:
                readiness_report['training_readiness'] = {'status': 'No training data available'}
                readiness_report['deployment_readiness']['overall_score'] = 0
            
        except Exception as e:
            readiness_report['error'] = str(e)
        
        # Generate recommendations
        readiness_report['recommendations'] = self.generate_readiness_recommendations(readiness_report)
        
        return readiness_report
    
    def get_model_performance_metrics(self, model_version: str) -> Dict:
        """Get model performance metrics"""
        try:
            conn = sqlite3.connect(self.ml_pipeline.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT model_type, validation_score, test_score, precision_score, 
                       recall_score, training_date, parameters
                FROM model_performance 
                WHERE model_version = ? 
                ORDER BY training_date DESC 
                LIMIT 1
            ''', (model_version,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'model_type': result[0],
                    'validation_score': result[1],
                    'test_score': result[2],
                    'precision_score': result[3],
                    'recall_score': result[4],
                    'training_date': result[5],
                    'parameters': json.loads(result[6]) if result[6] else {}
                }
        except Exception as e:
            return {'error': str(e)}
        
        return {}
    
    def calculate_readiness_score(self, readiness_report: Dict) -> float:
        """Calculate overall readiness score (0-100)"""
        score = 0
        
        training = readiness_report.get('training_readiness', {})
        deployment = readiness_report.get('deployment_readiness', {})
        
        # Sample adequacy (40 points)
        sample_count = training.get('sample_count', 0)
        if sample_count >= 200:
            score += 40
        elif sample_count >= 100:
            score += 30
        elif sample_count >= 50:
            score += 20
        elif sample_count >= 25:
            score += 10
        
        # Data quality (30 points)
        if training.get('class_balance_acceptable', False):
            score += 15
        
        completeness = training.get('data_completeness', 0)
        if completeness >= 95:
            score += 15
        elif completeness >= 90:
            score += 10
        elif completeness >= 80:
            score += 5
        
        # Model availability (30 points)
        if deployment.get('model_available', False):
            score += 20
            
            # Model performance bonus
            model_perf = readiness_report.get('model_performance', {})
            if model_perf.get('validation_score', 0) >= 0.7:
                score += 10
            elif model_perf.get('validation_score', 0) >= 0.6:
                score += 5
        
        return min(score, 100)
    
    def analyze_collection_performance(self) -> Dict:
        """Analyze data collection performance"""
        print("📊 Analyzing collection performance...")
        
        collection_report = {
            'timestamp': datetime.now().isoformat(),
            'collection_rate': {},
            'collection_trends': {},
            'efficiency_metrics': {},
            'recommendations': []
        }
        
        try:
            conn = sqlite3.connect(self.ml_pipeline.db_path)
            
            # Daily collection rate
            daily_query = '''
                SELECT DATE(created_at) as date, COUNT(*) as samples
                FROM sentiment_features
                WHERE created_at >= datetime('now', '-30 days')
                GROUP BY DATE(created_at)
                ORDER BY date
            '''
            
            daily_df = pd.read_sql_query(daily_query, conn)
            
            if not daily_df.empty:
                collection_report['collection_rate'] = {
                    'avg_daily_samples': float(daily_df['samples'].mean()),
                    'max_daily_samples': int(daily_df['samples'].max()),
                    'min_daily_samples': int(daily_df['samples'].min()),
                    'total_days': len(daily_df),
                    'total_samples': int(daily_df['samples'].sum())
                }
                
                # Collection trends
                if len(daily_df) >= 7:
                    recent_avg = daily_df['samples'].tail(7).mean()
                    earlier_avg = daily_df['samples'].head(7).mean()
                    trend = (recent_avg - earlier_avg) / earlier_avg * 100 if earlier_avg > 0 else 0
                    
                    collection_report['collection_trends'] = {
                        'recent_7_day_avg': float(recent_avg),
                        'earlier_7_day_avg': float(earlier_avg),
                        'trend_percentage': float(trend),
                        'trend_direction': 'improving' if trend > 0 else 'declining' if trend < 0 else 'stable'
                    }
            
            # Efficiency metrics (outcomes vs features)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM sentiment_features')
            total_features = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM trading_outcomes')
            total_outcomes = cursor.fetchone()[0]
            
            collection_report['efficiency_metrics'] = {
                'total_features_collected': total_features,
                'total_outcomes_recorded': total_outcomes,
                'outcome_recording_rate': total_outcomes / total_features * 100 if total_features > 0 else 0,
                'collection_efficiency': 'high' if total_outcomes / total_features >= 0.8 else 'medium' if total_outcomes / total_features >= 0.5 else 'low'
            }
            
            conn.close()
            
        except Exception as e:
            collection_report['error'] = str(e)
        
        # Generate recommendations
        collection_report['recommendations'] = self.generate_collection_recommendations(collection_report)
        
        return collection_report
    
    def generate_data_quality_recommendations(self, quality_report: Dict) -> List[str]:
        """Generate data quality recommendations"""
        recommendations = []
        
        adequacy = quality_report.get('sample_adequacy', {})
        
        if adequacy.get('total_samples', 0) < 50:
            recommendations.append("Increase data collection - need at least 50 samples for basic training")
        elif adequacy.get('total_samples', 0) < 100:
            recommendations.append("Continue data collection - aim for 100+ samples for robust training")
        
        if adequacy.get('class_balance_ratio', 1) < 0.3:
            recommendations.append("Address class imbalance - consider collecting more of the minority class")
        
        missing_values = adequacy.get('missing_values', {})
        if any(v > 0 for v in missing_values.values()):
            recommendations.append("Handle missing values in features")
        
        if adequacy.get('duplicate_rows', 0) > 0:
            recommendations.append("Remove duplicate samples to improve data quality")
        
        return recommendations
    
    def generate_readiness_recommendations(self, readiness_report: Dict) -> List[str]:
        """Generate model readiness recommendations"""
        recommendations = []
        
        training = readiness_report.get('training_readiness', {})
        deployment = readiness_report.get('deployment_readiness', {})
        
        if not training.get('min_samples_met', False):
            recommendations.append("Collect more training samples before model training")
        
        if not training.get('class_balance_acceptable', False):
            recommendations.append("Improve class balance in training data")
        
        if not deployment.get('model_available', False):
            recommendations.append("Train initial ML model")
        
        score = deployment.get('overall_score', 0)
        if score < 50:
            recommendations.append("System not ready for production - address data quality issues")
        elif score < 70:
            recommendations.append("System approaching readiness - continue improvements")
        elif score < 90:
            recommendations.append("System mostly ready - minor improvements needed")
        else:
            recommendations.append("System ready for production deployment")
        
        return recommendations
    
    def generate_collection_recommendations(self, collection_report: Dict) -> List[str]:
        """Generate collection performance recommendations"""
        recommendations = []
        
        rate = collection_report.get('collection_rate', {})
        trends = collection_report.get('collection_trends', {})
        efficiency = collection_report.get('efficiency_metrics', {})
        
        avg_daily = rate.get('avg_daily_samples', 0)
        if avg_daily < 5:
            recommendations.append("Increase collection frequency - aim for 5+ samples per day")
        elif avg_daily < 10:
            recommendations.append("Good collection rate - consider increasing to 10+ samples per day")
        
        if trends.get('trend_direction') == 'declining':
            recommendations.append("Collection rate declining - investigate and address issues")
        
        outcome_rate = efficiency.get('outcome_recording_rate', 0)
        if outcome_rate < 50:
            recommendations.append("Improve outcome recording - many signals lack follow-up")
        
        return recommendations
    
    def run_comprehensive_analysis(self) -> Dict:
        """Run complete comprehensive analysis"""
        print("🔍 Running comprehensive system analysis...")
        print("=" * 60)
        
        # Run all analyses
        data_quality = self.analyze_data_quality()
        model_readiness = self.analyze_model_readiness()
        collection_performance = self.analyze_collection_performance()
        
        # Combine results
        comprehensive_report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_quality': data_quality,
            'model_readiness': model_readiness,
            'collection_performance': collection_performance,
            'overall_assessment': self.generate_overall_assessment(data_quality, model_readiness, collection_performance)
        }
        
        return comprehensive_report
    
    def generate_overall_assessment(self, data_quality: Dict, model_readiness: Dict, collection_performance: Dict) -> Dict:
        """Generate overall system assessment"""
        assessment = {
            'system_health': 'unknown',
            'readiness_score': 0,
            'priority_actions': [],
            'next_steps': []
        }
        
        # Calculate overall readiness score
        readiness_score = model_readiness.get('deployment_readiness', {}).get('overall_score', 0)
        assessment['readiness_score'] = readiness_score
        
        # Determine system health
        if readiness_score >= 80:
            assessment['system_health'] = 'excellent'
        elif readiness_score >= 60:
            assessment['system_health'] = 'good'
        elif readiness_score >= 40:
            assessment['system_health'] = 'developing'
        else:
            assessment['system_health'] = 'needs_improvement'
        
        # Priority actions
        all_recommendations = (
            data_quality.get('recommendations', []) +
            model_readiness.get('recommendations', []) +
            collection_performance.get('recommendations', [])
        )
        
        assessment['priority_actions'] = list(set(all_recommendations))[:5]  # Top 5 unique recommendations
        
        # Next steps based on readiness
        if readiness_score < 30:
            assessment['next_steps'] = [
                "Focus on data collection",
                "Run quick_sample_boost.py to generate initial samples",
                "Set up regular data collection schedule"
            ]
        elif readiness_score < 60:
            assessment['next_steps'] = [
                "Continue data collection",
                "Improve data quality",
                "Consider initial model training"
            ]
        elif readiness_score < 80:
            assessment['next_steps'] = [
                "Train or retrain ML models",
                "Optimize model performance",
                "Prepare for production deployment"
            ]
        else:
            assessment['next_steps'] = [
                "Deploy to production",
                "Monitor model performance",
                "Implement automated retraining"
            ]
        
        return assessment
    
    def print_comprehensive_report(self, report: Dict):
        """Print comprehensive analysis report"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE SYSTEM ANALYSIS REPORT")
        print("="*80)
        
        # Overall assessment
        overall = report['overall_assessment']
        print(f"\n🎯 OVERALL ASSESSMENT")
        print(f"System Health: {overall['system_health'].upper()}")
        print(f"Readiness Score: {overall['readiness_score']}/100")
        
        # Data quality summary
        data_quality = report['data_quality']
        adequacy = data_quality.get('sample_adequacy', {})
        print(f"\n📈 DATA QUALITY SUMMARY")
        print(f"Total Samples: {adequacy.get('total_samples', 0)}")
        print(f"Features: {adequacy.get('feature_count', 0)}")
        print(f"Class Balance: {adequacy.get('class_balance_ratio', 0):.2f}")
        
        # Model readiness
        readiness = report['model_readiness']
        training = readiness.get('training_readiness', {})
        print(f"\n🤖 MODEL READINESS")
        print(f"Training Ready: {'Yes' if training.get('min_samples_met', False) else 'No'}")
        print(f"Model Available: {'Yes' if readiness.get('deployment_readiness', {}).get('model_available', False) else 'No'}")
        
        # Collection performance
        collection = report['collection_performance']
        rate = collection.get('collection_rate', {})
        print(f"\n📊 COLLECTION PERFORMANCE")
        print(f"Daily Average: {rate.get('avg_daily_samples', 0):.1f} samples/day")
        print(f"Outcome Recording: {collection.get('efficiency_metrics', {}).get('outcome_recording_rate', 0):.1f}%")
        
        # Priority actions
        print(f"\n🚀 PRIORITY ACTIONS")
        for i, action in enumerate(overall['priority_actions'], 1):
            print(f"{i}. {action}")
        
        # Next steps
        print(f"\n➡️  NEXT STEPS")
        for i, step in enumerate(overall['next_steps'], 1):
            print(f"{i}. {step}")
        
        print("="*80)
    
    def save_analysis_report(self, report: Dict, filename: str = None):
        """Save analysis report to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'reports/comprehensive_analysis_{timestamp}.json'
        
        os.makedirs('reports', exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Analysis report saved: {filename}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive system analyzer')
    parser.add_argument('--data-quality', action='store_true', help='Analyze data quality only')
    parser.add_argument('--model-readiness', action='store_true', help='Analyze model readiness only')
    parser.add_argument('--collection', action='store_true', help='Analyze collection performance only')
    parser.add_argument('--save', action='store_true', help='Save report to file')
    
    args = parser.parse_args()
    
    analyzer = ComprehensiveAnalyzer()
    
    if args.data_quality:
        report = analyzer.analyze_data_quality()
        print(json.dumps(report, indent=2))
    elif args.model_readiness:
        report = analyzer.analyze_model_readiness()
        print(json.dumps(report, indent=2))
    elif args.collection:
        report = analyzer.analyze_collection_performance()
        print(json.dumps(report, indent=2))
    else:
        # Run comprehensive analysis
        report = analyzer.run_comprehensive_analysis()
        analyzer.print_comprehensive_report(report)
        
        if args.save:
            analyzer.save_analysis_report(report)
