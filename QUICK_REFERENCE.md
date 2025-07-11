# 🎯 Quick Command Reference Card

## **Essential Daily Commands**

### **🌅 Morning (5 mins)**
```bash
python comprehensive_analyzer.py        # System status
python smart_collector.py &             # Start collection  
python launch_dashboard_auto.py         # Open dashboard
```

### **🌆 Evening (5 mins)**  
```bash
python advanced_daily_collection.py     # Daily report
python advanced_paper_trading.py --mode status  # Trading performance
```

### **📅 Weekly (15 mins)**
```bash
python scripts/retrain_ml_models.py     # Update ML models
python sentiment_threshold_calibrator.py # Optimize thresholds
python market_timing_optimizer.py       # Timing analysis
```

## **Quick Status Checks**

### **Sample Count**
```bash
python -c "from src.ml_training_pipeline import MLTrainingPipeline; p=MLTrainingPipeline(); X,y=p.prepare_training_dataset(min_samples=1); print(f'Samples: {len(X) if X else 0}')"
```

### **Model Performance**
```bash
python -c "from advanced_paper_trading import PaperTradingSystem; pts=PaperTradingSystem(); s=pts.get_performance_stats(); print(f'Win Rate: {s.get(\"win_rate\",0):.1%}, Return: {s.get(\"total_return\",0):.1%}')"
```

### **Collection Progress**
```bash
python -c "import json; f=open('data/ml_models/collection_progress.json','r') if __import__('os').path.exists('data/ml_models/collection_progress.json') else None; print(f'Today: {json.load(f).get(\"signals_today\",0)} signals' if f else 'No data'); f and f.close()"
```

## **Emergency Commands**

### **Restart Everything**
```bash
pkill -f "smart_collector\|launch_dashboard\|news_trading"
python smart_collector.py & python launch_dashboard_auto.py &
```

### **Force Model Retrain**
```bash
python scripts/retrain_ml_models.py --force
```

### **Quick Sample Boost**
```bash
python quick_sample_boost.py
```

## **Performance Monitoring**

### **Target Metrics**
- **Daily**: 5+ samples, >95% uptime, >70% confidence
- **Weekly**: >60% accuracy, positive returns, 25+ samples  
- **Monthly**: Profit trend, 100+ samples, optimization done

### **Alert Thresholds**
- **🚨 High**: Model <60%, No collection 2h+, Drawdown >5%
- **⚠️ Medium**: Low sample growth, declining confidence

## **Pro Tips**
1. **Aliases**: `alias ts="python comprehensive_analyzer.py"`
2. **Cron**: Automate morning/evening routines
3. **Dashboard**: Keep open during market hours (10 AM - 4 PM AEST)
4. **Logs**: `tail -f logs/enhanced_trading_system.log` for debugging

**📱 Bookmark this for daily reference!**
