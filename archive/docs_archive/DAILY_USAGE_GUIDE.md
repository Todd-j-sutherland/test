# 📊 Daily Usage Guide - Trading Analysis System

## 🚀 **Quick Start - Essential Daily Commands**

### **Morning Routine (5 minutes)**
```bash
# 1. Check system status and data collection progress
python comprehensive_analyzer.py

# 2. Start live data collection (runs in background)
python smart_collector.py &

# 3. Launch dashboard for monitoring
python launch_dashboard_auto.py
```

### **Evening Review (10 minutes)**
```bash
# 1. Generate daily report
python advanced_daily_collection.py

# 2. Check paper trading performance
python advanced_paper_trading.py --mode status

# 3. View analytics dashboard
python launch_dashboard_auto.py
```

---

## 📅 **Daily Workflow**

### **Every Morning (Before Market Open - 9:30 AM AEST)**
1. **System Health Check**
   ```bash
   python comprehensive_analyzer.py
   ```
   - Shows current ML model status
   - Displays data collection progress
   - Reports system readiness score

2. **Start Data Collection**
   ```bash
   python smart_collector.py
   ```
   - Automatically collects live sentiment data
   - Tracks trading signal outcomes
   - Builds ML training dataset

3. **Launch Monitoring Dashboard**
   ```bash
   python launch_dashboard_auto.py
   ```
   - Real-time sentiment analysis
   - Trading signals and alerts
   - ML prediction confidence scores

### **During Market Hours (10 AM - 4 PM AEST)**
- **Dashboard monitors automatically**
- **Smart collector runs in background**
- **Check alerts every 2-3 hours:**
  ```bash
  python news_trading_analyzer.py --symbols CBA.AX,WBC.AX,ANZ.AX,NAB.AX --quick-check
  ```

### **Every Evening (After Market Close - 5 PM AEST)**
1. **Generate Daily Analytics Report**
   ```bash
   python advanced_daily_collection.py
   ```
   - Sentiment trends for the day
   - Signal accuracy metrics
   - Data quality assessment

2. **Review Paper Trading Performance**
   ```bash
   python advanced_paper_trading.py --mode daily-report
   ```
   - Virtual trading results
   - Risk metrics and drawdown analysis
   - Performance vs benchmarks

---

## 📈 **Weekly Routine**

### **Every Sunday (Weekly Optimization)**
```bash
# 1. Run comprehensive system analysis
python comprehensive_analyzer.py --detailed

# 2. Retrain ML models with new data
python scripts/retrain_ml_models.py

# 3. Generate weekly performance report
python advanced_paper_trading.py --mode weekly-report

# 4. Optimize sentiment thresholds
python sentiment_threshold_calibrator.py

# 5. Analyze market timing patterns
python market_timing_optimizer.py
```

---

## 🎯 **Key Performance Monitoring**

### **Critical Metrics to Check Daily**
1. **ML Model Performance**
   ```bash
   python -c "
   from src.ml_training_pipeline import MLTrainingPipeline
   p = MLTrainingPipeline()
   X, y = p.prepare_training_dataset(min_samples=1)
   print(f'Training Samples: {len(X) if X is not None else 0}')
   "
   ```

2. **Signal Accuracy**
   ```bash
   python -c "
   from advanced_paper_trading import PaperTradingSystem
   pts = PaperTradingSystem()
   stats = pts.get_performance_stats()
   print(f'Win Rate: {stats.get(\"win_rate\", 0):.1%}')
   print(f'Total Return: {stats.get(\"total_return\", 0):.1%}')
   "
   ```

3. **Data Collection Rate**
   ```bash
   python -c "
   import json
   try:
       with open('data/ml_models/collection_progress.json', 'r') as f:
           progress = json.load(f)
       print(f'Samples Today: {progress.get(\"signals_today\", 0)}')
   except: print('No collection data found')
   "
   ```

---

## 🔧 **Maintenance Commands**

### **When System Needs Attention**

**If Model Performance Drops:**
```bash
# Retrain models with latest data
python scripts/retrain_ml_models.py

# Recalibrate sentiment thresholds
python sentiment_threshold_calibrator.py
```

**If Data Collection Stops:**
```bash
# Restart smart collector
pkill -f smart_collector.py
python smart_collector.py &

# Check for errors
tail -f logs/enhanced_trading_system.log
```

**If Dashboard is Slow:**
```bash
# Restart dashboard
pkill -f launch_dashboard_auto.py
python launch_dashboard_auto.py
```

---

## 📊 **Performance Optimization Schedule**

### **Daily (2 minutes)**
- Check system status
- Monitor collection progress
- Review key alerts

### **Weekly (15 minutes)**
- Retrain ML models
- Optimize thresholds
- Generate performance reports
- Analyze timing patterns

### **Monthly (30 minutes)**
- Deep performance analysis
- Model architecture review
- Feature importance analysis
- Risk management review

---

## 🚨 **Alert Conditions & Actions**

### **High Priority Alerts**
1. **Model Accuracy < 60%**
   ```bash
   python scripts/retrain_ml_models.py --force
   ```

2. **No Data Collection for 2+ Hours**
   ```bash
   python smart_collector.py --restart
   ```

3. **Paper Trading Drawdown > 5%**
   ```bash
   python advanced_paper_trading.py --risk-check
   python sentiment_threshold_calibrator.py --conservative
   ```

### **Medium Priority Alerts**
1. **Low Sample Count Growth**
   ```bash
   python quick_sample_boost.py
   ```

2. **Signal Confidence Declining**
   ```bash
   python sentiment_threshold_calibrator.py
   ```

---

## 📱 **Mobile-Friendly Quick Commands**

### **One-Line Status Check**
```bash
python -c "from comprehensive_analyzer import *; quick_status()"
```

### **Emergency Stop All**
```bash
pkill -f "smart_collector\|launch_dashboard\|news_trading"
```

### **Emergency Restart All**
```bash
python smart_collector.py & python launch_dashboard_auto.py &
```

---

## 💡 **Pro Tips for Maximum Performance**

### **Best Practices**
1. **Run morning routine before 9:30 AM** - Get ready for market open
2. **Keep dashboard open during market hours** - Monitor real-time signals
3. **Check evening reports daily** - Track performance trends
4. **Weekly model retraining** - Keep ML models fresh
5. **Monthly deep analysis** - Optimize strategy parameters

### **Time-Saving Shortcuts**
1. **Create aliases in your shell:**
   ```bash
   alias trading-start="python smart_collector.py & python launch_dashboard_auto.py"
   alias trading-status="python comprehensive_analyzer.py"
   alias trading-stop="pkill -f 'smart_collector\|launch_dashboard'"
   ```

2. **Set up cron jobs for automation:**
   ```bash
   # Add to crontab (crontab -e)
   0 9 * * 1-5 cd /Users/toddsutherland/Repos/trading_analysis && python smart_collector.py
   0 17 * * 1-5 cd /Users/toddsutherland/Repos/trading_analysis && python advanced_daily_collection.py
   0 8 * * 0 cd /Users/toddsutherland/Repos/trading_analysis && python scripts/retrain_ml_models.py
   ```

---

## 🎯 **Success Metrics to Track**

### **Daily Targets**
- ✅ 5+ new sentiment samples collected
- ✅ System uptime > 95%
- ✅ Signal confidence > 70%

### **Weekly Targets**
- ✅ Model accuracy > 60%
- ✅ Paper trading positive return
- ✅ 25+ new training samples

### **Monthly Targets**
- ✅ Consistent profitability trend
- ✅ 100+ training samples added
- ✅ System optimization completed

---

**🚀 Ready to maximize your trading system performance? Start with the morning routine and build the habit!**
