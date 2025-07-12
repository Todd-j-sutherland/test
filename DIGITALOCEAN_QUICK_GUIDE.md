# 🌊 DigitalOcean Trading System - Quick Guide

## 🔑 **Connect to Your Server**

```bash
# Connect to your trading server
ssh root@170.64.199.151

# Or create an alias for easy access
echo 'alias trading="ssh root@170.64.199.151"' >> ~/.zshrc
source ~/.zshrc
# Then just type: trading
```

---

## 📅 **Daily Commands Schedule**

### **🌅 MORNING (Start Your Trading Day)**
**When:** 8:00-9:00 AM (before market opens)  
**Why:** Starts data collection and analysis for the day

```bash
# Connect and start the system
ssh root@170.64.199.151
cd /root/test && source ../trading_venv/bin/activate
python daily_manager.py morning
```

**What this does:**
- ✅ Starts smart collector (gathers news every 30 minutes)
- ✅ Launches trading dashboard at http://170.64.199.151:8501
- ✅ Begins continuous sentiment analysis
- ✅ Tracks new trading signals

---

### **🌆 EVENING (End Your Trading Day)**
**When:** 6:00-7:00 PM (after market closes)  
**Why:** Gracefully stops processes and generates daily reports

```bash
# Connect and stop the system
ssh root@170.64.199.151
cd /root/test && source ../trading_venv/bin/activate
python daily_manager.py evening
```

**What this does:**
- ✅ Stops smart collector gracefully
- ✅ Generates daily performance report
- ✅ Saves collected data to files
- ✅ Shows trading summary for the day

---

### **📊 QUICK CHECK (Anytime)**
**When:** Whenever you want to check system health  
**Why:** Monitor data collection progress and system status

```bash
# Quick status check
ssh root@170.64.199.151
cd /root/test && source ../trading_venv/bin/activate
python daily_manager.py status
```

**What this shows:**
- 📈 Number of training samples collected
- 🎯 Trading signals found today
- 🤖 ML model performance
- ⚡ System health indicators

---

## 📆 **Weekly Commands**

### **🔧 WEEKEND MAINTENANCE**
**When:** Saturday or Sunday morning  
**Why:** Optimize ML models and clean up data

```bash
# Weekly optimization and retraining
ssh root@170.64.199.151
cd /root/test && source ../trading_venv/bin/activate
python daily_manager.py weekly
```

**What this does:**
- 🧠 Retrains ML models with new data
- 🗂️ Cleans up old logs and reports
- 📊 Generates comprehensive performance analysis
- 🔍 Identifies trading pattern improvements

---

## 🔄 **Process Management**

### **Check What's Running**
```bash
# See all Python processes
ssh root@170.64.199.151 "ps aux | grep python | grep -v grep"
```

**You should see:**
- `smart_collector.py` - Collecting news and signals (high memory usage is normal)
- `launch_dashboard_auto.py` - Managing the web dashboard
- `streamlit run core/news_analysis_dashboard.py` - The actual dashboard

### **Stop Everything (Emergency)**
```bash
# Emergency restart if something goes wrong
ssh root@170.64.199.151
cd /root/test && source ../trading_venv/bin/activate
python daily_manager.py restart
```

### **Manual Process Control**
```bash
# Kill specific processes if needed
ssh root@170.64.199.151 "pkill -f smart_collector"
ssh root@170.64.199.151 "pkill -f dashboard"

# Or kill all Python processes (nuclear option)
ssh root@170.64.199.151 "pkill -f python"
```

---

## 🌐 **Access Your Dashboard**

**URL:** http://170.64.199.151:8501

**When to check:**
- ✅ **After morning routine** - Verify dashboard is running
- ✅ **During trading hours** - Monitor live signals
- ✅ **End of day** - Review performance

**What you'll see:**
- 📊 Real-time bank sentiment analysis
- 📈 Trading signals and recommendations
- 📉 Historical performance charts
- 🎯 ML model accuracy metrics

---

## ⏰ **Optimal Daily Schedule**

### **Weekdays (Monday-Friday)**
```
8:30 AM  → Run morning command (starts data collection)
12:00 PM → Quick dashboard check (optional)
6:00 PM  → Run evening command (stops collection)
```

### **Weekends**
```
Saturday/Sunday AM → Run weekly command (ML optimization)
```

### **Holiday Schedule**
- **Market closed:** Skip morning/evening commands
- **Long weekends:** Run weekly command on Sunday

---

## 🚨 **Troubleshooting Commands**

### **System Health Check**
```bash
# Check server resources
ssh root@170.64.199.151 "free -h && df -h"

# Check if processes are responding
ssh root@170.64.199.151
cd /root/test && source ../trading_venv/bin/activate
python daily_manager.py status
```

### **Dashboard Not Loading**
```bash
# Restart just the dashboard
ssh root@170.64.199.151 "pkill -f streamlit"
ssh root@170.64.199.151
cd /root/test && source ../trading_venv/bin/activate
python tools/launch_dashboard_auto.py &
```

### **Smart Collector Stuck**
```bash
# Restart data collection
ssh root@170.64.199.151 "pkill -f smart_collector"
ssh root@170.64.199.151
cd /root/test && source ../trading_venv/bin/activate
python core/smart_collector.py &
```

### **Complete System Reset**
```bash
# If everything is broken
ssh root@170.64.199.151
cd /root/test && source ../trading_venv/bin/activate
python daily_manager.py restart
```

---

## 📊 **Understanding the Processes**

### **Smart Collector (Background Process)**
- **Purpose:** Continuously scans news and social media for bank-related sentiment
- **Frequency:** Every 30 minutes during market hours
- **Memory:** Uses 20-40% of server memory (normal)
- **Data:** Saves high-quality trading signals to build ML training dataset

### **Dashboard (Web Interface)**
- **Purpose:** Provides real-time visualization of trading analysis
- **Port:** 8501 (accessible via browser)
- **Updates:** Refreshes automatically with new data
- **Memory:** Uses 5-10% of server memory

### **ML Training Pipeline**
- **Purpose:** Learns from collected data to improve signal accuracy
- **Trigger:** Runs during weekly maintenance
- **Goal:** Achieve >70% prediction accuracy
- **Progress:** Tracks improvement over time

---

## 💡 **Pro Tips**

### **Daily Workflow**
1. **Morning:** Run morning command → Check dashboard loads
2. **Midday:** Glance at dashboard for any major signals
3. **Evening:** Run evening command → Review daily report

### **Performance Monitoring**
- **Week 1:** Expect 50-100 training samples
- **Month 1:** Target 500+ samples, 60%+ accuracy  
- **Month 3:** Target 1000+ samples, 70%+ accuracy

### **Cost Optimization**
- **Server cost:** $5/month (much cheaper than leaving laptop on)
- **Bandwidth:** Well under 1TB limit with normal usage
- **No surprise charges:** Fixed monthly pricing

### **Backup Strategy**
Your data is automatically backed up, but you can manually export:
```bash
# Download latest data to your Mac
scp -r root@170.64.199.151:/root/test/data ~/trading-backup-$(date +%Y%m%d)
```

---

## 🎯 **Success Metrics**

### **Daily Success:**
- ✅ Morning/evening commands run without errors
- ✅ Dashboard accessible and updating
- ✅ New training samples collected (check status command)

### **Weekly Success:**
- ✅ Weekly command completes successfully
- ✅ ML model accuracy improving over time
- ✅ System finding profitable trading signals

### **Monthly Success:**
- ✅ 500+ training samples collected
- ✅ 60%+ signal accuracy achieved
- ✅ Consistent daily operation without manual intervention

---

## 📞 **Quick Reference**

**Essential Commands:**
```bash
# Daily essentials
python daily_manager.py morning    # Start trading day
python daily_manager.py evening    # End trading day
python daily_manager.py status     # Quick health check

# Weekly maintenance
python daily_manager.py weekly     # Weekend optimization

# Emergency
python daily_manager.py restart    # Fix broken system
```

**Server Connection:**
```bash
ssh root@170.64.199.151            # Connect to server
cd /root/test                      # Go to trading directory
source ../trading_venv/bin/activate # Activate Python environment
```

**Dashboard URL:**
```
http://170.64.199.151:8501
```

**This simple workflow will keep your trading system running optimally with minimal daily effort!** 🚀

---

## 📥 **Transferring Trained Models to Local Machine**

### **🎯 What Gets Trained on the Server**
Your DigitalOcean server collects data and trains ML models that become more accurate over time. Here's what gets created:

**Training Data:**
- `data/ml_models/training_data.db` - SQLite database with all trading signals and outcomes
- `data/ml_models/training_data/` - Individual training sample files
- `data/sentiment_history/` - Historical sentiment analysis for trend detection

**Trained Models:**
- `data/ml_models/models/` - Actual trained ML models (RandomForest, XGBoost, etc.)
- `data/ml_models/feature_scaler.pkl` - Feature normalization parameters
- `data/ml_models/active_signals.json` - Currently tracked trading signals

---

### **💾 Download Everything (Complete Backup)**
**When:** Monthly or when you want to work locally
**Purpose:** Get all trained models and collected data

```bash
# Download complete data folder to your Mac
scp -r root@170.64.199.151:/root/test/data ~/trading-backup-$(date +%Y%m%d)
```

**What you get:**
- 📊 **All training data** (thousands of samples after a few months)
- 🤖 **Trained ML models** (ready to use locally)
- 📈 **Historical performance** data and charts
- 🎯 **Active signals** being tracked

**Local storage location:** `~/trading-backup-YYYYMMDD/`

---

### **🤖 Download Just the Models (Quick Transfer)**
**When:** You want to test models locally or use them in other projects
**Purpose:** Get just the trained ML models without all the raw data

```bash
# Create local ML models folder
mkdir -p ~/trading-models-$(date +%Y%m%d)

# Download just the trained models
scp -r root@170.64.199.151:/root/test/data/ml_models/models/ ~/trading-models-$(date +%Y%m%d)/
scp root@170.64.199.151:/root/test/data/ml_models/feature_scaler.pkl ~/trading-models-$(date +%Y%m%d)/
scp root@170.64.199.151:/root/test/data/ml_models/training_data.db ~/trading-models-$(date +%Y%m%d)/
```

**Local storage location:** `~/trading-models-YYYYMMDD/`

---

### **📊 Using Downloaded Models Locally**

#### **Option 1: Replace Your Local Data**
```bash
# Backup your current local data (if any)
mv ~/Repos/trading_analysis/data ~/Repos/trading_analysis/data_backup_$(date +%Y%m%d)

# Copy server data to your local project
cp -r ~/trading-backup-YYYYMMDD ~/Repos/trading_analysis/data
```

#### **Option 2: Load Specific Models in Code**
```python
# In your local Python scripts
import joblib
import pandas as pd

# Load the trained model
model_path = "~/trading-models-YYYYMMDD/latest_model.pkl"
trained_model = joblib.load(model_path)

# Load the feature scaler
scaler_path = "~/trading-models-YYYYMMDD/feature_scaler.pkl"
scaler = joblib.load(scaler_path)

# Now you can make predictions locally
# features = prepare_features(news_data)
# scaled_features = scaler.transform(features)
# prediction = trained_model.predict(scaled_features)
```

---

### **📈 Check Training Progress Before Downloading**
**Before downloading, see how much data has been collected:**

```bash
# Check how much training data is available
ssh root@170.64.199.151
cd /root/test && source ../trading_venv/bin/activate
python daily_manager.py status
```

**Good benchmarks:**
- **Week 1:** 50-100 samples (basic model)
- **Month 1:** 500+ samples (decent accuracy)
- **Month 3:** 1000+ samples (production ready)
- **Month 6:** 2000+ samples (highly accurate)

---

### **🔄 Sync Local and Server Data**

#### **Upload Local Improvements to Server**
If you improve the models locally and want to deploy them:

```bash
# Upload improved models back to server
scp -r ~/Repos/trading_analysis/data/ml_models/models/ root@170.64.199.151:/root/test/data/ml_models/
scp ~/Repos/trading_analysis/data/ml_models/feature_scaler.pkl root@170.64.199.151:/root/test/data/ml_models/
```

#### **Two-Way Sync Strategy**
```bash
# Download server data to local
scp -r root@170.64.199.151:/root/test/data ~/trading-server-sync

# Work locally, then upload changes
# (after local development)
scp -r ~/trading-server-sync/ml_models/models/ root@170.64.199.151:/root/test/data/ml_models/
```

---

### **💡 Local Development Workflow**

#### **Best Practice Setup:**
1. **Server:** Continuous data collection (24/7)
2. **Local:** Model development and testing (when needed)
3. **Sync:** Weekly downloads for local analysis

#### **Typical Monthly Workflow:**
```bash
# 1. Download latest server data
scp -r root@170.64.199.151:/root/test/data ~/trading-monthly-$(date +%Y%m%d)

# 2. Run local analysis
cd ~/Repos/trading_analysis
python -c "
import pandas as pd
import sqlite3
conn = sqlite3.connect('~/trading-monthly-YYYYMMDD/ml_models/training_data.db')
df = pd.read_sql('SELECT * FROM training_samples', conn)
print(f'Total samples: {len(df)}')
print(f'Success rate: {df.outcome.value_counts()}')
"

# 3. If you make improvements, upload them back
scp -r ~/Repos/trading_analysis/data/ml_models/models/ root@170.64.199.151:/root/test/data/ml_models/
```

---

### **📁 Local Data Organization**

**Recommended local folder structure:**
```
~/trading-analysis/
├── current/              # Latest downloaded data
├── backups/
│   ├── trading-backup-20250701/
│   ├── trading-backup-20250801/
│   └── trading-backup-20250901/
├── models-only/
│   ├── trading-models-20250701/
│   └── trading-models-20250801/
└── experiments/          # Your local model improvements
```

**Setup script:**
```bash
# Create organized structure
mkdir -p ~/trading-analysis/{current,backups,models-only,experiments}

# Download and organize
scp -r root@170.64.199.151:/root/test/data ~/trading-analysis/current/
cp -r ~/trading-analysis/current ~/trading-analysis/backups/trading-backup-$(date +%Y%m%d)
```

---

### **🎯 What Each File Contains**

#### **Training Database (`training_data.db`)**
- **Purpose:** SQLite database with all trading signals and their outcomes
- **Size:** Grows over time (100MB+ after several months)
- **Contents:** News headlines, sentiment scores, stock prices, profit/loss outcomes

#### **Model Files (`models/latest_model.pkl`)**
- **Purpose:** Trained machine learning models ready for predictions
- **Size:** Usually 1-10MB each
- **Contents:** RandomForest, XGBoost, or other trained algorithms

#### **Feature Scaler (`feature_scaler.pkl`)**
- **Purpose:** Normalizes input features to match training data
- **Size:** Small (< 1MB)
- **Contents:** Mean and standard deviation for each feature

#### **Active Signals (`active_signals.json`)**
- **Purpose:** Currently tracked trading signals waiting for outcomes
- **Size:** Small (< 1MB)
- **Contents:** Open positions that haven't been resolved yet

---

### **⚡ Quick Download Commands**

```bash
# Just get the latest trained model (fastest)
scp root@170.64.199.151:/root/test/data/ml_models/models/latest_model.pkl ~/Desktop/

# Get everything for analysis (complete backup)
scp -r root@170.64.199.151:/root/test/data ~/Desktop/trading-full-backup/

# Get just training data for analysis
scp root@170.64.199.151:/root/test/data/ml_models/training_data.db ~/Desktop/

# Get just models for deployment
scp -r root@170.64.199.151:/root/test/data/ml_models/models/ ~/Desktop/models/
```

**Your trained models will be stored locally and can be used independently of the server for analysis, backtesting, or integration into other trading systems!** 🚀

---

## 💤 **Turn Off Droplet at Night (Save Money)**

### **🌙 Automatic Nightly Shutdown**
**When:** After evening routine (7:00-8:00 PM)  
**Why:** Save ~$3/month by turning off during non-trading hours

```bash
# Evening routine + shutdown
ssh root@170.64.199.151
cd /root/test && source ../trading_venv/bin/activate
python daily_manager.py evening
sudo shutdown -h now
```

**What this does:**
- ✅ Gracefully stops all trading processes
- ✅ Saves all data and generates daily report
- ✅ Shuts down the server (stops billing)
- 💰 **Saves money:** Only pay for ~10 hours/day instead of 24

---

### **🌅 Morning Startup**
**When:** Before market opens (8:00-8:30 AM)  
**Why:** Start server and begin data collection

```bash
# Start droplet from your Mac
doctl compute droplet start <your-droplet-id>

# Wait 30-60 seconds for boot, then connect
ssh root@170.64.199.151
cd /root/test && source ../trading_venv/bin/activate
python daily_manager.py morning
```

---

### **📱 Using DigitalOcean CLI (doctl)**

#### **Install doctl on your Mac (one-time setup):**
```bash
# Install via Homebrew
brew install doctl

# Authenticate with your DigitalOcean account
doctl auth init
# (Enter your API token from DigitalOcean dashboard)

# Find your droplet ID
doctl compute droplet list
```

#### **Daily Commands:**
```bash
# Morning: Start the droplet
doctl compute droplet start <your-droplet-id>

# Evening: Stop the droplet (after running evening routine)
doctl compute droplet stop <your-droplet-id>

# Check status
doctl compute droplet list
```

---

### **🔧 Alternative: Use DigitalOcean Dashboard**

If you prefer clicking instead of command line:

1. **Go to:** https://cloud.digitalocean.com/droplets
2. **Find your droplet:** Look for IP 170.64.199.151
3. **Power controls:** Use the "..." menu → Power → Start/Stop

---

### **💰 Cost Savings Breakdown**

**Normal Operation (24/7):**
- **Cost:** $5.00/month (720 hours)
- **Usage:** Continuous operation

**Night Shutdown (12 hours/day):**
- **Cost:** ~$2.50/month (360 hours)
- **Savings:** ~$2.50/month (50% reduction)
- **Annual savings:** ~$30/year

**Weekend Shutdown (Fri 6PM - Mon 8AM):**
- **Additional savings:** ~$1/month
- **Total potential savings:** ~$42/year

---

### **⚠️ Important Considerations**

#### **Data Safety:**
- ✅ **Always run evening routine first** - saves all data
- ✅ **Data persists** when droplet is off (stored on disk)
- ✅ **No data loss** from proper shutdown

#### **Timing:**
- 🕰️ **Market hours:** 9:30 AM - 4:00 PM EST (14:30-20:00 UTC)
- 🕰️ **Safe shutdown:** After 6:00 PM EST (22:00 UTC)
- 🕰️ **Morning startup:** Before 8:30 AM EST (12:30 UTC)

#### **Automation Options:**
```bash
# Add to your evening routine (auto-shutdown)
echo 'sudo shutdown -h +5' >> ~/evening_shutdown.sh

# Create morning startup alias
echo 'alias start-trading="doctl compute droplet start <droplet-id> && sleep 60 && ssh root@170.64.199.151"' >> ~/.zshrc
```

---

### **🤖 Automated Schedule (Advanced)**

#### **Option 1: Cron Job for Shutdown**
Set up automatic shutdown on the server:

```bash
# On the droplet, add evening shutdown
ssh root@170.64.199.151
crontab -e

# Add this line (shuts down at 8 PM EST)
0 20 * * 1-5 /root/test/daily_manager.py evening && shutdown -h +2
```

#### **Option 2: Mac Automation**
Create scripts on your Mac for daily routine:

```bash
# ~/start-trading.sh
#!/bin/bash
doctl compute droplet start <droplet-id>
echo "Waiting for server to boot..."
sleep 60
ssh root@170.64.199.151 "cd /root/test && source ../trading_venv/bin/activate && python daily_manager.py morning"

# ~/stop-trading.sh
#!/bin/bash
ssh root@170.64.199.151 "cd /root/test && source ../trading_venv/bin/activate && python daily_manager.py evening"
sleep 10
doctl compute droplet stop <droplet-id>
```

**Make executable:**
```bash
chmod +x ~/start-trading.sh ~/stop-trading.sh
```

---

### **📋 Weekend & Holiday Schedule**

#### **Weekend Shutdown:**
```bash
# Friday evening
ssh root@170.64.199.151
cd /root/test && source ../trading_venv/bin/activate
python daily_manager.py evening
doctl compute droplet stop <droplet-id>

# Monday morning
doctl compute droplet start <droplet-id>
# (wait 60 seconds)
ssh root@170.64.199.151
cd /root/test && source ../trading_venv/bin/activate
python daily_manager.py morning
```

#### **Holiday Shutdown:**
```bash
# Before holiday
python daily_manager.py evening
doctl compute droplet stop <droplet-id>

# After holiday
doctl compute droplet start <droplet-id>
python daily_manager.py morning
```

---

### **⚡ Quick Reference Commands**

```bash
# Evening routine with shutdown
ssh root@170.64.199.151 "cd /root/test && source ../trading_venv/bin/activate && python daily_manager.py evening && sudo shutdown -h now"

# Morning startup sequence
doctl compute droplet start <droplet-id> && sleep 60 && ssh root@170.64.199.151 "cd /root/test && source ../trading_venv/bin/activate && python daily_manager.py morning"

# Check droplet status
doctl compute droplet list | grep 170.64.199.151
```

**Pro tip:** Replace `<droplet-id>` with your actual droplet ID from `doctl compute droplet list` 

**This can cut your server costs in half while maintaining full trading functionality during market hours!** 💰

---
