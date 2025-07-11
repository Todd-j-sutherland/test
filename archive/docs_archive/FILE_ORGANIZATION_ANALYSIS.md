# 🗂️ PROJECT FILE ORGANIZATION ANALYSIS

## 📊 **CURRENT vs LEGACY FILES**

Based on modification dates, dependencies, and functionality analysis, here's the breakdown of your project files:

---

## ✅ **CURRENT/ACTIVE FILES** (Keep These)

### **Root Directory - Core Files**
| File | Purpose | Status | Last Modified |
|------|---------|--------|---------------|
| `news_trading_analyzer.py` | **MAIN ENTRY POINT** - Primary analyzer with ML integration | ✅ **CURRENT** | 2025-07-10 |
| `news_analysis_dashboard.py` | **MAIN DASHBOARD** - Streamlit dashboard with ML predictions | ✅ **CURRENT** | 2025-07-10 |
| `demo_ml_integration.py` | **ML DEMO** - Tests ML integration functionality | ✅ **CURRENT** | 2025-07-10 |

### **Root Directory - Utility Files**
| File | Purpose | Status | Last Modified |
|------|---------|--------|---------------|
| `launch_dashboard.py` | Dashboard launcher with environment checks | ✅ **ACTIVE** | 2025-07-10 |
| `launch_dashboard_auto.py` | Automated dashboard launcher (bypasses email) | ✅ **ACTIVE** | 2025-07-10 |
| `run_analyzer.py` | Simple runner wrapper for news_trading_analyzer | ✅ **ACTIVE** | 2025-07-08 |

### **src/ Directory - Core Components**
| File | Purpose | Status |
|------|---------|--------|
| `news_sentiment.py` | **CORE** - News sentiment analysis with ML integration | ✅ **CURRENT** |
| `ml_training_pipeline.py` | **ML CORE** - Machine learning training system | ✅ **CURRENT** |
| `trading_outcome_tracker.py` | **ML CORE** - Trading outcome tracking for ML | ✅ **CURRENT** |
| `ml_backtester.py` | **ML CORE** - ML model backtesting | ✅ **CURRENT** |
| `technical_analysis.py` | **CORE** - Technical analysis integration | ✅ **CURRENT** |
| `ml_trading_config.py` | **CORE** - ML configuration and feature engineering | ✅ **CURRENT** |

---

## ⚠️ **LEGACY/SUPERSEDED FILES** (Consider Moving to Archive)

### **Root Directory - Legacy Files**
| File | Purpose | Status | Reason | Action |
|------|---------|--------|--------|--------|
| `generate_dashboard.py` | Old dashboard generator | 🔶 **LEGACY** | Superseded by `news_analysis_dashboard.py` | Move to `archive/` |
| `run_analysis_and_dashboard.py` | Old combined runner | 🔶 **LEGACY** | Superseded by individual launchers | Move to `archive/` |
| `backtesting_system.py` | Old backtesting system | 🔶 **LEGACY** | Superseded by `src/ml_backtester.py` | Move to `archive/` |

### **Root Directory - Temporary/Test Files**
| File | Purpose | Status | Action |
|------|---------|--------|--------|
| `_tmp.py` | Temporary test file | 🗑️ **TEMP** | Delete |
| `news_analysis_dashboard.html` | Static HTML output | 🗑️ **TEMP** | Delete (regenerated) |

---

## 📂 **RECOMMENDED FILE ORGANIZATION**

### **Step 1: Move Legacy Files to Archive**
```bash
# Move legacy files to archive
mv generate_dashboard.py archive/
mv run_analysis_and_dashboard.py archive/
mv backtesting_system.py archive/

# Delete temporary files
rm _tmp.py
rm news_analysis_dashboard.html  # This gets regenerated
```

### **Step 2: Clean Root Directory Structure**
After cleanup, your root directory should contain:

```
📁 trading_analysis/
├── 🚀 news_trading_analyzer.py          # MAIN ANALYZER
├── 📊 news_analysis_dashboard.py        # MAIN DASHBOARD  
├── 🧪 demo_ml_integration.py           # ML DEMO
├── 🎯 launch_dashboard.py              # LAUNCHER (with checks)
├── ⚡ launch_dashboard_auto.py         # LAUNCHER (automated)
├── 🔧 run_analyzer.py                  # SIMPLE RUNNER
├── 📋 requirements.txt                 # DEPENDENCIES
└── 📚 README.md                        # DOCUMENTATION
```

---

## 🎯 **USAGE PATTERNS & RECOMMENDATIONS**

### **Primary Workflows**
1. **Run Analysis**: 
   ```bash
   python news_trading_analyzer.py --symbols CBA.AX,WBC.AX
   # OR
   python run_analyzer.py  # Simple wrapper
   ```

2. **Launch Dashboard**:
   ```bash
   python launch_dashboard_auto.py  # Automated (recommended)
   # OR
   python launch_dashboard.py      # With manual checks
   ```

3. **Test ML Integration**:
   ```bash
   python demo_ml_integration.py
   ```

### **File Relationships**
```
news_trading_analyzer.py
├── imports: src/news_sentiment.py
├── imports: src/trading_outcome_tracker.py
└── uses: src/ml_training_pipeline.py

news_analysis_dashboard.py
├── imports: src/technical_analysis.py
└── displays: ML predictions from database

launch_dashboard_auto.py
└── runs: streamlit run news_analysis_dashboard.py
```

---

## 🔧 **DEPENDENCY ANALYSIS**

### **Core Dependencies (Keep)**
- `news_trading_analyzer.py` ← **Used by**: `run_analyzer.py`, `demo_ml_integration.py`
- `news_analysis_dashboard.py` ← **Used by**: `launch_dashboard.py`, `launch_dashboard_auto.py`

### **Legacy Dependencies (Safe to Archive)**
- `generate_dashboard.py` ← **Used by**: Nothing (legacy)
- `run_analysis_and_dashboard.py` ← **Used by**: Nothing (legacy)
- `backtesting_system.py` ← **Superseded by**: `src/ml_backtester.py`

---

## 📋 **CLEANUP CHECKLIST**

### ✅ **Phase 1: Immediate Cleanup**
- [ ] Move `generate_dashboard.py` to `archive/`
- [ ] Move `run_analysis_and_dashboard.py` to `archive/`
- [ ] Move `backtesting_system.py` to `archive/`
- [ ] Delete `_tmp.py`
- [ ] Delete `news_analysis_dashboard.html`

### ✅ **Phase 2: Verification**
- [ ] Test `python news_trading_analyzer.py` works
- [ ] Test `python launch_dashboard_auto.py` works  
- [ ] Test `python demo_ml_integration.py` works
- [ ] Verify no broken imports

### ✅ **Phase 3: Documentation Update**
- [ ] Update README.md with current file structure
- [ ] Update any scripts that reference moved files
- [ ] Add comments to launcher files explaining their purpose

---

## 🎯 **RECOMMENDED IMMEDIATE ACTION**

Run this cleanup script:

```bash
# Create archive subdirectory if needed
mkdir -p archive/legacy_root_files

# Move legacy files
mv generate_dashboard.py archive/legacy_root_files/
mv run_analysis_and_dashboard.py archive/legacy_root_files/
mv backtesting_system.py archive/legacy_root_files/

# Remove temporary files
rm -f _tmp.py
rm -f news_analysis_dashboard.html

# Verify core functionality still works
python demo_ml_integration.py
```

This will leave you with a clean, organized project structure focused on the current ML-integrated system!
