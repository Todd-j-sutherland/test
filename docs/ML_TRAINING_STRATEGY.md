# ML Training Strategy: Symbol Selection & Accuracy Optimization

## Current System Analysis

### ✅ Bank Symbol Standardization Status
- **All 7 symbols implemented**: CBA.AX, WBC.AX, ANZ.AX, NAB.AX, MQG.AX, SUN.AX, QBE.AX
- **Centralized configuration**: config/settings.py manages all symbols
- **Comprehensive keywords**: All 7 banks have detailed keyword lists
- **Feature engineering**: Advanced feature sets support all symbols

## ML Training Recommendations

### 🎯 Optimal Training Strategy

#### **Phase 1: Master the Financial Sector (Current - 3 months)**
- **Symbols**: 7 financial institutions (current set)
- **Rationale**: 
  - Sector coherence improves pattern recognition
  - Similar market dynamics and correlations
  - Established sentiment patterns and keywords
  - Reduces feature noise from different sectors

#### **Phase 2: Add Complementary Financials (Months 3-6)**
```python
EXPANSION_CANDIDATES = [
    'BEN.AX',  # Bendigo Adelaide Bank
    'BOQ.AX',  # Bank of Queensland
    'AMP.AX',  # AMP Limited
    'IFL.AX',  # IOOF Holdings
    'CWN.AX'   # Crown Resorts (entertainment/finance)
]
```

#### **Phase 3: Diversified Portfolio (Months 6+)**
```python
SECTOR_EXPANSION = {
    'mining': ['BHP.AX', 'RIO.AX', 'FMG.AX'],
    'retail': ['WOW.AX', 'COL.AX', 'JBH.AX'], 
    'telecom': ['TLS.AX', 'TPM.AX'],
    'tech': ['XRO.AX', 'APT.AX', 'ZIP.AX']
}
```

### 📊 Expected Accuracy Implications

#### **Why 7 Banks Optimize Current Accuracy:**

1. **Feature Coherence**: 
   - Similar price patterns across banks
   - Correlated sentiment drivers (interest rates, regulations)
   - Consistent market reaction patterns

2. **Data Quality Benefits**:
   - Established keyword dictionaries for all 7 banks
   - Consistent news source patterns
   - Similar market microstructure

3. **Model Learning Efficiency**:
   - Reduced feature noise
   - Clearer signal-to-noise ratio
   - More consistent training patterns

#### **Adding More Symbols - Trade-offs:**

**✅ Potential Benefits:**
- More training samples (if you expand data collection)
- Cross-sector correlation insights
- Broader market regime detection
- Portfolio diversification signals

**⚠️ Potential Drawbacks:**
- Feature dilution across different sectors
- Increased complexity without proportional signal improvement
- Different volatility patterns may confuse models
- Keyword mismatches for non-bank sectors

### 🧠 Advanced ML Considerations

#### **Your System's Sophisticated Features:**
```python
# Current feature engineering generates ~100+ features per symbol:
FEATURE_CATEGORIES = {
    'sentiment_base': 6,        # Core sentiment metrics
    'microstructure': 15,       # Market microstructure
    'cross_asset': 20,          # Cross-asset correlations  
    'alternative_data': 25,     # Alternative data sources
    'temporal': 30,             # Time-based features
    'interactions': 50+         # Feature interactions
}
```

#### **Model Performance Insights:**
- **Current Data**: 34 historical records across 7 symbols
- **Training Pipeline**: XGBoost ensemble with meta-learning
- **Feature Engineering**: Comprehensive 746-line implementation
- **Ensemble Learning**: 710-line transformer ensemble system

### 📈 Recommended Implementation

#### **Immediate Actions (This Week):**
1. **Verify Current Performance**: 
   ```bash
   python daily_manager.py test  # Check enhanced features
   python -m pytest tests/test_claude_enhancements.py  # Run ML tests
   ```

2. **Optimize Current 7-Symbol Setup**:
   - Focus on improving data collection quality
   - Enhance feature engineering for banks
   - Increase training sample frequency

#### **Medium Term (1-3 Months):**
1. **Expand Financial Universe**:
   - Add BEN.AX, BOQ.AX (regional banks)
   - Test accuracy impact with A/B testing
   - Monitor feature importance changes

2. **Cross-Validation Strategy**:
   ```python
   # Test symbol addition impact
   baseline_accuracy = train_model(symbols=['CBA.AX', 'WBC.AX', 'ANZ.AX', 'NAB.AX'])
   expanded_accuracy = train_model(symbols=Settings.BANK_SYMBOLS)  # All 7
   further_expanded = train_model(symbols=Settings.BANK_SYMBOLS + ['BEN.AX', 'BOQ.AX'])
   ```

#### **Long Term (3+ Months):**
1. **Sector Diversification**:
   - Gradually add mining/retail sectors
   - Implement sector-specific feature engineering
   - Use ensemble models with sector specialization

### 🎯 Key Success Metrics

```python
OPTIMIZATION_TARGETS = {
    'prediction_accuracy': '>75%',      # Win rate on trades
    'confidence_calibration': '<0.1',   # Confidence error
    'feature_importance_stability': '>0.8',  # Consistent top features
    'sharpe_ratio': '>1.5',            # Risk-adjusted returns
    'max_drawdown': '<15%'             # Risk management
}
```

### 💡 Conclusion

**Your current 7-bank setup is optimal for maximizing ML accuracy** because:

1. **Sector coherence** reduces noise and improves pattern recognition
2. **Established infrastructure** (keywords, features) supports all 7 symbols
3. **Sophisticated feature engineering** creates rich signal space per symbol
4. **Quality over quantity** approach suits your current data volume

**Next Steps**: Focus on collecting more high-quality samples within the financial sector before expanding to other industries.

---

## Quick Command Reference

```bash
# Test current ML system
python daily_manager.py test

# Check bank symbol management  
python src/bank_symbols.py

# Run comprehensive ML tests
python -m pytest tests/test_claude_enhancements.py -v

# Verify all symbols working
python daily_manager.py morning --symbols CBA.AX,MQG.AX,QBE.AX
```
