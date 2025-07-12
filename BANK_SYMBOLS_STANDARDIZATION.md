# BANK_SYMBOLS Standardization Summary

## Overview
All instances of ASX bank stock symbols have been standardized to use the canonical definition from `config/settings.py`.

## Canonical BANK_SYMBOLS Definition
Located in `/Users/toddsutherland/Repos/trading_analysis/config/settings.py`:
```python
BANK_SYMBOLS = [
    'CBA.AX',  # Commonwealth Bank
    'WBC.AX',  # Westpac
    'ANZ.AX',  # ANZ
    'NAB.AX',  # National Australia Bank
    'MQG.AX',  # Macquarie Group
    'SUN.AX',  # Suncorp Group
    'QBE.AX'   # QBE Insurance Group
]
```

## Files Updated to Use Canonical BANK_SYMBOLS

### Core System Files
1. **`core/news_trading_analyzer.py`**
   - Changed hardcoded list to `self.bank_symbols = self.settings.BANK_SYMBOLS`
   - Already had Settings import

2. **`core/news_analysis_dashboard.py`** 
   - Added `from config.settings import Settings`
   - Changed to `self.bank_symbols = self.settings.BANK_SYMBOLS`

3. **`core/smart_collector.py`**
   - Added `from config.settings import Settings`
   - Changed to `self.symbols = self.settings.BANK_SYMBOLS`

4. **`core/advanced_paper_trading.py`**
   - Added `from config.settings import Settings`
   - Default symbols now reference canonical list via argparse

### Source Files
5. **`src/technical_analysis.py`**
   - Added `from config.settings import Settings`
   - Test symbols now use `settings.BANK_SYMBOLS[:2]`

### Demo Files
6. **`demos/demo_enhanced_backtesting.py`**
   - Added `from config.settings import Settings` 
   - Changed to `symbols = settings.BANK_SYMBOLS[:2]` for demo

### Test Files
7. **`tests/test_suite_comprehensive.py`**
   - Added `from config.settings import Settings`
   - Changed to `valid_symbols = settings.BANK_SYMBOLS`

8. **`tests/test_trading_analyzer.py`**
   - Added `from config.settings import Settings`
   - Changed to `valid_symbols = settings.BANK_SYMBOLS`
   - Multiple symbol test uses `symbols = settings.BANK_SYMBOLS[:4]`

9. **`tests_files/test_dashboard_integration.py`**
   - Added `from config.settings import Settings`
   - Changed to `test_symbols = settings.BANK_SYMBOLS[:3]`

10. **`tests_files/test_technical_analysis.py`**
    - Added `from config.settings import Settings`
    - Changed to `test_symbols = settings.BANK_SYMBOLS[:3]`

## Files with Inconsistent Symbols (Still Need Manual Review)

### Archive Files (Legacy - Lower Priority)
- `archive/legacy_root_files/backtesting_system.py` - Uses reduced set ['CBA.AX', 'WBC.AX', 'ANZ.AX', 'NAB.AX']
- `archive/legacy_root_files/generate_dashboard.py` - Missing SUN.AX and QBE.AX
- `archive/news_analysis_dashboard_fixed.py` - Complete canonical list

### Configuration Files
- `tests/test_config.ini` - Uses ["CBA.AX", "WBC.AX", "ANZ.AX", "NAB.AX"] in mock_symbols

### Documentation Files (Low Priority)
- Various markdown files in `archive/docs_archive/` contain old symbol lists in code examples

## Files Already Using Canonical BANK_SYMBOLS
- `src/data_feed.py` - Uses `self.settings.BANK_SYMBOLS`
- `archive/main.py` - Uses `self.settings.BANK_SYMBOLS`
- `archive/src/fundamental_analysis.py` - Uses `self.settings.BANK_SYMBOLS`
- `archive/src/async_main.py` - Uses `self.settings.BANK_SYMBOLS`
- `archive/simple_news_analysis.py` - Uses `self.settings.BANK_SYMBOLS`

## Summary of Changes
- **10 active files** updated to use canonical BANK_SYMBOLS
- **All core system components** now reference the central definition
- **All active test files** updated for consistency
- **Demo files** updated to use canonical symbols
- **Import statements** added where needed for Settings class

## Benefits
1. **Single Source of Truth**: All symbols defined in one place
2. **Easy Maintenance**: Add/remove symbols in one location
3. **Consistency**: No more missing symbols in different parts of the system
4. **Future-Proof**: Easy to expand or modify the symbol list

## Next Steps (Optional)
1. Update `tests/test_config.ini` to reference all 7 symbols
2. Update archive files if they're still in use
3. Update documentation examples to show all 7 symbols
4. Consider creating a validation script to ensure no hardcoded symbol lists remain

## Validation
All active core files, demos, and tests now use the canonical BANK_SYMBOLS definition from `config/settings.py`. The system maintains full functionality while ensuring consistency across all components.
